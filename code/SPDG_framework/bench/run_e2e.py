import argparse
import json
import os
import time
from typing import Dict, List

import torch

from data_utils import create_dataloader
from full_attention_transformer import FullAttentionTransformer
from fixed_sparse_transformer import FixedSparseTransformer
from spdg_transformer import SPDGTransformer


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict:
    model.eval()
    latencies: List[float] = []
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            _sync(device)
            start = time.perf_counter()
            _ = model(input_ids, attention_mask)
            _sync(device)
            end = time.perf_counter()

            latencies.append(end - start)
            total_tokens += input_ids.numel()

    total_time = sum(latencies)
    return {
        "mean_latency_s": total_time / len(latencies),
        "tokens_per_s": total_tokens / total_time if total_time > 0 else 0.0,
        "num_batches": len(latencies),
    }


def run_e2e(device: torch.device, output_dir: str) -> List[Dict]:
    os.makedirs(output_dir, exist_ok=True)

    seq_len = 256
    batch_size = 8
    vocab_size = 10000

    dataloader = create_dataloader(
        dataset_name="synthetic",
        split="validation",
        batch_size=batch_size,
        shuffle=False,
        difficulty=0.5,
        max_length=seq_len,
        num_samples=128,
    )

    models = {
        "full": FullAttentionTransformer(
            vocab_size=vocab_size,
            d_model=256,
            n_heads=4,
            n_layers=2,
            seq_len=seq_len,
            dropout=0.0,
            n_classes=2,
        ).to(device),
        "fixed": FixedSparseTransformer(
            vocab_size=vocab_size,
            d_model=256,
            n_heads=4,
            n_layers=2,
            seq_len=seq_len,
            sparsity=0.1,
            pattern="local",
            dropout=0.0,
            n_classes=2,
        ).to(device),
        "spdg": SPDGTransformer(
            vocab_size=vocab_size,
            d_model=256,
            n_heads=4,
            n_layers=2,
            seq_len=seq_len,
            sparsity=0.1,
            pattern="local",
            dropout=0.0,
            n_classes=2,
        ).to(device),
    }

    results: List[Dict] = []
    for name, model in models.items():
        metrics = _measure_model(model, dataloader, device)
        metrics.update(
            {
                "model": name,
                "seq_len": seq_len,
                "batch_size": batch_size,
                "dataset": "synthetic",
            }
        )
        results.append(metrics)

    csv_path = os.path.join(output_dir, "e2e_bench.csv")
    json_path = os.path.join(output_dir, "e2e_bench.json")

    import pandas as pd

    pd.DataFrame(results).to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SPDG end-to-end benchmarks.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    default_output = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "results", "bench"))
    parser.add_argument("--output-dir", default=default_output)
    args = parser.parse_args()

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    run_e2e(device, args.output_dir)


if __name__ == "__main__":
    main()
