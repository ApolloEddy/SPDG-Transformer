import argparse
import json
import os
import time
from typing import List, Dict

import torch

from full_attention_transformer import FullAttentionTransformer
from fixed_sparse_transformer import FixedSparseTransformer
from spdg_transformer import SPDGTransformer


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_forward(model: torch.nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor, runs: int, warmup: int) -> float:
    model.eval()
    times: List[float] = []
    with torch.no_grad():
        for i in range(runs + warmup):
            _sync(input_ids.device)
            start = time.perf_counter()
            _ = model(input_ids, attention_mask)
            _sync(input_ids.device)
            end = time.perf_counter()
            if i >= warmup:
                times.append(end - start)
    return sum(times) / len(times)


def run_microbench(device: torch.device, output_dir: str) -> List[Dict]:
    os.makedirs(output_dir, exist_ok=True)

    configs = [
        {"seq_len": 128, "batch_size": 4, "n_heads": 4, "n_layers": 2, "d_model": 256},
        {"seq_len": 256, "batch_size": 4, "n_heads": 4, "n_layers": 2, "d_model": 256},
        {"seq_len": 256, "batch_size": 8, "n_heads": 8, "n_layers": 4, "d_model": 512},
    ]

    results: List[Dict] = []

    for cfg in configs:
        seq_len = cfg["seq_len"]
        batch_size = cfg["batch_size"]
        vocab_size = 10000

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        attention_mask = torch.ones(batch_size, seq_len, device=device)

        models = {
            "full": FullAttentionTransformer(
                vocab_size=vocab_size,
                d_model=cfg["d_model"],
                n_heads=cfg["n_heads"],
                n_layers=cfg["n_layers"],
                seq_len=seq_len,
                dropout=0.0,
                n_classes=2,
            ).to(device),
            "fixed": FixedSparseTransformer(
                vocab_size=vocab_size,
                d_model=cfg["d_model"],
                n_heads=cfg["n_heads"],
                n_layers=cfg["n_layers"],
                seq_len=seq_len,
                sparsity=0.1,
                pattern="local",
                dropout=0.0,
                n_classes=2,
            ).to(device),
            "spdg": SPDGTransformer(
                vocab_size=vocab_size,
                d_model=cfg["d_model"],
                n_heads=cfg["n_heads"],
                n_layers=cfg["n_layers"],
                seq_len=seq_len,
                sparsity=0.1,
                pattern="local",
                dropout=0.0,
                n_classes=2,
            ).to(device),
        }

        for name, model in models.items():
            mean_time = _time_forward(model, input_ids, attention_mask, runs=10, warmup=3)
            tokens = batch_size * seq_len
            throughput = tokens / mean_time

            results.append(
                {
                    "model": name,
                    "seq_len": seq_len,
                    "batch_size": batch_size,
                    "n_heads": cfg["n_heads"],
                    "n_layers": cfg["n_layers"],
                    "d_model": cfg["d_model"],
                    "mean_latency_s": mean_time,
                    "tokens_per_s": throughput,
                }
            )

    csv_path = os.path.join(output_dir, "microbench.csv")
    json_path = os.path.join(output_dir, "microbench.json")

    import pandas as pd

    pd.DataFrame(results).to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SPDG microbenchmarks.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    default_output = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "results", "bench"))
    parser.add_argument("--output-dir", default=default_output)
    args = parser.parse_args()

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    run_microbench(device, args.output_dir)


if __name__ == "__main__":
    main()
