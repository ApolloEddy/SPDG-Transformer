import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
import tarfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.append(str(CODE_ROOT))

from full_attention_transformer import FullAttentionTransformer
from fixed_sparse_transformer import FixedSparseTransformer
from spdg_transformer import SPDGTransformer


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=120) as response, open(target, "wb") as handle:
        handle.write(response.read())


def _extract_zip(archive_path: Path, output_dir: Path) -> None:
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(output_dir)


def _extract_tar_gz(archive_path: Path, output_dir: Path) -> None:
    with tarfile.open(archive_path, "r:gz") as tf:
        tf.extractall(output_dir)


def _stable_hash_token(token: str, vocab_size: int) -> int:
    digest = hashlib.sha1(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % (vocab_size - 2) + 2


def _tokenize_text(text: str, vocab_size: int, max_length: int) -> Tuple[List[int], List[int]]:
    tokens = re.findall(r"[A-Za-z0-9']+|[^\w\s]", text.lower())
    token_ids = [_stable_hash_token(token, vocab_size) for token in tokens[:max_length]]
    attention = [1] * len(token_ids)

    if len(token_ids) < max_length:
        pad = max_length - len(token_ids)
        token_ids.extend([0] * pad)
        attention.extend([0] * pad)

    return token_ids, attention


@dataclass
class DatasetSpec:
    name: str
    task_type: str
    train_path: Path
    eval_path: Path
    num_classes: int
    max_length: int


class TextClassificationDataset(Dataset):
    def __init__(self, records: Sequence[Tuple[str, int]], vocab_size: int, max_length: int):
        self.records = list(records)
        self.vocab_size = vocab_size
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text, label = self.records[idx]
        input_ids, attention_mask = _tokenize_text(text, self.vocab_size, self.max_length)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def ensure_sst2(root: Path) -> DatasetSpec:
    dataset_dir = root / "data" / "academic_benchmarks" / "SST-2"
    train_path = dataset_dir / "train.tsv"
    eval_path = dataset_dir / "dev.tsv"
    if not train_path.exists() or not eval_path.exists():
        archive_path = dataset_dir.parent / "SST-2.zip"
        if not archive_path.exists():
            _download("https://dl.fbaipublicfiles.com/glue/data/SST-2.zip", archive_path)
        _extract_zip(archive_path, dataset_dir.parent)
    return DatasetSpec(
        name="sst2",
        task_type="sentence_classification",
        train_path=train_path,
        eval_path=eval_path,
        num_classes=2,
        max_length=128,
    )


def ensure_imdb(root: Path) -> DatasetSpec:
    dataset_dir = root / "data" / "academic_benchmarks" / "aclImdb"
    train_path = dataset_dir / "_spdg_train.jsonl"
    eval_path = dataset_dir / "_spdg_test.jsonl"
    if not train_path.exists() or not eval_path.exists():
        archive_path = dataset_dir.parent / "aclImdb_v1.tar.gz"
        if not archive_path.exists():
            _download("https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz", archive_path)
        _extract_tar_gz(archive_path, dataset_dir.parent)
        _materialize_imdb_jsonl(dataset_dir, train_path, eval_path)
    return DatasetSpec(
        name="imdb",
        task_type="document_classification",
        train_path=train_path,
        eval_path=eval_path,
        num_classes=2,
        max_length=512,
    )


def _materialize_imdb_jsonl(dataset_dir: Path, train_path: Path, eval_path: Path) -> None:
    def collect(split: str) -> List[Tuple[str, int]]:
        items: List[Tuple[str, int]] = []
        for label_name, label in (("neg", 0), ("pos", 1)):
            label_dir = dataset_dir / split / label_name
            for file_path in sorted(label_dir.glob("*.txt")):
                items.append((file_path.read_text(encoding="utf-8", errors="ignore"), label))
        return items

    for out_path, split_name in ((train_path, "train"), (eval_path, "test")):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as handle:
            for text, label in collect(split_name):
                handle.write(json.dumps({"text": text, "label": label}, ensure_ascii=False) + "\n")


def load_records(spec: DatasetSpec, split: str, limit: int = 0) -> List[Tuple[str, int]]:
    source = spec.train_path if split == "train" else spec.eval_path
    records: List[Tuple[str, int]] = []

    if spec.name == "sst2":
        with open(source, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                sentence = row["sentence"].strip()
                label = int(row["label"])
                records.append((sentence, label))
                if limit and len(records) >= limit:
                    break
    elif spec.name == "imdb":
        with open(source, "r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                records.append((row["text"], int(row["label"])))
                if limit and len(records) >= limit:
                    break
    else:
        raise ValueError(f"Unsupported dataset: {spec.name}")

    return limit_records(records, limit, seed=42)


def limit_records(records: Sequence[Tuple[str, int]], limit: int, seed: int) -> List[Tuple[str, int]]:
    if not limit or limit >= len(records):
        return list(records)

    grouped: Dict[int, List[Tuple[str, int]]] = {}
    for text, label in records:
        grouped.setdefault(label, []).append((text, label))

    if len(grouped) <= 1:
        return list(records[:limit])

    rng = random.Random(seed)
    for items in grouped.values():
        rng.shuffle(items)

    labels = sorted(grouped)
    selected: List[Tuple[str, int]] = []
    while len(selected) < limit:
        progressed = False
        for label in labels:
            bucket = grouped[label]
            if bucket and len(selected) < limit:
                selected.append(bucket.pop())
                progressed = True
        if not progressed:
            break

    return selected


def build_dataloader(
    spec: DatasetSpec,
    split: str,
    vocab_size: int,
    batch_size: int,
    limit: int,
    shuffle: bool,
) -> DataLoader:
    records = load_records(spec, split, limit)
    dataset = TextClassificationDataset(records, vocab_size=vocab_size, max_length=spec.max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def build_model(model_name: str, vocab_size: int, num_classes: int, seq_len: int, d_model: int, n_heads: int, n_layers: int, sparsity: float, dropout: float) -> nn.Module:
    common = {
        "vocab_size": vocab_size,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "seq_len": seq_len,
        "dropout": dropout,
        "n_classes": num_classes,
    }
    if model_name == "spdg":
        return SPDGTransformer(sparsity=sparsity, pattern="local", **common)
    if model_name == "full":
        return FullAttentionTransformer(**common)
    if model_name == "fixed":
        return FixedSparseTransformer(sparsity=sparsity, pattern="local", **common)
    raise ValueError(f"Unknown model: {model_name}")


def evaluate_model(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    latencies: List[float] = []
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            start = time.perf_counter()
            logits, _, _ = model(input_ids, attention_mask)
            end = time.perf_counter()

            loss = criterion(logits, labels)
            predictions = logits.argmax(dim=-1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            loss_sum += loss.item() * labels.size(0)
            latencies.append(end - start)
            total_tokens += int(attention_mask.sum().item())

    return {
        "accuracy": correct / total if total else 0.0,
        "loss": loss_sum / total if total else 0.0,
        "mean_batch_latency_s": sum(latencies) / len(latencies) if latencies else 0.0,
        "tokens_per_second": total_tokens / sum(latencies) if latencies and sum(latencies) > 0 else 0.0,
        "num_examples": total,
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
) -> Dict[str, object]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        correct = 0
        total = 0
        loss_sum = 0.0
        start_time = time.perf_counter()

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits, _, _ = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            predictions = logits.argmax(dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            loss_sum += loss.item() * labels.size(0)

        train_metrics = {
            "train_accuracy": correct / total if total else 0.0,
            "train_loss": loss_sum / total if total else 0.0,
            "epoch_time_s": time.perf_counter() - start_time,
        }
        eval_metrics = evaluate_model(model, eval_loader, device)
        history.append({**train_metrics, **{f"eval_{k}": v for k, v in eval_metrics.items()}, "epoch": epoch})

    best = max(history, key=lambda item: item["eval_accuracy"])
    return {"history": history, "best": best}


def run_scaling_probe(device: torch.device, vocab_size: int, d_model: int, n_heads: int, n_layers: int, sparsity: float) -> List[Dict[str, float]]:
    seq_lengths = [128, 256, 512, 1024]
    batch_size = 2
    models = ("full", "fixed", "spdg")
    results: List[Dict[str, float]] = []

    for seq_len in seq_lengths:
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        for model_name in models:
            model = build_model(model_name, vocab_size, 2, seq_len, d_model, n_heads, n_layers, sparsity, 0.0).to(device)
            model.eval()
            for _ in range(2):
                with torch.no_grad():
                    model(input_ids, attention_mask)
            times: List[float] = []
            for _ in range(5):
                start = time.perf_counter()
                with torch.no_grad():
                    model(input_ids, attention_mask)
                end = time.perf_counter()
                times.append(end - start)
            results.append(
                {
                    "model": model_name,
                    "seq_len": seq_len,
                    "mean_latency_s": sum(times) / len(times),
                    "tokens_per_second": (batch_size * seq_len) / sum(times),
                }
            )
    return results


def summarize_tradeoff(results: Sequence[Dict[str, object]]) -> Dict[str, object]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in results:
        grouped.setdefault(row["model"], []).append(row)

    summary: Dict[str, object] = {}
    for model_name, rows in grouped.items():
        summary[model_name] = {
            "avg_eval_accuracy": sum(float(item["best_eval_accuracy"]) for item in rows) / len(rows),
            "avg_tokens_per_second": sum(float(item["best_eval_tokens_per_second"]) for item in rows) / len(rows),
            "datasets": [item["dataset"] for item in rows],
        }
    return summary


def save_outputs(output_dir: Path, classification_results: Sequence[Dict[str, object]], scaling_results: Sequence[Dict[str, float]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "classification_results.json", "w", encoding="utf-8") as handle:
        json.dump(list(classification_results), handle, indent=2, ensure_ascii=False)
    with open(output_dir / "scaling_results.json", "w", encoding="utf-8") as handle:
        json.dump(list(scaling_results), handle, indent=2, ensure_ascii=False)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "classification": summarize_tradeoff(classification_results),
                "scaling": scaling_results,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper-style SPDG benchmarks on academic datasets.")
    parser.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--datasets", nargs="+", default=["sst2", "imdb"])
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--train-limit", type=int, default=256)
    parser.add_argument("--eval-limit", type=int, default=256)
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "paper_benchmark"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vocab-size", type=int, default=20000)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--sparsity", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "full":
        if args.epochs == 1:
            args.epochs = 3
        if args.train_limit == 256:
            args.train_limit = 0
        if args.eval_limit == 256:
            args.eval_limit = 0

    seed_everything(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)

    specs = []
    for dataset_name in args.datasets:
        if dataset_name == "sst2":
            specs.append(ensure_sst2(ROOT))
        elif dataset_name == "imdb":
            specs.append(ensure_imdb(ROOT))
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

    classification_results: List[Dict[str, object]] = []
    for spec in specs:
        train_loader = build_dataloader(spec, "train", args.vocab_size, args.batch_size, args.train_limit, True)
        eval_loader = build_dataloader(spec, "eval", args.vocab_size, args.batch_size, args.eval_limit, False)
        for model_name in ("full", "fixed", "spdg"):
            model = build_model(
                model_name,
                args.vocab_size,
                spec.num_classes,
                spec.max_length,
                args.d_model,
                args.n_heads,
                args.n_layers,
                args.sparsity,
                0.1,
            )
            metrics = train_model(model, train_loader, eval_loader, device, args.epochs, args.learning_rate)
            best = metrics["best"]
            classification_results.append(
                {
                    "dataset": spec.name,
                    "task_type": spec.task_type,
                    "model": model_name,
                    "epochs": args.epochs,
                    "train_limit": args.train_limit,
                    "eval_limit": args.eval_limit,
                    "best_epoch": best["epoch"],
                    "best_eval_accuracy": best["eval_accuracy"],
                    "best_eval_loss": best["eval_loss"],
                    "best_eval_tokens_per_second": best["eval_tokens_per_second"],
                    "best_eval_mean_batch_latency_s": best["eval_mean_batch_latency_s"],
                    "best_train_accuracy": best["train_accuracy"],
                    "best_train_loss": best["train_loss"],
                    "history": metrics["history"],
                }
            )

    scaling_results = run_scaling_probe(device, args.vocab_size, args.d_model, args.n_heads, args.n_layers, args.sparsity)
    save_outputs(output_dir, classification_results, scaling_results)
    print(f"Saved paper benchmark results to {output_dir}")


if __name__ == "__main__":
    main()
