import argparse
import csv
import json
import os
import platform
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.append(str(CODE_ROOT))

from full_attention_transformer import FullAttentionTransformer
from fixed_sparse_transformer import FixedSparseTransformer
from spdg_components import SPDGTransformerLayer
from spdg_transformer import SPDGTransformer


GLUE_TASKS: Dict[str, Dict[str, object]] = {
    "sst2": {
        "text_keys": ("sentence",),
        "num_labels": 2,
        "metric_name": "accuracy",
        "max_length": 128,
        "validation_split": "validation",
    },
    "cola": {
        "text_keys": ("sentence",),
        "num_labels": 2,
        "metric_name": "matthews",
        "max_length": 128,
        "validation_split": "validation",
    },
    "mrpc": {
        "text_keys": ("sentence1", "sentence2"),
        "num_labels": 2,
        "metric_name": "f1",
        "max_length": 256,
        "validation_split": "validation",
    },
    "rte": {
        "text_keys": ("sentence1", "sentence2"),
        "num_labels": 2,
        "metric_name": "accuracy",
        "max_length": 256,
        "validation_split": "validation",
    },
    "qnli": {
        "text_keys": ("question", "sentence"),
        "num_labels": 2,
        "metric_name": "accuracy",
        "max_length": 256,
        "validation_split": "validation",
    },
    "mnli": {
        "text_keys": ("premise", "hypothesis"),
        "num_labels": 3,
        "metric_name": "accuracy",
        "max_length": 256,
        "validation_split": "validation_matched",
    },
}


@dataclass
class DatasetTiming:
    task: str
    split: str
    num_examples: int
    dataset_download_s: float
    tokenizer_load_s: float
    tokenization_s: float
    tokenization_examples_per_s: float
    tokenization_tokens_per_s: float
    avg_tokens_per_example: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AutoDL GLUE suite for SPDG-Transformer.")
    parser.add_argument("--mode", choices=["pilot", "full"], default="full")
    parser.add_argument("--tasks", nargs="+", default=["sst2", "cola", "mrpc", "rte", "qnli"])
    parser.add_argument("--ablation-task", default="sst2")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--tokenizer-name", default="bert-base-uncased")
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "autodl_glue_suite"))
    parser.add_argument("--cache-dir", default=str(ROOT / "data" / "hf_cache"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=1024)
    parser.add_argument("--sparsity", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--run-ablation", action="store_true")
    parser.add_argument("--run-scaling", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--save-checkpoints", action="store_true")
    return parser.parse_args()


def apply_mode_defaults(args: argparse.Namespace) -> None:
    if args.mode == "pilot":
        args.epochs = 1
        if args.train_limit == 0:
            args.train_limit = 512
        if args.eval_limit == 0:
            args.eval_limit = 512
        args.seeds = args.seeds[:1]
        args.run_ablation = True
        args.run_scaling = True
    else:
        args.run_ablation = True if not args.run_ablation else args.run_ablation
        args.run_scaling = True if not args.run_scaling else args.run_scaling


def ensure_dirs(output_dir: Path) -> Dict[str, Path]:
    paths = {
        "root": output_dir,
        "metrics": output_dir / "metrics",
        "figures": output_dir / "figures",
        "reports": output_dir / "reports",
        "checkpoints": output_dir / "checkpoints",
        "artifacts": output_dir / "artifacts",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def configure_hf_environment(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_dir / "datasets"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir / "transformers"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def get_peak_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def binary_f1(preds: Sequence[int], labels: Sequence[int]) -> float:
    tp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 1)
    fp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 0)
    fn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 1)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def matthews_corrcoef(preds: Sequence[int], labels: Sequence[int]) -> float:
    tp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 1)
    tn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 0)
    fp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 0)
    fn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 1)
    numerator = (tp * tn) - (fp * fn)
    denominator = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return numerator / denominator if denominator else 0.0


def compute_task_metrics(task_name: str, preds: Sequence[int], labels: Sequence[int]) -> Dict[str, float]:
    correct = sum(1 for p, y in zip(preds, labels) if p == y)
    accuracy = correct / len(labels) if labels else 0.0
    metrics = {"accuracy": accuracy}
    if task_name == "mrpc":
        metrics["f1"] = binary_f1(preds, labels)
    if task_name == "cola":
        metrics["matthews"] = matthews_corrcoef(preds, labels)
    return metrics


def load_dependencies():
    from datasets import load_dataset
    from transformers import AutoTokenizer

    return load_dataset, AutoTokenizer


def save_json(path: Path, payload: object) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def save_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prepare_glue_task(
    task_name: str,
    tokenizer_name: str,
    cache_dir: Path,
    train_limit: int,
    eval_limit: int,
) -> Tuple[Dict[str, object], DatasetTiming, DatasetTiming]:
    if task_name not in GLUE_TASKS:
        raise ValueError(f"Unsupported GLUE task: {task_name}")

    load_dataset, AutoTokenizer = load_dependencies()
    task_cfg = GLUE_TASKS[task_name]

    dataset_start = time.perf_counter()
    dataset = load_dataset("glue", task_name, cache_dir=str(cache_dir / "datasets"))
    dataset_download_s = time.perf_counter() - dataset_start

    tokenizer_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        cache_dir=str(cache_dir / "transformers"),
        use_fast=True,
    )
    tokenizer_load_s = time.perf_counter() - tokenizer_start

    text_keys = task_cfg["text_keys"]
    max_length = int(task_cfg["max_length"])

    def encode_batch(batch: Dict[str, List[object]]) -> Dict[str, object]:
        if len(text_keys) == 1:
            encoded = tokenizer(
                batch[text_keys[0]],
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )
        else:
            encoded = tokenizer(
                batch[text_keys[0]],
                batch[text_keys[1]],
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )
        encoded["labels"] = batch["label"]
        return encoded

    train_dataset = dataset["train"]
    eval_dataset = dataset[str(task_cfg["validation_split"])]

    if train_limit:
        train_dataset = train_dataset.select(range(min(train_limit, len(train_dataset))))
    if eval_limit:
        eval_dataset = eval_dataset.select(range(min(eval_limit, len(eval_dataset))))

    train_start = time.perf_counter()
    train_tokenized = train_dataset.map(encode_batch, batched=True, remove_columns=train_dataset.column_names)
    train_tokenization_s = time.perf_counter() - train_start

    eval_start = time.perf_counter()
    eval_tokenized = eval_dataset.map(encode_batch, batched=True, remove_columns=eval_dataset.column_names)
    eval_tokenization_s = time.perf_counter() - eval_start

    train_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    eval_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    train_tokens = int(sum(int(example["attention_mask"].sum().item()) for example in train_tokenized))
    eval_tokens = int(sum(int(example["attention_mask"].sum().item()) for example in eval_tokenized))

    train_timing = DatasetTiming(
        task=task_name,
        split="train",
        num_examples=len(train_tokenized),
        dataset_download_s=dataset_download_s,
        tokenizer_load_s=tokenizer_load_s,
        tokenization_s=train_tokenization_s,
        tokenization_examples_per_s=(len(train_tokenized) / train_tokenization_s) if train_tokenization_s else 0.0,
        tokenization_tokens_per_s=(train_tokens / train_tokenization_s) if train_tokenization_s else 0.0,
        avg_tokens_per_example=(train_tokens / len(train_tokenized)) if len(train_tokenized) else 0.0,
    )
    eval_timing = DatasetTiming(
        task=task_name,
        split="validation",
        num_examples=len(eval_tokenized),
        dataset_download_s=dataset_download_s,
        tokenizer_load_s=tokenizer_load_s,
        tokenization_s=eval_tokenization_s,
        tokenization_examples_per_s=(len(eval_tokenized) / eval_tokenization_s) if eval_tokenization_s else 0.0,
        tokenization_tokens_per_s=(eval_tokens / eval_tokenization_s) if eval_tokenization_s else 0.0,
        avg_tokens_per_example=(eval_tokens / len(eval_tokenized)) if len(eval_tokenized) else 0.0,
    )

    return {
        "task_name": task_name,
        "max_length": max_length,
        "num_labels": int(task_cfg["num_labels"]),
        "metric_name": str(task_cfg["metric_name"]),
        "train_dataset": train_tokenized,
        "eval_dataset": eval_tokenized,
    }, train_timing, eval_timing


def build_dataloader(dataset, batch_size: int, shuffle: bool, num_workers: int, device: torch.device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )


def build_variant_layers(
    model: SPDGTransformer,
    variant: str,
    seq_len: int,
    d_model: int,
    n_heads: int,
    dim_feedforward: int,
    dropout: float,
    sparsity: float,
) -> SPDGTransformer:
    layers = []
    for _ in range(model.n_layers):
        if variant == "random_prior":
            layer = SPDGTransformerLayer(
                d_model=d_model,
                n_heads=n_heads,
                seq_len=seq_len,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                sparsity=sparsity,
                pattern="random",
            )
        elif variant == "uncalibrated":
            layer = SPDGTransformerLayer(
                d_model=d_model,
                n_heads=n_heads,
                seq_len=seq_len,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                sparsity=sparsity,
                pattern="local",
                calibrate=False,
            )
        else:
            raise ValueError(f"Unsupported ablation variant: {variant}")
        layers.append(layer)
    model.layers = nn.ModuleList(layers)
    return model


def build_model(
    model_name: str,
    vocab_size: int,
    num_labels: int,
    seq_len: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    dim_feedforward: int,
    dropout: float,
    sparsity: float,
) -> nn.Module:
    common = {
        "vocab_size": vocab_size,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "seq_len": seq_len,
        "dropout": dropout,
        "n_classes": num_labels,
        "dim_feedforward": dim_feedforward,
    }
    if model_name == "full":
        return FullAttentionTransformer(**common)
    if model_name == "fixed":
        return FixedSparseTransformer(sparsity=sparsity, pattern="local", **common)
    if model_name == "spdg":
        return SPDGTransformer(sparsity=sparsity, pattern="local", **common)
    if model_name in {"random_prior", "uncalibrated"}:
        return build_variant_layers(
            SPDGTransformer(sparsity=sparsity, pattern="local", **common),
            model_name,
            seq_len,
            d_model,
            n_heads,
            dim_feedforward,
            dropout,
            sparsity,
        )
    raise ValueError(f"Unknown model name: {model_name}")


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    task_name: str,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_examples = 0
    latencies: List[float] = []
    labels_all: List[int] = []
    preds_all: List[int] = []
    total_tokens = 0

    reset_peak_memory(device)
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            synchronize(device)
            start = time.perf_counter()
            logits, _, _ = model(input_ids, attention_mask)
            synchronize(device)
            end = time.perf_counter()

            loss = criterion(logits, labels)
            preds = logits.argmax(dim=-1)

            total_loss += loss.item() * labels.size(0)
            total_examples += labels.size(0)
            total_tokens += int(attention_mask.sum().item())
            latencies.append(end - start)
            preds_all.extend(preds.cpu().tolist())
            labels_all.extend(labels.cpu().tolist())

    metrics = compute_task_metrics(task_name, preds_all, labels_all)
    metrics.update(
        {
            "loss": (total_loss / total_examples) if total_examples else 0.0,
            "mean_batch_latency_s": (sum(latencies) / len(latencies)) if latencies else 0.0,
            "tokens_per_second": (total_tokens / sum(latencies)) if latencies and sum(latencies) else 0.0,
            "examples_per_second": (total_examples / sum(latencies)) if latencies and sum(latencies) else 0.0,
            "peak_memory_mb": get_peak_memory_mb(device),
        }
    )
    return metrics


def train_single_run(
    model: nn.Module,
    task_name: str,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    gradient_accumulation: int,
    fp16: bool,
    checkpoint_path: Optional[Path],
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(fp16 and device.type == "cuda"))
    model.to(device)

    history: List[Dict[str, object]] = []
    best_epoch_record: Optional[Dict[str, object]] = None
    primary_metric_name = str(GLUE_TASKS[task_name]["metric_name"])

    for epoch in range(1, epochs + 1):
        model.train()
        reset_peak_memory(device)
        epoch_start = time.perf_counter()
        train_loss_sum = 0.0
        train_examples = 0
        train_correct = 0
        total_tokens = 0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(fp16 and device.type == "cuda")):
                logits, _, _ = model(input_ids, attention_mask)
                loss = criterion(logits, labels) / gradient_accumulation

            scaler.scale(loss).backward()

            if step % gradient_accumulation == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            preds = logits.argmax(dim=-1)
            batch_loss = loss.item() * gradient_accumulation
            train_loss_sum += batch_loss * labels.size(0)
            train_examples += labels.size(0)
            train_correct += (preds == labels).sum().item()
            total_tokens += int(attention_mask.sum().item())

        if train_loader and (len(train_loader) % gradient_accumulation) != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        epoch_time_s = time.perf_counter() - epoch_start
        train_peak_memory_mb = get_peak_memory_mb(device)
        eval_metrics = evaluate_model(model, eval_loader, task_name, device)

        record = {
            "epoch": epoch,
            "train_loss": (train_loss_sum / train_examples) if train_examples else 0.0,
            "train_accuracy": (train_correct / train_examples) if train_examples else 0.0,
            "train_examples_per_second": (train_examples / epoch_time_s) if epoch_time_s else 0.0,
            "train_tokens_per_second": (total_tokens / epoch_time_s) if epoch_time_s else 0.0,
            "epoch_time_s": epoch_time_s,
            "train_peak_memory_mb": train_peak_memory_mb,
            "eval_accuracy": eval_metrics.get("accuracy", 0.0),
            "eval_f1": eval_metrics.get("f1", 0.0),
            "eval_matthews": eval_metrics.get("matthews", 0.0),
            "eval_loss": eval_metrics["loss"],
            "eval_mean_batch_latency_s": eval_metrics["mean_batch_latency_s"],
            "eval_tokens_per_second": eval_metrics["tokens_per_second"],
            "eval_examples_per_second": eval_metrics["examples_per_second"],
            "eval_peak_memory_mb": eval_metrics["peak_memory_mb"],
        }
        history.append(record)

        if best_epoch_record is None or record[f"eval_{primary_metric_name}"] > best_epoch_record[f"eval_{primary_metric_name}"]:
            best_epoch_record = dict(record)
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "metrics": record}, checkpoint_path)

    if best_epoch_record is None:
        raise RuntimeError("Training produced no records.")
    return history, best_epoch_record


def run_scaling_probe(
    device: torch.device,
    output_dir: Path,
    d_model: int,
    n_heads: int,
    n_layers: int,
    dim_feedforward: int,
    sparsity: float,
) -> List[Dict[str, object]]:
    seq_lengths = [128, 256, 512, 1024, 2048]
    results: List[Dict[str, object]] = []
    vocab_size = 30522
    batch_size = 4 if device.type == "cuda" else 2

    for seq_len in seq_lengths:
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        for model_name in ("full", "fixed", "spdg"):
            model = build_model(
                model_name=model_name,
                vocab_size=vocab_size,
                num_labels=2,
                seq_len=seq_len,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                dim_feedforward=dim_feedforward,
                dropout=0.0,
                sparsity=sparsity,
            ).to(device)
            model.eval()

            for _ in range(3):
                with torch.no_grad():
                    model(input_ids, attention_mask)

            reset_peak_memory(device)
            latencies: List[float] = []
            for _ in range(10):
                synchronize(device)
                start = time.perf_counter()
                with torch.no_grad():
                    model(input_ids, attention_mask)
                synchronize(device)
                end = time.perf_counter()
                latencies.append(end - start)

            total_time = sum(latencies)
            results.append(
                {
                    "model": model_name,
                    "seq_len": seq_len,
                    "batch_size": batch_size,
                    "mean_latency_s": total_time / len(latencies),
                    "tokens_per_second": (batch_size * seq_len * len(latencies)) / total_time if total_time else 0.0,
                    "peak_memory_mb": get_peak_memory_mb(device),
                }
            )

    save_csv(output_dir / "scaling_probe.csv", results)
    save_json(output_dir / "scaling_probe.json", results)
    return results


def summarize_runs(run_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in run_rows:
        grouped[(str(row["task"]), str(row["model"]))].append(row)

    summary_rows: List[Dict[str, object]] = []
    for (task, model), rows in sorted(grouped.items()):
        primary_metric_name = str(GLUE_TASKS[task]["metric_name"])
        summary_rows.append(
            {
                "task": task,
                "model": model,
                "num_runs": len(rows),
                "avg_primary_metric": sum(float(row[f"best_eval_{primary_metric_name}"]) for row in rows) / len(rows),
                "avg_accuracy": sum(float(row["best_eval_accuracy"]) for row in rows) / len(rows),
                "avg_loss": sum(float(row["best_eval_loss"]) for row in rows) / len(rows),
                "avg_eval_tokens_per_second": sum(float(row["best_eval_tokens_per_second"]) for row in rows) / len(rows),
                "avg_eval_examples_per_second": sum(float(row["best_eval_examples_per_second"]) for row in rows) / len(rows),
                "avg_epoch_time_s": sum(float(row["best_epoch_time_s"]) for row in rows) / len(rows),
                "avg_eval_latency_s": sum(float(row["best_eval_mean_batch_latency_s"]) for row in rows) / len(rows),
                "avg_eval_peak_memory_mb": sum(float(row["best_eval_peak_memory_mb"]) for row in rows) / len(rows),
            }
        )
    return summary_rows


def generate_figures(output_dir: Path) -> List[str]:
    import matplotlib.pyplot as plt
    import pandas as pd

    metrics_dir = output_dir / "metrics"
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_paths: List[str] = []

    main_csv = metrics_dir / "main_runs.csv"
    summary_csv = metrics_dir / "main_summary.csv"
    if main_csv.exists() and summary_csv.exists():
        summary = pd.read_csv(summary_csv)

        fig, ax = plt.subplots(figsize=(12, 6))
        summary.pivot(index="task", columns="model", values="avg_primary_metric").plot(kind="bar", ax=ax)
        ax.set_ylabel("Primary GLUE Metric")
        ax.set_title("GLUE Primary Metric by Task")
        plt.tight_layout()
        path = figures_dir / "glue_primary_metric.png"
        plt.savefig(path, dpi=200)
        plt.close()
        figure_paths.append(str(path))

        fig, ax = plt.subplots(figsize=(12, 6))
        summary.pivot(index="task", columns="model", values="avg_eval_tokens_per_second").plot(kind="bar", ax=ax)
        ax.set_ylabel("Eval Tokens/s")
        ax.set_title("Evaluation Throughput by Task")
        plt.tight_layout()
        path = figures_dir / "glue_eval_throughput.png"
        plt.savefig(path, dpi=200)
        plt.close()
        figure_paths.append(str(path))

        fig, ax = plt.subplots(figsize=(8, 6))
        for model in sorted(summary["model"].unique()):
            model_df = summary[summary["model"] == model]
            ax.scatter(model_df["avg_eval_latency_s"], model_df["avg_primary_metric"], s=120, label=model)
            for _, row in model_df.iterrows():
                ax.annotate(row["task"], (row["avg_eval_latency_s"], row["avg_primary_metric"]))
        ax.set_xlabel("Eval Mean Batch Latency (s)")
        ax.set_ylabel("Primary GLUE Metric")
        ax.set_title("Accuracy-Latency Tradeoff")
        ax.legend()
        plt.tight_layout()
        path = figures_dir / "accuracy_latency_tradeoff.png"
        plt.savefig(path, dpi=200)
        plt.close()
        figure_paths.append(str(path))

    tokenization_csv = metrics_dir / "tokenization_timing.csv"
    if tokenization_csv.exists():
        token_df = pd.read_csv(tokenization_csv)

        fig, ax = plt.subplots(figsize=(12, 6))
        token_df.pivot(index="task", columns="split", values="tokenization_tokens_per_s").plot(kind="bar", ax=ax)
        ax.set_ylabel("Tokenization Tokens/s")
        ax.set_title("Tokenizer Throughput by Task")
        plt.tight_layout()
        path = figures_dir / "tokenization_speed.png"
        plt.savefig(path, dpi=200)
        plt.close()
        figure_paths.append(str(path))

        fig, ax = plt.subplots(figsize=(12, 6))
        prep = token_df.groupby("task", as_index=False)[["dataset_download_s", "tokenizer_load_s", "tokenization_s"]].max()
        prep.set_index("task").plot(kind="bar", stacked=True, ax=ax)
        ax.set_ylabel("Seconds")
        ax.set_title("Dataset Preparation Time Breakdown")
        plt.tight_layout()
        path = figures_dir / "dataset_prepare_time.png"
        plt.savefig(path, dpi=200)
        plt.close()
        figure_paths.append(str(path))

    ablation_csv = metrics_dir / "ablation_runs.csv"
    if ablation_csv.exists():
        ablation_df = pd.read_csv(ablation_csv)
        metric_col = "best_eval_accuracy"
        if "best_eval_f1" in ablation_df.columns and ablation_df["best_eval_f1"].fillna(0).max() > 0:
            metric_col = "best_eval_f1"

        fig, ax = plt.subplots(figsize=(10, 6))
        ablation_df.groupby("model", as_index=False)[metric_col].mean().plot(kind="bar", x="model", y=metric_col, ax=ax, legend=False)
        ax.set_ylabel(metric_col)
        ax.set_title("Ablation Performance")
        plt.tight_layout()
        path = figures_dir / "ablation_performance.png"
        plt.savefig(path, dpi=200)
        plt.close()
        figure_paths.append(str(path))

    scaling_csv = metrics_dir / "scaling_probe.csv"
    if scaling_csv.exists():
        scaling_df = pd.read_csv(scaling_csv)

        fig, ax = plt.subplots(figsize=(10, 6))
        for model in scaling_df["model"].unique():
            part = scaling_df[scaling_df["model"] == model]
            ax.plot(part["seq_len"], part["mean_latency_s"], marker="o", label=model)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=10)
        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Mean Latency (s)")
        ax.set_title("Scaling Probe Latency")
        ax.legend()
        plt.tight_layout()
        path = figures_dir / "scaling_latency.png"
        plt.savefig(path, dpi=200)
        plt.close()
        figure_paths.append(str(path))

        fig, ax = plt.subplots(figsize=(10, 6))
        for model in scaling_df["model"].unique():
            part = scaling_df[scaling_df["model"] == model]
            ax.plot(part["seq_len"], part["tokens_per_second"], marker="o", label=model)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Tokens/s")
        ax.set_title("Scaling Probe Throughput")
        ax.legend()
        plt.tight_layout()
        path = figures_dir / "scaling_throughput.png"
        plt.savefig(path, dpi=200)
        plt.close()
        figure_paths.append(str(path))

        if scaling_df["peak_memory_mb"].max() > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            for model in scaling_df["model"].unique():
                part = scaling_df[scaling_df["model"] == model]
                ax.plot(part["seq_len"], part["peak_memory_mb"], marker="o", label=model)
            ax.set_xscale("log", base=2)
            ax.set_xlabel("Sequence Length")
            ax.set_ylabel("Peak Memory (MB)")
            ax.set_title("Scaling Probe Peak Memory")
            ax.legend()
            plt.tight_layout()
            path = figures_dir / "scaling_peak_memory.png"
            plt.savefig(path, dpi=200)
            plt.close()
            figure_paths.append(str(path))

    return figure_paths


def build_environment_record(args: argparse.Namespace, device: torch.device) -> Dict[str, object]:
    record = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "cwd": str(ROOT),
        "args": vars(args),
        "env": {
            "HF_HOME": os.environ.get("HF_HOME"),
            "HF_DATASETS_CACHE": os.environ.get("HF_DATASETS_CACHE"),
            "TRANSFORMERS_CACHE": os.environ.get("TRANSFORMERS_CACHE"),
            "HF_ENDPOINT": os.environ.get("HF_ENDPOINT"),
            "HTTP_PROXY": os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy"),
            "HTTPS_PROXY": os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }
    if torch.cuda.is_available():
        record["gpu_name"] = torch.cuda.get_device_name(0)
        record["gpu_count"] = torch.cuda.device_count()
        record["cuda_version"] = torch.version.cuda
    return record


def generate_markdown_report(
    output_dir: Path,
    environment: Dict[str, object],
    tokenization_rows: Sequence[Dict[str, object]],
    main_summary_rows: Sequence[Dict[str, object]],
    ablation_rows: Sequence[Dict[str, object]],
    figure_paths: Sequence[str],
) -> Path:
    report_path = output_dir / "reports" / "autodl_glue_report.md"
    lines: List[str] = []
    lines.append("# AutoDL GLUE Suite Report")
    lines.append("")
    lines.append("## Environment")
    lines.append("")
    lines.append(f"- Device: `{environment.get('device')}`")
    lines.append(f"- Torch: `{environment.get('torch_version')}`")
    lines.append(f"- CUDA Available: `{environment.get('cuda_available')}`")
    if environment.get("gpu_name"):
        lines.append(f"- GPU: `{environment.get('gpu_name')}`")
    lines.append(f"- HF Cache: `{environment['env'].get('HF_HOME')}`")
    lines.append("")

    if tokenization_rows:
        lines.append("## Data Preparation")
        lines.append("")
        lines.append("| Task | Split | Examples | Download(s) | Tokenizer(s) | Tokenize(s) | Tok/s | Avg Tokens |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in tokenization_rows:
            lines.append(
                f"| {row['task']} | {row['split']} | {row['num_examples']} | "
                f"{row['dataset_download_s']:.2f} | {row['tokenizer_load_s']:.2f} | {row['tokenization_s']:.2f} | "
                f"{row['tokenization_tokens_per_s']:.2f} | {row['avg_tokens_per_example']:.2f} |"
            )
        lines.append("")

    if main_summary_rows:
        lines.append("## Main GLUE Results")
        lines.append("")
        lines.append("| Task | Model | Runs | Primary Metric | Accuracy | Eval Tok/s | Eval Latency(s) | Peak Mem(MB) |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in main_summary_rows:
            lines.append(
                f"| {row['task']} | {row['model']} | {row['num_runs']} | {row['avg_primary_metric']:.4f} | "
                f"{row['avg_accuracy']:.4f} | {row['avg_eval_tokens_per_second']:.2f} | "
                f"{row['avg_eval_latency_s']:.4f} | {row['avg_eval_peak_memory_mb']:.2f} |"
            )
        lines.append("")

    if ablation_rows:
        lines.append("## Ablation")
        lines.append("")
        lines.append("| Task | Model | Seed | Accuracy | F1 | Matthews | Eval Tok/s |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
        for row in ablation_rows:
            lines.append(
                f"| {row['task']} | {row['model']} | {row['seed']} | {row['best_eval_accuracy']:.4f} | "
                f"{row['best_eval_f1']:.4f} | {row['best_eval_matthews']:.4f} | {row['best_eval_tokens_per_second']:.2f} |"
            )
        lines.append("")

    if figure_paths:
        lines.append("## Figures")
        lines.append("")
        for figure_path in figure_paths:
            lines.append(f"- `{Path(figure_path).name}`")
        lines.append("")

    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return report_path


def main() -> None:
    args = parse_args()
    apply_mode_defaults(args)

    output_dir = Path(args.output_dir)
    ensure_dirs(output_dir)
    configure_hf_environment(Path(args.cache_dir))
    device = resolve_device(args.device)
    environment = build_environment_record(args, device)
    save_json(output_dir / "artifacts" / "environment.json", environment)

    tokenization_rows: List[Dict[str, object]] = []
    per_run_rows: List[Dict[str, object]] = []
    per_epoch_rows: List[Dict[str, object]] = []
    ablation_rows: List[Dict[str, object]] = []

    total_start = time.perf_counter()
    for task_name in args.tasks:
        task_bundle, train_timing, eval_timing = prepare_glue_task(
            task_name=task_name,
            tokenizer_name=args.tokenizer_name,
            cache_dir=Path(args.cache_dir),
            train_limit=args.train_limit,
            eval_limit=args.eval_limit,
        )
        tokenization_rows.extend([asdict(train_timing), asdict(eval_timing)])

        train_loader = build_dataloader(task_bundle["train_dataset"], args.batch_size, True, args.num_workers, device)
        eval_loader = build_dataloader(task_bundle["eval_dataset"], args.eval_batch_size, False, args.num_workers, device)

        for seed in args.seeds:
            seed_everything(seed)
            for model_name in ("full", "fixed", "spdg"):
                checkpoint_path = None
                if args.save_checkpoints:
                    checkpoint_path = output_dir / "checkpoints" / f"{task_name}_{model_name}_seed{seed}.pt"
                model = build_model(
                    model_name=model_name,
                    vocab_size=30522,
                    num_labels=int(task_bundle["num_labels"]),
                    seq_len=int(task_bundle["max_length"]),
                    d_model=args.d_model,
                    n_heads=args.n_heads,
                    n_layers=args.n_layers,
                    dim_feedforward=args.dim_feedforward,
                    dropout=args.dropout,
                    sparsity=args.sparsity,
                )
                history, best = train_single_run(
                    model=model,
                    task_name=task_name,
                    train_loader=train_loader,
                    eval_loader=eval_loader,
                    device=device,
                    epochs=args.epochs,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    gradient_accumulation=args.gradient_accumulation,
                    fp16=args.fp16,
                    checkpoint_path=checkpoint_path,
                )
                for row in history:
                    row.update({"task": task_name, "model": model_name, "seed": seed})
                    per_epoch_rows.append(row)
                per_run_rows.append(
                    {
                        "task": task_name,
                        "model": model_name,
                        "seed": seed,
                        "best_epoch": best["epoch"],
                        "best_eval_accuracy": best["eval_accuracy"],
                        "best_eval_f1": best["eval_f1"],
                        "best_eval_matthews": best["eval_matthews"],
                        "best_eval_loss": best["eval_loss"],
                        "best_eval_tokens_per_second": best["eval_tokens_per_second"],
                        "best_eval_examples_per_second": best["eval_examples_per_second"],
                        "best_eval_mean_batch_latency_s": best["eval_mean_batch_latency_s"],
                        "best_epoch_time_s": best["epoch_time_s"],
                        "best_eval_peak_memory_mb": best["eval_peak_memory_mb"],
                    }
                )

    if args.run_ablation:
        ablation_task = args.ablation_task
        task_bundle, _, _ = prepare_glue_task(
            task_name=ablation_task,
            tokenizer_name=args.tokenizer_name,
            cache_dir=Path(args.cache_dir),
            train_limit=args.train_limit,
            eval_limit=args.eval_limit,
        )
        train_loader = build_dataloader(task_bundle["train_dataset"], args.batch_size, True, args.num_workers, device)
        eval_loader = build_dataloader(task_bundle["eval_dataset"], args.eval_batch_size, False, args.num_workers, device)
        for seed in args.seeds:
            seed_everything(seed)
            for model_name in ("full", "fixed", "spdg", "random_prior", "uncalibrated"):
                model = build_model(
                    model_name=model_name,
                    vocab_size=30522,
                    num_labels=int(task_bundle["num_labels"]),
                    seq_len=int(task_bundle["max_length"]),
                    d_model=args.d_model,
                    n_heads=args.n_heads,
                    n_layers=args.n_layers,
                    dim_feedforward=args.dim_feedforward,
                    dropout=args.dropout,
                    sparsity=args.sparsity,
                )
                _, best = train_single_run(
                    model=model,
                    task_name=ablation_task,
                    train_loader=train_loader,
                    eval_loader=eval_loader,
                    device=device,
                    epochs=args.epochs,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    gradient_accumulation=args.gradient_accumulation,
                    fp16=args.fp16,
                    checkpoint_path=None,
                )
                ablation_rows.append(
                    {
                        "task": ablation_task,
                        "model": model_name,
                        "seed": seed,
                        "best_epoch": best["epoch"],
                        "best_eval_accuracy": best["eval_accuracy"],
                        "best_eval_f1": best["eval_f1"],
                        "best_eval_matthews": best["eval_matthews"],
                        "best_eval_tokens_per_second": best["eval_tokens_per_second"],
                        "best_eval_mean_batch_latency_s": best["eval_mean_batch_latency_s"],
                    }
                )

    metrics_dir = output_dir / "metrics"
    save_csv(metrics_dir / "tokenization_timing.csv", tokenization_rows)
    save_csv(metrics_dir / "training_history.csv", per_epoch_rows)
    save_csv(metrics_dir / "main_runs.csv", per_run_rows)
    main_summary_rows = summarize_runs(per_run_rows)
    save_csv(metrics_dir / "main_summary.csv", main_summary_rows)
    if ablation_rows:
        save_csv(metrics_dir / "ablation_runs.csv", ablation_rows)

    scaling_rows: List[Dict[str, object]] = []
    if args.run_scaling:
        scaling_rows = run_scaling_probe(
            device=device,
            output_dir=metrics_dir,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dim_feedforward=args.dim_feedforward,
            sparsity=args.sparsity,
        )

    save_json(
        output_dir / "artifacts" / "run_summary.json",
        {
            "environment": environment,
            "total_runtime_s": time.perf_counter() - total_start,
            "tokenization": tokenization_rows,
            "main_runs": per_run_rows,
            "main_summary": main_summary_rows,
            "ablation_runs": ablation_rows,
            "scaling_runs": scaling_rows,
        },
    )

    figure_paths = generate_figures(output_dir)
    report_path = generate_markdown_report(output_dir, environment, tokenization_rows, main_summary_rows, ablation_rows, figure_paths)

    print(f"AutoDL GLUE suite finished. Outputs: {output_dir}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
