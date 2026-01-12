import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def _plot_microbench(csv_path: str, output_path: str) -> None:
    df = pd.read_csv(csv_path)
    df["config"] = (
        "S"
        + df["seq_len"].astype(str)
        + "_B"
        + df["batch_size"].astype(str)
        + "_H"
        + df["n_heads"].astype(str)
        + "_L"
        + df["n_layers"].astype(str)
    )

    pivot = df.pivot(index="config", columns="model", values="tokens_per_s")
    pivot.plot(kind="bar", figsize=(10, 4))
    plt.ylabel("Tokens/s")
    plt.title("Microbench Throughput")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_e2e(csv_path: str, output_path: str) -> None:
    df = pd.read_csv(csv_path)
    df.plot(x="model", y="tokens_per_s", kind="bar", figsize=(6, 4), legend=False)
    plt.ylabel("Tokens/s")
    plt.title("End-to-End Throughput (Synthetic)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot benchmark results.")
    parser.add_argument("--bench-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    micro_csv = os.path.join(args.bench_dir, "microbench.csv")
    e2e_csv = os.path.join(args.bench_dir, "e2e_bench.csv")

    _plot_microbench(micro_csv, os.path.join(args.output_dir, "microbench_tokens_per_s.png"))
    _plot_e2e(e2e_csv, os.path.join(args.output_dir, "e2e_tokens_per_s.png"))


if __name__ == "__main__":
    main()
