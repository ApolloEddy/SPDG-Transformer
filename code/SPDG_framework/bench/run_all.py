import argparse
import os

import torch

from bench.run_microbench import run_microbench
from bench.run_e2e import run_e2e


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all SPDG benchmarks.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    default_output = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "results", "bench"))
    parser.add_argument("--output-dir", default=default_output)
    args = parser.parse_args()

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir

    run_microbench(device, output_dir)
    run_e2e(device, output_dir)


if __name__ == "__main__":
    main()
