#!/usr/bin/env python3
"""Small config sweep helper for the CPU smoke backend."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/press/synthetic_smoke.yaml"))
    parser.add_argument("--methods", nargs="+", default=["random", "token_norm"])
    parser.add_argument("--K", nargs="+", type=int, default=[6])
    parser.add_argument("--operator", default=None)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/press_sweep"))
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, str(root / "scripts" / "evaluate_press.py"), "--config", str((root / args.config).resolve())]
    for method in args.methods:
        for k in args.K:
            output = (root / args.output_root / f"{method}_k{k}").resolve()
            subprocess.run(command + ["--press", method, "--budget", str(k), "--output-dir", str(output)], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
