"""Launch the NAVSIM baseline trainer from an explicit JSON configuration."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from examples.wanvideo.driveva_train.train_navsim_v1 import main as train_main


def _argv(config: dict) -> list[str]:
    argv: list[str] = []
    for key, value in config.items():
        flag = f"--{key}"
        if value is None or value is False:
            continue
        if value is True:
            argv.append(flag)
        else:
            argv.extend([flag, str(value)])
    return argv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sanity-steps", type=int, default=None)
    args = parser.parse_args()
    document = json.loads(Path(args.config).read_text(encoding="utf-8"))
    train = dict(document["train"])
    if args.sanity_steps is not None:
        train["max_optimizer_steps"] = args.sanity_steps
        train["save_steps"] = args.sanity_steps
        train["output_path"] = str(Path(train["output_path"]) / f"sanity_{args.sanity_steps}")
        train["train_log_file"] = "train.log"
    return train_main(_argv(train))


if __name__ == "__main__":
    raise SystemExit(main())
