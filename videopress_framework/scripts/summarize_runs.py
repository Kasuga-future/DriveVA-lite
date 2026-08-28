#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args(argv)
    rows = []
    for run in args.runs:
        path = run / "summary.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"run": str(run), "press": value.get("press", {}).get("name"), "mode": value.get("mode"), "pdm": value.get("pdm"), "n_scenes": value.get("n_scenes")})
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
