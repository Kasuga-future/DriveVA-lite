#!/usr/bin/env python3
"""Rebuild suite tables/plots from completed evaluator artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = FRAMEWORK_ROOT.parent
if str(FRAMEWORK_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(1, str(PROJECT_ROOT))

from evaluation.artifacts import jsonable
from evaluation.statistics import aggregate_suite, write_suite_tables
from evaluation.visualization import generate_suite_visualizations


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate an existing VideoTokenPress suite")
    parser.add_argument("suite", type=Path, help="suite root or suite_manifest.json")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args(argv)
    suite = args.suite if args.suite.is_absolute() else FRAMEWORK_ROOT / args.suite
    summary = aggregate_suite(suite)
    root = suite.parent if suite.name == "suite_manifest.json" else suite
    table_paths = write_suite_tables(summary, root / "statistics")
    plot_paths = {} if args.skip_plots else generate_suite_visualizations(summary, root / "visualizations")
    report = {
        "suite_root": str(root.resolve()),
        "run_count": len(summary["run_rows"]),
        "method_count": len(summary["method_rows"]),
        "table_paths": table_paths,
        "plot_paths": plot_paths,
        "method_rows": summary["method_rows"],
    }
    (root / "suite_summary.json").write_text(
        json.dumps(jsonable(report), indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(jsonable(report), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

