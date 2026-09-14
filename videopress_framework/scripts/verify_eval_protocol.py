#!/usr/bin/env python3
"""Label a completed evaluation with its protocol and check it did not drift.

Absolute PDM is only comparable to published NAVSIM numbers on the primary
protocol ``navtest-7876`` (7,876 scenes, no-press 0.909839).  The project once
silently evaluated on the harder ``split-test-1920`` subset, which moved every
absolute number by about one point while leaving paired deltas unchanged.  See
``reports/eval_protocol_baseline_discrepancy_20260911.md``.

This script makes that class of mistake detectable after the fact:

* it recovers the protocol from the data paths recorded in the run,
* it compares the evaluated scene count with the protocol's expected count and
  exits non-zero on a mismatch,
* it prints every absolute PDM with its protocol label attached, next to the
  protocol's reference baselines, so a number can never be quoted bare.
"""

from __future__ import annotations

import argparse
import csv
import glob
import importlib.util
import json
import statistics
import sys
from pathlib import Path


FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FRAMEWORK_ROOT / "scripts"))

_spec = importlib.util.spec_from_file_location(
    "run_official_navsim_press", FRAMEWORK_ROOT / "scripts" / "run_official_navsim_press.py"
)
press = importlib.util.module_from_spec(_spec)
sys.modules["run_official_navsim_press"] = press
_spec.loader.exec_module(press)


def _recorded(run_root: Path) -> tuple[dict, str | None, dict]:
    """Recover (data paths, protocol label, evaluation scope) from a run root.

    The scope block is written by ``run_official_navsim_press.py`` into both
    ``suite_manifest.json`` and ``round*/method/config.json``.  Runs archived
    before that field existed return an empty scope, which is reported as
    "unknown" rather than silently treated as untruncated.
    """
    manifest = run_root / "suite_manifest.json"
    if manifest.is_file():
        payload = json.loads(manifest.read_text())
        return (
            payload.get("data", {}),
            (payload.get("eval_protocol") or {}).get("label"),
            dict(payload.get("evaluation_scope") or {}),
        )
    for config in sorted(glob.glob(str(run_root / "round*" / "*" / "config.json"))):
        payload = json.loads(Path(config).read_text())
        return (
            payload.get("data", {}),
            (payload.get("eval_protocol") or {}).get("label"),
            dict(payload.get("evaluation_scope") or {}),
        )
    return {}, None, {}


def _recorded_data(run_root: Path) -> tuple[dict, str | None]:
    data, label, _scope = _recorded(run_root)
    return data, label


def _label_for_paths(data: dict) -> tuple[str, dict | None]:
    for name, preset in press.EVAL_PROTOCOLS.items():
        # NOTE: the check must be `data.get(key) is None`, not
        # `str(data.get(key)) is not None` -- the latter is always True, so a
        # config missing one of the path keys raised KeyError on data[key]
        # instead of being classified as "custom" (audit 2026-09-12).
        if all(
            data.get(key) is not None
            and Path(str(data[key])).resolve() == Path(preset[key]).resolve()
            for key in press._PATH_KEYS
        ):
            return name, preset
    return "custom", None


def _method_pdm(path: Path) -> tuple[int, float] | None:
    matches = sorted(glob.glob(str(path / "pdm_score_*.csv")))
    if not matches:
        return None
    values = [
        float(row["pdm_score"])
        for row in csv.DictReader(open(matches[-1]))
        if row.get("token") != "average"
        and row.get("pdm_score") not in (None, "", "None")
        and str(row.get("valid", "True")).lower() in {"true", "1"}
    ]
    if not values:
        return None
    return len(values), statistics.fmean(values)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--expect-protocol",
        type=str,
        default=None,
        help="fail if the run turns out to be a different protocol",
    )
    args = parser.parse_args()

    data, recorded_label, scope = _recorded(args.run_root)
    if not data:
        raise SystemExit(f"no suite_manifest.json or round*/config.json under {args.run_root}")
    label, preset = _label_for_paths(data)
    if recorded_label and label == "custom":
        label = recorded_label

    expected = None if preset is None else int(preset["expected_scenes"])
    baselines = {} if preset is None else dict(preset["baselines"])

    # `evaluation_scope.is_truncated` was added after the first full-protocol
    # runs were archived, so its absence means "unknown", not "untruncated".
    truncated = None
    if "is_truncated" in scope:
        truncated = bool(scope["is_truncated"])

    print(f"run_root      : {args.run_root}")
    print(f"eval_protocol : {label}" + (f" (recorded: {recorded_label})" if recorded_label else ""))
    print(f"expected n    : {expected if expected is not None else 'unknown'}")
    if scope:
        print(
            "eval_scope    : "
            + ", ".join(f"{key}={scope[key]}" for key in sorted(scope))
        )
    else:
        print(
            "eval_scope    : absent (run predates evaluation_scope; truncation "
            "cannot be excluded from the artifact alone)"
        )
    if baselines:
        print(f"baselines     : " + ", ".join(f"{k}={v:.6f}" for k, v in baselines.items()))
    if preset is not None:
        print(f"note          : {preset['description']}")
    print()
    print(f"{'method':62s} {'n':>6s} {'PDM':>10s} {'protocol':>18s} {'n_ok':>6s}")
    results = []
    status = 0
    for method_dir in sorted((args.run_root / "round01").glob("*")):
        if not method_dir.is_dir():
            continue
        measured = _method_pdm(method_dir)
        if measured is None:
            continue
        n, pdm = measured
        n_ok = expected is None or n == expected
        if not n_ok:
            status = 1
        results.append(
            {
                "method": method_dir.name,
                "n_scenes": n,
                "pdm": pdm,
                "eval_protocol": label,
                "scene_count_ok": n_ok,
            }
        )
        print(
            f"{method_dir.name[:62]:62s} {n:6d} {pdm:10.6f} {label:>18s} "
            f"{'ok' if n_ok else 'MISMATCH':>6s}"
        )
    if not results:
        raise SystemExit("no completed methods found")

    if args.expect_protocol and label != args.expect_protocol:
        print(f"\nFAIL: expected protocol {args.expect_protocol}, found {label}")
        status = 1
    if expected is not None and any(not item["scene_count_ok"] for item in results):
        print(
            f"\nFAIL: scene count does not match {label}'s expected {expected}. "
            "Absolute PDM from this run is NOT protocol-comparable."
        )
        status = 1
    if truncated is True:
        print(
            "\nFAIL: evaluation_scope.is_truncated is True -- this run evaluated "
            f"fewer scenes than {label} declares "
            f"(--max-eval-tokens={scope.get('max_eval_tokens')} vs "
            f"expected={scope.get('expected_scenes')}). "
            "Its absolute PDM is NOT protocol-comparable and must not be cited."
        )
        status = 1
    elif truncated is None:
        print(
            "\nWARN: evaluation_scope.is_truncated is absent from this run's "
            "artifacts; the scene-count check above is the only truncation guard."
        )

    payload = {
        "run_root": str(args.run_root.resolve()),
        "eval_protocol": label,
        "expected_scenes": expected,
        "baselines": baselines,
        "evaluation_scope": scope,
        "truncated": truncated,
        "methods": results,
        "protocol_ok": status == 0,
        "citable": status == 0,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
