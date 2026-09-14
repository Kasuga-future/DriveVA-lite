#!/usr/bin/env python3
"""Sampling-seed instrument control for the best-of-N experiment.

Three passes over the *same* 4-scene official-test subset:

* ``seed1001``      -- sample seed 1001
* ``seed1001_rep01``-- sample seed 1001 again, separate process (control)
* ``seed2002``      -- sample seed 2002

The control must be bitwise identical (proves the planner has no other live RNG
source and that ``--sample-seed`` fully determines the sample), and the
different seed must produce a different trajectory (proves the harness can
actually produce independent samples).  Writes the raw evidence to
``seed_control_check.json`` next to the pass directories.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np

PAIR_A = "seed1001"
PAIR_B = "seed1001_rep01"
PAIR_C = "seed2002"


def load_trajectories(pass_dir: Path) -> dict[str, tuple[np.ndarray, int]]:
    out: dict[str, tuple[np.ndarray, int]] = {}
    for path in sorted((pass_dir / "round01" / "physical_no_press" / "trajectories").glob("*.npz")):
        token = path.name.split(".rank")[0]
        with np.load(path) as payload:
            out[token] = (np.asarray(payload["traj"], dtype=np.float64), int(payload["sample_seed"]))
    return out


def load_pdm(pass_dir: Path) -> dict[str, str]:
    matches = sorted(glob.glob(str(pass_dir / "round01" / "physical_no_press" / "pdm_score_*.csv")))
    with open(matches[-1], newline="") as handle:
        return {
            row["token"]: row["pdm_score"]
            for row in csv.DictReader(handle)
            if row["token"] != "average"
        }


def ade(a: np.ndarray, b: np.ndarray) -> float:
    x = a[0] if a.ndim == 3 else a
    y = b[0] if b.ndim == 3 else b
    n = min(len(x), len(y))
    return float(np.linalg.norm(x[:n, :2] - y[:n, :2], axis=1).mean())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    a = load_trajectories(root / PAIR_A)
    b = load_trajectories(root / PAIR_B)
    c = load_trajectories(root / PAIR_C)
    pdm_a, pdm_b, pdm_c = load_pdm(root / PAIR_A), load_pdm(root / PAIR_B), load_pdm(root / PAIR_C)

    shared = sorted(set(a) & set(b) & set(c))
    assert shared, "no scene is present in all three passes"
    report = {
        "passes": {"same_seed_a": PAIR_A, "same_seed_b": PAIR_B, "other_seed": PAIR_C},
        "n_scenes": len(shared),
        "sample_seed_per_pass": {
            PAIR_A: sorted({int(seed) for _, seed in a.values()}),
            PAIR_B: sorted({int(seed) for _, seed in b.values()}),
            PAIR_C: sorted({int(seed) for _, seed in c.values()}),
        },
        "same_seed_trajectory_bitwise_identical_rate": float(
            np.mean([np.array_equal(a[t][0], b[t][0]) for t in shared])
        ),
        "same_seed_pdm_identical": pdm_a == pdm_b,
        "different_seed_trajectory_bitwise_identical_rate": float(
            np.mean([np.array_equal(a[t][0], c[t][0]) for t in shared])
        ),
        "different_seed_ade_mean_m": float(np.mean([ade(a[t][0], c[t][0]) for t in shared])),
        "different_seed_ade_max_m": float(np.max([ade(a[t][0], c[t][0]) for t in shared])),
        "different_seed_pdm_identical": pdm_a == pdm_c,
        "per_scene": [
            {
                "scene_token": token,
                "same_seed_bitwise_identical": bool(np.array_equal(a[token][0], b[token][0])),
                "different_seed_bitwise_identical": bool(np.array_equal(a[token][0], c[token][0])),
                "different_seed_ade_m": ade(a[token][0], c[token][0]),
                "pdm_sample1001": pdm_a[token],
                "pdm_sample1001_repeat": pdm_b[token],
                "pdm_sample2002": pdm_c[token],
            }
            for token in shared
        ],
    }
    out_path = args.output or (root / "seed_control_check.json")
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "per_scene"}, indent=2))
    print(f"[control] wrote {out_path}")
    return 0 if report["same_seed_trajectory_bitwise_identical_rate"] == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
