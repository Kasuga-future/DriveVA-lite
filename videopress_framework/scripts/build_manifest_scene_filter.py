#!/usr/bin/env python3
"""Build a one-window-per-scene NAVSIM YAML filter from a split manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = PROJECT_ROOT / "examples" / "wanvideo" / "driveva_train"
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

from train_navsim_v1 import _read_jsonl_manifest, _representative_frame_tokens  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--num-history-frames", type=int, default=5)
    parser.add_argument("--num-future-frames", type=int, default=10)
    parser.add_argument("--frame-interval", type=int, default=1)
    parser.add_argument("--allow-missing-route", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = _read_jsonl_manifest(str(args.manifest))
    input_scene_count = len(rows)
    if not args.allow_missing_route:
        # Official PDM needs a valid route.  Keep the full split manifest as
        # the source of truth, but omit scene units that cannot produce any
        # route-valid evaluation window.
        rows = [
            row
            for row in rows
            if int((row.get("windows") or {}).get("route_valid_windows", 0)) > 0
        ]
    tokens = _representative_frame_tokens(
        rows,
        manifest_path=str(args.manifest),
        num_history_frames=args.num_history_frames,
        num_future_frames=args.num_future_frames,
        frame_interval=args.frame_interval,
        has_route=not args.allow_missing_route,
        windows_per_scene=1,
    )
    payload = {
        "num_history_frames": args.num_history_frames,
        "num_future_frames": args.num_future_frames,
        "frame_interval": args.frame_interval,
        "has_route": not args.allow_missing_route,
        "tokens": tokens,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    print(
        f"wrote {len(tokens)} unique scene windows to {args.output.resolve()} "
        f"(manifest_scenes={input_scene_count}, omitted_without_route={input_scene_count - len(rows)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
