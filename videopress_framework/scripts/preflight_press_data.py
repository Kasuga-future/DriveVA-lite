#!/usr/bin/env python3
"""Validate the migrated NavSIM metadata/sensor/cache paths without loading Wan."""

from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
import sys


FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
ROOT = FRAMEWORK_ROOT.parent
os.environ.setdefault("MPLCONFIGDIR", "/tmp/driveva-lite-matplotlib")
if str(FRAMEWORK_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(1, str(ROOT))

from examples.wanvideo.driveva_infer.navsim_dataset import _ensure_navsim_importable


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--log-path", type=Path, default=None)
    parser.add_argument("--sensor-path", type=Path, default=None)
    parser.add_argument("--metric-cache-path", type=Path, default=None)
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    log_path = args.log_path or repo_root / "data/navsim_v1.1/openscene-v1.1/meta_datas/test"
    sensor_path = args.sensor_path or repo_root / "data/navsim_v1.1/openscene-v1.1/sensor_blobs/test"
    cache_path = args.metric_cache_path or repo_root / "data/navsim_v1.1/metric_cache"
    paths = {"log_path": log_path, "sensor_path": sensor_path, "metric_cache_path": cache_path}
    missing = [f"{name}={path}" for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing migrated data paths:\n" + "\n".join(missing))

    _ensure_navsim_importable(repo_root)
    from navsim.common.dataclasses import SceneFilter, SensorConfig
    from navsim.common.dataloader import MetricCacheLoader, SceneLoader

    scene_filter = SceneFilter(
        num_history_frames=5,
        num_future_frames=10,
        frame_interval=1,
        has_route=True,
        max_scenes=None,
        log_names=None,
    )
    sensor_config = SensorConfig.build_all_sensors(include=True)
    init_sig = inspect.signature(SceneLoader.__init__)
    kwargs = {
        "data_path": log_path,
        "scene_filter": scene_filter,
        "sensor_config": sensor_config,
    }
    if "sensor_blobs_path" in init_sig.parameters:
        kwargs["sensor_blobs_path"] = sensor_path
        kwargs["load_image_path"] = True
    else:
        kwargs["original_sensor_path"] = sensor_path
    loader = SceneLoader(**kwargs)
    metric_cache = MetricCacheLoader(cache_path)
    scene_tokens = set(loader.tokens)
    cache_tokens = set(metric_cache.tokens)
    shared = sorted(scene_tokens & cache_tokens)
    payload = {
        "repo_root": str(repo_root),
        "log_path": str(log_path.resolve()),
        "sensor_path": str(sensor_path.resolve()),
        "metric_cache_path": str(cache_path.resolve()),
        "scene_tokens": len(scene_tokens),
        "metric_cache_tokens": len(cache_tokens),
        "intersection_tokens": len(shared),
        "first_shared_token": shared[0] if shared else None,
        "sensor_loader_signature": str(init_sig),
    }
    print(json.dumps(payload, indent=2))
    if not shared:
        raise RuntimeError("NavSIM scene/cache intersection is empty")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
