#!/usr/bin/env python3
"""Repair route-empty NAVSIM scene units without modifying the source split.

The repair map-matches the recorded ego path to roadblocks and stitches
consecutive matches through the directed roadblock graph.  Valid source files
are symlinked; only route-empty metadata pickles are rewritten.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle
import sys
import warnings
from typing import Any

from pyquaternion import Quaternion


FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = FRAMEWORK_ROOT.parent
if str(PROJECT_ROOT / "third_party") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "third_party"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--map-root", type=Path, required=True)
    parser.add_argument("--map-version", default="nuplan-maps-v1.0")
    parser.add_argument("--history-frames", type=int, default=5)
    parser.add_argument("--future-frames", type=int, default=10)
    parser.add_argument("--bfs-depth", type=int, default=20)
    return parser.parse_args()


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _map_matched_route(frames: list[dict[str, Any]], map_api: Any, bfs_depth: int) -> tuple[list[str], dict[str, int]]:
    from nuplan.common.actor_state.state_representation import StateSE2
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    from navsim.planning.simulation.planner.pdm_planner.utils.graph_search.bfs_roadblock import (
        BreadthFirstSearchRoadBlock,
    )
    from navsim.planning.simulation.planner.pdm_planner.utils.route_utils import (
        get_current_roadblock_candidates,
        remove_route_loops,
    )

    matched: list[str] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for frame in frames:
            xy = frame["ego2global_translation"]
            yaw = Quaternion(frame["ego2global_rotation"]).yaw_pitch_roll[0]
            block, _ = get_current_roadblock_candidates(
                StateSE2(float(xy[0]), float(xy[1]), float(yaw)), map_api, {}
            )
            if block is not None and (not matched or matched[-1] != str(block.id)):
                matched.append(str(block.id))
    if not matched:
        raise RuntimeError("map matching produced no roadblocks")

    route = [matched[0]]
    stitched = 0
    skipped_backward_or_ambiguous = 0
    for target in matched[1:]:
        if target == route[-1]:
            continue
        search = BreadthFirstSearchRoadBlock(route[-1], map_api, forward_search=True)
        (_blocks, path_ids), found = search.search(target, max_depth=bfs_depth)
        if found:
            for roadblock_id in path_ids[1:]:
                roadblock_id = str(roadblock_id)
                if roadblock_id != route[-1]:
                    route.append(roadblock_id)
            stitched += max(0, len(path_ids) - 2)
        else:
            # Overlapping intersections occasionally make the closest match
            # jump to a parallel/backward branch for one frame.  Skipping that
            # ambiguous observation preserves a connected directed route.
            skipped_backward_or_ambiguous += 1
    if not route:
        raise RuntimeError("route stitching produced an empty route")
    route_blocks = [
        map_api.get_map_object(roadblock_id, SemanticMapLayer.ROADBLOCK)
        or map_api.get_map_object(roadblock_id, SemanticMapLayer.ROADBLOCK_CONNECTOR)
        for roadblock_id in route
    ]
    if any(block is None for block in route_blocks):
        raise RuntimeError("stitched route contains an unresolved roadblock")
    route_blocks, loop_free_route = remove_route_loops(route_blocks, route)
    loop_truncated = len(route) - len(loop_free_route)
    route = [str(roadblock_id) for roadblock_id in loop_free_route]
    if not route:
        raise RuntimeError("route loop removal produced an empty route")
    return route, {
        "raw_map_matches": len(matched),
        "route_roadblocks": len(route),
        "stitched_intermediate": stitched,
        "skipped_ambiguous": skipped_backward_or_ambiguous,
        "loop_truncated": loop_truncated,
    }


def main() -> int:
    args = _parse_args()
    manifest = args.manifest.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    map_root = args.map_root.expanduser().resolve()
    if not manifest.is_file() or not map_root.is_dir():
        raise FileNotFoundError(manifest if not manifest.is_file() else map_root)
    metadata_root = output_root / "metadata"
    metadata_root.mkdir(parents=True, exist_ok=True)

    os.environ["NUPLAN_MAPS_ROOT"] = str(map_root)
    from nuplan.common.maps.nuplan_map.map_factory import get_maps_api

    rows = _load_rows(manifest)
    map_apis: dict[str, Any] = {}
    repaired_rows: list[dict[str, Any]] = []
    repaired_only: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    seen_scene_tokens: set[str] = set()
    for index, source_row in enumerate(rows, start=1):
        row = dict(source_row)
        scene_token = str(row["scene_token"])
        if scene_token in seen_scene_tokens:
            raise ValueError(f"duplicate scene_token in source manifest: {scene_token}")
        seen_scene_tokens.add(scene_token)
        source_path = Path(row["metadata_path"]).expanduser().resolve()
        target_path = metadata_root / source_path.name
        frames = pickle.loads(source_path.read_bytes())
        needs_repair = not any(bool(frame.get("roadblock_ids")) for frame in frames)
        if needs_repair:
            location = str(frames[0]["map_location"])
            if location not in map_apis:
                map_apis[location] = get_maps_api(str(map_root), args.map_version, location)
            route, stats = _map_matched_route(frames, map_apis[location], args.bfs_depth)
            for frame in frames:
                frame["roadblock_ids"] = list(route)
            with target_path.open("wb") as handle:
                pickle.dump(frames, handle, protocol=pickle.HIGHEST_PROTOCOL)
            candidate_windows = max(0, len(frames) - args.history_frames - args.future_frames + 1)
            row["windows"] = dict(row.get("windows") or {})
            row["windows"]["route_valid_windows"] = candidate_windows
            detail = {"scene_token": scene_token, "metadata": str(target_path), **stats}
            details.append(detail)
        else:
            if target_path.exists() or target_path.is_symlink():
                if target_path.resolve() != source_path:
                    raise FileExistsError(target_path)
            else:
                target_path.symlink_to(source_path)
        row["metadata_path"] = str(target_path.resolve() if target_path.is_symlink() else target_path)
        repaired_rows.append(row)
        if needs_repair:
            repaired_only.append(row)
        if index == 1 or index % 250 == 0 or index == len(rows):
            print(f"[route-repair] scenes={index}/{len(rows)} repaired={len(details)}", flush=True)

    output_root.mkdir(parents=True, exist_ok=True)
    for name, payload in (
        ("test_manifest_repaired.jsonl", repaired_rows),
        ("test_manifest_repaired_only.jsonl", repaired_only),
    ):
        (output_root / name).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in payload),
            encoding="utf-8",
        )
    report = {
        "source_manifest": str(manifest),
        "output_root": str(output_root),
        "scene_count": len(rows),
        "unchanged_scene_count": len(rows) - len(details),
        "repaired_scene_count": len(details),
        "map_locations": sorted(map_apis),
        "all_scenes_route_valid": len(details) == len(repaired_only),
        "details": details,
    }
    (output_root / "route_repair_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "details"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
