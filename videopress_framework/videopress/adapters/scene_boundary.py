"""Scene-boundary guard for the repository's NAVSIM ``SceneLoader``.

Some NAVSIM snapshots implement ``filter_scenes`` by slicing each log at a
fixed stride without checking the segment marker.  A slice can therefore end
with frames from one ``scene_token`` and continue at ``frame_idx == 0`` from
the next segment.  The official DriveVA evaluator remains the owner of scene
loading; this module only wraps its loader function at runtime and filters
those invalid windows.
"""

from __future__ import annotations

from typing import Any


def window_is_single_scene(frames: list[dict[str, Any]]) -> bool:
    """Return whether all frames belong to one NAVSIM scene segment."""

    if not frames:
        return False
    scene_tokens = {frame.get("scene_token") for frame in frames}
    scene_names = {frame.get("scene_name") for frame in frames}
    if None in scene_tokens or None in scene_names:
        return False
    return len(scene_tokens) == 1 and len(scene_names) == 1


def install_scene_boundary_guard() -> dict[str, Any]:
    """Install an idempotent post-filter around official ``filter_scenes``.

    The wrapper deliberately calls the original official function first, so
    all official log-name/token/route filtering semantics are retained.  It
    only removes returned windows whose complete frame list crosses either the
    ``scene_token`` or ``scene_name`` boundary.
    """

    import navsim.common.dataloader as dataloader

    current = dataloader.filter_scenes
    if getattr(current, "_driveva_lite_scene_boundary_guard", False):
        return {
            "installed": True,
            "already_installed": True,
            "guard": "scene_token+scene_name",
        }

    original = current

    def guarded_filter_scenes(data_path, scene_filter):
        filtered = original(data_path, scene_filter)
        required_length = int(scene_filter.num_frames)
        return {
            token: frames
            for token, frames in filtered.items()
            if len(frames) >= required_length and window_is_single_scene(frames)
        }

    guarded_filter_scenes.__name__ = "filter_scenes_scene_boundary_guarded"
    guarded_filter_scenes.__doc__ = (
        "Official NAVSIM filter_scenes plus same-scene-token/name window validation."
    )
    guarded_filter_scenes._driveva_lite_scene_boundary_guard = True
    guarded_filter_scenes._driveva_lite_original_filter = original
    dataloader.filter_scenes = guarded_filter_scenes
    return {
        "installed": True,
        "already_installed": False,
        "guard": "scene_token+scene_name",
    }
