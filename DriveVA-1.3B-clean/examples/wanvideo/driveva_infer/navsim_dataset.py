"""
Runtime helpers shared by DriveVA NavSIM and nuScenes inference scripts.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import torch
from PIL import Image


NAV_CMDS = ["turn left", "go straight", "turn right"]
DEFAULT_NEGATIVE_PROMPT = (
    "worst quality, low quality, blurry, jittery, distorted, motion blur, ghosting, "
    "flickering, stuttering, camera shake, unstable footage, warping, trailing artifacts, "
    "temporal inconsistency, jerky motion, choppy framerate"
)


def _purge_imported_navsim_modules() -> None:
    to_delete = [
        module_name
        for module_name in list(sys.modules.keys())
        if module_name == "navsim" or module_name.startswith("navsim.")
    ]
    for module_name in to_delete:
        del sys.modules[module_name]


def _ensure_navsim_importable(repo_root: Path) -> None:
    candidates = [
        repo_root / "third_party",
        Path(__file__).resolve().parents[3] / "third_party",
    ]

    candidates = list(dict.fromkeys(candidates))

    required_file = Path("navsim") / "common" / "dataclasses.py"

    for navsim_root in candidates:
        if navsim_root.exists() and (navsim_root / required_file).exists():
            selected_root = navsim_root.resolve()
            selected_root_str = str(selected_root)
            resolved_candidates = [p.resolve() for p in candidates if p.exists()]

            for candidate_root in resolved_candidates:
                candidate_root_str = str(candidate_root)
                while candidate_root_str in sys.path:
                    sys.path.remove(candidate_root_str)
            sys.path.insert(0, selected_root_str)

            loaded_navsim = sys.modules.get("navsim")
            loaded_root: Optional[Path] = None
            if loaded_navsim is not None:
                loaded_file = getattr(loaded_navsim, "__file__", None)
                if loaded_file:
                    try:
                        loaded_root = Path(str(loaded_file)).resolve().parents[1]
                    except Exception:
                        loaded_root = None
            if loaded_navsim is not None and loaded_root != selected_root:
                _purge_imported_navsim_modules()
            return

    raise FileNotFoundError(
        "Cannot locate a NavSIM package "
        "(expected navsim/common/dataclasses.py). Tried:\n  "
        + "\n  ".join(str(p) for p in candidates)
    )


def _read_image_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _mosaic_surround(paths: List[str | Path]) -> Image.Image:
    imgs = [_read_image_rgb(p) for p in paths]
    if len(imgs) == 0:
        raise ValueError("surround-view frame list is empty")
    base_w, base_h = imgs[0].size
    imgs = [img.resize((base_w, base_h), Image.BILINEAR) for img in imgs]
    while len(imgs) < 6:
        imgs.append(Image.new("RGB", (base_w, base_h), (0, 0, 0)))
    row1 = Image.new("RGB", (base_w * 3, base_h), (0, 0, 0))
    row2 = Image.new("RGB", (base_w * 3, base_h), (0, 0, 0))
    for idx, img in enumerate(imgs[:3]):
        row1.paste(img, (idx * base_w, 0))
    for idx, img in enumerate(imgs[3:6]):
        row2.paste(img, (idx * base_w, 0))
    mosaic = Image.new("RGB", (base_w * 3, base_h * 2), (0, 0, 0))
    mosaic.paste(row1, (0, 0))
    mosaic.paste(row2, (0, base_h))
    return mosaic


def _tensor_to_pil(frame: torch.Tensor, normalize_mode: Optional[str]) -> Image.Image:
    arr = frame.detach().cpu().float()
    if arr.ndim == 3:
        arr = arr.permute(1, 2, 0)
    if normalize_mode == "[0,1]":
        arr = arr * 255.0
    elif normalize_mode == "[-1,1]":
        arr = (arr + 1.0) * 127.5
    else:
        vmin = float(arr.min().item())
        vmax = float(arr.max().item())
        arr = arr * 255.0 if vmin >= -0.1 and vmax <= 1.1 else (arr + 1.0) * 127.5
    arr = arr.clamp(0, 255).to(torch.uint8).numpy()
    return Image.fromarray(arr).convert("RGB")


def _resolve_video_pil(
    frames: List[Any],
    height: int,
    width: int,
    surround_view: bool,
    normalize_mode: Optional[str] = None,
) -> List[Image.Image]:
    out: List[Image.Image] = []
    for frame in frames:
        if isinstance(frame, (list, tuple)):
            img = _mosaic_surround(list(frame)) if surround_view else _read_image_rgb(frame[0])
        elif isinstance(frame, (str, Path)):
            img = _read_image_rgb(frame)
        elif torch.is_tensor(frame):
            img = _tensor_to_pil(frame, normalize_mode)
        elif isinstance(frame, np.ndarray):
            img = Image.fromarray(frame.astype(np.uint8)).convert("RGB")
        elif isinstance(frame, Image.Image):
            img = frame.convert("RGB")
        else:
            raise TypeError(f"Unsupported frame type: {type(frame)}")
        out.append(img.resize((width, height), Image.BILINEAR))
    return out


def one_hot_to_cmd(one_hot: Any) -> str:
    if torch.is_tensor(one_hot):
        values = one_hot.detach().cpu().flatten().tolist()
    elif isinstance(one_hot, np.ndarray):
        values = one_hot.flatten().tolist()
    else:
        values = list(one_hot)
    for idx, value in enumerate(values[: len(NAV_CMDS)]):
        if int(value) == 1:
            return NAV_CMDS[idx]
    return "unknown"


def _build_prompt_fixed(*args: Any) -> str:
    """
    Build the inference prompt from driving command and ego dynamics.

    Supported signatures:
    - _build_prompt_fixed(cmd_onehot, speed_mps)
    - _build_prompt_fixed(history_xyh, cmd_onehot, speed_mps, accel_mps2)
    """
    accel_mps2: Optional[float] = None
    if len(args) == 2:
        cmd_onehot, speed_mps = args
    elif len(args) == 4:
        _, cmd_onehot, speed_mps, accel_mps2 = args
    else:
        raise TypeError(
            "_build_prompt_fixed expects (cmd_onehot, speed_mps) or "
            "(history_xyh, cmd_onehot, speed_mps, accel_mps2)"
        )

    cmd = one_hot_to_cmd(cmd_onehot).lower()
    speed_mps = float(speed_mps)
    accel_mps2 = None if accel_mps2 is None else float(accel_mps2)

    if speed_mps < 5.0:
        speed_desc = "at low speed"
    elif speed_mps < 15.0:
        speed_desc = "at moderate speed"
    else:
        speed_desc = "at highway speed"

    if "left" in cmd:
        motion_trend, turning_desc = "turning left", "with controlled steering"
    elif "right" in cmd:
        motion_trend, turning_desc = "turning right", "with controlled steering"
    elif "straight" in cmd:
        motion_trend, turning_desc = "driving straight ahead", "with stable lane keeping"
    else:
        motion_trend, turning_desc = "driving straight ahead", "with stable lane keeping"

    technical = f"[Technical: speed {speed_mps:.2f}m/s"
    if accel_mps2 is not None:
        technical += f", accel {accel_mps2:.2f}m/s^2"
    technical += "]"

    return (
        "A high-quality, photorealistic dashboard camera view of autonomous driving. "
        f"Based on the past 2 seconds video showing {motion_trend} {turning_desc}, "
        "predict and generate the next 4 seconds of realistic driving continuation, "
        f"following command: {cmd}. "
        f"Keep temporal consistency, realistic physics, and smooth motion, moving {speed_desc}. "
        f"{technical}"
    )
