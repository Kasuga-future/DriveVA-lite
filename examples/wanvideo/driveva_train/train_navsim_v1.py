"""
DriveVA NAVSIM v1 training entrypoint.
"""

from __future__ import annotations

import argparse
import atexit
import contextlib
import importlib
import json
import math
import os
import pickle
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

_REPO_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_IMPORT))

from diffsynth.models.utils import load_state_dict
from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, launch_training_task

from navsim_dataset import DEFAULT_NEGATIVE_PROMPT, NavsimDriveVAConfig, NavsimDriveVADataset
FRAMEWORK_ROOT = Path(__file__).resolve().parents[3] / "videopress_framework"
if str(FRAMEWORK_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK_ROOT))

from videopress.training.online_selector import (
    DynamicTokenSelector, counterfactual_group_bce, critical_box_token_mask,
    displacement_token_bce,
    gradient_input_scores,
    hard_topk_mask, horizon_weighted_trajectory_displacement,
    keep_ratio_at_step, online_topk_labels, parse_horizon_weights,
    parse_keep_schedule, selector_metrics, selector_pairwise_ranking_loss,
    signed_removal_scores, signed_soft_keep_labels,
    spatial_counterfactual_probe,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _flatten_sample_token(sample_token: Any) -> Optional[str]:
    """``data["token"]`` is a length-1 list per batch element after collation."""
    if sample_token is None:
        return None
    if isinstance(sample_token, (list, tuple)):
        if not sample_token:
            return None
        return str(sample_token[0])
    return str(sample_token)


def _dump_counterfactual_probe(
    dump_dir: Path,
    *,
    rank: int,
    global_step: int,
    sample_token: Optional[str],
    tokens: torch.Tensor,
    positions: torch.Tensor,
    ego_state: torch.Tensor,
    command: torch.Tensor,
    selector_timestep: Optional[torch.Tensor],
    logits: torch.Tensor,
    membership: torch.Tensor,
    group_index: int,
    baseline_loss: torch.Tensor,
    masked_loss: torch.Tensor,
    relative_delta: float,
    helpful_target: float,
    confidence: float,
    trajectory_metrics: Optional[dict] = None,
    extra: Optional[dict] = None,
    shard_tag: Optional[str] = None,
) -> str:
    """Persist one counterfactual probe so it can be re-analysed offline.

    The selector sees ``tokens``/``positions``/``ego_state``/``command``/
    ``timestep``; the teacher supplies exactly one scalar label per probe.
    Dumping both sides makes the label's learnability an offline question that
    does not need the 5B backbone again.
    """
    shard_dir = dump_dir / f"rank{int(rank)}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    # The tag is part of the name because one optimizer step can emit many
    # shards (one per tile, or one per removal size) that share the same step.
    tag = str(shard_tag) if shard_tag else f"tile-{int(group_index):02d}"
    path = shard_dir / f"step-{int(global_step):06d}-{tag}.pt"
    time_tensor = (
        torch.zeros(1)
        if selector_timestep is None
        else torch.as_tensor(selector_timestep).detach().float().reshape(-1)
    )
    payload = {
        "global_step": int(global_step),
        "rank": int(rank),
        "sample_token": _flatten_sample_token(sample_token),
        "tokens": tokens.detach().to(torch.float16).cpu(),
        "positions": positions.detach().float().cpu(),
        "ego_state": ego_state.detach().float().cpu(),
        "command": command.detach().float().cpu(),
        "selector_timestep": time_tensor.cpu(),
        "selector_logits": logits.detach().float().cpu(),
        "membership": membership.detach().bool().cpu(),
        "group_index": int(group_index),
        "baseline_loss_unweighted": float(baseline_loss.detach().float().reshape(-1)[0]),
        "masked_loss_unweighted": float(masked_loss.detach().float().reshape(-1)[0]),
        "relative_delta": float(relative_delta),
        "helpful_target": float(helpful_target),
        "confidence": float(confidence),
    }
    payload.update(
        {
            key: float(value)
            for key, value in (trajectory_metrics or {}).items()
            if isinstance(value, (int, float))
        }
    )
    if extra:
        payload.update(extra)
    tmp_path = path.with_suffix(".pt.tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)
    return str(path)


def _probe_relative_delta(baseline_loss: torch.Tensor, masked_loss: torch.Tensor) -> float:
    baseline = float(baseline_loss.detach().float().reshape(-1)[0])
    masked = float(masked_loss.detach().float().reshape(-1)[0])
    return (masked - baseline) / max(abs(baseline), 1e-6)


def _append_counterfactual_jsonl(path: Path, row: dict) -> None:
    """Append one compact JSON line per (scene, tile, latent) probe.

    The full ``.pt`` shard (``_dump_counterfactual_probe``) stores the 3072-d
    hidden states, which is what makes a 12-tile x 2-latent sweep expensive on
    disk.  The route-temporal-key-set probe only needs the scalar labels plus
    the tile geometry and the ego condition, so it writes those instead.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    for key, value in row.items():
        if value is None or isinstance(value, (bool, str, list, dict)):
            payload[key] = value
        elif isinstance(value, int):
            payload[key] = int(value)
        else:
            number = float(value)
            # Keep the file valid JSON (no bare NaN/Infinity literals).
            payload[key] = number if math.isfinite(number) else None
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _trajectory_divergence(
    baseline_pred: Optional[torch.Tensor],
    masked_pred: Optional[torch.Tensor],
    *,
    prefix_len: int = 0,
    horizon_weights=(),
    target_fps: float = 2.0,
) -> dict:
    """Geometric distance between the two predicted plans (review P3 item 3).

    The planning-loss delta is a signed functional that can cancel: a tile whose
    removal rotates the prediction without changing its error barely moves the
    MSE.  Plan displacement cannot cancel and is therefore a candidate for a
    much lower-variance teacher target.
    """
    if not torch.is_tensor(baseline_pred) or not torch.is_tensor(masked_pred):
        return {}
    baseline = baseline_pred.detach().float()
    masked = masked_pred.detach().float()
    if baseline.shape != masked.shape or baseline.ndim != 3:
        return {}
    if int(prefix_len) > 0 and baseline.shape[1] > int(prefix_len):
        baseline = baseline[:, int(prefix_len) :]
        masked = masked[:, int(prefix_len) :]
    if baseline.shape[1] == 0:
        return {}
    step_displacement = (baseline - masked).norm(dim=-1)
    scale = baseline.abs().mean().clamp_min(1e-6)
    metrics = {
        "counterfactual_traj_disp_mean": float(step_displacement.mean()),
        "counterfactual_traj_disp_max": float(step_displacement.max()),
        "counterfactual_traj_disp_final": float(step_displacement[:, -1].mean()),
        "counterfactual_traj_disp_relative": float(step_displacement.mean() / scale),
        "counterfactual_traj_endpoint_disp": float(
            (baseline[:, -1] - masked[:, -1]).norm(dim=-1).mean()
        ),
        "counterfactual_traj_scale": float(scale),
    }
    if horizon_weights:
        weighted, indices = horizon_weighted_trajectory_displacement(
            baseline,
            masked,
            horizon_weights,
            target_fps=target_fps,
        )
        metrics.update(
            {
                "counterfactual_traj_disp_long_horizon": float(weighted.mean()),
                "counterfactual_traj_disp_long_horizon_relative": float(
                    weighted.mean() / scale
                ),
                "counterfactual_traj_horizon_indices": indices,
                "counterfactual_traj_horizon_seconds": [
                    float(seconds) for seconds, _ in horizon_weights
                ],
                "counterfactual_traj_horizon_weights": [
                    float(weight) for _, weight in horizon_weights
                ],
            }
        )
    return metrics


def _planning_error_harm(
    baseline_pred: Optional[torch.Tensor],
    masked_pred: Optional[torch.Tensor],
    target: Optional[torch.Tensor],
    *,
    horizon_weights=(),
    target_fps: float = 2.0,
) -> dict:
    """Increase in metric-space planning error caused by removing a token tile.

    Positive harm means that deletion moves the predicted trajectory farther
    from ground truth and therefore supplies direct evidence that the removed
    tile is worth keeping.  Negative harm is retained for diagnostics but is
    clamped to zero by the selector target.
    """
    if not all(torch.is_tensor(value) for value in (baseline_pred, masked_pred, target)):
        return {}
    baseline = baseline_pred.detach().float()
    masked = masked_pred.detach().float()
    target = target.detach().float()
    if baseline.shape != masked.shape or baseline.shape != target.shape:
        return {}
    if baseline.ndim != 3 or baseline.shape[1] == 0:
        return {}
    # NAVSIM planning displacement is evaluated in the ground-plane x/y axes.
    baseline_step_error = (baseline[..., :2] - target[..., :2]).norm(dim=-1)
    masked_step_error = (masked[..., :2] - target[..., :2]).norm(dim=-1)
    baseline_ade = baseline_step_error.mean(dim=1)
    masked_ade = masked_step_error.mean(dim=1)
    harm_ade = masked_ade - baseline_ade
    metrics = {
        "counterfactual_planning_error_baseline_ade": float(baseline_ade.mean()),
        "counterfactual_planning_error_masked_ade": float(masked_ade.mean()),
        "counterfactual_planning_harm_ade": float(harm_ade.mean()),
        "counterfactual_planning_harm_ade_positive": float(
            harm_ade.clamp_min(0).mean()
        ),
    }
    if horizon_weights:
        baseline_weighted, indices = horizon_weighted_trajectory_displacement(
            baseline[..., :2],
            target[..., :2],
            horizon_weights,
            target_fps=target_fps,
        )
        masked_weighted, _ = horizon_weighted_trajectory_displacement(
            masked[..., :2],
            target[..., :2],
            horizon_weights,
            target_fps=target_fps,
        )
        harm = masked_weighted - baseline_weighted
        metrics.update(
            {
                "counterfactual_planning_error_baseline_long_horizon": float(
                    baseline_weighted.mean()
                ),
                "counterfactual_planning_error_masked_long_horizon": float(
                    masked_weighted.mean()
                ),
                "counterfactual_planning_harm_long_horizon": float(harm.mean()),
                "counterfactual_planning_harm_long_horizon_positive": float(
                    harm.clamp_min(0).mean()
                ),
                "counterfactual_planning_horizon_indices": indices,
            }
        )
    return metrics


_NAVSIM_EVAL_LOG_DEFAULT = "/path/to/navsim_v1.1/navsim_logs/test"
_NAVSIM_EVAL_SENSOR_DEFAULT = "/path/to/navsim_v1.1/sensor_blobs/test"
_NAVSIM_EVAL_CACHE_DEFAULT = (
    "/path/to/navsim_v1.1/metric_cache"
)


class _TeeStream:
    def __init__(self, console_stream, file_stream):
        self._console_stream = console_stream
        self._file_stream = file_stream

    def write(self, data):
        written = self._console_stream.write(data)
        self._file_stream.write(data)
        return written if isinstance(written, int) else len(str(data))

    def flush(self):
        self._console_stream.flush()
        self._file_stream.flush()

    def isatty(self):
        return bool(getattr(self._console_stream, "isatty", lambda: False)())

    @property
    def encoding(self):
        return getattr(self._console_stream, "encoding", None)

    def __getattr__(self, name):
        return getattr(self._console_stream, name)


def _is_rank0() -> bool:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return int(os.environ.get("RANK", "0")) == 0


def _enable_train_log_capture(output_path: str, train_log_file: str) -> None:
    if not _is_rank0() or not train_log_file:
        return
    output_dir = Path(output_path).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(train_log_file).expanduser()
    if not log_path.is_absolute():
        log_path = output_dir / log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fp = log_path.open("a", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = _TeeStream(original_stdout, log_fp)
    sys.stderr = _TeeStream(original_stderr, log_fp)

    def _restore_streams() -> None:
        try:
            log_fp.flush()
            log_fp.close()
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    atexit.register(_restore_streams)


def _str2bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def _normalize_train_ckpt_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    known_pipe_prefixes = (
        "dit.",
        "dit2.",
        "trajectory_encoder.",
        "trajectory_head.",
        "trajectory_decoder.",
        "text_encoder.",
        "vae.",
        "image_encoder.",
        "prompter.",
        "scheduler.",
        "motion_controller.",
        "vace.",
        "animate_adapter.",
        "audio_processor.",
    )
    normalized = {}
    for key, value in state_dict.items():
        norm_key = key[5:] if key.startswith("pipe.") else key
        if not norm_key.startswith(known_pipe_prefixes):
            norm_key = f"dit.{norm_key}"
        normalized[norm_key] = value
    return normalized


def _summarize_ckpt_keys(keys):
    if not keys:
        return ""
    counter = Counter(k.split(".", 1)[0] if "." in k else k for k in keys)
    return ", ".join(f"{name}:{count}" for name, count in counter.most_common())


def _env_first(*names: str, default: Any = None) -> Any:
    for name in names:
        value = os.environ.get(name)
        if value is not None and str(value) != "":
            return value
    return default


def _env_bool(*names: str, default: bool = False) -> bool:
    value = _env_first(*names, default="1" if default else "0")
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _add_optional_arg(argv: list[str], flag: str, value: Any) -> None:
    if value is not None and str(value) != "":
        argv.extend([flag, str(value)])


def _read_jsonl_manifest(path: str) -> list[dict[str, Any]]:
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"scene manifest not found: {manifest_path}")
    rows: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{manifest_path}:{line_no} must contain a JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"scene manifest is empty: {manifest_path}")
    return rows


def _manifest_scene_tokens(rows: list[dict[str, Any]], path: str) -> list[str]:
    tokens = [str(row["scene_token"]) for row in rows if row.get("scene_token") is not None]
    if len(tokens) != len(rows):
        raise ValueError(f"every row in {path} must contain scene_token")
    if len(set(tokens)) != len(tokens):
        raise ValueError(f"duplicate scene_token values in {path}")
    return tokens


def _forbidden_manifest_tokens(spec: str) -> tuple[set[str], list[dict[str, Any]]]:
    """Load a comma-separated set of manifests used as leakage guards."""

    paths = [value.strip() for value in str(spec).split(",") if value.strip()]
    if not paths:
        raise ValueError("forbidden_scene_manifest must name at least one manifest")
    union: set[str] = set()
    summaries: list[dict[str, Any]] = []
    for path in paths:
        rows = _read_jsonl_manifest(path)
        tokens = set(_manifest_scene_tokens(rows, path))
        union.update(tokens)
        summaries.append({"path": str(Path(path).expanduser().resolve()), "scenes": len(tokens)})
    return union, summaries


def _representative_frame_tokens(
    rows: list[dict[str, Any]],
    *,
    manifest_path: str,
    num_history_frames: int,
    num_future_frames: int,
    frame_interval: Optional[int],
    has_route: bool = True,
    windows_per_scene: int = 1,
) -> list[str]:
    """Choose deterministic, temporally spread windows per independent scene.

    Split manifests identify semantic scenes with ``scene_token`` while NAVSIM's
    SceneFilter expects the current-frame ``token``.  Selecting one central
    valid windows prevents a scene run from silently expanding into every
    highly correlated sliding-window sample while allowing controlled temporal
    diversity.
    """
    window = int(num_history_frames) + int(num_future_frames)
    stride = 1 if frame_interval is None else int(frame_interval)
    if window <= 0 or stride <= 0:
        raise ValueError("history/future window and frame_interval must be positive")
    if int(windows_per_scene) <= 0:
        raise ValueError("windows_per_scene must be positive")
    selected: list[str] = []
    for row_no, row in enumerate(rows, start=1):
        explicit = row.get("sample_token") or row.get("frame_token")
        if explicit is not None:
            selected.append(str(explicit))
            continue
        metadata_path = row.get("metadata_path")
        if not metadata_path:
            raise ValueError(f"{manifest_path}:{row_no} lacks metadata_path/sample_token")
        resolved = Path(str(metadata_path)).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"manifest metadata not found: {resolved}")
        with resolved.open("rb") as handle:
            frames = pickle.load(handle)
        starts = []
        for start in range(0, max(0, len(frames) - window + 1), stride):
            current = frames[start + int(num_history_frames) - 1]
            if has_route and not current.get("roadblock_ids"):
                continue
            starts.append(start)
        if not starts:
            raise ValueError(f"no valid training window in {resolved}")
        count = min(int(windows_per_scene), len(starts))
        if count == len(starts):
            selected_starts = starts
        else:
            # Interior quantiles avoid always choosing only the first/last
            # feasible moments while keeping selected windows well separated.
            indices = [min(len(starts) - 1, int((i + 1) * len(starts) / (count + 1))) for i in range(count)]
            selected_starts = [starts[index] for index in indices]
        selected.extend(
            str(frames[start + int(num_history_frames) - 1]["token"])
            for start in selected_starts
        )
    if len(set(selected)) != len(selected):
        raise ValueError(f"representative frame tokens are not unique in {manifest_path}")
    return selected


def _module_training_states(root: torch.nn.Module) -> Dict[torch.nn.Module, bool]:
    return {module: bool(module.training) for module in root.modules()}


def _restore_module_training_states(states: Dict[torch.nn.Module, bool]) -> None:
    for module, training in states.items():
        module.train(training)


@contextlib.contextmanager
def _use_ema_trainable_weights(model: "DriveVANavsimTrainingModule", enabled: bool):
    if not enabled or not hasattr(model, "has_ema") or not model.has_ema():
        yield False
        return

    shadow = getattr(model, "_ema_shadow", {})
    backup: Dict[str, torch.Tensor] = {}
    try:
        with torch.no_grad():
            for name, param in model.named_parameters():
                ema_value = shadow.get(name)
                if ema_value is None:
                    continue
                backup[name] = param.detach().cpu().clone()
                param.copy_(ema_value.to(device=param.device, dtype=param.dtype))
        yield True
    finally:
        with torch.no_grad():
            for name, param in model.named_parameters():
                raw_value = backup.get(name)
                if raw_value is not None:
                    param.copy_(raw_value.to(device=param.device, dtype=param.dtype))


class InProcessAutoEvalModelLogger(ModelLogger):
    def __init__(
        self,
        *args,
        train_args: argparse.Namespace,
        auto_eval: bool,
        auto_eval_ckpt_kind: str = "all",
        auto_eval_strict: bool = False,
        infer_all_output_root: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.train_args = train_args
        self.auto_eval = bool(auto_eval)
        self.auto_eval_ckpt_kind = str(auto_eval_ckpt_kind or "all").lower()
        self.auto_eval_strict = bool(auto_eval_strict)
        self.infer_all_output_root = infer_all_output_root or os.path.join(train_args.output_path, "infer_each_ckpt")
        self._eval_modules: Dict[str, Any] = {}
        if self.auto_eval_ckpt_kind not in {"raw", "ema", "all"}:
            raise ValueError(f"auto_eval_ckpt_kind must be raw, ema, or all; got {self.auto_eval_ckpt_kind}")

    def on_step_end(self, accelerator, model, save_steps=None):
        self.num_steps += 1
        if save_steps is not None and self.num_steps % int(save_steps) == 0:
            file_name = f"step-{self.num_steps}.safetensors"
            self.save_model(accelerator, model, file_name)
            self._run_auto_eval(accelerator, model, os.path.splitext(file_name)[0])

    def on_epoch_end(self, accelerator, model, epoch_id):
        file_name = f"epoch-{epoch_id}.safetensors"
        self.save_model(accelerator, model, file_name)
        self._run_auto_eval(accelerator, model, os.path.splitext(file_name)[0])

    def on_training_end(self, accelerator, model, save_steps=None):
        if save_steps is not None and self.num_steps % int(save_steps) != 0:
            file_name = f"step-{self.num_steps}.safetensors"
            self.save_model(accelerator, model, file_name)
            self._run_auto_eval(accelerator, model, os.path.splitext(file_name)[0])

    def _eval_module(self, name: str):
        if name not in self._eval_modules:
            self._eval_modules[name] = importlib.import_module(name)
        return self._eval_modules[name]

    def _eval_variants(self, unwrapped_model: DriveVANavsimTrainingModule) -> list[tuple[str, bool]]:
        if self.auto_eval_ckpt_kind == "raw":
            return [("", False)]
        if self.auto_eval_ckpt_kind == "ema":
            if hasattr(unwrapped_model, "has_ema") and unwrapped_model.has_ema():
                return [("-ema", True)]
            return []
        variants = [("", False)]
        if hasattr(unwrapped_model, "has_ema") and unwrapped_model.has_ema():
            variants.append(("-ema", True))
        return variants

    def _build_navsim_args(self, tag: str, ckpt_path: str) -> argparse.Namespace:
        mod = self._eval_module("examples.wanvideo.driveva_infer.eval_navsim_pdm")
        out_dir = os.path.join(self.infer_all_output_root, tag, "navsim_v1")
        argv = [
            "--repo_root", self.train_args.repo_root,
            "--navsim_log_path", str(_env_first("EVAL_V1_NAVSIM_LOG_PATH", "EVAL_NAVSIM_LOG_PATH", default=_NAVSIM_EVAL_LOG_DEFAULT)),
            "--sensor_blobs_path", str(_env_first("EVAL_V1_SENSOR_BLOBS_PATH", "EVAL_SENSOR_BLOBS_PATH", default=_NAVSIM_EVAL_SENSOR_DEFAULT)),
            "--metric_cache_path", str(_env_first("EVAL_V1_METRIC_CACHE_PATH", "EVAL_METRIC_CACHE_PATH", default=_NAVSIM_EVAL_CACHE_DEFAULT)),
            "--output_dir", out_dir,
            "--local_model_path", str(self.train_args.local_model_path),
            "--full_ckpt", ckpt_path,
            "--num_inference_steps", str(_env_first("NAVSIM_NUM_INFERENCE_STEPS", "EVAL_NUM_INFERENCE_STEPS", "NUM_INFERENCE_STEPS", default=3)),
            "--cfg_scale", str(_env_first("NAVSIM_CFG_SCALE", "EVAL_CFG_SCALE", "CFG_SCALE", default=1.0)),
            "--seed", str(_env_first("NAVSIM_SEED", "EVAL_SEED", "SEED", default=0)),
            "--num_history_frames", str(_env_first("EVAL_NUM_HISTORY_FRAMES", default=self.train_args.num_history_frames)),
            "--num_future_frames", str(_env_first("EVAL_NUM_FUTURE_FRAMES", default=10)),
            "--model_future_frames", str(_env_first("EVAL_MODEL_FUTURE_FRAMES", default=self.train_args.num_future_frames)),
            "--target_fps", str(_env_first("EVAL_TARGET_FPS", default=self.train_args.target_fps)),
            "--height", str(_env_first("EVAL_HEIGHT", default=self.train_args.height)),
            "--width", str(_env_first("EVAL_WIDTH", default=self.train_args.width)),
            "--trajectory_condition_mode", str(_env_first("EVAL_TRAJECTORY_CONDITION_MODE", default=self.train_args.trajectory_condition_mode)),
            "--scene_future_extra_seconds", str(_env_first("EVAL_V1_SCENE_FUTURE_EXTRA_SECONDS", "EVAL_SCENE_FUTURE_EXTRA_SECONDS", default=1.0)),
            "--pdm_num_poses", str(_env_first("EVAL_PDM_NUM_POSES", "PDM_NUM_POSES", default=40)),
            "--pdm_interval_length", str(_env_first("EVAL_PDM_INTERVAL_LENGTH", "PDM_INTERVAL_LENGTH", default=0.1)),
            "--traffic_agents_policy", str(_env_first("EVAL_TRAFFIC_AGENTS_POLICY", "TRAFFIC_AGENTS_POLICY", default="non_reactive")),
            "--viz_total_tokens", str(_env_first("EVAL_VIZ_TOTAL_TOKENS", default=100)),
            "--debug_prompt_steps", str(_env_first("EVAL_DEBUG_PROMPT_STEPS", default=0)),
            "--no_show_eval_progress",
        ]
        _add_optional_arg(argv, "--scene_filter_yaml", _env_first("EVAL_SCENE_FILTER_YAML", default=os.path.join(self.train_args.repo_root, "examples/wanvideo/driveva_infer/navsim_scene_filters/navtest.yaml")))
        if _env_bool("EVAL_SCENE_FILTER_YAML_FILTER_ONLY", default=True):
            argv.extend(["--scene_filter_yaml_filter_only", "1"])
        _add_optional_arg(argv, "--log_names", _env_first("EVAL_LOG_NAMES", "EVAL_V1_LOG_NAMES"))
        _add_optional_arg(argv, "--max_scenes", _env_first("EVAL_MAX_SCENES", "EVAL_V1_MAX_SCENES"))
        _add_optional_arg(argv, "--max_eval_tokens", _env_first("EVAL_MAX_EVAL_TOKENS", "MAX_EVAL_TOKENS"))
        _add_optional_arg(argv, "--num_eval_shards", _env_first("EVAL_NUM_SHARDS", "NUM_EVAL_SHARDS"))
        if _env_bool("EVAL_SAVE_VIZ", default=False):
            argv.append("--save_viz")
        if _env_bool("EVAL_INFER_TRAJECTORY_ONLY", default=True):
            argv.append("--infer_trajectory_only")
        else:
            argv.append("--no_infer_trajectory_only")
        if _env_bool("AUTO_EVAL_NUSCENES_METRICS", "ENABLE_NUSCENES_METRICS", default=False):
            argv.append("--enable_nuscenes_metrics")
            _add_optional_arg(argv, "--nuscenes_metric_horizons_s", _env_first("EVAL_NUSCENES_METRIC_HORIZONS_S", "NUSCENES_METRIC_HORIZONS_S", default="1,2,3"))
        return mod.parse_args(argv)

    def _build_nuscenes_args(self, tag: str, ckpt_path: str) -> Optional[argparse.Namespace]:
        dataroot = _env_first("NUSCENES_EVAL_DATAROOT", "NUSCENES_DATAROOT")
        if not dataroot:
            return None
        mod = self._eval_module("examples.wanvideo.driveva_infer.infer_nuscenes")
        out_dir = os.path.join(self.infer_all_output_root, tag, "nuscenes")
        argv = [
            "--nuscenes_dataroot", str(dataroot),
            "--nuscenes_version", str(_env_first("NUSCENES_EVAL_VERSION", "NUSCENES_VERSION", default="v1.0-trainval")),
            "--split", str(_env_first("NUSCENES_EVAL_SPLIT", "NUSCENES_SPLIT", default="val")),
            "--camera_name", str(_env_first("NUSCENES_EVAL_CAMERA_NAME", "CAMERA_NAME", default="CAM_FRONT")),
            "--output_dir", out_dir,
            "--local_model_path", str(self.train_args.local_model_path),
            "--full_ckpt", ckpt_path,
            "--num_inference_steps", str(_env_first("NUSCENES_EVAL_NUM_INFERENCE_STEPS", "NUM_INFERENCE_STEPS", default=3)),
            "--cfg_scale", str(_env_first("NUSCENES_EVAL_CFG_SCALE", "CFG_SCALE", default=1.0)),
            "--seed", str(_env_first("NUSCENES_EVAL_SEED", "SEED", default=0)),
            "--debug_prompt_steps", str(_env_first("NUSCENES_EVAL_DEBUG_PROMPT_STEPS", default=0)),
            "--trajectory_condition_mode", str(_env_first("NUSCENES_EVAL_TRAJECTORY_CONDITION_MODE", default=self.train_args.trajectory_condition_mode)),
            "--num_history_frames", str(_env_first("NUSCENES_EVAL_NUM_HISTORY_FRAMES", default=self.train_args.num_history_frames)),
            "--num_future_frames", str(_env_first("NUSCENES_EVAL_NUM_FUTURE_FRAMES", default=self.train_args.num_future_frames)),
            "--model_future_frames", str(_env_first("NUSCENES_EVAL_MODEL_FUTURE_FRAMES", default=self.train_args.num_future_frames)),
            "--scene_future_extra_seconds", str(_env_first("NUSCENES_EVAL_SCENE_FUTURE_EXTRA_SECONDS", default=0.0)),
            "--target_fps", str(_env_first("NUSCENES_EVAL_TARGET_FPS", default=self.train_args.target_fps)),
            "--metric_horizons_s", str(_env_first("NUSCENES_EVAL_METRIC_HORIZONS_S", default="1,2,3")),
            "--ego_box_length_m", str(_env_first("NUSCENES_EVAL_EGO_BOX_LENGTH_M", default=4.084)),
            "--ego_box_width_m", str(_env_first("NUSCENES_EVAL_EGO_BOX_WIDTH_M", default=1.85)),
            "--height", str(_env_first("NUSCENES_EVAL_HEIGHT", default=self.train_args.height)),
            "--width", str(_env_first("NUSCENES_EVAL_WIDTH", default=self.train_args.width)),
            "--no_show_eval_progress",
        ]
        _add_optional_arg(argv, "--policy_anno_json", _env_first("NUSCENES_EVAL_POLICY_ANNO_JSON", "POLICY_ANNO_JSON"))
        _add_optional_arg(argv, "--max_scenes", _env_first("NUSCENES_EVAL_MAX_SCENES"))
        _add_optional_arg(argv, "--max_eval_tokens", _env_first("NUSCENES_EVAL_MAX_EVAL_TOKENS", "MAX_EVAL_TOKENS"))
        if _env_bool("NUSCENES_EVAL_SAVE_VIZ", default=False):
            argv.append("--save_viz")
            _add_optional_arg(argv, "--viz_max_tokens", _env_first("NUSCENES_EVAL_VIZ_MAX_TOKENS", default=20))
        return mod.parse_args(argv)

    def _build_b2d_args(self, tag: str, ckpt_path: str) -> Optional[argparse.Namespace]:
        data_root = _env_first("B2D_DATA_ROOT", "BENCH2DRIVE_DATA_ROOT")
        ann_file = _env_first("B2D_EVAL_ANN_FILE", "B2D_ANN_FILE", "BENCH2DRIVE_ANN_FILE")
        if not data_root or not ann_file:
            return None
        mod = self._eval_module("examples.wanvideo.driveva_infer.infer_bench2drive")
        out_dir = os.path.join(self.infer_all_output_root, tag, "bench2drive")
        argv = [
            "--data_root", str(data_root),
            "--ann_file", str(ann_file),
            "--output_dir", out_dir,
            "--local_model_path", str(self.train_args.local_model_path),
            "--full_ckpt", ckpt_path,
            "--num_inference_steps", str(_env_first("B2D_EVAL_NUM_INFERENCE_STEPS", "NUM_INFERENCE_STEPS", default=3)),
            "--cfg_scale", str(_env_first("B2D_EVAL_CFG_SCALE", "CFG_SCALE", default=1.0)),
            "--seed", str(_env_first("B2D_EVAL_SEED", "SEED", default=0)),
            "--num_history_frames", str(_env_first("B2D_EVAL_NUM_HISTORY_FRAMES", default=self.train_args.num_history_frames)),
            "--num_future_frames", str(_env_first("B2D_EVAL_NUM_FUTURE_FRAMES", default=self.train_args.num_future_frames)),
            "--model_future_frames", str(_env_first("B2D_EVAL_MODEL_FUTURE_FRAMES", default=self.train_args.num_future_frames)),
            "--target_fps", str(_env_first("B2D_EVAL_TARGET_FPS", default=self.train_args.target_fps)),
            "--original_fps", str(_env_first("B2D_EVAL_ORIGINAL_FPS", "B2D_ORIGINAL_FPS", default=10)),
            "--frame_interval", str(_env_first("B2D_EVAL_FRAME_INTERVAL", "B2D_FRAME_INTERVAL", default=2)),
            "--height", str(_env_first("B2D_EVAL_HEIGHT", default=self.train_args.height)),
            "--width", str(_env_first("B2D_EVAL_WIDTH", default=self.train_args.width)),
            "--command_yaw_threshold_deg", str(_env_first("B2D_EVAL_COMMAND_YAW_THRESHOLD_DEG", default=8.0)),
            "--viz_max_scenes", str(_env_first("B2D_EVAL_VIZ_MAX_SCENES", default=30)),
            "--viz_plot_height", str(_env_first("B2D_EVAL_VIZ_PLOT_HEIGHT", default=420)),
            "--debug_prompt_steps", str(_env_first("B2D_EVAL_DEBUG_PROMPT_STEPS", default=0)),
            "--projection_debug_steps", str(_env_first("B2D_EVAL_PROJECTION_DEBUG_STEPS", default=0)),
            "--projection_debug_token", str(_env_first("B2D_EVAL_PROJECTION_DEBUG_TOKEN", default="")),
            "--distributed",
            "--no_show_eval_progress",
        ]
        _add_optional_arg(argv, "--max_scenes", _env_first("B2D_EVAL_MAX_SCENES"))
        if _env_bool("B2D_EVAL_SAVE_VIZ", default=False):
            argv.append("--save_viz")
        else:
            argv.append("--no_save_viz")
        if _env_bool("B2D_EVAL_COMPUTE_PLANNING_METRICS", default=True):
            argv.append("--compute_planning_metrics")
        else:
            argv.append("--no_compute_planning_metrics")
        return mod.parse_args(argv)

    def _run_auto_eval(self, accelerator, model, tag_base: str):
        if not self.auto_eval:
            return
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        pipe = unwrapped_model.pipe
        raw_ckpt_path = os.path.join(self.output_path, f"{tag_base}.safetensors")
        variants = self._eval_variants(unwrapped_model)
        if not variants:
            if accelerator.is_main_process:
                print("[train][eval][warn] auto eval requested but no matching checkpoint variant is available.")
            return

        module_states = _module_training_states(pipe)
        scheduler = getattr(pipe, "scheduler", None)
        scheduler_state = None
        if scheduler is not None:
            try:
                scheduler_state = {
                    "num_inference_steps": int(len(scheduler.timesteps)),
                    "training": bool(getattr(scheduler, "training", False)),
                    "shift": float(getattr(scheduler, "shift", 5.0)),
                }
            except Exception:
                scheduler_state = None
        attrs = {
            name: getattr(pipe, name)
            for name in (
                "target_fps",
                "num_history_frames",
                "trajectory_norm_mode",
                "trajectory_use_relative",
                "trajectory_condition_mode",
                "infer_replace_history_latents_before_decode",
            )
            if hasattr(pipe, name)
        }
        python_rng_state = random.getstate()
        numpy_rng_state = np.random.get_state()
        torch_cpu_rng_state = torch.random.get_rng_state()
        torch_cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        vae_tqdm = None
        try:
            import diffsynth.models.wan_video_vae as wan_video_vae_mod

            vae_tqdm = getattr(wan_video_vae_mod, "tqdm", None)
        except Exception:
            vae_tqdm = None
        try:
            for param in unwrapped_model.parameters():
                param.grad = None
            torch.cuda.empty_cache()
            pipe.eval()
            for suffix, use_ema in variants:
                tag = f"{tag_base}{suffix}"
                ckpt_path = os.path.join(self.output_path, f"{tag}.safetensors") if suffix else raw_ckpt_path
                if accelerator.is_main_process:
                    print(
                        "[train][eval] start:",
                        f"tag={tag}",
                        f"world_size={accelerator.num_processes}",
                        f"ema={use_ema}",
                        f"output={os.path.join(self.infer_all_output_root, tag)}",
                    )
                try:
                    with _use_ema_trainable_weights(unwrapped_model, enabled=use_ema):
                        with torch.inference_mode():
                            if _env_bool("RUN_NAVSIM", "AUTO_EVAL_NAVSIM_V1", default=True):
                                navsim_mod = self._eval_module("examples.wanvideo.driveva_infer.eval_navsim_pdm")
                                navsim_mod.run_eval(self._build_navsim_args(tag, ckpt_path), external_pipe=pipe)
                                accelerator.wait_for_everyone()
                            if _env_bool("RUN_NUSCENES", "AUTO_EVAL_NUSCENES", default=True):
                                nuscenes_args = self._build_nuscenes_args(tag, ckpt_path)
                                if nuscenes_args is not None:
                                    nusc_mod = self._eval_module("examples.wanvideo.driveva_infer.infer_nuscenes")
                                    nusc_mod.run_eval(nuscenes_args, external_pipe=pipe)
                                    accelerator.wait_for_everyone()
                                elif accelerator.is_main_process:
                                    print("[train][eval][nuscenes][warn] dataroot not set; skip.")
                            if _env_bool("RUN_B2D", "AUTO_EVAL_B2D_VIZ", default=True):
                                b2d_args = self._build_b2d_args(tag, ckpt_path)
                                if b2d_args is not None:
                                    b2d_mod = self._eval_module("examples.wanvideo.driveva_infer.infer_bench2drive")
                                    b2d_mod.run_eval(b2d_args, external_pipe=pipe)
                                    accelerator.wait_for_everyone()
                                elif accelerator.is_main_process:
                                    print("[train][eval][b2d][warn] data root or ann file not set; skip.")
                except Exception as exc:
                    if accelerator.is_main_process:
                        print(f"[train][eval][error] tag={tag} failed: {exc}")
                    if self.auto_eval_strict:
                        raise
                finally:
                    torch.cuda.empty_cache()
                    accelerator.wait_for_everyone()
        finally:
            for name, value in attrs.items():
                setattr(pipe, name, value)
            if vae_tqdm is not None:
                try:
                    import diffsynth.models.wan_video_vae as wan_video_vae_mod

                    wan_video_vae_mod.tqdm = vae_tqdm
                except Exception:
                    pass
            try:
                random.setstate(python_rng_state)
                np.random.set_state(numpy_rng_state)
                torch.random.set_rng_state(torch_cpu_rng_state)
                if torch_cuda_rng_state is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(torch_cuda_rng_state)
            except Exception as exc:
                if accelerator.is_main_process:
                    print(f"[train][eval][warn] failed to restore RNG states: {exc}")
            _restore_module_training_states(module_states)
            if scheduler is not None and scheduler_state is not None:
                try:
                    scheduler.set_timesteps(
                        scheduler_state["num_inference_steps"],
                        training=scheduler_state["training"],
                        shift=scheduler_state["shift"],
                    )
                except Exception as exc:
                    if accelerator.is_main_process:
                        print(f"[train][eval][warn] failed to restore scheduler state: {exc}")
            accelerator.wait_for_everyone()


class DriveVANavsimTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        *,
        local_model_path: Optional[str],
        trainable_models: Optional[str],
        lora_base_model: Optional[str],
        lora_target_modules: str,
        lora_rank: int,
        lora_checkpoint: Optional[str],
        use_gradient_checkpointing: bool,
        use_gradient_checkpointing_offload: bool,
        extra_inputs: Optional[str],
        max_timestep_boundary: float,
        min_timestep_boundary: float,
        target_fps: int,
        negative_prompt: str,
        use_trajectory: bool,
        train_future_video_noise_only: bool,
        infer_replace_history_latents_before_decode: bool,
        trajectory_condition_mode: str,
        num_history_frames: int,
        enable_online_selector: bool = False,
        selector_warmup_steps: int = 0,
        selector_layer: int = 15,
        selector_loss_weight: float = 1.0,
        selector_keep_schedule: str = "",
        selector_input_variant: str = "token_condition",
        selector_feature_mode: str = "all",
        selector_teacher_keep_ratio: float = 0.375,
        selector_gradient_interval: int = 1,
        selector_mask_start_step: int = -1,
        selector_only: bool = False,
        selector_checkpoint: Optional[str] = None,
        selector_teacher_mode: str = "gradient_abs",
        selector_signed_temperature: float = 1.0,
        selector_counterfactual_interval: int = 4,
        selector_counterfactual_weight: float = 1.0,
        selector_counterfactual_scale: float = 0.05,
        selector_counterfactual_tile_h: int = 3,
        selector_counterfactual_tile_w: int = 4,
        selector_counterfactual_physical: bool = False,
        selector_counterfactual_injection_point: str = "post_block",
        selector_teacher_timesteps: str = "",
        selector_teacher_seed: Optional[int] = None,
        selector_counterfactual_dump_dir: Optional[str] = None,
        selector_counterfactual_replicate: int = 0,
        selector_counterfactual_sweep_all: bool = False,
        selector_counterfactual_replays: int = 1,
        selector_counterfactual_scales: str = "",
        selector_counterfactual_abstain_eps: float = 0.0,
        selector_counterfactual_noise_seed: int = 1234,
        selector_counterfactual_latent_index: str = "0",
        selector_counterfactual_jsonl_dir: Optional[str] = None,
        selector_teacher_disp_scale: float = 0.01,
        selector_teacher_disp_normalize: str = "scene",
        selector_teacher_disp_min_spread: float = 0.0,
        selector_ranking_loss_weight: float = 0.0,
        selector_ranking_margin: float = 0.0,
        selector_ranking_max_pairs: int = 4096,
        selector_critical_token_mode: str = "none",
        selector_critical_token_dilation: float = 0.0,
        selector_long_horizon_weights: str = "",
    ):
        super().__init__()
        self.negative_prompt = negative_prompt
        self.target_fps = target_fps
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
        self.use_gradient_checkpointing_offload = bool(use_gradient_checkpointing_offload)
        self.extra_inputs = [x.strip() for x in extra_inputs.split(",") if x.strip()] if extra_inputs else []
        self.max_timestep_boundary = float(max_timestep_boundary)
        self.min_timestep_boundary = float(min_timestep_boundary)
        self.enable_online_selector = bool(enable_online_selector)
        self.selector_warmup_steps = max(0, int(selector_warmup_steps))
        self.selector_layer = int(selector_layer)
        self.selector_loss_weight = float(selector_loss_weight)
        self.selector_keep_schedule = parse_keep_schedule(selector_keep_schedule)
        self.selector_input_variant = str(selector_input_variant)
        self.selector_teacher_keep_ratio = float(selector_teacher_keep_ratio)
        self.selector_gradient_interval = max(1, int(selector_gradient_interval))
        self.selector_mask_start_step = int(selector_mask_start_step)
        self.selector_only = bool(selector_only)
        self.selector_feature_mode = str(selector_feature_mode)
        self.selector = (
            DynamicTokenSelector(feature_mode=self.selector_feature_mode)
            if self.enable_online_selector
            else None
        )
        self.selector_teacher_mode = str(selector_teacher_mode)
        self.selector_signed_temperature = float(selector_signed_temperature)
        self.selector_counterfactual_interval = max(1, int(selector_counterfactual_interval))
        self.selector_counterfactual_weight = float(selector_counterfactual_weight)
        self.selector_counterfactual_scale = float(selector_counterfactual_scale)
        self.selector_counterfactual_tile_h = int(selector_counterfactual_tile_h)
        self.selector_counterfactual_tile_w = int(selector_counterfactual_tile_w)
        self.selector_counterfactual_physical = bool(selector_counterfactual_physical)
        self.selector_counterfactual_injection_point = str(
            selector_counterfactual_injection_point
        )
        if self.selector_counterfactual_injection_point not in {
            "post_block",
            "pre_dit",
        }:
            raise ValueError(
                "selector counterfactual injection point must be post_block or pre_dit"
            )
        self.selector_teacher_seed = (
            None if selector_teacher_seed is None else int(selector_teacher_seed)
        )
        self.selector_counterfactual_dump_dir = (
            None
            if selector_counterfactual_dump_dir in (None, "")
            else Path(str(selector_counterfactual_dump_dir)).expanduser().resolve()
        )
        # 0 disables the determinism control.  N>0 repeats the masked forward N
        # extra times so the counterfactual label's numerical noise floor is
        # measured instead of assumed.
        self.selector_counterfactual_replicate = max(
            0, int(selector_counterfactual_replicate)
        )
        # Probe every spatial tile of the scene in one optimizer step instead of
        # one tile per step (review P1 label-coverage fix).
        self.selector_counterfactual_sweep_all = bool(selector_counterfactual_sweep_all)
        # Number of independent noise realisations averaged into one label
        # (review P3-1).  >1 replaces the single random planning-residual
        # direction with its expectation.
        self.selector_counterfactual_replays = max(1, int(selector_counterfactual_replays))
        # Optional comma-separated removal sizes in tiles; each is probed in the
        # same step so the perturbation-size dose-response is measurable.
        self.selector_counterfactual_scales = tuple(
            int(value.strip())
            for value in str(selector_counterfactual_scales).split(",")
            if value.strip()
        )
        if any(value <= 0 for value in self.selector_counterfactual_scales):
            raise ValueError("selector counterfactual scales must be positive tile counts")
        # Dead zone for the tile label (review P1): probes inside +-abstain_eps
        # get zero weight instead of a hard 0/1 target.
        self.selector_counterfactual_abstain_eps = float(selector_counterfactual_abstain_eps)
        if not math.isfinite(self.selector_counterfactual_abstain_eps) or self.selector_counterfactual_abstain_eps < 0:
            raise ValueError("selector counterfactual abstain_eps must be finite and non-negative")
        self.selector_counterfactual_noise_seed = int(selector_counterfactual_noise_seed)
        # Which conditioned history latent the counterfactual teacher probes,
        # counted back from the newest one: 0 == newest (historical default),
        # 1 == the previous history latent, ...  A comma-separated list probes
        # several latents in the same optimizer step (route-temporal-key-set
        # probe, 2026-09-11).  The default must reproduce the historical
        # single-latent behaviour exactly.
        self.selector_counterfactual_latent_indices = tuple(
            int(value.strip())
            for value in str(selector_counterfactual_latent_index).split(",")
            if value.strip()
        )
        if not self.selector_counterfactual_latent_indices:
            self.selector_counterfactual_latent_indices = (0,)
        if any(value < 0 for value in self.selector_counterfactual_latent_indices):
            raise ValueError(
                "selector counterfactual latent indices must be non-negative "
                "(they count back from the newest history latent)"
            )
        self.selector_counterfactual_jsonl_dir = (
            None
            if selector_counterfactual_jsonl_dir in (None, "")
            else Path(str(selector_counterfactual_jsonl_dir)).expanduser().resolve()
        )
        if self.selector_counterfactual_jsonl_dir is not None:
            self.selector_counterfactual_jsonl_dir.mkdir(parents=True, exist_ok=True)
        # Displacement teacher: relative plan displacement caused by removing a
        # tile, scaled into (0, 1) so the selector logit keeps "keep-worthiness"
        # semantics.  Default 0.01 is ~3x the mean measured displacement.
        self.selector_teacher_disp_scale = float(selector_teacher_disp_scale)
        if not math.isfinite(self.selector_teacher_disp_scale) or self.selector_teacher_disp_scale <= 0:
            raise ValueError("selector teacher disp_scale must be finite and positive")
        self.selector_teacher_disp_normalize = str(selector_teacher_disp_normalize)
        self.selector_teacher_disp_min_spread = float(selector_teacher_disp_min_spread)
        if (
            not math.isfinite(self.selector_teacher_disp_min_spread)
            or self.selector_teacher_disp_min_spread < 0
        ):
            raise ValueError("selector_teacher_disp_min_spread must be finite and non-negative")
        if self.selector_teacher_disp_normalize not in {"scene", "absolute"}:
            raise ValueError("selector teacher disp_normalize must be scene or absolute")
        self.selector_ranking_loss_weight = float(selector_ranking_loss_weight)
        self.selector_ranking_margin = float(selector_ranking_margin)
        self.selector_ranking_max_pairs = int(selector_ranking_max_pairs)
        if not math.isfinite(self.selector_ranking_loss_weight) or self.selector_ranking_loss_weight < 0:
            raise ValueError("selector ranking loss weight must be finite and non-negative")
        if not math.isfinite(self.selector_ranking_margin) or self.selector_ranking_margin < 0:
            raise ValueError("selector ranking margin must be finite and non-negative")
        if self.selector_ranking_max_pairs <= 0:
            raise ValueError("selector ranking max pairs must be positive")
        self.selector_critical_token_mode = str(selector_critical_token_mode)
        if self.selector_critical_token_mode not in {"none", "provided", "boxes"}:
            raise ValueError("selector critical token mode must be none, provided, or boxes")
        self.selector_critical_token_dilation = float(selector_critical_token_dilation)
        if not math.isfinite(self.selector_critical_token_dilation) or self.selector_critical_token_dilation < 0:
            raise ValueError("selector critical token dilation must be finite and non-negative")
        self.selector_long_horizon_weights = parse_horizon_weights(
            selector_long_horizon_weights
        )
        if self.selector_counterfactual_dump_dir is not None:
            self.selector_counterfactual_dump_dir.mkdir(parents=True, exist_ok=True)
        self.selector_teacher_timesteps = [
            float(value.strip())
            for value in str(selector_teacher_timesteps).split(",")
            if value.strip()
        ]
        if any(
            not math.isfinite(value) or value < 0.0 or value > 1000.0
            for value in self.selector_teacher_timesteps
        ):
            raise ValueError("selector teacher timesteps must be finite values in [0, 1000]")
        if self.selector is not None and selector_checkpoint:
            selector_path = Path(selector_checkpoint).expanduser().resolve()
            if not selector_path.is_file():
                raise FileNotFoundError(f"selector checkpoint not found: {selector_path}")
            state = load_state_dict(str(selector_path))
            if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
                state = state["state_dict"]
            normalized = {}
            for key, value in state.items():
                name = str(key)
                if name.startswith("module."):
                    name = name[len("module.") :]
                if name.startswith("selector."):
                    name = name[len("selector.") :]
                normalized[name] = value
            missing, unexpected = self.selector.load_state_dict(normalized, strict=False)
            incompatible_missing = [
                key
                for key in missing
                if not key.startswith(
                    ("context_mlp.", "context_scoring.", "timestep_mlp.", "timestep_scoring.")
                )
            ]
            if incompatible_missing or unexpected:
                raise ValueError(
                    f"incompatible selector checkpoint {selector_path}: "
                    f"missing={incompatible_missing}, unexpected={unexpected}"
                )
            print(f"[selector] initialized from {selector_path}")
        self.global_step = 0

        model_configs = [
            ModelConfig(
                model_id="Wan-AI/Wan2.2-TI2V-5B",
                origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
                offload_device="cpu",
                local_model_path=local_model_path,
                skip_download=True,
            ),
            ModelConfig(
                model_id="Wan-AI/Wan2.2-TI2V-5B",
                origin_file_pattern="diffusion_pytorch_model*.safetensors",
                offload_device="cpu",
                local_model_path=local_model_path,
                skip_download=True,
            ),
            ModelConfig(
                model_id="Wan-AI/Wan2.2-TI2V-5B",
                origin_file_pattern="Wan2.2_VAE.pth",
                offload_device="cpu",
                local_model_path=local_model_path,
                skip_download=True,
            ),
        ]
        tokenizer_config = ModelConfig(
            model_id="Wan-AI/Wan2.2-TI2V-5B",
            origin_file_pattern="google/*",
            local_model_path=local_model_path,
            skip_download=True,
        )
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cpu",
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            use_trajectory=use_trajectory,
        )
        self.pipe.target_fps = target_fps
        self.pipe.num_history_frames = max(0, int(num_history_frames))
        self.pipe.train_future_video_noise_only = bool(train_future_video_noise_only)
        self.pipe.infer_replace_history_latents_before_decode = bool(infer_replace_history_latents_before_decode)
        self.pipe.trajectory_norm_mode = "driveva_odo"
        self.pipe.trajectory_use_relative = False
        self.pipe.trajectory_condition_mode = str(trajectory_condition_mode).lower()

        self.switch_pipe_to_training_mode(
            self.pipe,
            trainable_models,
            lora_base_model,
            lora_target_modules,
            lora_rank,
            lora_checkpoint=lora_checkpoint,
        )

    # ------------------------------------------------------------------
    # Counterfactual teacher helpers (review P0/P1/P3, 2026-09-11)
    # ------------------------------------------------------------------
    def _counterfactual_probe_specs(
        self, event: int, world_size: int, rank: int, group_count: int
    ) -> list[dict]:
        """Return the interventions to probe in this step.

        Each spec is ``{"tiles": (...), "removal_count": n, "anchor_tile": t}``.
        ``selector_counterfactual_scales`` (perturbation-size sweep) takes
        precedence, then exhaustive tile sweep, then the historical single-tile
        rotation.
        """
        if self.selector_counterfactual_scales:
            specs = []
            for scale in self.selector_counterfactual_scales:
                count = max(1, min(int(scale), group_count))
                offset = (event * max(1, world_size) + rank) % group_count
                tiles = tuple(sorted((offset + step) % group_count for step in range(count)))
                specs.append(
                    {"tiles": tiles, "removal_count": count, "anchor_tile": offset}
                )
            return specs
        if self.selector_counterfactual_sweep_all:
            return [
                {"tiles": (tile,), "removal_count": 1, "anchor_tile": tile}
                for tile in range(group_count)
            ]
        anchor = (event * world_size + rank) % group_count
        return [{"tiles": (anchor,), "removal_count": 1, "anchor_tile": anchor}]

    def _counterfactual_mask(
        self, positions: torch.Tensor, spec: dict, *, latent_index: int = 0
    ):
        """Keep-mask and membership for one spec (possibly several tiles).

        ``positions`` is the (t, y, x) grid of the conditioned history latent
        selected by ``latent_index`` (0 == newest).  Every history latent shares
        the same patch grid, so the tile membership is latent-independent; the
        token range the mask is applied to is selected downstream by
        ``counterfactual_latent_index`` inside ``model_fn_wan_video``.
        """
        membership = None
        for tile in spec["tiles"]:
            _, single = spatial_counterfactual_probe(
                positions,
                int(tile),
                tile_h=self.selector_counterfactual_tile_h,
                tile_w=self.selector_counterfactual_tile_w,
            )
            membership = single if membership is None else (membership | single)
        # A spec whose tiles contain no tokens at all is a bug in the grid
        # arithmetic.  Removing *all* tiles is legitimate (whole-frame deletion)
        # and the pipeline restores the sequence length by scattering the kept
        # tokens back into a zero tensor.
        if membership is None or not membership.any(dim=1).all():
            raise ValueError(f"counterfactual spec {spec} selects no tokens")
        if int(latent_index) != int(spec.get("latent_index", latent_index)):
            raise ValueError(
                "counterfactual latent index mismatch between spec and probe loop"
            )
        return (~membership).to(dtype=positions.dtype), membership

    def _counterfactual_replay_noise(
        self, replay_index: int, reference: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """A fresh but reproducible video-noise realisation.

        Changing the video noise is what changes the planning residual
        direction, which is the quantity that turns the per-probe sign into a
        coin flip; averaging over realisations replaces it with its mean.
        """
        if reference is None:
            return None
        generator = torch.Generator("cpu").manual_seed(
            int(self.selector_counterfactual_noise_seed) + int(replay_index) * 104729
        )
        noise = torch.randn(
            tuple(reference.shape), generator=generator, dtype=torch.float32
        )
        return noise.to(device=reference.device, dtype=reference.dtype)

    @staticmethod
    def _with_noise(
        base: dict,
        noise_override: Optional[torch.Tensor],
        trajectory_noise: Optional[torch.Tensor],
    ) -> dict:
        out = dict(base)
        if noise_override is not None:
            out["noise"] = noise_override
        out["forced_trajectory_noise"] = trajectory_noise
        return out

    @staticmethod
    def _mean_relative_displacement(trajectory_records: list) -> float:
        values = [
            float(record["counterfactual_traj_disp_relative"])
            for record in trajectory_records
            if "counterfactual_traj_disp_relative" in record
        ]
        return float(sum(values) / len(values)) if values else 0.0

    @staticmethod
    def _mean_trajectory_metric(trajectory_records: list, key: str) -> float:
        values = [float(record[key]) for record in trajectory_records if key in record]
        if not values:
            raise RuntimeError(f"counterfactual trajectory metric {key!r} is missing")
        return float(sum(values) / len(values))

    @staticmethod
    def _counterfactual_shard_tag(spec: dict, specs: list) -> str:
        """Stable per-probe file tag.

        ``tile-NN`` (the historical name) whenever every spec addresses a
        distinct (latent, tile) pair, otherwise the removal-size tag.  When more
        than one history latent is probed in the same step the latent index is
        prefixed so the two probes of one tile cannot overwrite each other.
        """
        pairs = {(int(item.get("latent_index", 0)), int(item["anchor_tile"])) for item in specs}
        if len(specs) == 1 or len(pairs) == len(specs):
            tag = "tile-%02d" % int(spec["anchor_tile"])
        else:
            tag = "rm%02d" % int(spec["removal_count"])
        if len({latent for latent, _ in pairs}) > 1:
            tag = "latent%02d-%s" % (int(spec.get("latent_index", 0)), tag)
        return tag

    def _append_replay_labels(
        self, *, rank: int, specs: list, spec_deltas: list
    ) -> None:
        """Add the noise-averaged label to each spec's first-replay shard."""
        shard_dir = Path(self.selector_counterfactual_dump_dir) / f"rank{int(rank)}"
        for spec_position, spec in enumerate(specs):
            deltas = spec_deltas[spec_position]
            tag = self._counterfactual_shard_tag(spec, specs)
            path = shard_dir / f"step-{int(self.global_step):06d}-{tag}.pt"
            if not path.is_file():
                continue
            payload = torch.load(path, map_location="cpu", weights_only=False)
            payload["replay_deltas"] = [float(value) for value in deltas]
            payload["delta_mean_over_replays"] = float(sum(deltas) / len(deltas))
            payload["delta_std_over_replays"] = float(
                statistics.pstdev(deltas) if len(deltas) > 1 else 0.0
            )
            payload["sign_agreement_over_replays"] = float(
                max(
                    sum(1 for value in deltas if value >= 0),
                    sum(1 for value in deltas if value < 0),
                )
                / len(deltas)
            )
            tmp_path = path.with_suffix(".pt.tmp")
            torch.save(payload, tmp_path)
            os.replace(tmp_path, path)

    def _counterfactual_replicate_controls(
        self,
        models: dict,
        probe_inputs: dict,
        probe_base: dict,
        probe_mask: torch.Tensor,
        baseline_unweighted: torch.Tensor,
        replay_traj_noise: torch.Tensor,
        first_masked_loss: Optional[torch.Tensor] = None,
    ) -> dict:
        """Determinism controls: replicate the masked forward and probe the
        identity (all-keep) mask.  Any spread here is the resolution limit of
        the counterfactual label."""
        metrics: dict = {}
        replicate_deltas = []
        for _ in range(self.selector_counterfactual_replicate):
            with torch.no_grad():
                repeat_result = self.pipe.training_loss(
                    **models, **probe_inputs, return_loss_breakdown=True
                )
            repeat_unweighted = repeat_result.get(
                "trajectory_loss_unweighted", repeat_result["trajectory_loss"]
            )
            replicate_deltas.append(
                _probe_relative_delta(baseline_unweighted, repeat_unweighted)
            )
        metrics["counterfactual_control_mask_delta"] = float(replicate_deltas[0])
        metrics["counterfactual_control_mask_delta_max"] = float(
            max(abs(value) for value in replicate_deltas)
        )
        if first_masked_loss is not None:
            metrics["counterfactual_control_mask_loss_first"] = float(
                first_masked_loss.detach().float().reshape(-1)[0]
            )
            metrics["counterfactual_control_mask_loss_repeat"] = float(
                repeat_result.get(
                    "trajectory_loss_unweighted", repeat_result["trajectory_loss"]
                )
                .detach()
                .float()
                .reshape(-1)[0]
            )
        identity_inputs = dict(probe_inputs)
        identity_inputs["counterfactual_history_token_mask"] = torch.ones_like(
            probe_mask
        )
        with torch.no_grad():
            identity_result = self.pipe.training_loss(
                **models, **identity_inputs, return_loss_breakdown=True
            )
        identity_unweighted = identity_result.get(
            "trajectory_loss_unweighted", identity_result["trajectory_loss"]
        )
        metrics["counterfactual_control_identity_delta"] = float(
            _probe_relative_delta(baseline_unweighted, identity_unweighted)
        )
        metrics["counterfactual_control_identity_loss"] = float(
            identity_unweighted.detach().float().reshape(-1)[0]
        )
        return metrics

    def forward_preprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {"negative_prompt": self.negative_prompt}
        inputs_shared = {
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": data["num_frames"],
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
            "target_fps": self.target_fps,
        }
        if self.enable_online_selector and self.selector_teacher_seed is not None:
            # Match official inference: video uses seed N and trajectory uses
            # the independent N+1 stream. This removes RNG noise from signed
            # counterfactual labels while preserving the normal training path.
            inputs_shared["seed"] = self.selector_teacher_seed
            inputs_shared["fixed_trajectory_noise_seed"] = self.selector_teacher_seed + 1

        if "trajectory" in data:
            inputs_shared["trajectory"] = data["trajectory"]
        if "ego_vel" in data:
            inputs_shared["ego_vel"] = data["ego_vel"]
            inputs_shared["ego_state"] = data["ego_vel"][..., :2]
        if "driving_command" in data:
            inputs_shared["driving_command"] = data["driving_command"]
            inputs_shared["command"] = data["driving_command"]
        if self.pipe.trajectory_condition_mode in {"auto", "history"} and "history_positions" in data:
            inputs_shared["history_positions"] = data["history_positions"]

        for extra_input in self.extra_inputs:
            if extra_input == "longcat_video":
                inputs_shared["longcat_video"] = data.get("longcat_video")
            elif extra_input == "trajectory":
                inputs_shared["trajectory"] = data.get("trajectory")
            elif extra_input == "ego_vel":
                inputs_shared["ego_vel"] = data.get("ego_vel")
            elif extra_input == "history_positions":
                inputs_shared["history_positions"] = data.get("history_positions")
            else:
                inputs_shared[extra_input] = data[extra_input]

        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(
                unit,
                self.pipe,
                inputs_shared,
                inputs_posi,
                inputs_nega,
            )
        return {**inputs_shared, **inputs_posi}

    def forward(self, data, inputs=None, global_step=None):
        if inputs is None:
            inputs = self.forward_preprocess(data)
        if global_step is not None:
            self.global_step = int(global_step)
        teacher_step = self.enable_online_selector and self.global_step >= self.selector_warmup_steps and self.global_step % self.selector_gradient_interval == 0
        counterfactual_step = (
            teacher_step
            and self.selector_teacher_mode
            in {"signed_hybrid", "displacement", "planning_harm"}
            and self.global_step % self.selector_counterfactual_interval == 0
        )
        if counterfactual_step:
            # The counterfactual teacher records exactly one tile record and one
            # target per probe spec, so the tile/logit stacking below is only
            # well defined for a per-device batch of one.  Fail loudly here
            # instead of producing a silently misaligned supervision tensor.
            batch = None
            for key in ("noise", "input_latents", "latents"):
                value = inputs.get(key) if isinstance(inputs, dict) else None
                if torch.is_tensor(value) and value.ndim >= 1:
                    batch = int(value.shape[0])
                    break
            if batch is None and isinstance(data, dict) and data.get("video") is not None:
                batch = len(data["video"])
            if batch is not None and batch != 1:
                raise ValueError(
                    "the counterfactual token teacher requires per-device batch "
                    f"size 1, got {batch}; its tile records are not batch-aligned"
                )
        if counterfactual_step and self.selector_teacher_timesteps:
            world_size = 1
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                world_size = torch.distributed.get_world_size()
            group_count = self.selector_counterfactual_tile_h * self.selector_counterfactual_tile_w
            counterfactual_event = self.global_step // self.selector_counterfactual_interval
            events_per_spatial_sweep = (
                1
                if self.selector_counterfactual_sweep_all
                else max(1, math.ceil(group_count / world_size))
            )
            timestep_index = (
                counterfactual_event // events_per_spatial_sweep
            ) % len(self.selector_teacher_timesteps)
            inputs["forced_training_timestep"] = self.selector_teacher_timesteps[
                timestep_index
            ]
        sparse_step = self.enable_online_selector and self.selector_mask_start_step >= 0 and self.global_step >= self.selector_mask_start_step
        if teacher_step:
            inputs["capture_history_tokens"] = True
            inputs["capture_planning_graph"] = not counterfactual_step
            inputs["capture_training_replay"] = counterfactual_step
            inputs["selector_layer"] = self.selector_layer
        if counterfactual_step:
            # The main (baseline) forward captures the token grid of the first
            # requested history latent; every probe below then re-points the
            # mask at its own latent.  Default index 0 keeps the historical
            # newest-latent behaviour bit-identical.
            inputs["counterfactual_latent_index"] = int(
                self.selector_counterfactual_latent_indices[0]
            )
        if sparse_step and not teacher_step and hasattr(self, "_selector_mask"):
            inputs["history_token_mask"] = self._selector_mask
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        if counterfactual_step:
            with torch.no_grad():
                result = self.pipe.training_loss(**models, **inputs, return_loss_breakdown=True)
        else:
            result = self.pipe.training_loss(**models, **inputs, return_loss_breakdown=True)
        if not self.enable_online_selector or self.selector is None:
            return result
        tokens = getattr(self.pipe, "_last_history_tokens", None)
        positions = getattr(self.pipe, "_last_token_positions", None)
        planning_loss = result.get("planning_loss_for_teacher")
        if not teacher_step or tokens is None:
            return result
        ratio = keep_ratio_at_step(self.global_step, self.selector_warmup_steps, self.selector_keep_schedule)
        teacher_start = time.perf_counter()
        ego_state = inputs.get("ego_state")
        command = inputs.get("command")
        if positions is None or ego_state is None or command is None:
            raise RuntimeError("Online selector requires real token positions, ego_state, and driving command")
        if ego_state.ndim == 1: ego_state = ego_state.unsqueeze(0)
        if command.ndim == 1: command = command.unsqueeze(0)
        positions = positions.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        critical_token_mask = None
        if self.selector_critical_token_mode == "provided":
            if not isinstance(data, dict) or data.get("critical_token_mask") is None:
                raise RuntimeError(
                    "selector critical-token mode=provided requires data['critical_token_mask']"
                )
            critical_token_mask = torch.as_tensor(
                data["critical_token_mask"], device=tokens.device, dtype=torch.bool
            )
            if critical_token_mask.ndim == 1:
                critical_token_mask = critical_token_mask.unsqueeze(0)
            logits_shape = (tokens.shape[0], tokens.shape[1])
            if critical_token_mask.shape != logits_shape:
                raise ValueError(
                    "critical_token_mask must match selector candidates "
                    f"{logits_shape}, got {tuple(critical_token_mask.shape)}"
                )
        elif self.selector_critical_token_mode == "boxes":
            if not isinstance(data, dict) or data.get("critical_object_boxes_yxyx") is None:
                raise RuntimeError(
                    "selector critical-token mode=boxes requires "
                    "data['critical_object_boxes_yxyx']"
                )
            critical_token_mask = critical_box_token_mask(
                positions,
                torch.as_tensor(data["critical_object_boxes_yxyx"]),
                box_valid=data.get("critical_object_box_valid"),
                dilation=self.selector_critical_token_dilation,
            )
        if not hasattr(self, "_selector_condition_logged"):
            print("[selector] condition:", f"history_tokens={tuple(tokens.shape)}", f"token_positions={tuple(positions.shape)}", f"ego_state={tuple(ego_state.shape)}", f"driving_command={tuple(command.shape)}", f"token_mean={tokens.detach().float().mean():.5f}", f"token_std={tokens.detach().float().std():.5f}", f"position_range=({positions.min():.3f},{positions.max():.3f})", f"ego_sample={ego_state[0].tolist()}", f"command_sample={command[0].tolist()}")
            self._selector_condition_logged = True
        selector_timestep = getattr(self.pipe, "_last_training_timestep", None)
        logits = self.selector(
            tokens.detach(), positions, ego_state, command, timestep=selector_timestep
        )
        counterfactual_metrics = {}
        if counterfactual_step:
            world_size = 1
            rank = 0
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                world_size = torch.distributed.get_world_size()
                rank = torch.distributed.get_rank()
            group_count = self.selector_counterfactual_tile_h * self.selector_counterfactual_tile_w
            # ``counterfactual_step`` only fires every N optimizer steps.  Using
            # the raw global step here can alias with the tile-grid size, so
            # index by counterfactual events instead (review defect C).
            counterfactual_event = self.global_step // self.selector_counterfactual_interval
            # A "probe spec" is one intervention: a removal size (in tiles) and
            # the concrete tile subset that is removed.  The same spec is
            # repeated for every requested history latent; the extra
            # ``latent_index`` key selects which conditioned history latent's
            # token range the mask is applied to.
            base_specs = self._counterfactual_probe_specs(
                counterfactual_event, world_size, rank, group_count
            )
            specs = [
                {**spec, "latent_index": int(latent_index)}
                for latent_index in self.selector_counterfactual_latent_indices
                for spec in base_specs
            ]
            conditioned_latents = None
            longcat_latents = inputs.get("longcat_latents")
            if torch.is_tensor(longcat_latents) and longcat_latents.ndim >= 3:
                conditioned_latents = int(longcat_latents.shape[2])
            if conditioned_latents is not None:
                for latent_index in self.selector_counterfactual_latent_indices:
                    if latent_index > conditioned_latents - 1:
                        raise ValueError(
                            "counterfactual latent index "
                            f"{latent_index} exceeds the {conditioned_latents} "
                            "conditioned history latents available in this batch"
                        )
            replay_timestep = getattr(self.pipe, "_last_training_timestep", None)
            replay_traj_noise = getattr(self.pipe, "_last_trajectory_noise", None)
            if replay_timestep is None or replay_traj_noise is None:
                raise RuntimeError("counterfactual teacher did not capture replay state")
            probe_base = dict(inputs)
            probe_base.update(
                {
                    "capture_history_tokens": False,
                    "capture_planning_graph": False,
                    "capture_training_replay": False,
                    "forced_training_timestep": replay_timestep,
                    # -1 is the exact deployment intervention: physically
                    # remove tokens before the first DiT block.  Keep the old
                    # post-block mode available for reproducibility.
                    "counterfactual_layer": (
                        -1
                        if self.selector_counterfactual_injection_point == "pre_dit"
                        else self.selector_layer
                    ),
                    "counterfactual_physical_prune": self.selector_counterfactual_physical,
                    "return_traj_pred": True,
                    "use_gradient_checkpointing": False,
                    "use_gradient_checkpointing_offload": False,
                }
            )
            reference_unweighted = result.get(
                "trajectory_loss_unweighted", result["trajectory_loss"]
            )
            replay_timestep_float = replay_timestep.detach().float()
            # label[spec][replay] -> signed relative loss change
            spec_deltas = [[] for _ in specs]
            spec_trajectory = [[] for _ in specs]
            spec_baseline_loss = []
            member_rows = []
            baseline_loss_values = []
            tile_records = []

            for replay_index in range(self.selector_counterfactual_replays):
                if replay_index == 0:
                    # Reuse the baseline that the ordinary training path already
                    # produced, so the default single-replay behaviour is
                    # bit-identical to the pre-2026-09-11 teacher.
                    baseline_result = result
                    baseline_unweighted = reference_unweighted
                    noise_override = None
                    traj_noise_override = replay_traj_noise
                else:
                    # A different noise realisation is exactly what changes the
                    # planning residual direction, which is the quantity that
                    # makes the per-probe sign a coin flip (see report
                    # dynamic_token_learnability_verdict_20260911 section 7).
                    noise_override = self._counterfactual_replay_noise(
                        replay_index, inputs.get("noise")
                    )
                    traj_noise_override = torch.randn_like(
                        replay_traj_noise.detach().float()
                    ).to(dtype=replay_traj_noise.dtype, device=replay_traj_noise.device)
                    with torch.no_grad():
                        baseline_result = self.pipe.training_loss(
                            **models,
                            **self._with_noise(
                                probe_base, noise_override, traj_noise_override
                            ),
                            return_loss_breakdown=True,
                        )
                    baseline_unweighted = baseline_result.get(
                        "trajectory_loss_unweighted",
                        baseline_result["trajectory_loss"],
                    )
                baseline_loss_values.append(
                    float(baseline_unweighted.detach().float().reshape(-1)[0])
                )
                for spec_position, spec in enumerate(specs):
                    probe_mask, membership = self._counterfactual_mask(
                        positions, spec, latent_index=int(spec["latent_index"])
                    )
                    probe_inputs = self._with_noise(
                        probe_base, noise_override, traj_noise_override
                    )
                    probe_inputs["counterfactual_history_token_mask"] = probe_mask
                    probe_inputs["counterfactual_latent_index"] = int(
                        spec["latent_index"]
                    )
                    with torch.no_grad():
                        masked_result = self.pipe.training_loss(
                            **models, **probe_inputs, return_loss_breakdown=True
                        )
                    masked_unweighted = masked_result.get(
                        "trajectory_loss_unweighted", masked_result["trajectory_loss"]
                    )
                    measured_delta = _probe_relative_delta(
                        baseline_unweighted, masked_unweighted
                    )
                    spec_deltas[spec_position].append(measured_delta)
                    trajectory_divergence = _trajectory_divergence(
                        baseline_result.get("trajectory_pred"),
                        masked_result.get("trajectory_pred"),
                        prefix_len=int(
                            baseline_result.get("trajectory_prefix_len", 0) or 0
                        ),
                        horizon_weights=self.selector_long_horizon_weights,
                        target_fps=self.target_fps,
                    )
                    trajectory_divergence.update(
                        _planning_error_harm(
                            baseline_result.get("trajectory_x0_pred"),
                            masked_result.get("trajectory_x0_pred"),
                            baseline_result.get("trajectory_gt_abs"),
                            horizon_weights=self.selector_long_horizon_weights,
                            target_fps=self.target_fps,
                        )
                    )
                    spec_trajectory[spec_position].append(trajectory_divergence)
                    if replay_index != 0:
                        continue
                    # Per-probe record, metrics and diagnostics use the first
                    # realisation so that single-replay runs are unchanged.
                    group_logits = logits.detach()[membership]
                    group_probabilities = torch.sigmoid(group_logits.float())
                    record = {
                        "group_index": int(spec["anchor_tile"]),
                        "latent_index": int(spec["latent_index"]),
                        "removal_count": int(spec["removal_count"]),
                        "removal_tiles": list(spec["tiles"]),
                        "measured_delta": measured_delta,
                        "baseline_loss": float(
                            baseline_unweighted.detach().float().reshape(-1)[0]
                        ),
                        "masked_loss": float(
                            masked_unweighted.detach().float().reshape(-1)[0]
                        ),
                        "helpful_target": float((measured_delta >= 0)),
                        "confidence": float(
                            min(
                                abs(measured_delta)
                                / float(self.selector_counterfactual_scale),
                                1.0,
                            )
                        ),
                        "group_logit": float(group_logits.mean().item()),
                        "group_probability_mean": float(
                            group_probabilities.mean().item()
                        ),
                        "group_probability_std": float(
                            group_probabilities.std(unbiased=False).item()
                        ),
                        "trajectory": trajectory_divergence,
                    }
                    tile_records.append(record)
                    member_rows.append(membership)
                    if self.selector_counterfactual_jsonl_dir is not None:
                        with torch.no_grad():
                            tile_rows = membership.detach()[0].bool()
                            tile_pos_mean = (
                                positions.detach()[0][tile_rows].float().mean(dim=0)
                            )
                        history_positions = (
                            data.get("history_positions")
                            if isinstance(data, dict)
                            else None
                        )
                        history_rows = None
                        if torch.is_tensor(history_positions):
                            history_rows = (
                                history_positions.detach()
                                .float()
                                .reshape(-1, history_positions.shape[-1])[:, :3]
                                .cpu()
                                .tolist()
                            )
                        _append_counterfactual_jsonl(
                            self.selector_counterfactual_jsonl_dir
                            / f"rank{int(rank)}.jsonl",
                            {
                                "scene_token": _flatten_sample_token(
                                    data.get("token") if isinstance(data, dict) else None
                                ),
                                "global_step": int(self.global_step),
                                "tile": int(spec["anchor_tile"]),
                                "latent_index": int(spec["latent_index"]),
                                "relative_delta": float(measured_delta),
                                "counterfactual_traj_disp_relative": float(
                                    trajectory_divergence.get(
                                        "counterfactual_traj_disp_relative", float("nan")
                                    )
                                ),
                                "counterfactual_traj_disp_mean": float(
                                    trajectory_divergence.get(
                                        "counterfactual_traj_disp_mean", float("nan")
                                    )
                                ),
                                "counterfactual_traj_disp_max": float(
                                    trajectory_divergence.get(
                                        "counterfactual_traj_disp_max", float("nan")
                                    )
                                ),
                                "counterfactual_traj_disp_long_horizon_relative": float(
                                    trajectory_divergence.get(
                                        "counterfactual_traj_disp_long_horizon_relative",
                                        float("nan"),
                                    )
                                ),
                                "counterfactual_planning_harm_ade": float(
                                    trajectory_divergence.get(
                                        "counterfactual_planning_harm_ade",
                                        float("nan"),
                                    )
                                ),
                                "counterfactual_planning_harm_long_horizon": float(
                                    trajectory_divergence.get(
                                        "counterfactual_planning_harm_long_horizon",
                                        float("nan"),
                                    )
                                ),
                                "tile_pos_t": float(tile_pos_mean[0]),
                                "tile_pos_y": float(tile_pos_mean[1]),
                                "tile_pos_x": float(tile_pos_mean[2]),
                                "ego_vx": float(ego_state.detach()[0, 0]),
                                "ego_vy": float(ego_state.detach()[0, 1]),
                                "command": [
                                    float(value)
                                    for value in command.detach()[0].reshape(-1)[:3]
                                ],
                                "baseline_loss": float(
                                    baseline_unweighted.detach().float().reshape(-1)[0]
                                ),
                                "masked_loss": float(
                                    masked_unweighted.detach().float().reshape(-1)[0]
                                ),
                                "conditioned_latents": conditioned_latents,
                                # Layer at which the tile mask was applied.  The
                                # replay forward takes this from
                                # ``self.selector_layer``, so recording it makes
                                # the cross-layer importance scan (P2 of
                                # reports/next_steps_plan_20260911.md)
                                # self-traceable instead of relying on a config
                                # sidecar.
                                "counterfactual_layer": int(
                                    probe_base["counterfactual_layer"]
                                ),
                                "counterfactual_injection_point": (
                                    self.selector_counterfactual_injection_point
                                ),
                                "teacher_timestep": float(
                                    replay_timestep_float.reshape(-1)[0]
                                ),
                                "history_positions": history_rows,
                                "run_id": os.environ.get("DRIVEVA_RUN_ID"),
                            },
                        )
                    if self.selector_counterfactual_replicate > 0 and spec_position == 0:
                        counterfactual_metrics.update(
                            self._counterfactual_replicate_controls(
                                models,
                                probe_inputs,
                                probe_base,
                                probe_mask,
                                baseline_unweighted,
                                replay_traj_noise,
                                masked_unweighted,
                            )
                        )
                    if self.selector_counterfactual_dump_dir is not None:
                        _dump_counterfactual_probe(
                            self.selector_counterfactual_dump_dir,
                            rank=rank,
                            global_step=self.global_step,
                            sample_token=data.get("token")
                            if isinstance(data, dict)
                            else None,
                            tokens=tokens,
                            positions=positions,
                            ego_state=ego_state,
                            command=command,
                            selector_timestep=selector_timestep,
                            logits=logits,
                            membership=membership,
                            group_index=int(spec["anchor_tile"]),
                            baseline_loss=baseline_unweighted,
                            masked_loss=masked_unweighted,
                            relative_delta=measured_delta,
                            helpful_target=record["helpful_target"],
                            confidence=record["confidence"],
                            trajectory_metrics=trajectory_divergence,
                            extra={
                                "removal_count": int(spec["removal_count"]),
                                "removal_tiles": list(spec["tiles"]),
                                "spec_index": int(spec_position),
                                "replays": int(self.selector_counterfactual_replays),
                                "run_id": os.environ.get("DRIVEVA_RUN_ID"),
                            },
                            shard_tag=self._counterfactual_shard_tag(spec, specs),
                        )
            # ---- multi-replay label (review P3-1 noise averaging) -----------
            if self.selector_counterfactual_dump_dir is not None and (
                self.selector_counterfactual_replays > 1
            ):
                # The averaged label only exists once every replay has run, so it
                # is appended to the shard written by the first replay.
                self._append_replay_labels(
                    rank=rank,
                    specs=specs,
                    spec_deltas=spec_deltas,
                )
            for spec_position, spec in enumerate(specs):
                deltas = spec_deltas[spec_position]
                if self.selector_counterfactual_replays > 1:
                    counterfactual_metrics[
                        f"counterfactual_mean_delta_spec{spec_position}"
                    ] = float(sum(deltas) / len(deltas))
                    counterfactual_metrics[
                        f"counterfactual_std_delta_spec{spec_position}"
                    ] = float(
                        statistics.pstdev(deltas) if len(deltas) > 1 else 0.0
                    )
                    counterfactual_metrics[
                        f"counterfactual_sign_agreement_spec{spec_position}"
                    ] = float(
                        max(sum(1 for value in deltas if value >= 0),
                            sum(1 for value in deltas if value < 0))
                        / len(deltas)
                    )
            if self.selector_counterfactual_replays > 1:
                agreement = [
                    counterfactual_metrics[
                        f"counterfactual_sign_agreement_spec{position}"
                    ]
                    for position in range(len(specs))
                ]
                mean_deltas = [
                    counterfactual_metrics[f"counterfactual_mean_delta_spec{position}"]
                    for position in range(len(specs))
                ]
                single_deltas = [spec_deltas[position][0] for position in range(len(specs))]
                counterfactual_metrics["counterfactual_replays"] = float(
                    self.selector_counterfactual_replays
                )
                counterfactual_metrics["counterfactual_sign_agreement_mean"] = float(
                    sum(agreement) / len(agreement)
                )
                counterfactual_metrics["counterfactual_mean_delta"] = float(
                    sum(mean_deltas) / len(mean_deltas)
                )
                counterfactual_metrics["counterfactual_replay_std_mean"] = float(
                    sum(
                        counterfactual_metrics[
                            f"counterfactual_std_delta_spec{position}"
                        ]
                        for position in range(len(specs))
                    )
                    / len(specs)
                )
                # How much of the single-realisation spread survives averaging:
                # the ratio of between-spec spread to within-spec spread is the
                # signal-to-noise of the averaged label.
                counterfactual_metrics["counterfactual_single_delta_std"] = float(
                    statistics.pstdev(single_deltas)
                    if len(single_deltas) > 1
                    else 0.0
                )
                counterfactual_metrics["counterfactual_mean_delta_std"] = float(
                    statistics.pstdev(mean_deltas) if len(mean_deltas) > 1 else 0.0
                )
            # ---- supervision -------------------------------------------------
            primary_positions = list(range(len(specs)))
            stacked_membership = torch.cat(
                [member_rows[position] for position in primary_positions], dim=0
            )
            stacked_logits = logits.repeat(len(primary_positions), 1)
            if self.selector_counterfactual_replays > 1:
                label_deltas = torch.tensor(
                    [
                        counterfactual_metrics[
                            f"counterfactual_mean_delta_spec{position}"
                        ]
                        for position in primary_positions
                    ],
                    device=reference_unweighted.device,
                    dtype=reference_unweighted.dtype,
                )
                baseline_tensor = torch.tensor(
                    baseline_loss_values,
                    device=reference_unweighted.device,
                    dtype=reference_unweighted.dtype,
                ).mean()
                masked_tensor = baseline_tensor * (1.0 + label_deltas)
            else:
                masked_tensor = torch.tensor(
                    [record["masked_loss"] for record in tile_records],
                    device=reference_unweighted.device,
                    dtype=reference_unweighted.dtype,
                )
                baseline_tensor = reference_unweighted
            teacher_token_scores = None
            if self.selector_teacher_mode in {"displacement", "planning_harm"}:
                # Magnitude teacher (2026-09-11 verdict, section 12.2): the sign
                # of the loss change is unlearnable, its geometric magnitude is
                # not.  Displacement is averaged over replays when replays > 1.
                if self.selector_teacher_mode == "planning_harm":
                    displacement_key = (
                        "counterfactual_planning_harm_long_horizon_positive"
                        if self.selector_long_horizon_weights
                        else "counterfactual_planning_harm_ade_positive"
                    )
                else:
                    displacement_key = (
                        "counterfactual_traj_disp_long_horizon_relative"
                        if self.selector_long_horizon_weights
                        else "counterfactual_traj_disp_relative"
                    )
                if self.selector_counterfactual_replays > 1:
                    displacement_values = [
                        self._mean_trajectory_metric(
                            spec_trajectory[position], displacement_key
                        )
                        for position in range(len(specs))
                    ]
                else:
                    displacement_values = [
                        float(
                            record["trajectory"].get(
                                displacement_key, 0.0
                            )
                        )
                        for record in tile_records
                    ]
                displacement_tensor = torch.tensor(
                    displacement_values,
                    device=reference_unweighted.device,
                    dtype=torch.float32,
                )
                selector_loss, counterfactual_metrics_bce = displacement_token_bce(
                    stacked_logits,
                    stacked_membership,
                    displacement_tensor,
                    disp_scale=self.selector_teacher_disp_scale,
                    normalize=self.selector_teacher_disp_normalize,
                    min_spread=self.selector_teacher_disp_min_spread,
                )
                if (
                    self.selector_teacher_disp_normalize == "scene"
                    and displacement_tensor.numel() > 1
                ):
                    disp_low = displacement_tensor.min()
                    disp_high = displacement_tensor.max()
                    disp_spread = float((disp_high - disp_low).clamp_min(0.0).item())
                    if disp_spread < self.selector_teacher_disp_min_spread:
                        # Same abstention rule as displacement_token_bce (which
                        # also reports the ratio/spread): the ranking diagnostics
                        # must not pretend that a numerically degenerate spread
                        # is a real ordering.
                        target_values = torch.zeros_like(displacement_tensor)
                    else:
                        target_values = (
                            (displacement_tensor - disp_low)
                            / (disp_high - disp_low).clamp_min(1e-6)
                        ).clamp(0.0, 1.0)
                else:
                    target_values = (
                        displacement_tensor / self.selector_teacher_disp_scale
                    ).clamp(0.0, 1.0)
                # Expand the tile-level causal targets back to the 390-token
                # grid solely for ranking diagnostics (and the optional
                # pairwise loss).  Full 12-tile sweeps cover every token once.
                teacher_token_scores = torch.zeros_like(logits, dtype=torch.float32)
                teacher_token_counts = torch.zeros_like(logits, dtype=torch.float32)
                for position, membership in enumerate(member_rows):
                    membership_float = membership.to(dtype=torch.float32)
                    teacher_token_scores = teacher_token_scores + (
                        membership_float * target_values[position]
                    )
                    teacher_token_counts = teacher_token_counts + membership_float
                teacher_token_scores = teacher_token_scores / teacher_token_counts.clamp_min(1)
                counterfactual_metrics["counterfactual_displacement_mean"] = float(
                    displacement_tensor.mean()
                )
                counterfactual_metrics["counterfactual_displacement_target_key"] = (
                    displacement_key
                )
                if self.selector_teacher_mode == "planning_harm":
                    counterfactual_metrics["counterfactual_planning_harm_mean"] = float(
                        displacement_tensor.mean()
                    )
            else:
                selector_loss, counterfactual_metrics_bce = counterfactual_group_bce(
                    stacked_logits,
                    stacked_membership,
                    baseline_tensor,
                    masked_tensor,
                    relative_scale=self.selector_counterfactual_scale,
                    abstain_eps=self.selector_counterfactual_abstain_eps,
                )
            counterfactual_metrics.update(counterfactual_metrics_bce)
            selector_loss = self.selector_counterfactual_weight * selector_loss
            result["selector_bce_unweighted"] = torch.tensor(
                float(counterfactual_metrics.get("counterfactual_unweighted_bce", 0.0)),
                device=selector_loss.device,
            )
            scores = teacher_token_scores
            labels = (
                online_topk_labels(scores, self.selector_teacher_keep_ratio)
                if scores is not None
                else (logits.detach() >= 0).float()
            )
            counterfactual_metrics["counterfactual_group_index"] = float(
                sum(record["group_index"] for record in tile_records) / len(tile_records)
            )
            counterfactual_metrics["counterfactual_tiles_per_step"] = float(
                len(primary_positions)
            )
            counterfactual_metrics["counterfactual_removal_count_mean"] = float(
                sum(record["removal_count"] for record in tile_records)
                / len(tile_records)
            )
            counterfactual_metrics["counterfactual_measured_delta_single"] = float(
                sum(record["measured_delta"] for record in tile_records)
                / len(tile_records)
            )
            # Spec-0, first-realisation delta: the historical single-probe label,
            # kept so the existing summary tooling and the determinism control
            # compare like with like.
            counterfactual_metrics["counterfactual_measured_delta"] = float(
                spec_deltas[0][0]
            )
            counterfactual_metrics["counterfactual_baseline_loss_unweighted"] = float(
                sum(baseline_loss_values) / len(baseline_loss_values)
            )
            counterfactual_metrics["counterfactual_masked_loss_unweighted"] = float(
                sum(record["masked_loss"] for record in tile_records)
                / len(tile_records)
            )
            counterfactual_metrics["counterfactual_group_probability_mean"] = float(
                sum(record["group_probability_mean"] for record in tile_records)
                / len(tile_records)
            )
            counterfactual_metrics["counterfactual_group_probability_std"] = float(
                sum(record["group_probability_std"] for record in tile_records)
                / len(tile_records)
            )
            counterfactual_metrics["counterfactual_probability_mean"] = float(
                torch.sigmoid(logits.detach().float()).mean().item()
            )
            counterfactual_metrics.update(
                {
                    "counterfactual_timestep_mean": float(
                        replay_timestep_float.mean().item()
                    ),
                    "counterfactual_timestep_min": float(
                        replay_timestep_float.min().item()
                    ),
                    "counterfactual_timestep_max": float(
                        replay_timestep_float.max().item()
                    ),
                }
            )
            for key in (
                "counterfactual_traj_disp_mean",
                "counterfactual_traj_disp_max",
                "counterfactual_traj_disp_final",
                "counterfactual_traj_disp_relative",
                "counterfactual_traj_disp_long_horizon",
                "counterfactual_traj_disp_long_horizon_relative",
                "counterfactual_traj_endpoint_disp",
                "counterfactual_traj_scale",
                "counterfactual_planning_error_baseline_ade",
                "counterfactual_planning_error_masked_ade",
                "counterfactual_planning_harm_ade",
                "counterfactual_planning_harm_ade_positive",
                "counterfactual_planning_error_baseline_long_horizon",
                "counterfactual_planning_error_masked_long_horizon",
                "counterfactual_planning_harm_long_horizon",
                "counterfactual_planning_harm_long_horizon_positive",
            ):
                values = [
                    record["trajectory"][key]
                    for record in tile_records
                    if key in record["trajectory"]
                ]
                if values:
                    counterfactual_metrics[key] = float(sum(values) / len(values))
        else:
            if self.selector_teacher_mode in {"displacement", "planning_harm"}:
                raise RuntimeError(
                    "the planning-causal teacher needs the counterfactual probe; "
                    "set --selector-counterfactual-interval 1"
                )
            if planning_loss is None:
                raise RuntimeError("signed/gradient teacher requires planning loss graph")
            if self.selector_teacher_mode == "signed_hybrid":
                scores = signed_removal_scores(planning_loss, tokens, retain_graph=True)
                labels = signed_soft_keep_labels(
                    scores, temperature=self.selector_signed_temperature
                )
            else:
                scores = gradient_input_scores(
                    planning_loss,
                    tokens,
                    # DDP plus non-reentrant gradient checkpointing keeps shared
                    # autograd bookkeeping until the outer loss backward completes.
                    retain_graph=True,
                )
                labels = online_topk_labels(scores, self.selector_teacher_keep_ratio)
            selector_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        metric_labels = (
            online_topk_labels(scores, self.selector_teacher_keep_ratio)
            if scores is not None
            else labels
        )
        # Keep ``selector_bce`` semantically stable for historical dashboards:
        # it is the pre-ranking supervision term, while ``selector_total_loss``
        # is the quantity actually optimized when the optional ranking term is
        # enabled.
        selector_bce = selector_loss
        ranking_loss = selector_loss.new_zeros(())
        if self.selector_ranking_loss_weight > 0 and scores is not None:
            ranking_loss = selector_pairwise_ranking_loss(
                logits,
                metric_labels,
                margin=self.selector_ranking_margin,
                max_pairs=self.selector_ranking_max_pairs,
            )
            selector_loss = selector_loss + self.selector_ranking_loss_weight * ranking_loss
        result["selector_bce"] = selector_bce.detach()
        result["selector_ranking_loss"] = ranking_loss.detach()
        result["selector_total_loss"] = selector_loss.detach()
        sm = selector_metrics(logits, metric_labels)
        result["selector_topk_overlap"] = torch.tensor(sm["selector_topk_overlap"], device=selector_loss.device)
        result["selector_pairwise_accuracy"] = torch.tensor(
            sm["selector_pairwise_accuracy"], device=selector_loss.device
        )
        result["selector_ndcg_at_k"] = torch.tensor(
            sm["selector_ndcg_at_k"], device=selector_loss.device
        )
        result["selector_positive_logit_mean"] = sm["selector_mean_positive_logit"]
        result["selector_negative_logit_mean"] = sm["selector_mean_negative_logit"]
        result["current_keep_ratio"] = float(ratio)
        result["teacher_positive_ratio"] = float(labels.mean())
        if scores is not None:
            result["gradient_score_mean"] = float(scores.mean().detach())
            result["gradient_score_std"] = float(scores.std().detach())
            result["gradient_score_max"] = float(scores.max().detach())
            result["gradient_score_nonzero_ratio"] = float((scores > 0).float().mean().detach())
            result["gradient_topk_count"] = int(metric_labels.sum(dim=1)[0].item())
        result.update(counterfactual_metrics)
        result["gradient_teacher_time_ms"] = (time.perf_counter() - teacher_start) * 1000.0
        result["planning_loss"] = result.get("trajectory_loss")
        if self.selector_only:
            # The teacher backward has already consumed the frozen downstream
            # graph.  Optimise only the lightweight selector BCE.
            result["loss"] = self.selector_loss_weight * selector_loss
        else:
            result["loss"] = result["loss"] + self.selector_loss_weight * selector_loss
        # Consumers can apply this to the history portion while preserving
        # sequence length.  The selector is trained on the *newest* conditioned
        # history latent (``selector_counterfactual_latent_indices[0]``), so its
        # mask covers ``tokens.shape[1]`` candidates; ``model_fn`` masks the
        # whole ``num_cond_latents * h * w`` history block.  Expanding the
        # candidate mask with ones outside the newest latent keeps the older
        # history protected and makes the two shapes agree -- without this the
        # train-time masking path raised a broadcast error and could not be
        # enabled at all.
        selector_mask = hard_topk_mask(
            logits, ratio, protected_mask=critical_token_mask
        ).detach()
        num_cond_latents = 0
        longcat_latents = inputs.get("longcat_latents")
        if torch.is_tensor(longcat_latents) and longcat_latents.ndim >= 3:
            num_cond_latents = int(longcat_latents.shape[2])
        tokens_per_latent = int(tokens.shape[1])
        full_history_length = num_cond_latents * tokens_per_latent
        if num_cond_latents > 1 and full_history_length > tokens_per_latent:
            history_mask = torch.ones(
                (selector_mask.shape[0], full_history_length),
                device=selector_mask.device,
                dtype=selector_mask.dtype,
            )
            history_mask[:, full_history_length - tokens_per_latent :] = selector_mask
        else:
            history_mask = selector_mask
        result["history_mask"] = history_mask
        result["history_mask_length"] = int(history_mask.shape[1])
        result["history_mask_candidate_length"] = tokens_per_latent
        result["selector_protected_token_count"] = float(
            critical_token_mask.sum().item() if critical_token_mask is not None else 0
        )
        self._selector_mask = result["history_mask"]
        return result

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DriveVA on NAVSIM v1.")

    parser.add_argument("--repo_root", type=str, required=True)
    parser.add_argument("--navsim_log_path", type=str, required=True)
    parser.add_argument("--sensor_blobs_path", type=str, required=True)
    parser.add_argument("--cache_path", type=str, default=None)
    parser.add_argument("--use_cache_only", action="store_true")
    parser.add_argument("--force_cache_computation", action="store_true")
    parser.add_argument("--train_log_names", type=str, default=None)
    parser.add_argument("--train_scene_manifest", type=str, default=None)
    parser.add_argument(
        "--forbidden_scene_manifest",
        type=str,
        default=None,
        help="Comma-separated manifests whose semantic scene tokens must not occur in training.",
    )
    parser.add_argument(
        "--allow_missing_route",
        action="store_true",
        help="Allow manifest-selected training scenes whose current frame has no route roadblocks.",
    )
    parser.add_argument("--windows_per_scene", type=int, default=1)
    parser.add_argument("--max_scenes", type=int, default=None)
    parser.add_argument("--frame_interval", type=int, default=None)
    parser.add_argument("--skip_missing_files", action="store_true")
    parser.add_argument("--print_navsim_tokens", action="store_true")

    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_history_frames", type=int, default=5)
    parser.add_argument("--num_future_frames", type=int, default=8)
    parser.add_argument("--surround_view", action="store_true")
    parser.add_argument("--target_fps", type=int, default=2)

    parser.add_argument("--local_model_path", type=str, default=None)
    parser.add_argument("--full_ckpt", type=str, default=None)
    parser.add_argument("--trainable_models", type=str, default=None)
    parser.add_argument("--lora_base_model", type=str, default=None)
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2")
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_checkpoint", type=str, default=None)

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--num_epochs", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--dataset_num_workers", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lr_scheduler_type", type=str, default="constant", choices=["cosine", "linear", "constant"])
    parser.add_argument("--find_unused_parameters", dest="find_unused_parameters", action="store_true")
    parser.add_argument("--no_find_unused_parameters", dest="find_unused_parameters", action="store_false")
    parser.set_defaults(find_unused_parameters=True)
    parser.add_argument("--gradient_clip_norm", type=float, default=None)
    parser.add_argument("--ddp_timeout_seconds", type=int, default=1800)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--warmup_start_factor", type=float, default=0.01)
    parser.add_argument("--log_every_steps", type=int, default=50)

    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--ema_update_after_step", type=int, default=0)
    parser.add_argument("--ema_update_every", type=int, default=1)
    parser.add_argument("--ema_on_cpu", action="store_true")
    parser.add_argument("--save_ema", action="store_true")
    parser.add_argument("--save_raw_ckpt", dest="save_raw_ckpt", action="store_true")
    parser.add_argument("--no_save_raw_ckpt", dest="save_raw_ckpt", action="store_false")
    parser.set_defaults(save_raw_ckpt=True)

    parser.add_argument("--train_future_video_noise_only", nargs="?", const=True, default=True, type=_str2bool)
    parser.add_argument("--no_train_future_video_noise_only", dest="train_future_video_noise_only", action="store_false")
    parser.add_argument("--infer_replace_history_latents_before_decode", nargs="?", const=True, default=True, type=_str2bool)
    parser.add_argument(
        "--no_infer_replace_history_latents_before_decode",
        dest="infer_replace_history_latents_before_decode",
        action="store_false",
    )
    parser.add_argument("--use_gradient_checkpointing", nargs="?", const=True, default=True, type=_str2bool)
    parser.add_argument("--no_use_gradient_checkpointing", dest="use_gradient_checkpointing", action="store_false")
    parser.add_argument("--use_gradient_checkpointing_offload", action="store_true")

    parser.add_argument("--output_path", type=str, default="./outputs/train_navsim_v1")
    parser.add_argument("--train_log_file", type=str, default="train.log")
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--auto_eval", dest="auto_eval", action="store_true")
    parser.add_argument("--no_auto_eval", dest="auto_eval", action="store_false")
    parser.set_defaults(auto_eval=False)
    parser.add_argument("--auto_eval_ckpt_kind", type=str, default="all", choices=["raw", "ema", "all"])
    parser.add_argument("--auto_eval_strict", action="store_true")
    parser.add_argument("--infer_all_output_root", type=str, default=None)

    parser.add_argument("--extra_inputs", default="longcat_video,trajectory,ego_vel")
    parser.add_argument("--use_trajectory", action="store_true")
    parser.add_argument(
        "--trajectory_condition_mode",
        type=str,
        default="velocity",
        choices=["auto", "history", "velocity"],
    )
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0)
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0)
    parser.add_argument("--enable-online-selector", dest="enable_online_selector", action="store_true")
    parser.add_argument("--selector-warmup-steps", type=int, default=0)
    parser.add_argument("--selector-layer", type=int, default=15)
    parser.add_argument("--selector-loss-weight", type=float, default=1.0)
    parser.add_argument("--selector-input-variant", type=str, default="token_condition", choices=["token_condition"])
    parser.add_argument(
        "--selector-feature-mode",
        type=str,
        default="all",
        choices=["all", "condition_position_time"],
    )
    parser.add_argument("--selector-keep-schedule", type=str, default="")
    parser.add_argument("--selector-teacher-keep-ratio", type=float, default=0.375)
    parser.add_argument("--selector-gradient-interval", type=int, default=1)
    parser.add_argument("--selector-mask-start-step", type=int, default=-1)
    parser.add_argument("--selector-only", action="store_true")
    parser.add_argument("--selector-checkpoint", type=str, default=None)
    parser.add_argument(
        "--selector-teacher-mode",
        type=str,
        default="gradient_abs",
        choices=["gradient_abs", "signed_hybrid", "displacement", "planning_harm"],
    )
    parser.add_argument("--selector-signed-temperature", type=float, default=1.0)
    parser.add_argument("--selector-counterfactual-interval", type=int, default=4)
    parser.add_argument("--selector-counterfactual-weight", type=float, default=1.0)
    parser.add_argument("--selector-counterfactual-scale", type=float, default=0.05)
    parser.add_argument("--selector-counterfactual-tile-h", type=int, default=3)
    parser.add_argument("--selector-counterfactual-tile-w", type=int, default=4)
    parser.add_argument("--selector-counterfactual-physical", action="store_true")
    parser.add_argument(
        "--selector-counterfactual-injection-point",
        type=str,
        default="post_block",
        choices=["post_block", "pre_dit"],
        help=(
            "where the teacher removes history tokens; pre_dit matches the "
            "deployment selector before block 0"
        ),
    )
    parser.add_argument("--selector-teacher-timesteps", type=str, default="")
    parser.add_argument("--selector-teacher-seed", type=int, default=None)
    parser.add_argument(
        "--selector-counterfactual-dump-dir",
        type=str,
        default=None,
        help="optional directory receiving one .pt shard per counterfactual probe",
    )
    parser.add_argument(
        "--selector-counterfactual-replicate",
        type=int,
        default=0,
        help="repeat the masked forward N times to measure the label noise floor",
    )
    parser.add_argument(
        "--selector-counterfactual-sweep-all",
        action="store_true",
        help="probe every spatial tile of the scene in one step (full label coverage)",
    )
    parser.add_argument(
        "--selector-counterfactual-replays",
        type=int,
        default=1,
        help="independent noise realisations averaged into one label",
    )
    parser.add_argument(
        "--selector-counterfactual-scales",
        type=str,
        default="",
        help="comma-separated removal sizes in tiles, probed in the same step",
    )
    parser.add_argument("--selector-counterfactual-abstain-eps", type=float, default=0.0)
    parser.add_argument("--selector-counterfactual-noise-seed", type=int, default=1234)
    parser.add_argument(
        "--selector-counterfactual-latent-index",
        type=str,
        default=str(_env_first("SELECTOR_COUNTERFACTUAL_LATENT_INDEX", default="0")),
        help=(
            "conditioned history latent the counterfactual teacher probes, "
            "counted back from the newest history latent (0 = newest, the "
            "historical default); a comma-separated list probes several "
            "latents in the same step"
        ),
    )
    parser.add_argument(
        "--selector-counterfactual-jsonl-dir",
        type=str,
        default=_env_first("SELECTOR_COUNTERFACTUAL_JSONL_DIR", default=""),
        help=(
            "optional directory receiving one compact JSON line per "
            "(scene, tile, latent) probe instead of the 3072-d .pt shards"
        ),
    )
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--selector-teacher-disp-scale", type=float, default=0.01)
    parser.add_argument(
        "--selector-teacher-disp-normalize",
        type=str,
        default="scene",
        choices=["scene", "absolute"],
    )
    parser.add_argument(
        "--selector-teacher-disp-min-spread",
        type=float,
        default=float(_env_first("SELECTOR_TEACHER_DISP_MIN_SPREAD", default=0.0)),
        help=(
            "scene-relative teacher normalisation abstains when the measured "
            "tile spread is below this value; 0.0 preserves the legacy min-max "
            "behaviour"
        ),
    )
    parser.add_argument(
        "--selector-ranking-loss-weight",
        type=float,
        default=0.0,
        help="auxiliary top-k ranking loss weight; 0 preserves the legacy objective",
    )
    parser.add_argument("--selector-ranking-margin", type=float, default=0.0)
    parser.add_argument("--selector-ranking-max-pairs", type=int, default=4096)
    parser.add_argument(
        "--selector-critical-token-mode",
        choices=["none", "provided", "boxes"],
        default="none",
        help="force provided token mask or projected yxyx object boxes into top-k",
    )
    parser.add_argument("--selector-critical-token-dilation", type=float, default=0.0)
    parser.add_argument(
        "--selector-long-horizon-weights",
        type=str,
        default="",
        help="comma-separated seconds:weight target, e.g. 1:0.2,2:0.3,3:0.5",
    )

    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    process_seed = int(args.seed) + int(os.environ.get("RANK", "0"))
    random.seed(process_seed)
    np.random.seed(process_seed % (2**32))
    torch.manual_seed(process_seed)
    if args.selector_only and not args.enable_online_selector:
        raise ValueError("--selector-only requires --enable-online-selector")
    if args.selector_only and not args.full_ckpt:
        raise ValueError("--selector-only requires a pretrained --full_ckpt teacher")
    if args.selector_only and args.lora_checkpoint:
        raise ValueError("--selector-only is incompatible with --lora_checkpoint")
    if args.selector_only and args.selector_gradient_interval != 1:
        raise ValueError("--selector-only requires --selector-gradient-interval 1")
    if args.selector_only and args.selector_warmup_steps != 0:
        raise ValueError("--selector-only requires --selector-warmup-steps 0")
    if args.enable_online_selector and not 0 <= args.selector_layer < 30:
        raise ValueError("--selector-layer must be in [0, 29]")
    if args.enable_online_selector and not 0.0 < args.selector_teacher_keep_ratio <= 1.0:
        raise ValueError("--selector-teacher-keep-ratio must be in (0, 1]")
    if args.selector_signed_temperature <= 0:
        raise ValueError("--selector-signed-temperature must be positive")
    if args.selector_counterfactual_interval <= 0:
        raise ValueError("--selector-counterfactual-interval must be positive")
    if args.selector_counterfactual_weight < 0 or args.selector_counterfactual_scale <= 0:
        raise ValueError("counterfactual weight must be non-negative and scale positive")
    if args.selector_counterfactual_tile_h <= 0 or args.selector_counterfactual_tile_w <= 0:
        raise ValueError("counterfactual tile dimensions must be positive")
    latent_index_tokens = [
        value.strip()
        for value in str(args.selector_counterfactual_latent_index).split(",")
        if value.strip()
    ]
    if not latent_index_tokens:
        raise ValueError("--selector-counterfactual-latent-index must not be empty")
    for token in latent_index_tokens:
        if not token.lstrip("+-").isdigit() or int(token) < 0:
            raise ValueError(
                "--selector-counterfactual-latent-index must be a non-negative "
                f"integer or comma-separated list of them, got '{token}'"
            )
    if args.enable_online_selector and args.selector_mask_start_step >= 0 and args.selector_mask_start_step <= args.selector_warmup_steps:
        raise ValueError("--selector-mask-start-step must be greater than --selector-warmup-steps")
    if args.lora_base_model is not None and args.lora_base_model.strip().lower() in {"none", "null", ""}:
        args.lora_base_model = None
    if int(args.target_fps) <= 0:
        raise ValueError(f"--target_fps must be > 0, got {args.target_fps}")
    if args.selector_only:
        args.trainable_models = ""
    elif args.use_trajectory and args.trainable_models is None:
        args.trainable_models = "trajectory_encoder,trajectory_head"

    extra_inputs = [x.strip() for x in args.extra_inputs.split(",") if x.strip()] if args.extra_inputs else []
    if args.use_trajectory and "trajectory" not in extra_inputs:
        extra_inputs.append("trajectory")
    if "ego_vel" not in extra_inputs:
        extra_inputs.append("ego_vel")
    if args.trajectory_condition_mode in {"auto", "history"} and "history_positions" not in extra_inputs:
        extra_inputs.append("history_positions")
    if args.trajectory_condition_mode == "velocity":
        extra_inputs = [x for x in extra_inputs if x != "history_positions"]
    if "longcat_video" not in extra_inputs:
        extra_inputs.append("longcat_video")
    args.extra_inputs = ",".join(extra_inputs) if extra_inputs else None

    _enable_train_log_capture(args.output_path, args.train_log_file)
    os.makedirs(args.output_path, exist_ok=True)
    with open(os.path.join(args.output_path, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    train_logs = [x.strip() for x in args.train_log_names.split(",") if x.strip()] if args.train_log_names else None
    train_scene_tokens = None
    if args.train_scene_manifest:
        train_rows = _read_jsonl_manifest(args.train_scene_manifest)
        semantic_train_tokens = _manifest_scene_tokens(train_rows, args.train_scene_manifest)
        train_scene_tokens = _representative_frame_tokens(
            train_rows,
            manifest_path=args.train_scene_manifest,
            num_history_frames=args.num_history_frames,
            num_future_frames=args.num_future_frames,
            frame_interval=args.frame_interval,
            has_route=not args.allow_missing_route,
            windows_per_scene=args.windows_per_scene,
        )
        if args.forbidden_scene_manifest:
            forbidden_tokens, forbidden_summaries = _forbidden_manifest_tokens(
                args.forbidden_scene_manifest
            )
            overlap = sorted(set(semantic_train_tokens) & forbidden_tokens)
            if overlap:
                raise ValueError(
                    f"train/forbidden scene manifests overlap by {len(overlap)} scene_token values; "
                    f"examples={overlap[:5]}"
                )
            print(
                "[train][split-audit]",
                f"train_scenes={len(semantic_train_tokens)}",
                f"forbidden_scenes={len(forbidden_tokens)}",
                f"forbidden_manifests={forbidden_summaries}",
                "scene_token_overlap=0",
            )
    dataset = NavsimDriveVADataset(
        NavsimDriveVAConfig(
            repo_root=args.repo_root,
            navsim_log_path=args.navsim_log_path,
            sensor_blobs_path=args.sensor_blobs_path,
            cache_path=args.cache_path,
            use_cache_only=args.use_cache_only,
            force_cache_computation=args.force_cache_computation,
            num_history_frames=args.num_history_frames,
            num_future_frames=args.num_future_frames,
            frame_interval=args.frame_interval,
            has_route=not args.allow_missing_route,
            train_log_names=train_logs,
            train_scene_tokens=train_scene_tokens,
            max_scenes=args.max_scenes,
            image_height=args.height,
            image_width=args.width,
            surround_view=args.surround_view,
            skip_missing_files=args.skip_missing_files,
            quiet_scene_loader=not args.print_navsim_tokens,
        ),
        split="train",
    )
    expected_manifest_samples = (
        len(train_scene_tokens)
        if train_scene_tokens is not None and args.max_scenes is None
        else min(len(train_scene_tokens), int(args.max_scenes))
        if train_scene_tokens is not None
        else None
    )
    if expected_manifest_samples is not None and len(dataset) != expected_manifest_samples:
        raise RuntimeError(
            f"scene manifest requested {expected_manifest_samples} samples but NAVSIM loaded {len(dataset)}"
        )

    model = DriveVANavsimTrainingModule(
        local_model_path=args.local_model_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        target_fps=args.target_fps,
        negative_prompt=args.negative_prompt,
        use_trajectory=args.use_trajectory,
        train_future_video_noise_only=args.train_future_video_noise_only,
        infer_replace_history_latents_before_decode=args.infer_replace_history_latents_before_decode,
        trajectory_condition_mode=args.trajectory_condition_mode,
        num_history_frames=args.num_history_frames,
        enable_online_selector=args.enable_online_selector,
        selector_warmup_steps=args.selector_warmup_steps,
        selector_layer=args.selector_layer,
        selector_loss_weight=args.selector_loss_weight,
        selector_keep_schedule=args.selector_keep_schedule,
        selector_input_variant=args.selector_input_variant,
        selector_feature_mode=args.selector_feature_mode,
        selector_teacher_keep_ratio=args.selector_teacher_keep_ratio,
        selector_gradient_interval=args.selector_gradient_interval,
        selector_mask_start_step=args.selector_mask_start_step,
        selector_only=args.selector_only,
        selector_checkpoint=args.selector_checkpoint,
        selector_teacher_mode=args.selector_teacher_mode,
        selector_signed_temperature=args.selector_signed_temperature,
        selector_counterfactual_interval=args.selector_counterfactual_interval,
        selector_counterfactual_weight=args.selector_counterfactual_weight,
        selector_counterfactual_scale=args.selector_counterfactual_scale,
        selector_counterfactual_tile_h=args.selector_counterfactual_tile_h,
        selector_counterfactual_tile_w=args.selector_counterfactual_tile_w,
        selector_counterfactual_physical=args.selector_counterfactual_physical,
        selector_counterfactual_injection_point=(
            args.selector_counterfactual_injection_point
        ),
        selector_teacher_timesteps=args.selector_teacher_timesteps,
        selector_teacher_seed=args.selector_teacher_seed,
        selector_counterfactual_dump_dir=args.selector_counterfactual_dump_dir,
        selector_counterfactual_replicate=args.selector_counterfactual_replicate,
        selector_counterfactual_sweep_all=args.selector_counterfactual_sweep_all,
        selector_counterfactual_replays=args.selector_counterfactual_replays,
        selector_counterfactual_scales=args.selector_counterfactual_scales,
        selector_counterfactual_abstain_eps=args.selector_counterfactual_abstain_eps,
        selector_counterfactual_noise_seed=args.selector_counterfactual_noise_seed,
        selector_counterfactual_latent_index=args.selector_counterfactual_latent_index,
        selector_counterfactual_jsonl_dir=args.selector_counterfactual_jsonl_dir,
        selector_teacher_disp_scale=args.selector_teacher_disp_scale,
        selector_teacher_disp_normalize=args.selector_teacher_disp_normalize,
        selector_teacher_disp_min_spread=args.selector_teacher_disp_min_spread,
        selector_ranking_loss_weight=args.selector_ranking_loss_weight,
        selector_ranking_margin=args.selector_ranking_margin,
        selector_ranking_max_pairs=args.selector_ranking_max_pairs,
        selector_critical_token_mode=args.selector_critical_token_mode,
        selector_critical_token_dilation=args.selector_critical_token_dilation,
        selector_long_horizon_weights=args.selector_long_horizon_weights,
    )

    if args.full_ckpt:
        if not os.path.exists(args.full_ckpt):
            raise FileNotFoundError(f"full_ckpt not found: {args.full_ckpt}")
        print(f"[train] loading full checkpoint: {args.full_ckpt}")
        state_dict = _normalize_train_ckpt_keys(load_state_dict(args.full_ckpt))
        missing, unexpected = model.pipe.load_state_dict(state_dict, strict=False)
        print(f"[train] full_ckpt loaded: keys={len(state_dict)}, missing={len(missing)}, unexpected={len(unexpected)}")
        if len(missing) > 0:
            print(f"[train] full_ckpt missing summary: {_summarize_ckpt_keys(list(missing))}")
        if len(unexpected) > 0:
            print(f"[train] full_ckpt unexpected summary: {_summarize_ckpt_keys(list(unexpected))}")

    remove_prefix = "pipe.dit." if args.lora_base_model is not None else "pipe."
    logger_cls = InProcessAutoEvalModelLogger if args.auto_eval else ModelLogger
    logger_kwargs = {}
    if args.auto_eval:
        logger_kwargs.update(
            train_args=args,
            auto_eval=True,
            auto_eval_ckpt_kind=args.auto_eval_ckpt_kind,
            auto_eval_strict=args.auto_eval_strict,
            infer_all_output_root=args.infer_all_output_root,
        )
    model_logger = logger_cls(
        args.output_path,
        remove_prefix_in_ckpt=remove_prefix,
        save_raw_ckpt=args.save_raw_ckpt,
        save_ema_ckpt=args.save_ema,
        **logger_kwargs,
    )

    print(
        "[train] navsim:",
        f"samples={len(dataset)}",
        f"history={args.num_history_frames}",
        f"future={args.num_future_frames}",
        f"target_fps={args.target_fps}",
        f"trajectory={bool(args.use_trajectory)}",
        f"condition_mode={args.trajectory_condition_mode}",
        f"auto_eval={bool(args.auto_eval)}",
    )
    # Selector architectures consume different amounts of RNG during model
    # construction. Reset here so held-out runs compare identical data order,
    # timesteps, and trajectory noise.
    random.seed(process_seed)
    np.random.seed(process_seed % (2**32))
    torch.manual_seed(process_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(process_seed)
    launch_training_task(dataset, model, model_logger, args=args, save_steps=args.save_steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
