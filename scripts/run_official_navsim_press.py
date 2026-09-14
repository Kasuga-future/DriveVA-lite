#!/usr/bin/env python3
"""Run VideoTokenPress through the official DriveVA/NAVSIM evaluation path.

This runner deliberately keeps the official evaluator as the owner of scene
loading, feature construction, Wan inference, trajectory conversion and PDM
scoring.  It supplies an externally constructed official pipeline only so a
runtime-only press can be installed on that pipeline instance.  No file under
``diffsynth`` or ``examples`` is modified by this integration.

The official evaluator's causal attention and gradient methods need a probe
forward because their scores are not available at the pre-concatenation
VIDEO_INPUT boundary.  The runner performs that probe, stores the frozen
ranking in ``ScoreCache``, and then lets the normal intervention forward use
the cached ranking.  Other methods use one official forward per scene.
"""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from dataclasses import asdict
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import types
from types import SimpleNamespace
from typing import Any, Callable, Iterable

import numpy as np
import torch


FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = FRAMEWORK_ROOT.parent
POC_TEST_DERIVED_STATUS = ["POC_ONLY", "TEST_DERIVED", "NOT_FOR_OFFICIAL_REPORTING"]
if str(FRAMEWORK_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(1, str(PROJECT_ROOT))

from evaluation.artifacts import environment_snapshot, jsonable
from evaluation.statistics import aggregate_suite, write_suite_tables
from evaluation.visualization import generate_suite_visualizations
from scripts.run_full_compression_suite import method_specs
from videopress.adapters.driveva import DriveVAAdapter
from videopress.adapters.scene_boundary import install_scene_boundary_guard, window_is_single_scene
from videopress.core.context import TokenContext
from videopress.core.domain import build_domain
from videopress.core.layout import TokenLayout
from videopress.core.runtime import InjectionPoint, VideoPressRuntime
from videopress.factory import build_press
from videopress.probes.score_cache import ScoreCache
from videopress.scorers.planning_gradient import (
    OBJECTIVE_TYPE as PLANNING_OBJECTIVE_TYPE,
    SCORE_REDUCTION as PLANNING_SCORE_REDUCTION,
    original_gradient_input_reduction,
    trajectory_projection_objective,
)
from videopress.utils.tensor import canonicalize_qkv


def _load_official_eval_module():
    """Load the repository-local official evaluator by absolute path."""

    path = PROJECT_ROOT / "examples" / "wanvideo" / "driveva_infer" / "eval_navsim_pdm.py"
    spec = importlib.util.spec_from_file_location("driveva_lite_official_eval", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load official evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _configure_official_data_environment(repo_root: Path) -> dict[str, str]:
    """Force the official process to use this DriveVA data mount.

    The host may have another project exporting ``OPENSCENE_DATA_ROOT`` or
    ``NUPLAN_MAPS_ROOT``.  NAVSIM reads these values at module-import time, so
    inheriting them can make a correct DriveVA run fail before inference with a
    misleading missing-map error.  The evaluator arguments remain the source
    of truth for log/sensor/cache paths; these variables align the libraries
    that construct maps and scenarios with the same repository.
    """

    repo_root = repo_root.expanduser().resolve()
    openscene_root = repo_root / "data" / "navsim_v1.1" / "openscene-v1.1"
    map_candidates = (
        repo_root / "data" / "nuplan" / "nuplan-maps-v1.0",
        repo_root / "data" / "nuplan" / "maps",
    )
    map_root = next(
        (candidate for candidate in map_candidates if (candidate / "nuplan-maps-v1.0.json").exists()),
        None,
    )
    if not openscene_root.exists():
        raise FileNotFoundError(f"DriveVA OPENSCENE_DATA_ROOT does not exist: {openscene_root}")
    if map_root is None:
        raise FileNotFoundError(
            "DriveVA NUPLAN_MAPS_ROOT does not contain nuplan-maps-v1.0.json; tried:\n"
            + "\n".join(str(candidate) for candidate in map_candidates)
        )
    values = {
        "OPENSCENE_DATA_ROOT": str(openscene_root),
        "NUPLAN_MAPS_ROOT": str(map_root),
        "NUPLAN_DATA_ROOT": str(repo_root / "data" / "nuplan"),
    }
    for key, value in values.items():
        os.environ[key] = value
    return values


def _str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean: {value}")


def _optional_path(value: str | None, default: Path) -> str:
    return str(Path(value).expanduser().resolve() if value else default.resolve())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the complete VideoTokenPress matrix with official DriveVA/NAVSIM PDM."
    )
    parser.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--navsim-log-path",
        type=Path,
        default=PROJECT_ROOT / "data/navsim_v1.1/openscene-v1.1/meta_datas/test",
    )
    parser.add_argument(
        "--sensor-blobs-path",
        type=Path,
        default=PROJECT_ROOT / "data/navsim_v1.1/openscene-v1.1/sensor_blobs/test",
    )
    parser.add_argument(
        "--metric-cache-path",
        type=Path,
        default=PROJECT_ROOT / "data/navsim_v1.1/metric_cache",
    )
    parser.add_argument(
        "--full-ckpt",
        type=Path,
        default=PROJECT_ROOT / "checkpoints/pdms90_9.safetensors",
    )
    parser.add_argument("--local-model-path", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument(
        "--scene-filter-yaml",
        type=Path,
        default=PROJECT_ROOT / "examples/wanvideo/driveva_infer/navsim_scene_filters/navtest.yaml",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=FRAMEWORK_ROOT / "outputs/official_navsim_full",
    )
    parser.add_argument(
        "--score-cache-root",
        type=Path,
        default=None,
        help="Optional root for frozen probe score caches; useful for large runs on NVMe.",
    )
    parser.add_argument("--methods", default=None, help="comma-separated method names; default is all 18 methods")
    parser.add_argument(
        "--domain",
        default="last_history",
        choices=("last_history", "history", "all_history"),
        help="token selection domain; history/all_history covers both history latents",
    )
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--seed-base", type=int, default=20260828)
    parser.add_argument("--max-eval-tokens", type=int, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=3)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-history-frames", type=int, default=5)
    parser.add_argument("--num-future-frames", type=int, default=10)
    parser.add_argument("--model-future-frames", type=int, default=8)
    parser.add_argument("--target-fps", type=int, default=2)
    parser.add_argument("--pdm-num-poses", type=int, default=40)
    parser.add_argument("--pdm-interval-length", type=float, default=0.1)
    parser.add_argument("--save-viz", action="store_true")
    parser.add_argument("--viz-total-tokens", type=int, default=100)
    parser.add_argument("--viz-max-tokens", type=int, default=20)
    parser.add_argument("--enable-nuscenes-metrics", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--allow-existing-output", action="store_true")
    parser.add_argument(
        "--poc-test-derived",
        action="store_true",
        help="mark every emitted artifact as POC_ONLY / TEST_DERIVED / NOT_FOR_OFFICIAL_REPORTING",
    )
    parser.add_argument(
        "--force-full-scene-set",
        action="store_true",
        help="fail if the official metric-cache intersection is smaller than the scene-filter token set",
    )
    parser.add_argument(
        "--gradient-debug-compare",
        action="store_true",
        help="also compute framework Gradient x Input diagnostics for the planning method",
    )
    return parser.parse_args(argv)


def _latest_official_csv(run_dir: Path) -> Path:
    paths = sorted(run_dir.glob("pdm_score_*.csv"))
    if not paths:
        raise FileNotFoundError(f"official evaluator wrote no PDM CSV in {run_dir}")
    return paths[-1]


def _finite(value: Any, default: float = float("nan")) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _metadata_subset(metadata: dict[str, Any]) -> dict[str, Any]:
    """Keep event records compact while retaining all reporting fields."""

    keys = {
        "press",
        "scorer",
        "selector",
        "operator",
        "domain",
        "injection_point",
        "n_history",
        "n_eligible",
        "n_candidate",
        "candidate_start",
        "candidate_end",
        "n_kept",
        "history_keep_ratio",
        "eligible_keep_ratio",
        "protected_count",
        "selected_count",
        "q_length",
        "k_length_before",
        "k_length_after",
        "v_length_after",
        "theoretical_attn_ratio",
        "post_rope",
        "video_only_layout",
        "feature",
        "configured_domain",
        "resolved_domain",
        "domain_override",
        "scene_token",
        "scene_tokens",
        "segment_scene_token",
        "segment_scene_name",
        "segment_frame_idx_start",
        "segment_frame_idx_end",
        "segment_frame_count",
        "segment_frame_idx_contiguous",
        "selection_candidate_valid",
        "selection_candidate_unique",
        "selected_global_count",
        "selected_global_min",
        "selected_global_max",
    }
    return {key: jsonable(metadata[key]) for key in keys if key in metadata}


def _runtime_event_summary(runtime: VideoPressRuntime) -> dict[str, Any]:
    events = list(runtime.events)
    metadata = [_metadata_subset(event.result.metadata or {}) for event in events]
    last = metadata[-1] if metadata else {}

    def distinct(key: str) -> list[Any]:
        values = []
        for item in metadata:
            value = item.get(key)
            if value not in values:
                values.append(value)
        return values

    return {
        "event_count": len(events),
        "layer_count": len({event.key.layer_idx for event in events}),
        "diffusion_rank_count": len({event.key.diffusion_rank for event in events}),
        "selector_latency_ms": float(runtime.selector_latency_ms),
        "last": last,
        "invalid_selection_count": sum(
            1 for item in metadata if item.get("selection_candidate_valid") is False
        ),
        "noncontiguous_segment_count": sum(
            1 for item in metadata if item.get("segment_frame_idx_contiguous") is False
        ),
        "distinct_segment_scene_tokens": distinct("segment_scene_token"),
        "distinct_segment_scene_names": distinct("segment_scene_name"),
        "distinct_k_length_after": distinct("k_length_after"),
        "distinct_v_length_after": distinct("v_length_after"),
        "distinct_theoretical_attn_ratio": distinct("theoretical_attn_ratio"),
        "distinct_n_kept": distinct("n_kept"),
    }


class _RunState:
    def __init__(
        self,
        runtime: VideoPressRuntime,
        method_dir: Path,
        method_name: str,
        artifact_status: list[str] | None = None,
    ):
        self.runtime = runtime
        self.method_dir = method_dir
        self.method_name = method_name
        self.artifact_status = list(artifact_status or [])
        self.current_token: str | None = None
        self.segment_scene_token: str | None = None
        self.segment_scene_name: str | None = None
        self.segment_frame_idx_start: int | None = None
        self.segment_frame_idx_end: int | None = None
        self.segment_frame_count: int = 0
        self.segment_frame_idx_contiguous: bool = False
        self.event_path = method_dir / f"press_events.rank{_rank()}.jsonl"
        self.probe_count = 0
        self.probe_details: list[dict[str, Any]] = []

    def bind_scene(self, loader, token: str) -> None:
        """Bind the anchor token and its raw same-segment window atomically."""

        token = str(token)
        frames = getattr(loader, "scene_frames_dicts", {}).get(token)
        if not frames:
            raise RuntimeError(f"official scene loader has no frame window for token={token}")
        if not window_is_single_scene(frames):
            scene_tokens = sorted({str(frame.get("scene_token")) for frame in frames})
            scene_names = sorted({str(frame.get("scene_name")) for frame in frames})
            raise RuntimeError(
                f"scene window crosses a segment boundary token={token} "
                f"scene_tokens={scene_tokens} scene_names={scene_names}"
            )
        frame_indices = [int(frame["frame_idx"]) for frame in frames]
        self.current_token = token
        self.segment_scene_token = str(frames[0].get("scene_token"))
        self.segment_scene_name = str(frames[0].get("scene_name"))
        self.segment_frame_idx_start = frame_indices[0]
        self.segment_frame_idx_end = frame_indices[-1]
        self.segment_frame_count = len(frames)
        self.segment_frame_idx_contiguous = all(
            right - left == 1 for left, right in zip(frame_indices, frame_indices[1:])
        )

    def segment_metadata(self) -> dict[str, Any]:
        return {
            "segment_scene_token": self.segment_scene_token,
            "segment_scene_name": self.segment_scene_name,
            "segment_frame_idx_start": self.segment_frame_idx_start,
            "segment_frame_idx_end": self.segment_frame_idx_end,
            "segment_frame_count": self.segment_frame_count,
            "segment_frame_idx_contiguous": self.segment_frame_idx_contiguous,
        }

    def sample(self):
        token = self.current_token or "unknown"
        return SimpleNamespace(
            scene_token=token,
            metadata={
                "scene_tokens": [token],
                "method": self.method_name,
                **self.segment_metadata(),
            },
        )

    def write_event(self, *, valid: bool, error: str | None = None) -> None:
        summary = _runtime_event_summary(self.runtime)
        row = {
            "artifact_status": self.artifact_status,
            "scene_token": self.current_token or "unknown",
            "method": self.method_name,
            "valid": bool(valid),
            "error": error,
            "probe_count": self.probe_count,
            "probe_details": list(self.probe_details),
            "segment": self.segment_metadata(),
            "runtime": summary,
        }
        self.method_dir.mkdir(parents=True, exist_ok=True)
        with self.event_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(jsonable(row), sort_keys=True) + "\n")
        self.probe_details = []


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _device_from_dist(official_eval, dist_info: dict[str, int]) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("official VideoTokenPress evaluation requires CUDA")
    torch.cuda.set_device(int(dist_info["local_rank"]))
    return torch.device(f"cuda:{int(dist_info['local_rank'])}")


def _build_official_pipeline(official_eval, args: argparse.Namespace, device: torch.device):
    from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline

    model_configs = [
        ModelConfig(
            model_id="Wan-AI/Wan2.2-TI2V-5B",
            origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
            offload_device=str(device),
            local_model_path=str(args.local_model_path),
            skip_download=True,
        ),
        ModelConfig(
            model_id="Wan-AI/Wan2.2-TI2V-5B",
            origin_file_pattern="diffusion_pytorch_model*.safetensors",
            offload_device=str(device),
            local_model_path=str(args.local_model_path),
            skip_download=True,
        ),
        ModelConfig(
            model_id="Wan-AI/Wan2.2-TI2V-5B",
            origin_file_pattern="Wan2.2_VAE.pth",
            offload_device=str(device),
            local_model_path=str(args.local_model_path),
            skip_download=True,
        ),
    ]
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=str(device),
        model_configs=model_configs,
        use_trajectory=True,
    )
    pipe.eval()

    if not args.full_ckpt.is_file():
        raise FileNotFoundError(args.full_ckpt)
    if _rank() == 0:
        print(f"[official-press] loading checkpoint: {args.full_ckpt}", flush=True)
    state_dict = official_eval._normalize_checkpoint_keys(official_eval.load_state_dict(str(args.full_ckpt)))
    missing, unexpected = pipe.load_state_dict(state_dict, strict=False)
    if _rank() == 0:
        print(
            f"[official-press] checkpoint loaded missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )
    return pipe


def _layout_from_model_call(kwargs: dict[str, Any], adapter: DriveVAAdapter) -> TokenLayout | None:
    latents = kwargs.get("latents")
    dit = kwargs.get("dit")
    if not torch.is_tensor(latents) or latents.ndim != 5 or dit is None:
        return None
    patch_size = tuple(int(value) for value in getattr(dit, "patch_size", (1, 2, 2)))
    if len(patch_size) != 3 or any(value <= 0 for value in patch_size):
        return None
    longcat = kwargs.get("longcat_latents")
    traj = kwargs.get("traj_tokens")
    num_cond = int(longcat.shape[2]) if torch.is_tensor(longcat) and longcat.ndim >= 3 else 0
    traj_len = int(traj.shape[1]) if torch.is_tensor(traj) and traj.ndim >= 2 else 0
    prefix_len = int(kwargs.get("traj_prefix_len", 0) or 0)
    return adapter.build_layout(
        int(latents.shape[2]) // patch_size[0],
        int(latents.shape[3]) // patch_size[1],
        int(latents.shape[4]) // patch_size[2],
        num_cond,
        traj_len,
        prefix_len,
    )


def _cache_probe_scores(
    *,
    runtime: VideoPressRuntime,
    press: Any,
    scene_token: str,
    captures: dict[int, dict[str, Any]],
    cache: ScoreCache,
    adapter: DriveVAAdapter,
    kind: str,
) -> list[dict[str, Any]]:
    scorer = getattr(press, "scorer", None)
    if scorer is None:
        raise RuntimeError(f"{kind} probe requires a scorer")
    details = []
    for diffusion_rank in sorted(captures):
        capture = captures[diffusion_rank]
        q_raw = capture.get("q")
        k_raw = capture.get("k")
        v_raw = capture.get("v")
        layout = capture.get("layout")
        if q_raw is None or k_raw is None or layout is None:
            raise RuntimeError(
                f"{kind} probe did not capture q/k/layout for scene={scene_token} rank={diffusion_rank}"
            )
        q, _ = canonicalize_qkv(q_raw, int(capture.get("num_heads") or 0))
        k, _ = canonicalize_qkv(k_raw, int(capture.get("num_heads") or 0))
        v = None
        if v_raw is not None:
            v, _ = canonicalize_qkv(v_raw, int(capture.get("num_heads") or 0))
        if q.shape[2] != layout.total_length or k.shape[2] != layout.total_length:
            raise RuntimeError(
                f"probe layout/attention length mismatch: layout={layout.total_length} q={q.shape[2]} k={k.shape[2]}"
            )
        tokens = torch.zeros(
            (q.shape[0], layout.total_length, q.shape[-1]),
            device=q.device,
            dtype=q.dtype,
        )
        domain_name = getattr(press, "domain", None) or "last_history"
        domain = build_domain(domain_name, layout, tokens.device)
        context = TokenContext(
            tokens=tokens,
            layout=layout,
            domain=domain,
            scene_token=scene_token,
            diffusion_rank=int(diffusion_rank),
            q=q,
            k=k,
            v=v,
            metadata={"scene_tokens": [scene_token], "probe": kind},
        )
        scores = press.score(context)
        ranking = press.ranking(scores)
        key = runtime.score_key(context)
        cache.save(
            key,
            scores,
            ranking=ranking,
            metadata={
                "probe": kind,
                "scene_token": scene_token,
                "diffusion_rank": int(diffusion_rank),
                "layer": getattr(scorer, "layer", None),
                "layout": layout.to_dict(),
            },
        )
        details.append(
            {
                "kind": kind,
                "diffusion_rank": int(diffusion_rank),
                "layer": getattr(scorer, "layer", None),
                "n_candidate": int(domain.n_candidate),
                "score_cache": str(cache.path_for(key)),
            }
        )
    if not details:
        raise RuntimeError(f"{kind} probe captured no attention/model calls for scene={scene_token}")
    return details


def _run_attention_probe(
    *,
    pipe: Any,
    state: _RunState,
    args: argparse.Namespace,
    adapter: DriveVAAdapter,
    invoke_kwargs: dict[str, Any],
) -> None:
    """Run one uncompressed official forward and cache post-RoPE attention scores."""

    press = state.runtime.press
    scorer = getattr(press, "scorer", None)
    target_layer = getattr(scorer, "layer", None)
    if target_layer is None:
        target_layer = 15
    captures: dict[int, dict[str, Any]] = {}
    original_model_fn = pipe.model_fn
    attention_restores: list[tuple[Any, Any]] = []

    def model_fn_probe(*call_args, **call_kwargs):
        layout = _layout_from_model_call(call_kwargs, adapter)
        timestep = call_kwargs.get("timestep")
        rank = int(timestep.reshape(-1)[0].item()) if torch.is_tensor(timestep) and timestep.numel() else None
        if rank is not None:
            captures.setdefault(rank, {})["layout"] = layout
        return original_model_fn(*call_args, **call_kwargs)

    pipe.model_fn = model_fn_probe
    try:
        for model_name in ("dit", "dit2"):
            model = getattr(pipe, model_name, None)
            blocks = getattr(model, "blocks", None) if model is not None else None
            if blocks is None or not (0 <= int(target_layer) < len(blocks)):
                continue
            attention = getattr(blocks[int(target_layer)], "self_attn", None)
            module = getattr(attention, "attn", None) if attention is not None else None
            if module is None:
                continue
            original_forward = module.forward

            def wrapped_forward(_module, q, k, v, *, _original=original_forward, _attention=attention):
                timestep = getattr(pipe, "_videopress_probe_timestep", None)
                rank = int(timestep) if timestep is not None else None
                if rank is not None:
                    payload = captures.setdefault(rank, {})
                    payload["q"] = q.detach().clone()
                    payload["k"] = k.detach().clone()
                    payload["v"] = v.detach().clone()
                    payload["num_heads"] = int(getattr(_attention, "num_heads", 0))
                    payload["model_name"] = model_name
                return _original(q, k, v)

            # The model_fn wrapper identifies the active timestep before the
            # attention hook is reached.  It is attached below per call.
            def wrapped_model_fn_with_timestep(*call_args, _base=model_fn_probe, **call_kwargs):
                timestep = call_kwargs.get("timestep")
                pipe._videopress_probe_timestep = (
                    int(timestep.reshape(-1)[0].item())
                    if torch.is_tensor(timestep) and timestep.numel()
                    else None
                )
                try:
                    return _base(*call_args, **call_kwargs)
                finally:
                    pipe._videopress_probe_timestep = None

            module.forward = types.MethodType(wrapped_forward, module)
            attention_restores.append((module, original_forward))

        # Replace the model function after all hooks are installed.  The
        # current timestep wrapper remains transparent to the official pipe.
        pipe.model_fn = wrapped_model_fn_with_timestep
        pipe(**invoke_kwargs)
    finally:
        for module, original_forward in attention_restores:
            module.forward = original_forward
        pipe.model_fn = original_model_fn
        if hasattr(pipe, "_videopress_probe_timestep"):
            delattr(pipe, "_videopress_probe_timestep")

    details = _cache_probe_scores(
        runtime=state.runtime,
        press=press,
        scene_token=str(state.current_token),
        captures=captures,
        cache=state.runtime.score_cache,
        adapter=adapter,
        kind="attention",
    )
    state.probe_count += 1
    state.probe_details.extend(details)


def _gradient_scores(
    scorer: Any,
    gradients: torch.Tensor,
    tokens: torch.Tensor,
    layout: TokenLayout,
    domain_name: str,
) -> torch.Tensor:
    candidate = build_domain(domain_name, layout, gradients.device).candidate_indices
    grad_candidate = gradients.index_select(1, candidate).float()
    token_candidate = tokens.index_select(1, candidate).float()
    if getattr(scorer, "name", "") == "gradient_input":
        attribution = grad_candidate * token_candidate
        if str(getattr(scorer, "reduction", "l2")) == "abs_sum":
            return attribution.abs().sum(dim=-1)
        return torch.linalg.vector_norm(attribution, dim=-1)
    return torch.linalg.vector_norm(grad_candidate, dim=-1)


def _pearson(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.float().reshape(-1)
    right = right.float().reshape(-1)
    left = left - left.mean()
    right = right - right.mean()
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    return float("nan") if float(denominator) == 0.0 else float(((left * right).sum() / denominator).item())


def _selection_diagnostics(framework: torch.Tensor, planning: torch.Tensor, k: int) -> dict[str, Any]:
    def ranks(values: torch.Tensor) -> torch.Tensor:
        order = values.argsort(stable=True)
        output = torch.empty(values.numel(), device=values.device, dtype=torch.float32)
        output[order] = torch.arange(values.numel(), device=values.device, dtype=torch.float32)
        return output

    framework_set = set(framework.argsort(descending=True, stable=True)[:k].cpu().tolist())
    planning_set = set(planning.argsort(descending=True, stable=True)[:k].cpu().tolist())
    intersection = len(framework_set & planning_set)
    union = len(framework_set | planning_set)
    return {
        "pearson": _pearson(framework, planning),
        "spearman": _pearson(ranks(framework), ranks(planning)),
        "topk_intersection": intersection,
        "topk_overlap_ratio": float(intersection / k) if k else 1.0,
        "topk_jaccard": float(intersection / union) if union else 1.0,
        "masks_different": framework_set != planning_set,
    }


def _run_gradient_probe(
    *,
    pipe: Any,
    state: _RunState,
    args: argparse.Namespace,
    adapter: DriveVAAdapter,
    invoke_kwargs: dict[str, Any],
) -> None:
    """Probe trajectory-head sensitivity to patchified history tokens."""

    press = state.runtime.press
    scorer = getattr(press, "scorer", None)
    if scorer is None:
        raise RuntimeError("gradient probe requires a scorer")
    domain_name = str(getattr(press, "domain", None) or "last_history")
    captures: dict[int, dict[str, Any]] = {}
    original_model_fn = pipe.model_fn

    def model_fn_probe(*call_args, **call_kwargs):
        dit = call_kwargs.get("dit")
        if dit is None or not hasattr(dit, "patchify"):
            return original_model_fn(*call_args, **call_kwargs)
        layout = _layout_from_model_call(call_kwargs, adapter)
        capture: dict[str, Any] = {"layout": layout}
        original_patchify = dit.patchify

        def wrapped_patchify(_model, x, *, _original=original_patchify):
            patched = _original(x)
            if not torch.is_tensor(patched) or patched.ndim != 5:
                raise RuntimeError("gradient probe expected patchify() -> [B,C,F,H,W]")
            tokens = patched.permute(0, 2, 3, 4, 1).reshape(
                patched.shape[0], -1, patched.shape[1]
            )
            # Use the exact tensor returned by the official patchify call as
            # the autograd target. A separately-created flattened view is not
            # guaranteed to be the tensor recorded by the model graph.
            capture["patched"] = patched
            capture["tokens"] = tokens
            return patched

        dit.patchify = types.MethodType(wrapped_patchify, dit)
        try:
            with torch.enable_grad():
                output = original_model_fn(*call_args, **call_kwargs)
                if not isinstance(output, dict) or output.get("traj") is None:
                    raise RuntimeError("gradient probe requires the official trajectory output")
                trajectory = output["traj"]
                patched = capture.get("patched")
                tokens = capture.get("tokens")
                if patched is None or tokens is None or layout is None:
                    raise RuntimeError("gradient probe did not capture video tokens/layout")
                planning_method = getattr(scorer, "name", "") == "planning_gradient_input"
                debug_compare = bool(planning_method and args.gradient_debug_compare)
                framework_objective = trajectory.float().pow(2).mean()
                trajectory_points = None
                planning_objective = None
                if planning_method:
                    planning_objective, trajectory_points = trajectory_projection_objective(
                        trajectory, int(call_kwargs.get("traj_prefix_len", 0) or 0)
                    )
                objective = planning_objective if planning_method else framework_objective
                gradients = torch.autograd.grad(
                    objective,
                    patched,
                    retain_graph=debug_compare,
                    create_graph=False,
                    allow_unused=False,
                )[0]
                gradient_tokens = gradients.permute(0, 2, 3, 4, 1).reshape(
                    gradients.shape[0], -1, gradients.shape[1]
                )
                timestep = call_kwargs.get("timestep")
                if not torch.is_tensor(timestep) or not timestep.numel():
                    raise RuntimeError("gradient probe did not receive a timestep")
                rank = int(timestep.reshape(-1)[0].item())
                candidate = build_domain(domain_name, layout, gradient_tokens.device).candidate_indices
                candidate_gradients = gradient_tokens.index_select(1, candidate)
                candidate_tokens = tokens.index_select(1, candidate)
                scores = (
                    original_gradient_input_reduction(candidate_gradients, candidate_tokens)
                    if planning_method
                    else _gradient_scores(scorer, gradient_tokens, tokens, layout, domain_name)
                )
                payload = {
                    "layout": layout,
                    "scores": scores.detach(),
                    "candidate_tokens": candidate_tokens.detach(),
                    "n_candidate": int(candidate.numel()),
                    "gradient_target_shape": tuple(patched.shape),
                    "candidate_gradient_shape": tuple(candidate_gradients.shape),
                    "trajectory_shape": tuple(trajectory.shape),
                    "trajectory_points_shape": tuple(trajectory_points.shape) if trajectory_points is not None else None,
                    "objective_value": float(objective.detach().float().item()),
                }
                if debug_compare:
                    framework_gradients = torch.autograd.grad(
                        framework_objective, patched, retain_graph=False, create_graph=False, allow_unused=False
                    )[0]
                    framework_tokens = framework_gradients.permute(0, 2, 3, 4, 1).reshape(
                        framework_gradients.shape[0], -1, framework_gradients.shape[1]
                    )
                    payload["framework_scores"] = _gradient_scores(
                        SimpleNamespace(name="gradient_input", reduction="l2"),
                        framework_tokens,
                        tokens,
                        layout,
                        domain_name,
                    ).detach()
                captures[rank] = payload
                return output
        finally:
            dit.patchify = original_patchify

    pipe.model_fn = model_fn_probe
    try:
        pipe(**invoke_kwargs)
    finally:
        pipe.model_fn = original_model_fn

    cache = state.runtime.score_cache
    details = []
    for diffusion_rank in sorted(captures):
        capture = captures[diffusion_rank]
        scores = capture["scores"]
        layout = capture["layout"]
        tokens = torch.zeros(
            (scores.shape[0], layout.total_length, 1),
            device=scores.device,
            dtype=scores.dtype,
        )
        # A zero-token context is sufficient to construct the stable cache
        # key.  The actual ranking was computed from the official gradient.
        context = TokenContext(
            tokens=tokens,
            layout=layout,
            domain=build_domain(domain_name, layout, tokens.device),
            scene_token=str(state.current_token),
            diffusion_rank=int(diffusion_rank),
            metadata={"scene_tokens": [str(state.current_token)], "probe": "gradient"},
        )
        key = state.runtime.score_key(context)
        ranking = scores.argsort(dim=-1, descending=True, stable=True)
        k = int(round(context.domain.n_candidate * 0.5))
        selected_global = context.domain.candidate_indices[ranking[:, :k]]
        candidate_keep_mask = torch.zeros_like(scores, dtype=torch.bool)
        candidate_keep_mask.scatter_(1, ranking[:, :k], True)
        candidate_positions = torch.tensor(
            [
                (int(global_index) // layout.tokens_per_latent,)
                + divmod(int(global_index) % layout.tokens_per_latent, layout.video_w)
                for global_index in context.domain.candidate_indices.detach().cpu().tolist()
            ],
            dtype=torch.int16,
        )
        keep_mask = torch.zeros((scores.shape[0], layout.total_length), dtype=torch.bool, device=scores.device)
        keep_mask.scatter_(1, selected_global, True)
        planning_method = getattr(scorer, "name", "") == "planning_gradient_input"
        artifact_metadata = {
            "artifact_status": list(state.artifact_status),
            "method_name": getattr(scorer, "name", "gradient_input"),
            "anchor_id": str(state.current_token),
            "diffusion_timestep": int(diffusion_rank),
            "domain": domain_name,
            "candidate_count": int(context.domain.n_candidate),
            "K": k,
            "objective_type": PLANNING_OBJECTIVE_TYPE if planning_method else "mean_traj_noise_prediction_squared",
            "objective_value": capture["objective_value"],
            "score_reduction": PLANNING_SCORE_REDUCTION if planning_method else getattr(scorer, "reduction", "l2"),
            "trajectory_tensor_shape": capture["trajectory_shape"],
            "trajectory_points_shape": capture["trajectory_points_shape"],
            "gradient_target_shape": capture["gradient_target_shape"],
            "candidate_gradient_shape": capture["candidate_gradient_shape"],
        }
        if "framework_scores" in capture:
            artifact_metadata["framework_comparison"] = _selection_diagnostics(
                capture["framework_scores"][0], scores[0], k
            )
        if planning_method:
            artifact_dir = state.method_dir / "planning_gradient_artifacts" / str(state.current_token)
            artifact_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    **artifact_metadata,
                    "scores": scores.detach().cpu(),
                    "candidate_tokens": capture["candidate_tokens"].detach().cpu(),
                    "candidate_positions": candidate_positions,
                    "rank_local_descending": ranking.detach().cpu(),
                    "topk_mask_candidate": candidate_keep_mask.detach().cpu(),
                    "selected_indices": selected_global.detach().cpu(),
                    "mask": keep_mask.detach().cpu(),
                    "mask_semantics": "True=kept; every False token is zeroed",
                },
                artifact_dir / f"timestep_{int(diffusion_rank)}.pt",
            )
            print("[planning-gradient-debug] " + json.dumps(jsonable({
                **artifact_metadata,
                "score_min": float(scores.min().item()),
                "score_max": float(scores.max().item()),
                "score_mean": float(scores.mean().item()),
                "score_std": float(scores.std().item()),
                "score_has_nan": bool(torch.isnan(scores).any().item()),
                "score_has_inf": bool(torch.isinf(scores).any().item()),
                "selected_indices": selected_global[0].detach().cpu().tolist(),
            }), sort_keys=True), flush=True)
        cache.save(
            key,
            scores,
            ranking=ranking,
            metadata={
                "probe": "original_trajectory_projection_gradient_input" if planning_method else "gradient_trajectory_l2_zero_target",
                "scene_token": str(state.current_token),
                "diffusion_rank": int(diffusion_rank),
                "layout": layout.to_dict(),
                **artifact_metadata,
            },
        )
        details.append(
            {
                "kind": "gradient",
                "diffusion_rank": int(diffusion_rank),
                "score_cache": str(cache.path_for(key)),
                "objective": artifact_metadata["objective_type"],
            }
        )
    if not details:
        raise RuntimeError(f"gradient probe captured no calls for scene={state.current_token}")
    state.probe_count += 1
    state.probe_details.extend(details)


class _PipelineProxy:
    """Delegate every official pipeline attribute but intercept scene calls."""

    def __init__(self, pipe: Any, state: _RunState, adapter: DriveVAAdapter, args: argparse.Namespace):
        object.__setattr__(self, "_pipe", pipe)
        object.__setattr__(self, "_state", state)
        object.__setattr__(self, "_adapter", adapter)
        object.__setattr__(self, "_args", args)

    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, "_pipe"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "_pipe"), name, value)

    def eval(self):
        self._pipe.eval()
        return self

    def __call__(self, *call_args, **call_kwargs):
        state = self._state
        runtime = state.runtime
        if not state.current_token or state.current_token == "unknown":
            raise RuntimeError(
                "official pipeline call has no active scene token; refusing to reuse a stale scene binding"
            )
        sample = state.sample()
        runtime.begin_sample(sample)
        press = runtime.press
        scorer = getattr(press, "scorer", None) if press is not None else None
        point = InjectionPoint.parse(getattr(press, "injection_point", InjectionPoint.VIDEO_INPUT)) if press else None
        requires_probe = bool(getattr(scorer, "requires_probe", False)) or (
            point is InjectionPoint.VIDEO_INPUT
            and str(getattr(scorer, "probe_mode", "none")) in {"online", "ProbeMode.ONLINE"}
        )
        try:
            if requires_probe:
                if runtime.score_cache is None:
                    raise RuntimeError("probe method requires a ScoreCache")
                runtime.remove(self._pipe)
                invoke_kwargs = dict(call_kwargs)
                if call_args:
                    raise RuntimeError("official VideoTokenPress proxy only supports keyword pipeline calls")
                if bool(getattr(scorer, "requires_probe", False)):
                    _run_gradient_probe(
                        pipe=self._pipe,
                        state=state,
                        args=self._args,
                        adapter=self._adapter,
                        invoke_kwargs=invoke_kwargs,
                    )
                else:
                    _run_attention_probe(
                        pipe=self._pipe,
                        state=state,
                        args=self._args,
                        adapter=self._adapter,
                        invoke_kwargs=invoke_kwargs,
                    )
                runtime.install(self._pipe)
                runtime.begin_sample(sample)
            result = self._pipe(*call_args, **call_kwargs)
            state.write_event(valid=True)
            return result
        except Exception as exc:
            state.write_event(valid=False, error=str(exc))
            raise


def _patch_official_scene_hooks(official_eval, state_box: dict[str, _RunState]) -> None:
    original_loader_builder = official_eval._build_scene_loader
    original_scene_builder = official_eval._build_scene_without_print

    def build_scene_loader(*call_args, **call_kwargs):
        loader = original_loader_builder(*call_args, **call_kwargs)
        original_get = loader.get_agent_input_from_token

        def get_agent_input_from_token(token):
            state = state_box["state"]
            state.bind_scene(loader, str(token))
            return original_get(token)

        loader.get_agent_input_from_token = get_agent_input_from_token
        return loader

    def build_scene_without_print(scene_loader, scene_cls, token):
        state_box["state"].bind_scene(scene_loader, str(token))
        return original_scene_builder(scene_loader, scene_cls, token)

    official_eval._build_scene_loader = build_scene_loader
    official_eval._build_scene_without_print = build_scene_without_print


def _official_args(args: argparse.Namespace, output_dir: Path, official_eval) -> argparse.Namespace:
    argv = [
        "--repo_root",
        str(args.repo_root.resolve()),
        "--navsim_log_path",
        str(args.navsim_log_path.resolve()),
        "--sensor_blobs_path",
        str(args.sensor_blobs_path.resolve()),
        "--metric_cache_path",
        str(args.metric_cache_path.resolve()),
        "--output_dir",
        str(output_dir.resolve()),
        "--scene_filter_yaml",
        str(args.scene_filter_yaml.resolve()),
        "--scene_filter_yaml_filter_only",
        "1",
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--num_history_frames",
        str(args.num_history_frames),
        "--num_future_frames",
        str(args.num_future_frames),
        "--model_future_frames",
        str(args.model_future_frames),
        "--target_fps",
        str(args.target_fps),
        "--pdm_num_poses",
        str(args.pdm_num_poses),
        "--pdm_interval_length",
        str(args.pdm_interval_length),
        "--local_model_path",
        str(args.local_model_path.resolve()),
        "--full_ckpt",
        str(args.full_ckpt.resolve()),
        "--num_inference_steps",
        str(args.num_inference_steps),
        "--cfg_scale",
        str(args.cfg_scale),
        "--seed",
        str(args.seed),
        "--no_show_eval_progress",
        "--no_print_errors",
        "--no_print_alignment_params",
        "--infer_trajectory_only",
    ]
    if args.max_eval_tokens is not None:
        argv.extend(["--max_eval_tokens", str(args.max_eval_tokens)])
    if args.save_viz:
        argv.extend([
            "--save_viz",
            "--viz_total_tokens",
            str(args.viz_total_tokens),
            "--viz_max_tokens",
            str(args.viz_max_tokens),
        ])
    if args.enable_nuscenes_metrics:
        argv.append("--enable_nuscenes_metrics")
    return official_eval.parse_args(argv)


def _join_official_records(method_dir: Path, method_spec: dict[str, Any], round_index: int) -> dict[str, Any]:
    csv_path = _latest_official_csv(method_dir)
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        official_rows = list(csv.DictReader(handle))
    official_rows = [row for row in official_rows if str(row.get("token", "")) != "average"]

    events: dict[str, dict[str, Any]] = {}
    for event_path in sorted(method_dir.glob("press_events.rank*.jsonl")):
        for line in event_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            events[str(item.get("scene_token"))] = item

    records = []
    for row in official_rows:
        token = str(row.get("token"))
        event = events.get(token, {})
        runtime = event.get("runtime", {}) if isinstance(event, dict) else {}
        last = runtime.get("last", {}) if isinstance(runtime, dict) else {}
        pdm = _finite(row.get("pdm_score"), default=_finite(row.get("score")))
        valid = _bool_value(row.get("valid"))
        if "valid" not in row:
            valid = bool(event.get("valid", True))
        record = {
            "scene_token": token,
            "valid": valid,
            "pdm": pdm,
            "trajectory_l2": float("nan"),
            "endpoint_l2": float("nan"),
            "latency_ms": _finite(row.get("infer_time_ms")),
            "model_latency_ms": _finite(row.get("infer_time_ms")),
            "e2e_latency_ms": _finite(row.get("infer_time_ms")),
            "selector_latency_ms": _finite(runtime.get("selector_latency_ms")),
            "peak_memory_mb": _finite(row.get("gpu_mem_alloc_peak_mb")),
            # Promote compression fields from the compact event metadata so
            # evaluation.statistics can aggregate them without knowing the
            # runner's event-file schema.
            "eligible_keep_ratio": _finite(last.get("eligible_keep_ratio")),
            "history_keep_ratio": _finite(last.get("history_keep_ratio")),
            # Physical KV operators expose the eligible domain directly; use
            # it as the candidate count when they do not emit a separate
            # n_candidate field.
            "n_candidate": _finite(last.get("n_candidate", last.get("n_eligible"))),
            "K": _finite(last.get("n_kept")),
            "press_name": method_spec["name"],
            "scorer": method_spec["press"].get("scorer", {}).get("name")
            if isinstance(method_spec["press"].get("scorer"), dict)
            else None,
            "operator": method_spec["press"].get("operator", {}).get("name")
            if isinstance(method_spec["press"].get("operator"), dict)
            else None,
            "domain": method_spec["press"].get("domain"),
            "metadata": {
                **last,
                "runtime_event_count": runtime.get("event_count", 0),
                "probe_count": event.get("probe_count", 0),
                "probe_details": event.get("probe_details", []),
            },
            "official": {
                key: jsonable(value)
                for key, value in row.items()
                if key not in {"token", "valid", "rank"}
            },
        }
        records.append(record)

    (method_dir / "records.jsonl").write_text(
        "".join(json.dumps(jsonable(record), sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    press_config = deepcopy(method_spec["press"])
    summary = {
        "method": method_spec["name"],
        "round": int(round_index),
        "mode": method_spec["mode"],
        "backend": "official_navsim",
        "press": press_config,
        "official_csv": str(csv_path),
        "n_scenes": len(records),
        "valid_scenes": sum(bool(record["valid"]) for record in records),
        "probe_protocol": "frozen_score_cache" if any(record["metadata"].get("probe_count", 0) for record in records) else "none",
    }
    (method_dir / "summary.json").write_text(
        json.dumps(jsonable(summary), indent=2, allow_nan=False), encoding="utf-8"
    )
    return summary


def _write_method_config(method_dir: Path, args: argparse.Namespace, method_spec: dict[str, Any], round_index: int) -> None:
    method_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "artifact_status": POC_TEST_DERIVED_STATUS if args.poc_test_derived else [],
        "backend": "official_navsim",
        "round": round_index,
        "method": method_spec["name"],
        "mode": method_spec["mode"],
        "press": method_spec["press"],
        "data": {
            "repo_root": str(args.repo_root.resolve()),
            "navsim_log_path": str(args.navsim_log_path.resolve()),
            "sensor_blobs_path": str(args.sensor_blobs_path.resolve()),
            "metric_cache_path": str(args.metric_cache_path.resolve()),
            "scene_filter_yaml": str(args.scene_filter_yaml.resolve()),
        },
        "score_cache_root": str(args.score_cache_root.resolve()) if args.score_cache_root else None,
        "model": {
            "full_ckpt": str(args.full_ckpt.resolve()),
            "local_model_path": str(args.local_model_path.resolve()),
            "num_inference_steps": args.num_inference_steps,
            "model_future_frames": args.model_future_frames,
        },
    }
    (method_dir / "config.json").write_text(
        json.dumps(jsonable(config), indent=2, allow_nan=False), encoding="utf-8"
    )
    (method_dir / "environment.json").write_text(
        json.dumps(jsonable(environment_snapshot(args.repo_root)), indent=2), encoding="utf-8"
    )


def _resolve_output_root(path: Path, allow_existing: bool) -> Path:
    path = path if path.is_absolute() else FRAMEWORK_ROOT / path
    if allow_existing:
        path.mkdir(parents=True, exist_ok=True)
        return path
    if not path.exists():
        path.mkdir(parents=True)
        return path
    index = 1
    while True:
        candidate = path.parent / f"{path.name}_rerun{index:02d}"
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
        index += 1


def _write_manifest(root: Path, manifest: dict[str, Any]) -> None:
    (root / "suite_manifest.json").write_text(
        json.dumps(jsonable(manifest), indent=2, allow_nan=False), encoding="utf-8"
    )


def _check_official_scene_coverage(official_eval, official_args: argparse.Namespace, args: argparse.Namespace) -> dict[str, int]:
    official_eval._ensure_navsim_importable(Path(official_args.repo_root))
    from navsim.common.dataclasses import SceneFilter, SensorConfig
    from navsim.common.dataloader import MetricCacheLoader

    overrides = official_eval._load_scene_filter_yaml(
        official_args.scene_filter_yaml,
        filter_only=official_args.scene_filter_yaml_filter_only,
    )
    scene_filter = SceneFilter(
        num_history_frames=official_args.num_history_frames,
        num_future_frames=official_args.num_future_frames,
        frame_interval=official_args.frame_interval,
        has_route=True,
        max_scenes=official_args.max_scenes,
        log_names=None,
    )
    for key, value in overrides.items():
        setattr(scene_filter, key, value)
    from navsim.common.dataloader import SceneLoader

    loader = SceneLoader(
        data_path=Path(official_args.navsim_log_path),
        sensor_blobs_path=Path(official_args.sensor_blobs_path),
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_all_sensors(include=True),
        load_image_path=True,
    )
    metric_cache_loader = MetricCacheLoader(Path(official_args.metric_cache_path))
    scene_tokens = set(loader.tokens)
    cache_tokens = set(metric_cache_loader.tokens)
    intersection = scene_tokens & cache_tokens
    counts = {
        "scene_filter_tokens": len(scene_tokens),
        "metric_cache_tokens": len(cache_tokens),
        "intersection_tokens": len(intersection),
        "missing_metric_cache_tokens": len(scene_tokens - cache_tokens),
    }
    if _rank() == 0:
        print("[official-press] coverage:", json.dumps(counts, sort_keys=True), flush=True)
    if args.force_full_scene_set and counts["missing_metric_cache_tokens"]:
        raise RuntimeError(
            "official metric cache is incomplete; build it with the official NAVSIM metric-caching command "
            f"before --force-full-scene-set (missing={counts['missing_metric_cache_tokens']})"
        )
    return counts


def run(args: argparse.Namespace) -> int:
    if args.rounds < 1:
        raise ValueError("--rounds must be >= 1")
    if args.max_eval_tokens is not None and args.max_eval_tokens < 1:
        raise ValueError("--max-eval-tokens must be >= 1 when set")
    for path in (
        args.navsim_log_path,
        args.sensor_blobs_path,
        args.metric_cache_path,
        args.full_ckpt,
        args.local_model_path,
        args.scene_filter_yaml,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    data_environment = _configure_official_data_environment(args.repo_root)
    official_eval = _load_official_eval_module()
    # The guard wraps navsim.common.dataloader, so the official evaluator's
    # normal repository-local import setup must run before installing it.
    official_eval._ensure_navsim_importable(args.repo_root.resolve())
    scene_boundary_guard = install_scene_boundary_guard()
    dist_info = official_eval._init_distributed()
    device = _device_from_dist(official_eval, dist_info)
    # Resolve the output directory once.  In a torchrun job every rank enters
    # this function, so resolving independently would make rank 0 choose
    # ``*_rerun01`` while rank 1 observes that directory and chooses
    # ``*_rerun02``.  That splits event journals and official CSV inputs and
    # makes the resulting suite impossible to audit.  Broadcast rank 0's
    # resolved path before any rank creates method outputs.
    requested_output_root = (
        args.output_root if args.output_root.is_absolute() else FRAMEWORK_ROOT / args.output_root
    )
    if int(dist_info["world_size"]) > 1 and torch.distributed.is_initialized():
        root_payload = [
            str(_resolve_output_root(requested_output_root, args.allow_existing_output))
            if int(dist_info["rank"]) == 0
            else None
        ]
        torch.distributed.broadcast_object_list(root_payload, src=0)
        root = Path(str(root_payload[0]))
    else:
        root = _resolve_output_root(requested_output_root, args.allow_existing_output)
    state_box: dict[str, _RunState] = {}
    _patch_official_scene_hooks(official_eval, state_box)

    all_specs = method_specs(args.seed_base, domain=args.domain)
    selected = None if not args.methods else {name.strip() for name in args.methods.split(",") if name.strip()}
    specs = [spec for spec in all_specs if selected is None or spec["name"] in selected]
    if not specs:
        raise ValueError("--methods selected no known method")

    manifest: dict[str, Any] = {
        "artifact_status": POC_TEST_DERIVED_STATUS if args.poc_test_derived else [],
        "suite_name": "official_navsim_videopress",
        "version": 1,
        "backend": "official_navsim",
        "repo_root": str(args.repo_root.resolve()),
        "device": str(device),
        "world_size": int(dist_info["world_size"]),
        "rounds": int(args.rounds),
        "scene_boundary_guard": scene_boundary_guard,
        "data_environment": data_environment,
        "scene_filter": {
            "num_history_frames": int(args.num_history_frames),
            "num_future_frames": int(args.num_future_frames),
            "frame_interval": 1,
            "has_route": True,
            "yaml_filter_only": True,
            "window_length": int(args.num_history_frames + args.num_future_frames),
        },
        "methods": [spec["name"] for spec in specs],
        "data": {
            "navsim_log_path": str(args.navsim_log_path.resolve()),
            "sensor_blobs_path": str(args.sensor_blobs_path.resolve()),
            "metric_cache_path": str(args.metric_cache_path.resolve()),
            "scene_filter_yaml": str(args.scene_filter_yaml.resolve()),
        },
        "score_cache_root": str(args.score_cache_root.resolve()) if args.score_cache_root else None,
        "runs": [],
    }
    if _rank() == 0:
        _write_manifest(root, manifest)

    # Build the model once and reuse its official weights for every method.
    pipe = _build_official_pipeline(official_eval, args, device)

    for round_index in range(1, args.rounds + 1):
        round_seed = int(args.seed_base + round_index - 1)
        round_specs = method_specs(round_seed, domain=args.domain)
        by_name = {spec["name"]: spec for spec in round_specs}
        for selected_spec in specs:
            spec = by_name[selected_spec["name"]]
            method_dir = root / f"round{round_index:02d}" / spec["name"]
            if _rank() == 0:
                print(
                    f"[official-press] start round={round_index} method={spec['name']} mode={spec['mode']}",
                    flush=True,
                )
            _write_method_config(method_dir, args, spec, round_index)
            score_cache = None
            scorer = spec["press"].get("scorer") if isinstance(spec["press"], dict) else None
            if isinstance(scorer, dict) and (
                scorer.get("name") in {"action_attention", "action_attention_vnorm", "gradient_norm", "gradient_input", "planning_gradient_input"}
            ):
                if args.score_cache_root is None:
                    score_cache_dir = method_dir / "score_cache"
                else:
                    score_cache_dir = (
                        args.score_cache_root.resolve()
                        / f"round{round_index:02d}"
                        / spec["name"]
                        / "score_cache"
                    )
                score_cache = ScoreCache(score_cache_dir)
            press = build_press(spec["press"])
            adapter = DriveVAAdapter()
            runtime = VideoPressRuntime(
                press=press,
                mode=spec["mode"],
                adapter=adapter,
                score_cache=score_cache,
            )
            state = _RunState(
                runtime,
                method_dir,
                spec["name"],
                POC_TEST_DERIVED_STATUS if args.poc_test_derived else None,
            )
            state_box["state"] = state
            proxy = _PipelineProxy(pipe, state, adapter, args)
            runtime.install(pipe)
            official_args = _official_args(args, method_dir, official_eval)
            if round_index == 1 and selected_spec is specs[0]:
                coverage = _check_official_scene_coverage(official_eval, official_args, args)
                if _rank() == 0:
                    manifest["coverage"] = coverage
                    _write_manifest(root, manifest)
            try:
                official_eval.run_eval(official_args, external_pipe=proxy)
            finally:
                runtime.remove(pipe)
            if _rank() == 0:
                summary = _join_official_records(method_dir, spec, round_index)
                manifest["runs"].append(
                    {
                        "method": spec["name"],
                        "round": round_index,
                        "protocol": spec["mode"],
                        "backend": "official_navsim",
                        "output_dir": str(method_dir.relative_to(root)),
                        "official_csv": summary["official_csv"],
                        "n_scenes": summary["n_scenes"],
                        "valid_scenes": summary["valid_scenes"],
                    }
                )
                _write_manifest(root, manifest)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()

    if _rank() == 0:
        summary = aggregate_suite(root)
        tables = write_suite_tables(summary, root / "statistics")
        plots = {} if args.skip_plots else generate_suite_visualizations(summary, root / "visualizations")
        report = {
            "suite_root": str(root.resolve()),
            "backend": "official_navsim",
            "run_count": len(summary["run_rows"]),
            "method_count": len(summary["method_rows"]),
            "coverage": manifest.get("coverage"),
            "tables": tables,
            "plots": plots,
            "method_rows": summary["method_rows"],
        }
        (root / "suite_summary.json").write_text(
            json.dumps(jsonable(report), indent=2, allow_nan=False), encoding="utf-8"
        )
        print(json.dumps(jsonable(report), indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
