#!/usr/bin/env python3
"""Route A integration smoke on the REAL DriveVA model and a real NAVSIM scene.

This is the bridge between "Route A is implemented" and "Route A trains on
NAVSIM".  It loads the released DriveVA checkpoint through the official eval
harness, attaches :class:`RouteADynamicSelect` to the live DiT via the guarded
``dit._tokenpress_route_a`` hook in ``model_fn_wan_video``, and runs the real
evaluation loop for a single scene.

What it proves (and what it does not):

* proves the insertion point is correct at the real geometry -- 30 blocks,
  ``dim=3072``, a 1569-token sequence (1560 video + 9 trajectory), real
  ``t_mod``/``freqs``/RoPE layout, real ``dit.head`` and ``dit.unpatchify``;
* proves the threshold gate, the sparse backend and the dense recovery decoder
  run on real weights in the model dtype without shape or dtype errors;
* proves the physically shortened forward is inference-correct enough to
  produce a valid PDM record;
* does NOT prove anything about quality -- the scorer is randomly initialised,
  so the PDM is expected to be poor.  This is a wiring test, not a result.

Usage::

    CUDA_VISIBLE_DEVICES=3 python scripts/route_a_real_forward_smoke.py \\
        --max-eval-tokens 1 --bottleneck 18 --keep-hint 0.5
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = FRAMEWORK_ROOT.parent
sys.path.insert(0, str(FRAMEWORK_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from videopress.retraining import (  # noqa: E402
    RouteAConfig,
    RouteALayoutSpec,
    SafetyClampConfig,
)

NVME = Path("/mnt/nvme/chenpeijian/autodrive/DriveVA/data/navsim_v1.1")
# ``local_model_path`` is the directory that CONTAINS ``Wan-AI/`` -- the runner
# appends the model id itself, so pointing at the leaf directory makes the
# tokenizer lookup resolve to nothing.
DEFAULT_FULL_CKPT = PROJECT_ROOT / "checkpoints/pdms90_9.safetensors"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models"
DEFAULT_SCENE_FILTER = (
    PROJECT_ROOT / "examples/wanvideo/driveva_infer/navsim_scene_filters/navtest.yaml"
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--max-eval-tokens", type=int, default=1)
    p.add_argument("--bottleneck", type=int, default=18)
    p.add_argument("--keep-hint", type=float, default=0.5,
                   help="used only to set the safety clamp's ceiling, so the gate "
                        "cannot keep the whole sequence; NOT a fixed budget")
    p.add_argument("--history-threshold", type=float, default=0.5)
    p.add_argument("--future-threshold", type=float, default=0.5)
    p.add_argument("--recovery-layers", type=int, default=2)
    p.add_argument("--output-root", type=Path, default=FRAMEWORK_ROOT / "outputs/route_a_real_smoke")
    p.add_argument("--full-ckpt", type=Path, default=DEFAULT_FULL_CKPT)
    p.add_argument("--local-model-path", type=Path, default=DEFAULT_MODEL_PATH)
    p.add_argument("--scene-filter-yaml", type=Path, default=DEFAULT_SCENE_FILTER)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    from scripts import run_official_navsim_press as official_runner

    eval_mod = official_runner._load_official_eval_module()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[route-a-smoke] device={device}", flush=True)

    build_args = SimpleNamespace(
        local_model_path=args.local_model_path,
        full_ckpt=args.full_ckpt,
        pre_dit_register_lora_checkpoint=None,
        pre_dit_register_lora_target_modules="",
        pre_dit_register_lora_rank=32,
    )
    started = time.time()
    pipe = official_runner._build_official_pipeline(eval_mod, build_args, device)
    print(f"[route-a-smoke] pipeline built in {time.time() - started:.1f}s", flush=True)

    dit = pipe.dit
    dim = int(getattr(dit, "dim"))
    num_blocks = len(dit.blocks)
    param_dtype = next(dit.parameters()).dtype
    print(f"[route-a-smoke] dit dim={dim} blocks={num_blocks} dtype={param_dtype}", flush=True)

    layout = RouteALayoutSpec()  # 780 history + 780 future + 9 traj = 1569
    clamp = SafetyClampConfig(
        min_kept_history=8,
        min_kept_future=32,
        max_kept_total=max(64, int(round(layout.video_tokens * float(args.keep_hint)))),
    )
    config = RouteAConfig(
        token_dim=dim,
        bottleneck_layer=int(args.bottleneck),
        num_blocks=num_blocks,
        layout=layout,
        history_threshold=float(args.history_threshold),
        future_threshold=float(args.future_threshold),
        recovery_layers=int(args.recovery_layers),
        recovery_heads=16 if dim % 16 == 0 else 8,
        safety_clamp=clamp,
    )
    module = config.build().to(device=device, dtype=param_dtype)
    total = sum(p.numel() for p in module.parameters())
    print(f"[route-a-smoke] Route A params={total/1e6:.1f}M bottleneck=L{args.bottleneck} "
          f"clamp={clamp}", flush=True)

    # Attach through the guarded hook added to model_fn_wan_video.
    dit._tokenpress_route_a = module
    dit._tokenpress_route_a_physical = True
    dit._tokenpress_route_a_capture_layers = ()

    out_dir = args.output_root / "official_eval"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    argv = [
        "--repo_root", str(PROJECT_ROOT),
        "--navsim_log_path", str(NVME / "openscene-v1.1/meta_datas/test"),
        "--sensor_blobs_path", str(NVME / "openscene-v1.1/sensor_blobs/test"),
        "--metric_cache_path", str(NVME / "metric_cache_full"),
        "--output_dir", str(out_dir),
        "--full_ckpt", str(args.full_ckpt),
        "--local_model_path", str(args.local_model_path),
        "--scene_filter_yaml", str(args.scene_filter_yaml),
        "--max_eval_tokens", str(args.max_eval_tokens),
        "--num_inference_steps", "3",
        "--height", "480",
        "--width", "832",
        "--num_history_frames", "5",
        "--num_future_frames", "10",
        "--model_future_frames", "8",
        "--infer_trajectory_only",
        "--no_print_alignment_params",
        "--no_show_eval_progress",
    ]
    eval_args = eval_mod.parse_args(argv)

    started = time.time()
    try:
        eval_mod.run_eval(eval_args, external_pipe=pipe)
    finally:
        dit._tokenpress_route_a = None

    print(f"[route-a-smoke] run_eval finished in {time.time() - started:.1f}s", flush=True)
    csvs = sorted(out_dir.glob("*.csv"))
    ok = False
    for csv in csvs:
        text = csv.read_text().splitlines()
        print(f"[route-a-smoke] {csv.name}:")
        for line in text:
            print(f"    {line}")
        if any(line.split(",")[1:2] == ["True"] for line in text[1:]):
            ok = True
    print(f"[route-a-smoke] VALID_RECORD={ok}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
