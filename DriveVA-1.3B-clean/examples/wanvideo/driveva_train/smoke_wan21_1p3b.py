"""Run one real NAVSIM batch through the native Wan2.1-1.3B training path.

This intentionally has no synthetic fallback: a missing Wan2.1 checkpoint or
unreadable NAVSIM sample is an error, not a successful smoke test.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT))
from examples.wanvideo.backbone_config import WAN21_T2V_1P3B, get_backbone_config, validate_checkpoint_files


def _shape(value: Any) -> str:
    if torch.is_tensor(value):
        return str(tuple(value.shape))
    if isinstance(value, (list, tuple)):
        return f"list[{len(value)}]"
    return type(value).__name__


def _assert_finite(name: str, value: Any) -> None:
    if torch.is_tensor(value) and not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} contains NaN or Inf")


def _count_parameters(module: torch.nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in module.parameters() if not p.requires_grad)
    return trainable, frozen


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", required=True)
    parser.add_argument("--navsim_log_path", required=True)
    parser.add_argument("--sensor_blobs_path", required=True)
    parser.add_argument("--local_model_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_history_frames", type=int, default=5)
    parser.add_argument("--num_future_frames", type=int, default=8)
    parser.add_argument("--debug-shapes", action="store_true")
    args = parser.parse_args()

    validate_checkpoint_files(get_backbone_config(WAN21_T2V_1P3B), args.local_model_path)
    from examples.wanvideo.driveva_train.navsim_dataset import NavsimDriveVAConfig, NavsimDriveVADataset
    from examples.wanvideo.driveva_train.train_navsim_v1 import DriveVANavsimTrainingModule, DEFAULT_NEGATIVE_PROMPT

    if not torch.cuda.is_available():
        raise RuntimeError("Wan2.1-1.3B smoke test requires a CUDA GPU; no CUDA device is available.")
    torch.cuda.reset_peak_memory_stats()
    dataset = NavsimDriveVADataset(NavsimDriveVAConfig(
        repo_root=args.repo_root, navsim_log_path=args.navsim_log_path,
        sensor_blobs_path=args.sensor_blobs_path, num_history_frames=args.num_history_frames,
        num_future_frames=args.num_future_frames, image_height=args.height, image_width=args.width,
        skip_missing_files=False, quiet_scene_loader=True,
    ), split="train")
    sample = dataset[0]
    model = DriveVANavsimTrainingModule(
        local_model_path=args.local_model_path, trainable_models="trajectory_encoder,trajectory_head",
        lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32,
        lora_checkpoint=None, use_gradient_checkpointing=True, use_gradient_checkpointing_offload=False,
        extra_inputs="longcat_video,trajectory,ego_vel", max_timestep_boundary=1.0,
        min_timestep_boundary=0.0, target_fps=2, negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        use_trajectory=True, train_future_video_noise_only=True,
        infer_replace_history_latents_before_decode=False, trajectory_condition_mode="velocity",
        num_history_frames=args.num_history_frames, backbone_type=WAN21_T2V_1P3B,
    ).to("cuda")
    model.pipe.device = torch.device("cuda")
    model.pipe.debug_shapes = bool(args.debug_shapes)
    model.train()
    if args.debug_shapes:
        raw_shape = (len(sample["video"]), sample["video"][0].height, sample["video"][0].width, 3)
        history_shape = (len(sample["longcat_video"]), sample["longcat_video"][0].height, sample["longcat_video"][0].width, 3)
        current = sample["longcat_video"][-1]
        print(f"raw RGB shape: {raw_shape}")
        print(f"history frame shape: {history_shape}")
        print(f"current frame shape: {(current.height, current.width, 3)}")
    inputs = model.forward_preprocess(sample)
    for key in ("input_latents", "longcat_latents", "traj_tokens"):
        if key in inputs:
            _assert_finite(key, inputs[key])
            if args.debug_shapes:
                print(f"{key} shape: {_shape(inputs[key])}")
    output = model(sample, inputs=inputs)
    loss = output["loss"] if isinstance(output, dict) else output
    _assert_finite("loss", loss)
    if isinstance(output, dict):
        for name, value in output.items():
            _assert_finite(name, value)
    loss.backward()
    trainable, frozen = _count_parameters(model)
    result = {
        "loss": float(loss.detach().cpu()), "trajectory_gt_shape": _shape(sample.get("trajectory")),
        "debug_tensor_shapes": {name: list(shape) for name, shape in model.pipe.debug_tensor_shapes.items()},
        "backward_completed": True, "nan_or_inf": False,
        "trainable_parameters": trainable, "frozen_parameters": frozen,
        "cuda_allocated_bytes": torch.cuda.memory_allocated(),
        "cuda_reserved_bytes": torch.cuda.memory_reserved(),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    print(json.dumps(result, indent=2))
    os.makedirs(args.output_path, exist_ok=True)
    with open(Path(args.output_path) / "smoke_result.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
