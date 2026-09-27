"""Short real-data Wan2.1 DriveVA training and checkpoint reload sanity check."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from examples.wanvideo.backbone_config import WAN21_T2V_1P3B
from examples.wanvideo.driveva_infer.trajectory_modules import TrajectoryEncoder, TrajectoryHead
from examples.wanvideo.driveva_train.navsim_dataset import NavsimDriveVAConfig, NavsimDriveVADataset
from examples.wanvideo.driveva_train.train_navsim_v1 import DEFAULT_NEGATIVE_PROMPT, DriveVANavsimTrainingModule


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", required=True)
    parser.add_argument("--navsim_log_path", required=True)
    parser.add_argument("--sensor_blobs_path", required=True)
    parser.add_argument("--local_model_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    args = parser.parse_args()

    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = NavsimDriveVADataset(NavsimDriveVAConfig(
        repo_root=args.repo_root, navsim_log_path=args.navsim_log_path,
        sensor_blobs_path=args.sensor_blobs_path, num_history_frames=5,
        num_future_frames=8, image_height=480, image_width=832,
        skip_missing_files=False, quiet_scene_loader=True,
    ), split="train")
    model = DriveVANavsimTrainingModule(
        local_model_path=args.local_model_path,
        trainable_models="trajectory_encoder,trajectory_head",
        lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32,
        lora_checkpoint=None, use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs="longcat_video,trajectory,ego_vel", max_timestep_boundary=1.0,
        min_timestep_boundary=0.0, target_fps=2, negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        use_trajectory=True, train_future_video_noise_only=True,
        infer_replace_history_latents_before_decode=False,
        trajectory_condition_mode="velocity", num_history_frames=5,
        backbone_type=WAN21_T2V_1P3B,
    ).to("cuda")
    model.pipe.device = torch.device("cuda")
    model.train()
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=args.learning_rate, weight_decay=0.01)
    torch.cuda.reset_peak_memory_stats()
    records = []
    for step in range(1, args.steps + 1):
        sample = dataset[(step - 1) % len(dataset)]
        inputs = model.forward_preprocess(sample)
        output = model(sample, inputs=inputs)
        loss = output["loss"]
        action_loss = output["trajectory_loss"]
        if not torch.isfinite(loss) or not torch.isfinite(action_loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        record = {
            "step": step,
            "total_loss": float(loss.detach().cpu()),
            "action_loss": float(action_loss.detach().cpu()),
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
        }
        records.append(record)
        print(json.dumps(record), flush=True)

    checkpoint = output_dir / f"step-{args.steps}.safetensors"
    state = {
        name.removeprefix("pipe."): param.detach().cpu().contiguous()
        for name, param in model.named_parameters() if param.requires_grad
    }
    save_file(state, checkpoint)
    loaded = load_file(checkpoint, device="cpu")
    dim = int(model.pipe.dit.dim)
    reload_encoder = TrajectoryEncoder(point_dim=3, output_dim=dim)
    reload_head = TrajectoryHead(dim, out_dim=3)
    enc_state = {k.removeprefix("trajectory_encoder."): v for k, v in loaded.items() if k.startswith("trajectory_encoder.")}
    head_state = {k.removeprefix("trajectory_head."): v for k, v in loaded.items() if k.startswith("trajectory_head.")}
    reload_encoder.load_state_dict(enc_state, strict=True)
    reload_head.load_state_dict(head_state, strict=True)
    reload_ok = all(torch.equal(state[k], loaded[k]) for k in state)
    summary = {
        "steps": args.steps,
        "initial_total_loss": records[0]["total_loss"],
        "final_total_loss": records[-1]["total_loss"],
        "initial_action_loss": records[0]["action_loss"],
        "final_action_loss": records[-1]["action_loss"],
        "nan_or_inf": False,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "memory_first_allocated_bytes": records[0]["allocated_bytes"],
        "memory_last_allocated_bytes": records[-1]["allocated_bytes"],
        "checkpoint": str(checkpoint),
        "checkpoint_reload_ok": reload_ok,
    }
    (output_dir / "loss_curve.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
