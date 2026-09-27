"""Minimal backbone registry shared by DriveVA training and evaluation."""

from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

WAN22_TI2V_5B = "wan2.2-ti2v-5b"
WAN21_T2V_1P3B = "wan2.1-t2v-1.3b"


@dataclass(frozen=True)
class DriveVABackboneConfig:
    backbone_type: str
    model_id: str
    default_model_dir: Path
    vae_filename: str
    model_type: str
    hidden_dim: int
    num_heads: int
    latent_channels: int
    patch_size: Tuple[int, int, int]
    uses_ti2v_image_condition: bool


_REPO_ROOT = Path(__file__).resolve().parents[2]
BACKBONES = {
    WAN22_TI2V_5B: DriveVABackboneConfig(
        backbone_type=WAN22_TI2V_5B,
        model_id="Wan-AI/Wan2.2-TI2V-5B",
        default_model_dir=_REPO_ROOT / "models" / "Wan-AI" / "Wan2.2-TI2V-5B",
        vae_filename="Wan2.2_VAE.pth",
        model_type="ti2v",
        hidden_dim=3072,
        num_heads=24,
        latent_channels=48,
        patch_size=(1, 2, 2),
        uses_ti2v_image_condition=True,
    ),
    WAN21_T2V_1P3B: DriveVABackboneConfig(
        backbone_type=WAN21_T2V_1P3B,
        model_id="Wan-AI/Wan2.1-T2V-1.3B",
        default_model_dir=_REPO_ROOT / "models" / "Wan2.1-T2V-1.3B",
        vae_filename="Wan2.1_VAE.pth",
        model_type="t2v",
        hidden_dim=1536,
        num_heads=12,
        latent_channels=16,
        patch_size=(1, 2, 2),
        uses_ti2v_image_condition=False,
    ),
}


def get_backbone_config(backbone_type: str) -> DriveVABackboneConfig:
    try:
        return BACKBONES[backbone_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported backbone_type={backbone_type!r}; choose from {sorted(BACKBONES)}") from exc


def resolve_model_dir(config: DriveVABackboneConfig, local_model_path: str | None) -> Path:
    if local_model_path is None:
        return config.default_model_dir
    supplied = Path(local_model_path).expanduser()
    direct = supplied if supplied.name == config.model_id.rsplit("/", 1)[-1] else supplied / config.model_id
    if direct.exists():
        return direct
    flat = supplied / config.model_id.rsplit("/", 1)[-1]
    return flat if flat.exists() else direct


def checkpoint_files(config: DriveVABackboneConfig, local_model_path: str | None) -> dict[str, object]:
    model_dir = resolve_model_dir(config, local_model_path)
    return {
        "model_dir": model_dir,
        "text_encoder": model_dir / "models_t5_umt5-xxl-enc-bf16.pth",
        "dit": sorted(Path(p) for p in glob.glob(str(model_dir / "diffusion_pytorch_model*.safetensors"))),
        "vae": model_dir / config.vae_filename,
        "tokenizer": model_dir / "google" / "umt5-xxl",
    }


def validate_checkpoint_files(config: DriveVABackboneConfig, local_model_path: str | None) -> dict[str, object]:
    files = checkpoint_files(config, local_model_path)
    missing = []
    for key in ("text_encoder", "vae", "tokenizer"):
        if not Path(files[key]).exists():
            missing.append(f"{key}: {files[key]}")
    if not files["dit"]:
        missing.append(f"dit: {files['model_dir']}/diffusion_pytorch_model*.safetensors")
    incomplete = sorted(Path(files["model_dir"]).glob("*.incomplete"))
    if incomplete:
        missing.extend(f"incomplete: {path}" for path in incomplete)
    if missing:
        raise FileNotFoundError("Backbone checkpoint is incomplete:\n  " + "\n  ".join(missing))
    return files


def build_model_configs(config: DriveVABackboneConfig, local_model_path: str | None, offload_device: str):
    from diffsynth.pipelines.wan_video_new import ModelConfig

    files = validate_checkpoint_files(config, local_model_path)
    dit_paths = [str(path) for path in files["dit"]]
    return [
        ModelConfig(path=str(files["text_encoder"]), offload_device=offload_device),
        ModelConfig(path=dit_paths[0] if len(dit_paths) == 1 else dit_paths, offload_device=offload_device),
        ModelConfig(path=str(files["vae"]), offload_device=offload_device),
    ], ModelConfig(path=str(files["tokenizer"])), files


def validate_loaded_backbone(dit, config: DriveVABackboneConfig) -> None:
    actual_heads = dit.blocks[0].self_attn.num_heads
    actual_patch_size = tuple(dit.patch_size)
    actual = (dit.dim, actual_heads, dit.in_dim, actual_patch_size)
    expected = (config.hidden_dim, config.num_heads, config.latent_channels, config.patch_size)
    if actual != expected:
        raise ValueError(f"Loaded backbone does not match {config.backbone_type}: expected={expected}, actual={actual}")
