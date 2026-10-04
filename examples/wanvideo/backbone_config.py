"""Backbone registry for the official DriveVA training/evaluation entry points."""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import glob

WAN22_TI2V_5B = "wan2.2-ti2v-5b"
WAN21_T2V_1P3B = "wan2.1-t2v-1.3b"


@dataclass(frozen=True)
class DriveVABackboneConfig:
    backbone_type: str
    model_id: str
    default_model_dir: Path
    vae_filename: str
    hidden_dim: int
    num_heads: int
    latent_channels: int
    patch_size: Tuple[int, int, int]


_REPO_ROOT = Path(__file__).resolve().parents[2]
BACKBONES = {
    WAN22_TI2V_5B: DriveVABackboneConfig(WAN22_TI2V_5B, "Wan-AI/Wan2.2-TI2V-5B", _REPO_ROOT / "models" / "Wan-AI" / "Wan2.2-TI2V-5B", "Wan2.2_VAE.pth", 3072, 24, 48, (1, 2, 2)),
    WAN21_T2V_1P3B: DriveVABackboneConfig(WAN21_T2V_1P3B, "Wan-AI/Wan2.1-T2V-1.3B", _REPO_ROOT / "models" / "Wan2.1-T2V-1.3B", "Wan2.1_VAE.pth", 1536, 12, 16, (1, 2, 2)),
}


def get_backbone_config(backbone_type: str) -> DriveVABackboneConfig:
    try:
        return BACKBONES[backbone_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported backbone_type={backbone_type!r}; choose from {sorted(BACKBONES)}") from exc


def _model_dir(config: DriveVABackboneConfig, local_model_path: str | None) -> Path:
    if local_model_path is None:
        return config.default_model_dir
    supplied = Path(local_model_path).expanduser()
    candidates = [supplied if supplied.name == config.model_id.rsplit("/", 1)[-1] else supplied / config.model_id,
                  supplied / config.model_id.rsplit("/", 1)[-1]]
    return next((p for p in candidates if p.exists()), candidates[0])


def build_model_configs(config: DriveVABackboneConfig, local_model_path: str | None, offload_device):
    from diffsynth.pipelines.wan_video_new import ModelConfig
    model_dir = _model_dir(config, local_model_path)
    text = model_dir / "models_t5_umt5-xxl-enc-bf16.pth"
    vae = model_dir / config.vae_filename
    dit = sorted(Path(p) for p in glob.glob(str(model_dir / "diffusion_pytorch_model*.safetensors")))
    tokenizer = model_dir / "google" / "umt5-xxl"
    missing = [str(p) for p in (text, vae, tokenizer) if not p.exists()]
    if not dit:
        missing.append(str(model_dir / "diffusion_pytorch_model*.safetensors"))
    if missing:
        raise FileNotFoundError("Backbone checkpoint is incomplete:\n  " + "\n  ".join(missing))
    return ([ModelConfig(path=str(text), offload_device=offload_device),
             ModelConfig(path=[str(p) for p in dit] if len(dit) > 1 else str(dit[0]), offload_device=offload_device),
             ModelConfig(path=str(vae), offload_device=offload_device)],
            ModelConfig(path=str(tokenizer)), model_dir)


def validate_loaded_backbone(dit, config: DriveVABackboneConfig):
    actual = (dit.dim, dit.blocks[0].self_attn.num_heads, dit.in_dim, tuple(dit.patch_size))
    expected = (config.hidden_dim, config.num_heads, config.latent_channels, config.patch_size)
    if actual != expected:
        raise ValueError(f"Loaded backbone does not match {config.backbone_type}: expected={expected}, actual={actual}")
