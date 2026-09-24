"""Route A training stages A0-A4 (plan sections 14, 28-30, 43).

The plan splits Route A training into four stages plus a compression-layer
curriculum, because unfreezing everything at once destroys the planning function
long before the compression architecture has learned anything:

``A0``
    Dense teacher cache + student initialised from the same DriveVA checkpoint.
``A1``
    Selector warm-up: backbone frozen, only the selector and the dense recovery
    decoder train, with a low threshold and a tiny sparsity weight.
``A2``
    LoRA adaptation of the DiT so the backbone starts writing information into
    tokens it expects to keep.
``A3``
    Full DiT adaptation with the sparsity weight ramped up.
``A4``
    Move the bottleneck earlier (18 -> 15 -> 12) and re-fine-tune.

This module encodes the *policy* (what is trainable, at which LR, with which
threshold and sparsity weight) so a driver only has to apply it.  It is
deliberately free of any model construction so it can be unit tested.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

from .threshold_gate import SparsityCurriculum, jittered_thresholds


STAGE_A0 = "A0"
STAGE_A1 = "A1"
STAGE_A2 = "A2"
STAGE_A3 = "A3"
STAGE_A4 = "A4"
STAGES = (STAGE_A0, STAGE_A1, STAGE_A2, STAGE_A3, STAGE_A4)

#: Compression-layer curriculum from plan section 9.  Never jump straight from
#: L18 to L8; each step is re-fine-tuned before the next one.
DEFAULT_LAYER_CURRICULUM: Tuple[int, ...] = (18, 15, 12)

#: Initial lambda sweep from plan section 30.
DEFAULT_SPARSITY_SWEEP: Tuple[float, ...] = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)


@dataclass
class StageSpec:
    """Trainable-set + optimisation policy for one stage."""

    name: str
    train_selector: bool = True
    train_decoder: bool = True
    train_lora: bool = False
    train_dit: bool = False
    train_trajectory_modules: bool = False
    compression_lr: float = 1.0e-4
    dit_lr: Optional[float] = None
    trajectory_lr: float = 2.0e-5
    sparsity_weight: float = 0.0
    sparsity_warmup_steps: int = 0
    sparsity_ramp_steps: int = 0
    threshold_jitter: float = 0.0
    note: str = ""

    def __post_init__(self) -> None:
        if self.name not in STAGES:
            raise ValueError(f"unknown stage {self.name!r}; expected one of {STAGES}")
        if self.train_lora and self.train_dit:
            raise ValueError("a stage trains LoRA or the full DiT, not both")

    @property
    def trainable_new_modules(self) -> bool:
        return bool(self.train_selector or self.train_decoder)

    def sparsity_curriculum(self) -> SparsityCurriculum:
        return SparsityCurriculum(
            warmup_steps=int(self.sparsity_warmup_steps),
            ramp_end_step=int(self.sparsity_warmup_steps) + int(self.sparsity_ramp_steps),
            max_weight=float(self.sparsity_weight),
            weights=DEFAULT_SPARSITY_SWEEP,
        )


def default_stage_specs() -> Dict[str, StageSpec]:
    """The plan's A0-A3 policy with the recommended hyper-parameters."""
    return {
        STAGE_A0: StageSpec(
            name=STAGE_A0,
            train_selector=False,
            train_decoder=False,
            sparsity_weight=0.0,
            note="cache dense teacher outputs; no student update yet",
        ),
        STAGE_A1: StageSpec(
            name=STAGE_A1,
            train_selector=True,
            train_decoder=True,
            compression_lr=1.0e-4,
            sparsity_weight=0.0,
            threshold_jitter=0.02,
            note="selector + recovery warm-up on a frozen backbone",
        ),
        STAGE_A2: StageSpec(
            name=STAGE_A2,
            train_selector=True,
            train_decoder=True,
            train_lora=True,
            compression_lr=1.0e-4,
            dit_lr=5.0e-5,
            sparsity_weight=1.0e-4,
            sparsity_warmup_steps=0,
            sparsity_ramp_steps=2000,
            threshold_jitter=0.05,
            note="LoRA adaptation so the DiT writes into kept tokens",
        ),
        STAGE_A3: StageSpec(
            name=STAGE_A3,
            train_selector=True,
            train_decoder=True,
            train_dit=True,
            train_trajectory_modules=True,
            compression_lr=5.0e-5,
            dit_lr=5.0e-6,
            trajectory_lr=2.0e-5,
            sparsity_weight=1.0e-3,
            sparsity_warmup_steps=0,
            sparsity_ramp_steps=4000,
            threshold_jitter=0.05,
            note="full DiT adaptation with the sparsity weight ramped up",
        ),
        STAGE_A4: StageSpec(
            name=STAGE_A4,
            train_selector=True,
            train_decoder=True,
            train_dit=True,
            train_trajectory_modules=True,
            compression_lr=5.0e-5,
            dit_lr=5.0e-6,
            trajectory_lr=2.0e-5,
            sparsity_weight=1.0e-3,
            sparsity_warmup_steps=0,
            sparsity_ramp_steps=2000,
            threshold_jitter=0.05,
            note="re-fine-tune after moving the bottleneck earlier",
        ),
    }


@dataclass
class RouterStageSchedule:
    """Maps a global step to the active :class:`StageSpec` plus bottleneck layer."""

    specs: Dict[str, StageSpec] = field(default_factory=default_stage_specs)
    stage_steps: Dict[str, int] = field(
        default_factory=lambda: {
            STAGE_A0: 0,
            STAGE_A1: 2000,
            STAGE_A2: 6000,
            STAGE_A3: 12000,
        }
    )
    layer_curriculum: Tuple[int, ...] = DEFAULT_LAYER_CURRICULUM
    steps_per_layer: int = 6000

    def __post_init__(self) -> None:
        for name in self.stage_steps:
            if name not in self.specs:
                raise ValueError(f"stage_steps references unknown stage {name!r}")
        if int(self.steps_per_layer) <= 0:
            raise ValueError("steps_per_layer must be positive")
        if not self.layer_curriculum:
            raise ValueError("layer_curriculum must not be empty")

    @property
    def total_steps(self) -> int:
        a3_start = int(self.stage_steps.get(STAGE_A3, 0))
        return a3_start + int(self.steps_per_layer) * len(self.layer_curriculum)

    def stage_at(self, step: int) -> str:
        step = int(step)
        a3_start = int(self.stage_steps.get(STAGE_A3, 0))
        # A4 is the layer-curriculum phase: it begins one ``steps_per_layer``
        # budget after A3 starts, i.e. when the curriculum moves to layer 15.
        if STAGE_A4 in self.specs and step >= a3_start + int(self.steps_per_layer):
            return STAGE_A4
        ordered = sorted(
            ((int(start), name) for name, start in self.stage_steps.items()),
            key=lambda item: item[0],
        )
        current = ordered[0][1]
        for start, name in ordered:
            if step >= start:
                current = name
        return current

    def spec_at(self, step: int) -> StageSpec:
        return self.specs[self.stage_at(step)]

    def layer_at(self, step: int) -> int:
        step = int(step)
        a3_start = int(self.stage_steps.get(STAGE_A3, 0))
        if step < a3_start:
            return int(self.layer_curriculum[0])
        index = (step - a3_start) // int(self.steps_per_layer)
        index = min(index, len(self.layer_curriculum) - 1)
        return int(self.layer_curriculum[index])

    def describe(self, step: int) -> Dict[str, object]:
        spec = self.spec_at(step)
        return {
            "step": int(step),
            "stage": spec.name,
            "bottleneck_layer": self.layer_at(step),
            "sparsity_weight": spec.sparsity_curriculum().value(step - int(self.stage_steps.get(spec.name, 0)))
            if spec.sparsity_weight
            else 0.0,
            "train_selector": spec.train_selector,
            "train_decoder": spec.train_decoder,
            "train_lora": spec.train_lora,
            "train_dit": spec.train_dit,
        }


def apply_stage(
    spec: StageSpec,
    *,
    compression_module: nn.Module,
    dit: Optional[nn.Module] = None,
    lora_modules: Sequence[nn.Module] = (),
    trajectory_modules: Sequence[nn.Module] = (),
) -> Dict[str, int]:
    """Set ``requires_grad`` for every parameter group and report trainable sizes.

    The new modules live inside ``compression_module`` (scorer, gate, recovery),
    so freezing the backbone is a matter of toggling the DiT / LoRA /
    trajectory parameter groups.  The gate's thresholds stay as buffers, not
    parameters, so calibration cannot be overwritten by the optimiser.

    The returned counts are *trainable* parameter counts, i.e. they respect both
    the stage flag and the selector/decoder refinement below, so a stage that
    freezes the backbone reports ``dit == 0``.
    """
    groups: Dict[str, Sequence[Optional[nn.Module]]] = {
        "compression": (compression_module,),
        "dit": (dit,),
        "lora": tuple(lora_modules),
        "trajectory": tuple(trajectory_modules),
    }
    enabled = {
        "compression": spec.trainable_new_modules,
        "dit": spec.train_dit,
        "lora": spec.train_lora,
        "trajectory": spec.train_trajectory_modules,
    }
    for name, modules in groups.items():
        for module in modules:
            if module is None:
                continue
            for param in module.parameters():
                param.requires_grad_(bool(enabled[name]))

    # A stage may warm up only one of the two new modules (plan section 14-A1
    # trains selector + decoder together, but this keeps the knob available).
    if spec.train_selector and not spec.train_decoder:
        for param in compression_module.recovery.parameters():
            param.requires_grad_(False)
    if spec.train_decoder and not spec.train_selector:
        for param in compression_module.scorer.parameters():
            param.requires_grad_(False)

    counts: Dict[str, int] = {}
    for name, modules in groups.items():
        counts[name] = sum(
            int(param.numel())
            for module in modules
            if module is not None
            for param in module.parameters()
            if param.requires_grad
        )
    return counts


def build_optimizer(
    spec: StageSpec,
    *,
    compression_module: nn.Module,
    dit: Optional[nn.Module] = None,
    lora_modules: Sequence[nn.Module] = (),
    trajectory_modules: Sequence[nn.Module] = (),
    weight_decay: float = 0.01,
) -> torch.optim.Optimizer:
    """AdamW with the plan's separate learning rates (section 28)."""
    groups: List[Dict[str, object]] = []

    def add(params, lr: Optional[float]):
        if lr is None:
            return
        collected = [p for p in params if p.requires_grad]
        if collected:
            groups.append({"params": collected, "lr": float(lr), "weight_decay": float(weight_decay)})

    if spec.trainable_new_modules:
        add(compression_module.parameters(), spec.compression_lr)
    if spec.train_dit and dit is not None:
        add(dit.parameters(), spec.dit_lr)
    if spec.train_lora:
        for module in lora_modules:
            if module is not None:
                add(module.parameters(), spec.dit_lr)
    if spec.train_trajectory_modules:
        for module in trajectory_modules:
            if module is not None:
                add(module.parameters(), spec.trajectory_lr)
    if not groups:
        raise ValueError(f"stage {spec.name} has no trainable parameter group")
    return torch.optim.AdamW(groups)


def jitter_for_step(
    spec: StageSpec,
    thresholds: Sequence[float],
    step: int,
    *,
    generator: Optional[torch.Generator] = None,
) -> List[float]:
    """Apply the stage's threshold jitter, disabled during warm-up."""
    if spec.threshold_jitter <= 0 or int(step) < int(spec.sparsity_warmup_steps):
        return [float(t) for t in thresholds]
    return jittered_thresholds(thresholds, spec.threshold_jitter, generator=generator)
