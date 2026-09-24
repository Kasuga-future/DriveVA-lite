#!/usr/bin/env python3
"""Route A training driver and controlled-simulation training report.

Two modes:

``--mode sim`` (default)
    A *controlled redundancy simulation* on the **production** Wan ``DiTBlock``,
    ``TrajectoryHead`` and ``Head`` implementations.  It answers a precise
    question about the Route A objective: on a task where a known small subset of
    video tokens is sufficient, does the threshold scorer + sparsity curriculum
    actually learn to keep that subset and shrink the dynamic length, while the
    distillation objective stays low?

    The redundancy is injected explicitly so the answer is measurable: the
    *teacher* (frozen dense forward) sees only a per-scene random subset of
    ``--signal-tokens`` video tokens; the *student* sees all of them and must
    learn which ones matter.  Selection overlap with the injected signal set is
    then a direct, ground-truth measure of selector quality -- something the
    NAVSIM benchmark cannot give, because there the "right" tokens are unknown
    (the 2026-09-20 token-level oracle search established exactly that).

    This is a mechanics + learnability result, **not** a PDM result.  It says
    nothing about whether 12-18% retention is reachable on NAVSIM.

``--mode navsim``
    The real retraining entry point.  It reuses the DriveVA NAVSIM data path and
    is intentionally guarded: it needs a free GPU and the NAVSIM metadata/blobs,
    and it refuses to run without them rather than falling back silently.

Run examples::

    python scripts/train_route_a_smoke.py --mode sim --steps 300 --device cpu
    python scripts/train_route_a_smoke.py --mode sim --steps 300 --device cuda:0
    python scripts/train_route_a_smoke.py --mode navsim --train-manifest ... --device cuda:0
"""
from __future__ import annotations

import argparse
import copy
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch


FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = FRAMEWORK_ROOT.parent
sys.path.insert(0, str(FRAMEWORK_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from videopress.retraining import (  # noqa: E402
    CompressionStatsRecorder,
    RouteAConfig,
    RouteALayoutSpec,
    SafetyClampConfig,
    StageSpec,
    apply_stage,
    build_driveva_video_positions,
    build_optimizer,
    compute_route_a_loss,
    gate_health,
    jitter_for_step,
    SparsityGuard,
)


# --------------------------------------------------------------------------
# argument parsing
# --------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["sim", "navsim"], default="sim")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-scenes", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260924)

    # synthetic task
    parser.add_argument("--signal-tokens", type=int, default=120,
                        help="per-scene video tokens that actually carry the teacher's signal")
    parser.add_argument("--signal-scale", type=float, default=3.0)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--traj-scale", type=float, default=0.0,
                        help=">0 makes trajectory tokens carry their own noise, weakening "
                             "the dependence of the trajectory target on video selection")

    # model shape (defaults stay small so the smoke runs on CPU)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ffn-dim", type=int, default=384)
    parser.add_argument("--bottleneck", type=int, default=8)
    parser.add_argument("--capture-layers", type=str, default="4,8,11")
    parser.add_argument("--selector-hidden", type=int, default=256)
    parser.add_argument("--action-mode", choices=["pooled", "attention"], default="pooled")
    parser.add_argument("--recovery-layers", type=int, default=2)
    parser.add_argument("--history-threshold", type=float, default=0.5)
    parser.add_argument("--future-threshold", type=float, default=0.5)
    parser.add_argument("--max-kept-total", type=int, default=384)
    parser.add_argument("--min-kept-history", type=int, default=8)
    parser.add_argument("--min-kept-future", type=int, default=32)

    # training
    parser.add_argument("--stage", choices=["A1", "A2", "A3"], default="A3")
    parser.add_argument("--sparsity-weight", type=float, default=3.0e-3)
    parser.add_argument("--sparsity-guard", dest="sparsity_guard", action="store_true",
                        help="refuse to raise lambda_sparse while the STE gate is saturated")
    parser.add_argument("--no-sparsity-guard", dest="sparsity_guard", action="store_false")
    parser.add_argument("--guard-min-responsive", type=float, default=0.05)
    parser.add_argument("--guard-min-ste-gain", type=float, default=1e-5)
    parser.add_argument("--guard-patience", type=int, default=0)
    parser.add_argument("--lambda-sweep", type=str, default=None,
                        help="comma-separated lambda_sparse values; overrides --sparsity-weight "
                             "and runs one arm per value so a quality/compression frontier "
                             "can be read off directly")
    parser.add_argument("--sparsity-warmup", type=int, default=0)
    parser.add_argument("--sparsity-ramp", type=int, default=200)
    parser.add_argument("--compression-lr", type=float, default=1.0e-3)
    parser.add_argument("--dit-lr", type=float, default=None,
                        help="backbone LR; unset uses the stage default (A1 frozen, "
                             "A2 5e-5, A3 5e-6).  Pass 0 to force a frozen backbone.")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--min-temperature", type=float, default=0.05)
    parser.add_argument("--threshold-jitter", type=float, default=0.05)
    parser.add_argument("--dense-gate", action="store_true",
                        help="train with the whole sequence masked instead of physically "
                             "gathered, so dropped candidates also receive gradient")
    parser.add_argument("--physical-shortening-final-steps", type=int, default=0,
                        help="switch back to physical shortening for the last N steps")
    parser.add_argument("--use-gradient-checkpointing", action="store_true")
    parser.add_argument("--cpu-attention-fallback", action="store_true",
                        help="force the SDPA attention fallback (automatic on CPU)")

    # control arm
    parser.add_argument("--control-arm", action="store_true",
                        help="also train a lambda_sparse=0 arm as the quality reference")

    parser.set_defaults(sparsity_guard=True)
    parser.add_argument("--output", default=None)
    return parser.parse_args(argv)


# --------------------------------------------------------------------------
# synthetic controlled-redundancy task
# --------------------------------------------------------------------------
class ControlledRedundancyTask:
    """Teacher sees only the signal tokens; the student sees every token."""

    def __init__(
        self,
        layout: RouteALayoutSpec,
        dim: int,
        *,
        signal_tokens: int,
        signal_scale: float,
        noise_scale: float,
        traj_scale: float,
        device: torch.device,
        seed: int,
    ):
        if not 0 < int(signal_tokens) < layout.video_tokens:
            raise ValueError("signal_tokens must be inside the video token range")
        self.layout = layout
        self.dim = int(dim)
        self.signal_tokens = int(signal_tokens)
        self.signal_scale = float(signal_scale)
        self.noise_scale = float(noise_scale)
        self.traj_scale = float(traj_scale)
        self.device = device
        self.generator = torch.Generator(device="cpu").manual_seed(int(seed))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(x_student, x_teacher, signal_mask)``.

        Trajectory tokens are deliberately near-zero by default (``--traj-scale``)
        so the teacher's planning output is driven *by the video tokens* and the
        student's trajectory error is therefore a genuine function of which video
        tokens it kept.  With random trajectory tokens the trajectory head would
        read its own noisy inputs and be almost selection-insensitive.
        """
        spec = self.layout
        gen = self.generator
        video = torch.randn(batch_size, spec.video_tokens, self.dim, generator=gen)
        video = video * self.noise_scale
        signal_mask = torch.zeros(batch_size, spec.video_tokens, dtype=torch.bool)
        x_student = video.clone()
        x_teacher = video.clone()
        for row in range(batch_size):
            index = torch.randperm(spec.video_tokens, generator=gen)[: self.signal_tokens]
            signal_mask[row, index] = True
            x_student[row, index] *= self.signal_scale
            # The teacher never sees the non-signal video tokens, so the dense
            # teacher output is reproducible from the signal subset alone.
            x_teacher[row, ~signal_mask[row]] = 0.0
        traj = torch.randn(batch_size, spec.traj_tokens, self.dim, generator=gen)
        traj = traj * self.traj_scale
        return (
            torch.cat([x_student, traj], dim=1).to(self.device),
            torch.cat([x_teacher, traj], dim=1).to(self.device),
            signal_mask.to(self.device),
        )


# --------------------------------------------------------------------------
# model scaffolding
# --------------------------------------------------------------------------
def build_backbone(args: argparse.Namespace, *, seed: int, device: torch.device):
    from diffsynth.models.wan_video_dit import DiTBlock, Head, precompute_freqs_cis
    from examples.wanvideo.driveva_infer.trajectory_modules import TrajectoryHead

    torch.manual_seed(int(seed))
    if args.dim % args.heads:
        raise ValueError("--dim must be divisible by --heads")
    blocks = torch.nn.ModuleList(
        [
            DiTBlock(False, args.dim, args.heads, args.ffn_dim)
            for _ in range(args.layers)
        ]
    ).to(device)
    traj_head = TrajectoryHead(args.dim, args.dim).to(device)
    video_head = Head(args.dim, 4, (1, 2, 2), 1e-6).to(device)
    return blocks, traj_head, video_head, precompute_freqs_cis


def make_inputs(
    layout: RouteALayoutSpec,
    args: argparse.Namespace,
    device: torch.device,
    generator: torch.Generator,
):
    seq = layout.total_tokens
    context = torch.randn(1, 16, args.dim, generator=generator).to(device)
    t_mod = torch.randn(1, seq, 6, args.dim, generator=generator).to(device)
    freqs = _freqs(args, layout, device)
    # The Wan head is position-wise over the video grid only, so its time
    # embedding is video-length exactly as in the production pipeline.
    head_time = torch.randn(1, layout.video_tokens, args.dim, generator=generator).to(device)
    return context, t_mod, freqs, head_time


def _freqs(args: argparse.Namespace, layout: RouteALayoutSpec, device: torch.device):
    from diffsynth.models.wan_video_dit import precompute_freqs_cis

    return (
        precompute_freqs_cis(args.dim // args.heads, end=layout.total_tokens)
        .unsqueeze(1)
        .to(device)
    )


def dense_forward(blocks, x, context, t_mod, freqs, *, capture_layers=()):
    hidden: Dict[int, torch.Tensor] = {}
    for index, block in enumerate(blocks):
        x = block(x, context, t_mod, freqs)
        if index in capture_layers:
            hidden[int(index)] = x
    return x, hidden


# --------------------------------------------------------------------------
# one training arm
# --------------------------------------------------------------------------
def run_arm(
    args: argparse.Namespace,
    *,
    arm_name: str,
    sparsity_weight: float,
    task: ControlledRedundancyTask,
    device: torch.device,
    capture_layers: Sequence[int],
) -> Dict[str, object]:
    layout = task.layout
    blocks, traj_head, video_head, _ = build_backbone(args, seed=args.seed, device=device)
    teacher_blocks = copy.deepcopy(blocks)
    for param in teacher_blocks.parameters():
        param.requires_grad_(False)
    teacher_traj_head = copy.deepcopy(traj_head)
    teacher_video_head = copy.deepcopy(video_head)
    for module in (teacher_traj_head, teacher_video_head):
        for param in module.parameters():
            param.requires_grad_(False)

    config = RouteAConfig(
        token_dim=args.dim,
        bottleneck_layer=args.bottleneck,
        num_blocks=args.layers,
        layout=layout,
        history_threshold=args.history_threshold,
        future_threshold=args.future_threshold,
        temperature=args.temperature,
        min_temperature=args.min_temperature,
        selector_hidden=args.selector_hidden,
        action_mode=args.action_mode,
        selector_heads=args.heads,
        recovery_layers=args.recovery_layers,
        recovery_heads=args.heads,
        safety_clamp=SafetyClampConfig(
            min_kept_history=args.min_kept_history,
            min_kept_future=args.min_kept_future,
            max_kept_total=args.max_kept_total,
        ),
    )
    module = config.build().to(device)

    train_dit = float(resolved_dit_lr(args)) > 0.0
    spec = StageSpec(
        name="A3" if train_dit else "A1",
        train_selector=True,
        train_decoder=True,
        train_dit=train_dit,
        compression_lr=args.compression_lr,
        dit_lr=float(resolved_dit_lr(args)) if train_dit else None,
        sparsity_weight=float(sparsity_weight),
        sparsity_warmup_steps=int(args.sparsity_warmup),
        sparsity_ramp_steps=int(args.sparsity_ramp),
        threshold_jitter=float(args.threshold_jitter),
    )
    apply_stage(spec, compression_module=module, dit=blocks)
    optimizer = build_optimizer(
        spec,
        compression_module=module,
        dit=blocks if train_dit else None,
        weight_decay=args.weight_decay,
    )
    curriculum = spec.sparsity_curriculum()
    guard = (
        SparsityGuard(
            min_responsive=float(args.guard_min_responsive),
            min_ste_gain=float(args.guard_min_ste_gain),
            patience=int(args.guard_patience),
        )
        if args.sparsity_guard
        else None
    )

    input_generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    context, t_mod, freqs, head_time = make_inputs(layout, args, device, input_generator)
    context = context.expand(args.batch_size, -1, -1).contiguous()
    t_mod = t_mod.expand(args.batch_size, -1, -1, -1).contiguous()
    head_time = head_time.expand(args.batch_size, -1, -1).contiguous()

    positions = build_driveva_video_positions(layout, device=device, dtype=torch.float32)
    timestep = torch.tensor([716.0], device=device).expand(args.batch_size)

    def shorten_now(step: int) -> bool:
        if not args.dense_gate:
            return True
        tail = int(args.physical_shortening_final_steps)
        return bool(tail) and step > int(args.steps) - tail

    history: List[Dict[str, float]] = []
    module.train()
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    for step in range(1, int(args.steps) + 1):
        x_student, x_teacher, signal_mask = task.sample(args.batch_size)

        with torch.no_grad():
            teacher_out, teacher_hidden = dense_forward(
                teacher_blocks, x_teacher, context, t_mod, freqs,
                capture_layers=capture_layers,
            )
            teacher_traj = teacher_traj_head(teacher_out[:, layout.traj_slice])
            teacher_video = teacher_video_head(
                teacher_out[:, layout.video_slice], head_time[:, layout.video_slice]
            )
            teacher_action_hidden = {
                layer: hidden[:, layout.traj_slice] for layer, hidden in teacher_hidden.items()
            }

        lambda_sparse = curriculum.value(step)
        if float(sparsity_weight) > 0:
            thresholds = jitter_for_step(
                spec,
                [args.history_threshold, args.future_threshold],
                step,
                generator=input_generator,
            )
            module.gate.set_thresholds(thresholds)

        out = module(
            blocks,
            x_student,
            context,
            t_mod,
            freqs,
            timestep=timestep,
            positions=positions,
            trajectory_head=traj_head,
            head=video_head,
            head_t_mod=head_time,
            capture_layers=capture_layers,
            use_checkpoint=bool(args.use_gradient_checkpointing),
            physical_shortening=shorten_now(step),
        )

        # Synthetic ground-truth flow targets.  They exercise the FM code path;
        # the learning signal in this simulation is the distillation pair.
        traj_fm_target = torch.randn_like(out.traj_pred)
        video_fm_target = torch.randn_like(out.video_flow)
        loss = compute_route_a_loss(
            student_traj_flow=out.traj_pred,
            teacher_traj_flow=teacher_traj,
            traj_fm_target=traj_fm_target,
            student_video_flow=out.video_flow,
            teacher_video_flow=teacher_video,
            video_fm_target=video_fm_target,
            student_action_hidden=out.captured_action_hidden,
            teacher_action_hidden=teacher_action_hidden,
            # Plan section 5 penalises mean(sigma(logit)); the gate exposes that
            # probability directly as ``scores``.
            sparse_scores=out.gate.scores,
            lambda_sparse=lambda_sparse,
            reference=out.logits,
        )
        if guard is not None:
            guard.step(
                lambda_sparse,
                gate_health(
                    out.logits.detach(),
                    module.gate.thresholds(),
                    float(args.temperature),
                ),
            )

        (loss.total / max(1, int(args.gradient_accumulation))).backward()
        if step % max(1, int(args.gradient_accumulation)) == 0:
            if float(args.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in module.parameters() if p.requires_grad], float(args.grad_clip)
                )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if step == 1 or step % max(1, int(args.steps) // 10) == 0:
            overlap = _selection_overlap(out.kept_video_indices, signal_mask)
            row = {
                    "step": step,
                    "lambda_sparse": float(lambda_sparse),
                    "loss": float(loss.total.detach()),
                    **{k: float(v.detach()) for k, v in loss.terms.items()},
                    "kept_video": int(out.kept_video_indices.shape[1]),
                    "kept_total": int(out.kept_total),
                    "retention": float(out.kept_video_indices.shape[1]) / layout.video_tokens,
                    "signal_overlap": overlap,
            }
            if guard is not None and guard.last_health is not None:
                row["gate_ste_gain"] = guard.last_health.ste_gain
                row["gate_responsive"] = guard.last_health.responsive
                row["gate_score_mean"] = guard.last_health.score_mean
                row["gate_saturated_off"] = guard.last_health.saturated_off
                row["guard_interventions"] = guard.interventions
            history.append(row)

    elapsed = time.perf_counter() - started
    evaluation = evaluate_arm(
        args, module=module, blocks=blocks, traj_head=traj_head, video_head=video_head,
        teacher_blocks=teacher_blocks, teacher_traj_head=teacher_traj_head,
        teacher_video_head=teacher_video_head, task=task, device=device,
        capture_layers=capture_layers, context=context, t_mod=t_mod, freqs=freqs,
        head_time=head_time, positions=positions, timestep=timestep,
    )
    return {
        "arm": arm_name,
        "sparsity_weight": float(sparsity_weight),
        "train_dit": train_dit,
        "dit_lr": float(resolved_dit_lr(args)),
        "stage": spec.name,
        "steps": int(args.steps),
        "elapsed_seconds": elapsed,
        "seconds_per_step": elapsed / max(1, int(args.steps)),
        "history": history,
        "evaluation": evaluation,
        "dense_gate": bool(args.dense_gate),
        "physical_shortening_final_steps": int(args.physical_shortening_final_steps),
        "sparsity_guard": None if guard is None else guard.describe(),
        "trainable_parameters": {
            "compression": sum(
                p.numel() for p in module.parameters() if p.requires_grad
            ),
            "dit": sum(p.numel() for p in blocks.parameters() if p.requires_grad),
        },
    }


#: Stage-default backbone learning rates (plan section 28).
STAGE_DIT_LR = {"A1": 0.0, "A2": 5.0e-5, "A3": 5.0e-6}


def resolved_dit_lr(args: argparse.Namespace) -> float:
    """Explicit --dit-lr wins; otherwise use the stage default."""
    if args.dit_lr is not None:
        return float(args.dit_lr)
    return float(STAGE_DIT_LR[args.stage])


def _selection_overlap(kept_video_index: torch.Tensor, signal_mask: torch.Tensor) -> float:
    """Mean fraction of kept tokens that are members of the injected signal set."""
    if kept_video_index.numel() == 0:
        return 0.0
    ratios = []
    for row in range(kept_video_index.shape[0]):
        kept = kept_video_index[row]
        ratios.append(float(signal_mask[row, kept].float().mean()))
    return sum(ratios) / len(ratios)


@torch.no_grad()
def evaluate_arm(
    args: argparse.Namespace,
    *,
    module,
    blocks,
    traj_head,
    video_head,
    teacher_blocks,
    teacher_traj_head,
    teacher_video_head,
    task: ControlledRedundancyTask,
    device: torch.device,
    capture_layers: Sequence[int],
    context,
    t_mod,
    freqs,
    head_time,
    positions,
    timestep,
) -> Dict[str, object]:
    module.eval()
    layout = task.layout
    recorder = CompressionStatsRecorder()
    traj_errors: List[float] = []
    video_errors: List[float] = []
    overlaps: List[float] = []
    pixel_spread: List[float] = []
    for scene in range(int(args.eval_scenes)):
        x_student, x_teacher, signal_mask = task.sample(1)
        with torch.no_grad():
            teacher_out, teacher_hidden = dense_forward(
                teacher_blocks, x_teacher, context[:1], t_mod[:1], freqs,
                capture_layers=capture_layers,
            )
            teacher_traj = teacher_traj_head(teacher_out[:, layout.traj_slice])
            teacher_video = teacher_video_head(
                teacher_out[:, layout.video_slice], head_time[:1, layout.video_slice]
            )
            out = module(
                blocks, x_student, context[:1], t_mod[:1], freqs,
                timestep=timestep[:1], positions=positions,
                trajectory_head=traj_head, head=video_head,
                head_t_mod=head_time[:1], capture_layers=capture_layers,
            )
        traj_errors.append(float((out.traj_pred - teacher_traj).pow(2).mean()))
        video_errors.append(float((out.video_flow - teacher_video).pow(2).mean()))
        overlaps.append(_selection_overlap(out.kept_video_indices, signal_mask))
        kept = out.kept_video_indices
        if kept.shape[1] > 1:
            pos = positions[kept[0]]
            pixel_spread.append(float(pos[:, 1:].std(dim=0).mean()))
        recorder.record(
            scores=out.gate.scores,
            kept_counts=[
                int((kept < layout.history_tokens).sum()),
                int((kept >= layout.history_tokens).sum()),
            ],
            candidate_counts=list(layout.domain_sizes),
            scene_id=f"sim-{scene}",
            round_index=0,
            sigma=float(timestep[0]) / 1000.0,
            thresholds=out.gate.thresholds,
            difficulty=traj_errors[-1],
        )
    report = recorder.report()
    return {
        "scenes": int(args.eval_scenes),
        "traj_mse_vs_teacher": _stats(traj_errors),
        "video_mse_vs_teacher": _stats(video_errors),
        "kept_signal_overlap": _stats(overlaps),
        "kept_position_spread": _stats(pixel_spread),
        "compression": report,
    }


def _stats(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {}
    tensor = torch.tensor(list(values), dtype=torch.float32)
    return {
        "mean": float(tensor.mean()),
        "std": float(tensor.std(unbiased=False)),
        "min": float(tensor.min()),
        "max": float(tensor.max()),
    }


# --------------------------------------------------------------------------
# navsim entry point (guarded)
# --------------------------------------------------------------------------
def run_navsim(args: argparse.Namespace) -> int:
    raise SystemExit(
        "Route A NAVSIM retraining needs a free GPU plus the NAVSIM manifest, metadata\n"
        "and sensor blobs.  Point --train-manifest at\n"
        "  videopress_framework/outputs/navsim_split_audit/train_manifest.jsonl\n"
        "and reuse examples/wanvideo/driveva_train/train_navsim_v1.py as the data/model\n"
        "driver: it already builds the DriveVA pipe, the flow-matching targets and the\n"
        "teacher capture layers.  Replace its online-selector head with RouteADynamicSelect\n"
        "and its selector loss with compute_route_a_loss.  See\n"
        "ROUTE_A_IMPLEMENTATION_AND_TRAINING_REPORT.md for the exact patch points and the\n"
        "resource requirements.  This guard exists so the smoke never silently pretends to\n"
        "be a NAVSIM run."
    )


# --------------------------------------------------------------------------
def install_cpu_attention_fallback() -> str:
    """Replace the CUDA-only flash-attention kernel with SDPA for CPU runs.

    ``diffsynth.models.wan_video_dit.flash_attention`` prefers the installed
    ``flash_attn`` package, which has no CPU kernel, so the production blocks
    cannot run on CPU as-is.  The repository already ships an identical-math
    ``F.scaled_dot_product_attention`` branch (its ``compatibility_mode`` /
    fallback path); this installs exactly that branch.  It is a *device*
    workaround for verification, not a numerical approximation, and it is never
    used for GPU runs.
    """
    from einops import rearrange
    import torch.nn.functional as F
    import diffsynth.models.wan_video_dit as dit_module

    def sdpa_flash_attention(q, k, v, num_heads, compatibility_mode=False):
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        out = F.scaled_dot_product_attention(q, k, v)
        return rearrange(out, "b n s d -> b s (n d)", n=num_heads)

    dit_module.flash_attention = sdpa_flash_attention
    return "sdpa"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.mode == "navsim":
        return run_navsim(args)
    if args.layers <= args.bottleneck:
        raise ValueError("--layers must exceed --bottleneck")
    device = torch.device(args.device)
    attention_backend = "flash_attn"
    if device.type == "cpu" or args.cpu_attention_fallback:
        attention_backend = install_cpu_attention_fallback()

    layout = RouteALayoutSpec()
    if args.signal_tokens >= layout.video_tokens:
        raise ValueError("--signal-tokens must be below the video token count")
    capture_layers = sorted(
        {int(v) for v in str(args.capture_layers).split(",") if v.strip()}
    )
    invalid = [v for v in capture_layers if not 0 <= v < args.layers]
    if invalid:
        raise ValueError(f"--capture-layers outside [0, {args.layers - 1}]: {invalid}")

    task = ControlledRedundancyTask(
        layout,
        args.dim,
        signal_tokens=args.signal_tokens,
        signal_scale=args.signal_scale,
        noise_scale=args.noise_scale,
        traj_scale=args.traj_scale,
        device=device,
        seed=args.seed,
    )
    chance_overlap = float(args.signal_tokens) / float(layout.video_tokens)

    payload: Dict[str, object] = {
        "mode": "controlled_redundancy_simulation",
        "disclaimer": (
            "Mechanics and learnability only.  The redundancy is injected, so a drop in "
            "kept tokens here does NOT imply a working NAVSIM compression point."
        ),
        "argv": sys.argv[1:],
        "platform": platform.platform(),
        "device": str(device),
        "attention_backend": attention_backend,
        "torch": torch.__version__,
        "layout": asdict(layout),
        "config": {
            key: value
            for key, value in vars(args).items()
            if key not in {"mode"}
        },
        "capture_layers": capture_layers,
        "chance_signal_overlap": chance_overlap,
        "signal_tokens": int(args.signal_tokens),
        "arms": [],
    }

    if args.lambda_sweep:
        weights = [
            float(item)
            for item in str(args.lambda_sweep).replace(";", ",").split(",")
            if item.strip()
        ]
        if not weights:
            raise ValueError("--lambda-sweep must contain at least one value")
        arms = [(f"lambda{weight:g}", weight) for weight in weights]
    else:
        arms = [("sparsity", float(args.sparsity_weight))]
        if args.control_arm:
            arms.append(("control_lambda0", 0.0))

    for name, weight in arms:
        result = run_arm(
            args,
            arm_name=name,
            sparsity_weight=weight,
            task=task,
            device=device,
            capture_layers=capture_layers,
        )
        payload["arms"].append(result)

    _print_summary(payload)

    output = args.output
    if output is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output = str(
            FRAMEWORK_ROOT / "outputs" / f"route_a_sim_{stamp}" / "route_a_sim_report.json"
        )
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {target}")
    return 0


def _print_summary(payload: Dict[str, object]) -> None:
    print("=" * 78)
    print("Route A controlled-redundancy simulation")
    print("=" * 78)
    for arm in payload["arms"]:
        evaluation = arm["evaluation"]
        comp = evaluation["compression"]
        print(f"\narm={arm['arm']}  lambda_sparse={arm['sparsity_weight']}  "
              f"stage={arm['stage']}  train_dit={arm['train_dit']}")
        print(f"  seconds/step           : {arm['seconds_per_step']:.3f}")
        print(f"  traj MSE vs teacher    : {evaluation['traj_mse_vs_teacher']['mean']:.5f}")
        print(f"  video MSE vs teacher   : {evaluation['video_mse_vs_teacher']['mean']:.5f}")
        print(f"  kept-signal overlap    : {evaluation['kept_signal_overlap']['mean']:.4f} "
              f"(chance {payload['chance_signal_overlap']:.4f})")
        print(f"  kept video tokens mean : {comp['length']['mean']:.1f} "
              f"(P10 {comp['length']['p10']:.0f} / P90 {comp['length']['p90']:.0f})")
        print(f"  retention mean         : {comp['domains']['history']['retention_mean']:.3f} history, "
              f"{comp['domains']['future']['retention_mean']:.3f} future")
        print(f"  truly dynamic          : {comp['dynamic_check']['dynamic']} "
              f"(spread {comp['dynamic_check']['spread']:.1f}, "
              f"needs > {comp['dynamic_check']['required_spread']:.1f})")


if __name__ == "__main__":
    raise SystemExit(main())
