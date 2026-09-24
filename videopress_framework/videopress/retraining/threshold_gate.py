"""Threshold gate for Route A (Dynamic Select).

Plan reference: sections 3-6, 29-30 of
``DriveVA Dynamic Video Token Compression - Retraining Implementation Plan v2``.

The gate turns per-token scores into a *dynamic* keep mask whose length is
``K(x) = sum_i 1[score_i >= tau]``.  There is deliberately no fixed token
budget anywhere: the only knobs are per-domain thresholds plus a very loose
safety clamp that exists to keep early training numerically stable.

Training uses a straight-through estimator because the hard threshold is not
differentiable (plan section 4)::

    scores = sigmoid(logits)
    soft   = sigmoid((scores - tau) / temperature)
    hard   = scores >= tau
    mask   = hard.detach() - soft.detach() + soft

so the forward pass is exactly the deployed hard rule while the backward pass
sees the smooth sigmoid.  The threshold therefore lives on the *probability*
scale, which is what plan section 29 calibrates (``threshold_*: 0.5``).

Deployment only needs ``hard`` (``keep = scores >= tau``); the soft branch is
never required at inference time.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn


DEFAULT_HISTORY_THRESHOLD = 0.5
DEFAULT_FUTURE_THRESHOLD = 0.5


@dataclass
class SafetyClampConfig:
    """Loose training-time guard rails (plan section 6.1).

    These are explicitly *not* compression levels.  They only stop the trivial
    degenerate states ``K == 0`` and ``K == N`` while the scorer is still
    random.  ``None`` disables a bound.
    """

    min_kept_history: Optional[int] = 8
    min_kept_future: Optional[int] = 32
    max_kept_total: Optional[int] = 384

    def __post_init__(self) -> None:
        for name in ("min_kept_history", "min_kept_future", "max_kept_total"):
            value = getattr(self, name)
            if value is not None and int(value) < 0:
                raise ValueError(f"{name} must be non-negative or None")


@dataclass
class ThresholdGateOutput:
    """Result of one threshold-gate call."""

    logits: torch.Tensor
    scores: torch.Tensor
    hard_mask: torch.Tensor
    soft_mask: torch.Tensor
    mask: torch.Tensor
    #: Kept count per domain, taken from the binding row (see
    #: :func:`binding_row_domain_counts`).  ``sum(kept_counts)`` is exactly the
    #: rectangular video length a batched forward runs.
    kept_counts: List[int]
    candidate_counts: List[int]
    clamped_min: List[int]
    clamped_max: List[int]
    thresholds: List[float]

    @property
    def kept_total(self) -> int:
        return int(sum(self.kept_counts))

    def as_dict(self) -> Dict[str, object]:
        return {
            "kept_counts": list(self.kept_counts),
            "candidate_counts": list(self.candidate_counts),
            "kept_total": self.kept_total,
            "candidate_total": int(sum(self.candidate_counts)),
            "retention": self.kept_total / max(1, int(sum(self.candidate_counts))),
            "clamped_min": list(self.clamped_min),
            "clamped_max": list(self.clamped_max),
            "thresholds": list(self.thresholds),
        }


def _apply_min_clamp(
    hard: torch.Tensor,
    scores: torch.Tensor,
    offset: int,
    size: int,
    minimum: Optional[int],
) -> int:
    """Force at least ``minimum`` tokens of this domain to be kept."""
    if minimum is None or int(minimum) <= 0:
        return 0
    row_hard = hard[:, offset : offset + size]
    row_scores = scores[:, offset : offset + size]
    needed = int(minimum) - int(row_hard.sum(dim=-1).min().item())
    if needed <= 0:
        return 0
    topk = row_scores.detach().topk(min(int(minimum), size), dim=-1).indices
    row_hard.scatter_(1, topk, True)
    return needed


def _apply_max_clamp(
    hard: torch.Tensor,
    scores: torch.Tensor,
    sizes: Sequence[int],
    maximum: Optional[int],
) -> int:
    """Force at most ``maximum`` tokens in total to be kept."""
    if maximum is None:
        return 0
    dropped_total = 0
    for row in range(hard.shape[0]):
        counts = [int(hard[row, o : o + s].sum().item()) for o, s in _offsets(sizes)]
        excess = sum(counts) - int(maximum)
        if excess <= 0:
            continue
        flat_scores = scores[row].detach().clone()
        flat_scores[~hard[row]] = float("inf")
        drop = flat_scores.topk(min(excess, int(hard[row].sum().item())), largest=False).indices
        hard[row, drop] = False
        dropped_total += int(drop.numel())
    return dropped_total


def _offsets(sizes: Sequence[int]) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    offset = 0
    for size in sizes:
        out.append((offset, int(size)))
        offset += int(size)
    return out


def binding_row_domain_counts(
    hard_mask: torch.Tensor, sizes: Sequence[int]
) -> List[int]:
    """Per-domain kept counts of the *binding* row.

    A batched forward is rectangular, so its video length is the largest
    per-row total.  Reporting the per-domain *maxima* independently would make
    the domain counts sum to more than that length whenever the rows distribute
    their budget differently.  Reporting the row that attains the maximum keeps
    ``sum(result) == rectangular video length`` exactly, and for the single-row
    case used by the plan's training recipe it is simply that row's counts.
    """
    if hard_mask.ndim != 2:
        raise ValueError("hard_mask must be [B, N]")
    offsets = _offsets(sizes)
    per_row = torch.stack(
        [hard_mask[:, offset : offset + size].sum(dim=-1) for offset, size in offsets],
        dim=-1,
    )
    totals = per_row.sum(dim=-1)
    row = int(totals.argmax().item())
    return [int(value) for value in per_row[row].tolist()]


class STEThresholdGate(nn.Module):
    """Per-domain threshold gate with straight-through gradients.

    ``domain_sizes`` describes the candidate layout, e.g. ``(780, 780)`` for
    ``history`` followed by ``future``.  One threshold per domain is stored so
    history and future can be calibrated independently (plan section 6).
    """

    def __init__(
        self,
        domain_sizes: Sequence[int],
        thresholds: Sequence[float] = (DEFAULT_HISTORY_THRESHOLD, DEFAULT_FUTURE_THRESHOLD),
        temperature: float = 0.2,
        min_temperature: float = 0.05,
        clamp: Optional[SafetyClampConfig] = None,
    ):
        super().__init__()
        sizes = [int(s) for s in domain_sizes]
        if not sizes or any(s <= 0 for s in sizes):
            raise ValueError("domain_sizes must be positive")
        self.domain_sizes = sizes
        self.n_domain = len(sizes)
        values = [float(v) for v in thresholds]
        if len(values) == 1 and self.n_domain > 1:
            values = values * self.n_domain
        if len(values) != self.n_domain:
            raise ValueError(
                f"expected {self.n_domain} thresholds, got {len(values)}"
            )
        for value in values:
            if not 0.0 < value < 1.0:
                raise ValueError(f"threshold must be in (0, 1), got {value}")
        if not math.isfinite(float(temperature)) or float(temperature) <= 0:
            raise ValueError("temperature must be finite and positive")
        if not math.isfinite(float(min_temperature)) or float(min_temperature) <= 0:
            raise ValueError("min_temperature must be finite and positive")
        self.register_buffer(
            "threshold_values", torch.tensor(values, dtype=torch.float32)
        )
        self.temperature = float(temperature)
        self.min_temperature = float(min_temperature)
        self.clamp = clamp if clamp is not None else SafetyClampConfig()

    # ---------------------------------------------------------------- helpers
    @property
    def n_candidate(self) -> int:
        return int(sum(self.domain_sizes))

    def thresholds(self) -> List[float]:
        return [float(v) for v in self.threshold_values.tolist()]

    def set_thresholds(self, values: Sequence[float]) -> None:
        values = [float(v) for v in values]
        if len(values) != self.n_domain:
            raise ValueError(f"expected {self.n_domain} thresholds, got {len(values)}")
        with torch.no_grad():
            self.threshold_values.copy_(torch.tensor(values, dtype=torch.float32))

    def ramp_temperature(self, progress: float) -> float:
        """Anneal ``temperature -> min_temperature`` (plan section 29)."""
        p = min(max(float(progress), 0.0), 1.0)
        return float(self.temperature + (self.min_temperature - self.temperature) * p)

    def _expanded_thresholds(self, scores: torch.Tensor) -> torch.Tensor:
        parts = [
            torch.full((size,), float(self.threshold_values[i]), device=scores.device, dtype=scores.dtype)
            for i, size in enumerate(self.domain_sizes)
        ]
        return torch.cat(parts, dim=0).unsqueeze(0)

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        scores: torch.Tensor,
        *,
        temperature: Optional[float] = None,
        safety_clamp: bool = True,
    ) -> ThresholdGateOutput:
        if scores.ndim != 2:
            raise ValueError(f"scores must be [B, N], got {tuple(scores.shape)}")
        if int(scores.shape[1]) != self.n_candidate:
            raise ValueError(
                f"scores width {int(scores.shape[1])} != candidate total {self.n_candidate}"
            )
        logits = scores.float()
        tau = self._expanded_thresholds(logits)
        temp = float(self.temperature if temperature is None else temperature)
        if temp <= 0:
            raise ValueError("temperature must be positive")

        # Plan section 4: the threshold is applied to the *probability* score,
        # not to the raw logit.
        probabilities = torch.sigmoid(logits)
        soft = torch.sigmoid((probabilities - tau) / temp)
        hard = probabilities >= tau

        clamped_min: List[int] = [0] * self.n_domain
        clamped_max: List[int] = [0]
        if safety_clamp and self.clamp is not None:
            # Domain 0 is the history block, every later block is a future
            # latent; the plan keeps separate minima because history and future
            # hidden statistics differ (plan section 6).
            for index, (offset, size) in enumerate(_offsets(self.domain_sizes)):
                minimum = (
                    self.clamp.min_kept_history
                    if index == 0
                    else self.clamp.min_kept_future
                )
                clamped_min[index] = _apply_min_clamp(
                    hard, probabilities, offset, size, minimum
                )
            clamped_max[0] = _apply_max_clamp(
                hard, probabilities, self.domain_sizes, self.clamp.max_kept_total
            )

        mask = hard.detach().to(logits.dtype) - soft.detach() + soft
        # ``kept_counts`` is reported for the binding row (see
        # ``binding_row_domain_counts``), so ``kept_total`` equals the
        # rectangular video length the batched forward actually runs.
        kept = binding_row_domain_counts(hard, self.domain_sizes)
        hard_out = hard.to(logits.dtype)
        return ThresholdGateOutput(
            logits=logits,
            scores=probabilities,
            hard_mask=hard_out,
            soft_mask=soft,
            mask=mask,
            kept_counts=kept,
            candidate_counts=list(self.domain_sizes),
            clamped_min=clamped_min,
            clamped_max=clamped_max,
            thresholds=self.thresholds(),
        )

    def extra_repr(self) -> str:
        return (
            f"domain_sizes={tuple(self.domain_sizes)}, "
            f"thresholds={self.thresholds()}, temperature={self.temperature}"
        )


def sparsity_loss(mask: torch.Tensor) -> torch.Tensor:
    """Lagrangian sparsity pressure ``mean(score)`` (plan sections 5, 32).

    The plan writes the penalty on the *scores*, not on the hard mask, so the
    term is differentiable everywhere and keeps pressing redundant tokens down
    even when they already sit below the threshold.

    .. warning::

       This term is a *uniform downward* gradient on every score.  On its own it
       cannot create a ranking, and if it is strong enough to push
       ``sigmoid(logit)`` out of the gate's responsive band it also kills the
       straight-through gradient (see :func:`gate_health` and
       :class:`SparsityGuard`).  Measured on the controlled-redundancy
       simulation, an unguarded ``lambda = 3e-3`` drove every keep probability
       towards 0 within 16 steps and the selector lost the token ranking it had
       already learned.  Plan section 14-A1 anticipates this ("keep tau low,
       lambda_sparse tiny" during warm-up); :class:`SparsityGuard` is the
       explicit runtime form of that instruction.
    """
    if mask.ndim != 2:
        raise ValueError("mask must be [B, N]")
    return mask.mean()


@dataclass
class GateHealth:
    """Diagnostics for whether the straight-through gate is still trainable."""

    score_mean: float
    score_std: float
    saturated_off: float
    saturated_on: float
    responsive: float
    ste_gain: float
    threshold_mean: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "score_mean": self.score_mean,
            "score_std": self.score_std,
            "saturated_off": self.saturated_off,
            "saturated_on": self.saturated_on,
            "responsive": self.responsive,
            "ste_gain": self.ste_gain,
            "threshold_mean": self.threshold_mean,
        }

    def is_degenerate(self, *, min_responsive: float = 0.05, min_ste_gain: float = 1e-5) -> bool:
        return (
            self.responsive < float(min_responsive)
            or self.ste_gain < float(min_ste_gain)
        )


def gate_health(
    logits: torch.Tensor,
    thresholds: Sequence[float],
    temperature: float,
    *,
    off_eps: float = 0.01,
    on_eps: float = 0.99,
) -> GateHealth:
    """Measure whether the gate can still pass gradients to the scorer.

    ``ste_gain`` is the mean ``|d mask / d logit|``.  It is the product of two
    sigmoid derivatives -- one from ``sigmoid(logit)`` and one from
    ``sigmoid((score - tau)/temperature)`` -- so it collapses whenever the keep
    probabilities saturate at 0 (sparsity pressure) or at 1 (an over-confident
    scorer).  A gate with ``ste_gain ~ 0`` can no longer learn a ranking from the
    task losses; only ``lambda_sparse`` keeps acting, which is precisely the
    degenerate state to detect.
    """
    if logits.ndim != 2:
        raise ValueError("logits must be [B, N]")
    if not math.isfinite(float(temperature)) or float(temperature) <= 0:
        raise ValueError("temperature must be finite and positive")
    values = [float(v) for v in thresholds]
    if not values:
        raise ValueError("at least one threshold is required")
    width = int(logits.shape[1])
    if width % len(values):
        raise ValueError("thresholds must evenly describe the candidate width")
    per_domain = width // len(values)
    logits = logits.float()
    tau = torch.cat(
        [
            torch.full((per_domain,), value, device=logits.device, dtype=logits.dtype)
            for value in values
        ]
    ).unsqueeze(0)
    scores = torch.sigmoid(logits)
    soft = torch.sigmoid((scores - tau) / float(temperature))
    gain = (soft * (1.0 - soft)) / float(temperature) * scores * (1.0 - scores)
    return GateHealth(
        score_mean=float(scores.mean()),
        score_std=float(scores.std(unbiased=False)),
        saturated_off=float((scores < float(off_eps)).float().mean()),
        saturated_on=float((scores > float(on_eps)).float().mean()),
        responsive=float(
            ((scores >= float(off_eps)) & (scores <= float(on_eps))).float().mean()
        ),
        ste_gain=float(gain.mean()),
        threshold_mean=float(sum(values) / len(values)),
    )


class SparsityGuard:
    """Stop raising ``lambda_sparse`` once the gate stops being trainable.

    The plan's sparsity curriculum (section 30) ramps a coefficient against a
    penalty that drives every keep probability towards zero.  Past a point that
    is self-defeating: the straight-through gradient vanishes, the scorer can no
    longer learn which tokens matter, and the run silently collapses to whatever
    the safety clamp happens to select.

    The guard is deliberately conservative and observable rather than clever: it
    tracks the most recent :class:`GateHealth`, refuses to increase ``lambda``
    while the gate is degenerate, and records how often it intervened.  Plan
    section 14-A1's "low tau, tiny lambda_sparse" warm-up is the same instruction
    expressed as a schedule; this is the feedback-controlled form.
    """

    def __init__(
        self,
        *,
        min_responsive: float = 0.05,
        min_ste_gain: float = 1e-5,
        patience: int = 0,
    ):
        if not 0.0 <= float(min_responsive) <= 1.0:
            raise ValueError("min_responsive must be in [0, 1]")
        if float(min_ste_gain) < 0:
            raise ValueError("min_ste_gain must be non-negative")
        if int(patience) < 0:
            raise ValueError("patience must be non-negative")
        self.min_responsive = float(min_responsive)
        self.min_ste_gain = float(min_ste_gain)
        self.patience = int(patience)
        self.last_lambda = 0.0
        self.interventions = 0
        self._degenerate_streak = 0
        self.last_health: Optional[GateHealth] = None

    def step(self, proposed_lambda: float, health: GateHealth) -> float:
        """Return the lambda actually to use this step."""
        if not math.isfinite(float(proposed_lambda)) or float(proposed_lambda) < 0:
            raise ValueError("proposed_lambda must be finite and non-negative")
        self.last_health = health
        degenerate = health.is_degenerate(
            min_responsive=self.min_responsive, min_ste_gain=self.min_ste_gain
        )
        self._degenerate_streak = self._degenerate_streak + 1 if degenerate else 0
        capped = degenerate and self._degenerate_streak > self.patience
        if capped:
            # Never increase, and never let the pressure grow while the gate is
            # dead.  Holding the current value (rather than forcing zero) keeps
            # the parameter connected to the loss for DDP.
            value = min(float(proposed_lambda), self.last_lambda)
            if float(proposed_lambda) > self.last_lambda:
                self.interventions += 1
        else:
            value = float(proposed_lambda)
        self.last_lambda = value
        return value

    def describe(self) -> Dict[str, object]:
        return {
            "min_responsive": self.min_responsive,
            "min_ste_gain": self.min_ste_gain,
            "patience": self.patience,
            "last_lambda": self.last_lambda,
            "interventions": self.interventions,
            "last_health": None if self.last_health is None else self.last_health.as_dict(),
        }


def select_kept_indices(
    hard_mask: torch.Tensor,
    *,
    video_start: int = 0,
    always_keep: Sequence[int] = (),
) -> List[torch.Tensor]:
    """Sorted global indices of kept tokens plus the always-kept positions."""
    if hard_mask.ndim != 2:
        raise ValueError("hard_mask must be [B, N]")
    kept: List[torch.Tensor] = []
    extra = torch.tensor(sorted(int(i) for i in always_keep), dtype=torch.long)
    for row in range(hard_mask.shape[0]):
        index = (hard_mask[row] > 0).nonzero(as_tuple=False).flatten() + int(video_start)
        if extra.numel():
            index = torch.cat([index, extra.to(index.device)])
        kept.append(torch.sort(index.unique())[0])
    return kept


def jittered_thresholds(
    thresholds: Sequence[float], amplitude: float, generator: Optional[torch.Generator] = None
) -> List[float]:
    """``tau_train = tau_base + U(-a, a)`` (plan section 29).

    The plan is explicit that this exists to stop the scorer over-fitting one
    threshold value, not to create discrete K buckets.
    """
    amplitude = float(amplitude)
    if not math.isfinite(amplitude) or amplitude < 0:
        raise ValueError("amplitude must be finite and non-negative")
    if amplitude == 0:
        return [float(t) for t in thresholds]
    out = []
    for value in thresholds:
        noise = (torch.rand(1, generator=generator).item() * 2.0 - 1.0) * amplitude
        out.append(min(max(float(value) + noise, 1e-3), 1.0 - 1e-3))
    return out


class SparsityCurriculum:
    """``lambda_sparse`` warmup + linear ramp (plan section 30)."""

    def __init__(
        self,
        warmup_steps: int = 0,
        ramp_end_step: int = 0,
        max_weight: float = 1e-3,
        weights: Optional[Sequence[float]] = None,
    ):
        if int(warmup_steps) < 0:
            raise ValueError("warmup_steps must be non-negative")
        if int(ramp_end_step) < int(warmup_steps):
            raise ValueError("ramp_end_step must be >= warmup_steps")
        if float(max_weight) < 0:
            raise ValueError("max_weight must be non-negative")
        self.warmup_steps = int(warmup_steps)
        self.ramp_end_step = int(ramp_end_step)
        self.max_weight = float(max_weight)
        self.weights = [float(v) for v in (weights or ())]

    def value(self, step: int) -> float:
        step = int(step)
        if step < self.warmup_steps:
            return 0.0
        if self.ramp_end_step <= self.warmup_steps:
            return self.max_weight
        span = self.ramp_end_step - self.warmup_steps
        progress = min(max((step - self.warmup_steps) / span, 0.0), 1.0)
        return self.max_weight * progress

    def sweep_value(self, step: int, weight: float) -> float:
        """Curriculum shape re-scaled to one entry of the lambda sweep."""
        base = self.value(step)
        if self.max_weight == 0:
            return 0.0
        return base / self.max_weight * float(weight)

    def describe(self) -> Dict[str, object]:
        return {
            "warmup_steps": self.warmup_steps,
            "ramp_end_step": self.ramp_end_step,
            "max_weight": self.max_weight,
            "sweep": list(self.weights),
        }
