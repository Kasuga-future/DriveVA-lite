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


def per_domain_zscore(
    logits: torch.Tensor,
    domain_sizes: Sequence[int],
    *,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Standardise the logits of each domain *within each sample*.

    This is what makes a probability threshold usable at all.  On real NAVSIM
    weights the scorer's scores turn out to be dominated by a per-scene offset:
    the pooled score std over 1560 candidates is about 0.023 while the shift
    between scenes is of the same order, so a single global ``tau`` leaves the
    gate with only two states -- keep everything or keep nothing.  A 40-step
    smoke recorded 0.026, 0.039, 0.46, 0.96, 0.29, 1.00, 0.92, 1.00 ... on
    consecutive steps with a *calibrated* threshold.

    Standardising per sample removes the scene-level offset and forces the gate
    to act on the only thing that carries a compression decision: the ranking of
    candidates inside that scene.  The cost is explicit and is recorded in the
    compression report: retention then depends only on the *shape* of the score
    distribution, so K becomes nearly constant and scene-dependent length is no
    longer produced by the gate itself.
    """
    sizes = [int(v) for v in domain_sizes]
    if logits.ndim != 2:
        raise ValueError(f"logits must be [B, N], got {tuple(logits.shape)}")
    if sum(sizes) != int(logits.shape[1]):
        raise ValueError(
            f"domain sizes {sizes} do not sum to logit width {int(logits.shape[1])}"
        )
    parts = []
    offset = 0
    for size in sizes:
        chunk = logits[:, offset : offset + size]
        mean = chunk.mean(dim=1, keepdim=True)
        std = chunk.std(dim=1, keepdim=True, unbiased=False)
        parts.append((chunk - mean) / (std + float(eps)))
        offset += size
    return torch.cat(parts, dim=1)


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
        normalize_scores: bool = False,
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
        self.normalize_scores = bool(normalize_scores)
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
        if self.normalize_scores:
            logits = per_domain_zscore(logits, self.domain_sizes)
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


class RetentionController:
    """Dual-ascent controller for the realised keep ratio.

    The plan's sparsity curriculum ramps a *fixed* ``lambda_sparse`` and hopes
    the resulting keep ratio lands near the compression budget.  The first
    NAVSIM A1 run showed why that is not enough: with no sparsity term at all
    the gate drifted from 99.7% retention at step 1250 to 50% at step 3768, so
    the run measured an uncontrolled, moving operating point rather than the
    policy under test.

    This controller treats the retention budget as a constraint and solves for
    ``lambda`` instead of guessing it::

        lambda <- clip(lambda + gain * (observed - target), 0, max_lambda)

    ``observed`` is the realised (hard) keep ratio, i.e. the quantity that
    actually runs at inference; a dead band of ``tolerance`` around the target
    prevents the controller from chasing estimator noise, and ``max_step``
    bounds one update so a single pathological batch cannot slam ``lambda`` to
    its ceiling.

    The class is deliberately a plain Python object with no tensors: it is an
    outer control loop over the optimiser, not part of the graph.
    """

    def __init__(
        self,
        *,
        target: float = 0.25,
        initial_lambda: float = 0.0,
        max_lambda: float = 5.0,
        gain: float = 0.02,
        tolerance: float = 0.02,
        max_step: float = 0.05,
        ema: float = 0.9,
    ):
        for name, value in (("target", target), ("tolerance", tolerance), ("ema", ema)):
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if not 0.0 < float(target) < 1.0:
            raise ValueError(f"target must be in (0, 1), got {target}")
        if not 0.0 <= float(tolerance) < 1.0:
            raise ValueError(f"tolerance must be in [0, 1), got {tolerance}")
        if float(initial_lambda) < 0 or float(max_lambda) < 0:
            raise ValueError("lambda bounds must be non-negative")
        if float(initial_lambda) > float(max_lambda):
            raise ValueError("initial_lambda must be <= max_lambda")
        if float(gain) < 0 or float(max_step) < 0:
            raise ValueError("gain and max_step must be non-negative")
        if not 0.0 <= float(ema) < 1.0:
            raise ValueError(f"ema must be in [0, 1), got {ema}")
        self.target = float(target)
        self.max_lambda = float(max_lambda)
        self.gain = float(gain)
        self.tolerance = float(tolerance)
        self.max_step = float(max_step)
        self.ema = float(ema)
        self.lambda_value = float(initial_lambda)
        self.updates = 0
        self.skips = 0
        self._ema_observed: Optional[float] = None
        self.last_error = 0.0

    def observe(self, observed_ratio: float) -> float:
        """Feed one realised keep ratio and return the lambda to apply next."""
        value = float(observed_ratio)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"observed_ratio must be finite and non-negative, got {value}")
        self._ema_observed = (
            value
            if self._ema_observed is None
            else self.ema * self._ema_observed + (1.0 - self.ema) * value
        )
        error = self._ema_observed - self.target
        self.last_error = error
        if abs(error) <= self.tolerance:
            self.skips += 1
            return self.lambda_value
        step = max(-self.max_step, min(self.max_step, self.gain * error))
        self.lambda_value = min(
            max(self.lambda_value + step, 0.0), self.max_lambda
        )
        self.updates += 1
        return self.lambda_value

    def describe(self) -> Dict[str, object]:
        return {
            "target": self.target,
            "lambda": self.lambda_value,
            "ema_observed": self._ema_observed,
            "last_error": self.last_error,
            "updates": self.updates,
            "skips": self.skips,
            "max_lambda": self.max_lambda,
            "gain": self.gain,
            "tolerance": self.tolerance,
            "max_step": self.max_step,
            "ema": self.ema,
        }


class QuantileThresholdCalibrator:
    """Calibrate ``tau`` from the pooled score quantiles (plan section 25).

    Feedback control on ``tau`` does *not* work here, and a 12-scene wiring
    smoke on real NAVSIM weights showed exactly why.  The realised keep ratio is
    a step function of ``tau``: with 1560 candidates whose scores have std 0.026,
    the whole transition from "keep nothing" to "keep everything" spans about
    0.1, so the loop gain is of order 10 per unit ``tau``.  A proportional
    controller therefore bang-bangs: history sat at the safety floor (0.041)
    while its ``tau`` climbed monotonically, and future alternated between 0.082
    and 1.000 on consecutive steps.

    The stable formulation is to stop controlling and start *estimating*: pool
    the scores seen since the last calibration and set each domain's threshold to
    the empirical ``1 - target`` quantile::

        tau_d = Quantile_{1-target}( scores_d )

    That is unbiased for the average keep ratio, is immune to the per-step
    discontinuity because it is computed from thousands of candidates at once,
    and leaves K free to vary per scene -- a fixed per-scene quantile would make
    K constant and defeat the dynamic-length goal.  Calibration is deliberately
    infrequent (hundreds of steps) so ``tau`` tracks the slow drift of the score
    scale rather than per-batch noise.
    """

    def __init__(
        self,
        *,
        target: float = 0.25,
        initial: Sequence[float] = (0.5, 0.5),
        interval: int = 100,
        warmup_samples: int = 4,
        smoothing: float = 0.5,
        lower: float = 0.001,
        upper: float = 0.999,
    ):
        values = [float(v) for v in initial]
        if not values:
            raise ValueError("at least one initial threshold is required")
        if not 0.0 < float(target) < 1.0:
            raise ValueError(f"target must be in (0, 1), got {target}")
        if int(interval) < 1:
            raise ValueError("interval must be >= 1")
        if int(warmup_samples) < 1:
            raise ValueError("warmup_samples must be >= 1")
        if not 0.0 <= float(smoothing) <= 1.0:
            raise ValueError("smoothing must be in [0, 1]")
        if not 0.0 < float(lower) < float(upper) < 1.0:
            raise ValueError("threshold bounds must satisfy 0 < lower < upper < 1")
        self.target = float(target)
        self.interval = int(interval)
        self.warmup_samples = int(warmup_samples)
        self.smoothing = float(smoothing)
        self.lower = float(lower)
        self.upper = float(upper)
        self.values = [min(max(v, self.lower), self.upper) for v in values]
        self.calibrations = 0
        self.last_quantiles: Optional[List[Optional[float]]] = None
        self._pool: Optional[List[List[torch.Tensor]]] = None
        self._last_calibration_step: Optional[int] = None

    # ------------------------------------------------------------------ API
    def observe(
        self,
        scores: torch.Tensor,
        domain_sizes: Sequence[int],
        step: int,
    ) -> Optional[List[float]]:
        """Pool one batch's scores; return new ``tau`` when a calibration is due.

        ``None`` means "keep the current thresholds".
        """
        if scores.ndim != 2:
            raise ValueError(f"scores must be [B, N], got {tuple(scores.shape)}")
        sizes = [int(v) for v in domain_sizes]
        if sum(sizes) != int(scores.shape[1]):
            raise ValueError(
                f"domain sizes {sizes} do not sum to candidate width {int(scores.shape[1])}"
            )
        flat = scores.detach().float()
        if self._pool is None:
            self._pool = [[] for _ in sizes]
        offset = 0
        for index, size in enumerate(sizes):
            self._pool[index].append(flat[:, offset : offset + size].reshape(-1).cpu())
            offset += size
        step = int(step)
        if self._last_calibration_step is None:
            due = len(self._pool[0]) >= self.warmup_samples
        else:
            due = (step - self._last_calibration_step) >= self.interval
        if not due:
            return None
        return self._calibrate(step)

    def _calibrate(self, step: int) -> List[float]:
        assert self._pool is not None
        quantile = 1.0 - self.target
        pooled: List[Optional[float]] = []
        for samples in self._pool:
            values = torch.cat(samples) if samples else torch.empty(0)
            if values.numel() < 8:
                pooled.append(None)
                continue
            pooled.append(float(torch.quantile(values, quantile)))
        self.last_quantiles = list(pooled)
        for index, value in enumerate(pooled):
            if value is None:
                continue
            blended = (
                value
                if self.calibrations == 0
                else (1.0 - self.smoothing) * self.values[index]
                + self.smoothing * value
            )
            self.values[index] = min(max(blended, self.lower), self.upper)
        self.calibrations += 1
        self._last_calibration_step = int(step)
        self._pool = [[] for _ in self.values]
        return list(self.values)

    def describe(self) -> Dict[str, object]:
        return {
            "target": self.target,
            "thresholds": list(self.values),
            "interval": self.interval,
            "warmup_samples": self.warmup_samples,
            "smoothing": self.smoothing,
            "calibrations": self.calibrations,
            "last_quantiles": self.last_quantiles,
            "last_calibration_step": self._last_calibration_step,
            "bounds": [self.lower, self.upper],
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
