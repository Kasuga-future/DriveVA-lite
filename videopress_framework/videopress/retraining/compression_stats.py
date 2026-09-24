"""Runtime compression statistics and dynamic-length analysis (plan 34-35).

Two requirements from the plan drive this module.

1. *Record*, per scene and per inference round, the candidate/kept counts, the
   score distribution, and the threshold actually used (plan section 34).
2. *Prove the length is genuinely dynamic* (plan section 35).  If every scene
   keeps roughly the same number of tokens, the threshold scorer has silently
   degenerated into a fixed-budget selector and the whole Route A premise is
   void.  The plan therefore asks for the correlations
   ``corr(K, difficulty)`` and ``corr(K, sigma)`` and for a per-category
   breakdown.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("percentile of an empty sequence")
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * float(q)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Spearman rank correlation, ``None`` when either side is constant."""
    if len(xs) != len(ys):
        raise ValueError("spearman inputs must have equal length")
    n = len(xs)
    if n < 3:
        return None

    def ranks(values: Sequence[float]) -> List[float]:
        order = sorted(range(n), key=lambda i: float(values[i]))
        out = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and float(values[order[j + 1]]) == float(values[order[i]]):
                j += 1
            average = (i + j) / 2.0
            for k in range(i, j + 1):
                out[order[k]] = average
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0.0 or dy == 0.0:
        return None
    return num / (dx * dy)


@dataclass
class RoundRecord:
    """One ``(scene, round)`` compression event (plan section 34 schema)."""

    scene_id: str
    round: int
    sigma: Optional[float] = None
    history_candidates: int = 0
    future_candidates: int = 0
    history_kept: int = 0
    future_kept: int = 0
    score_mean: float = 0.0
    score_std: float = 0.0
    score_p10: float = 0.0
    score_p50: float = 0.0
    score_p90: float = 0.0
    threshold: object = None
    keep_ratio: float = 1.0
    difficulty: Optional[float] = None
    category: Optional[str] = None
    extra: Dict[str, object] = field(default_factory=dict)

    @property
    def total_video_kept(self) -> int:
        return int(self.history_kept) + int(self.future_kept)

    @property
    def total_candidates(self) -> int:
        return int(self.history_candidates) + int(self.future_candidates)

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["total_video_kept"] = self.total_video_kept
        payload["total_candidates"] = self.total_candidates
        return payload


class CompressionStatsRecorder:
    """Accumulate :class:`RoundRecord` entries and summarise them."""

    def __init__(self) -> None:
        self.records: List[RoundRecord] = []

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    def clear(self) -> None:
        self.records.clear()

    def record(
        self,
        *,
        scores: torch.Tensor,
        kept_counts: Sequence[int],
        candidate_counts: Sequence[int],
        scene_id: str,
        round_index: int,
        sigma: Optional[float] = None,
        thresholds: Optional[Sequence[float]] = None,
        difficulty: Optional[float] = None,
        category: Optional[str] = None,
        extra: Optional[Dict[str, object]] = None,
    ) -> RoundRecord:
        if scores.ndim != 2:
            raise ValueError("scores must be [B,N]")
        if len(kept_counts) != len(candidate_counts):
            raise ValueError("kept_counts and candidate_counts must align")
        flat = scores.detach().float().reshape(-1)
        quantiles = torch.quantile(
            flat, torch.tensor([0.10, 0.50, 0.90], device=flat.device)
        ) if flat.numel() > 1 else flat.repeat(3)
        history_kept = int(kept_counts[0]) if kept_counts else 0
        future_kept = int(sum(kept_counts[1:])) if len(kept_counts) > 1 else 0
        history_candidates = int(candidate_counts[0]) if candidate_counts else 0
        future_candidates = int(sum(candidate_counts[1:])) if len(candidate_counts) > 1 else 0
        total_candidates = max(1, history_candidates + future_candidates)
        entry = RoundRecord(
            scene_id=str(scene_id),
            round=int(round_index),
            sigma=None if sigma is None else float(sigma),
            history_candidates=history_candidates,
            future_candidates=future_candidates,
            history_kept=history_kept,
            future_kept=future_kept,
            score_mean=float(flat.mean()) if flat.numel() else 0.0,
            score_std=float(flat.std(unbiased=False)) if flat.numel() else 0.0,
            score_p10=float(quantiles[0]),
            score_p50=float(quantiles[1]),
            score_p90=float(quantiles[2]),
            threshold=list(thresholds) if thresholds is not None else None,
            keep_ratio=(history_kept + future_kept) / total_candidates,
            difficulty=None if difficulty is None else float(difficulty),
            category=None if category is None else str(category),
            extra=dict(extra or {}),
        )
        self.records.append(entry)
        return entry

    # --------------------------------------------------------------- reports
    def length_summary(self) -> Dict[str, float]:
        """Mean/median/P10/P50/P90/P95 of the dynamic total (plan section 15)."""
        lengths = [r.total_video_kept for r in self.records]
        if not lengths:
            return {}
        return {
            "count": len(lengths),
            "mean": sum(lengths) / len(lengths),
            "min": float(min(lengths)),
            "max": float(max(lengths)),
            "std": float(torch.tensor(lengths, dtype=torch.float32).std(unbiased=False)),
            "p10": _percentile(lengths, 0.10),
            "p50": _percentile(lengths, 0.50),
            "p90": _percentile(lengths, 0.90),
            "p95": _percentile(lengths, 0.95),
        }

    def domain_summary(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for label, kept_key, cand_key in (
            ("history", "history_kept", "history_candidates"),
            ("future", "future_kept", "future_candidates"),
        ):
            kept = [getattr(r, kept_key) for r in self.records]
            cands = [getattr(r, cand_key) for r in self.records]
            if not kept:
                continue
            out[label] = {
                "kept_mean": sum(kept) / len(kept),
                "kept_p10": _percentile(kept, 0.10),
                "kept_p50": _percentile(kept, 0.50),
                "kept_p90": _percentile(kept, 0.90),
                "candidate_mean": sum(cands) / max(1, len(cands)),
                "retention_mean": sum(kept) / max(1, sum(cands)),
            }
        return out

    def length_sigma_correlation(self, *, per_scene: bool = True) -> Optional[float]:
        """``corr(K, sigma)`` (plan section 35)."""
        if per_scene:
            grouped: Dict[str, RoundRecord] = {}
            for record in self.records:
                grouped.setdefault(record.scene_id, record)
            rows = [r for r in grouped.values() if r.sigma is not None]
        else:
            rows = [r for r in self.records if r.sigma is not None]
        if len(rows) < 3:
            return None
        return _spearman([float(r.sigma) for r in rows], [float(r.total_video_kept) for r in rows])

    def length_difficulty_correlation(self) -> Optional[float]:
        """``corr(K, difficulty)``; difficulty is caller-supplied (PDM, cost, ...)."""
        rows = [r for r in self.records if r.difficulty is not None]
        if len(rows) < 3:
            return None
        return _spearman(
            [float(r.difficulty) for r in rows], [float(r.total_video_kept) for r in rows]
        )

    def category_summary(self) -> Dict[str, Dict[str, float]]:
        groups: Dict[str, List[RoundRecord]] = {}
        for record in self.records:
            if record.category:
                groups.setdefault(record.category, []).append(record)
        return {
            name: {
                "count": len(rows),
                "kept_mean": sum(r.total_video_kept for r in rows) / len(rows),
                "kept_p10": _percentile([r.total_video_kept for r in rows], 0.10),
                "kept_p90": _percentile([r.total_video_kept for r in rows], 0.90),
            }
            for name, rows in sorted(groups.items())
        }

    def is_truly_dynamic(self, *, min_spread_ratio: float = 0.05) -> Dict[str, object]:
        """Flag a scorer that has degenerated into a fixed-budget selector.

        ``min_spread_ratio`` compares the inter-decile spread of K against the
        mean K.  A fixed selector has spread 0; the plan expects easy scenes to
        drop well below and hard scenes to rise above the mean.
        """
        summary = self.length_summary()
        if not summary:
            return {"dynamic": False, "reason": "no records"}
        spread = summary["p90"] - summary["p10"]
        threshold = float(min_spread_ratio) * max(1.0, summary["mean"])
        return {
            "dynamic": bool(spread > threshold),
            "p10": summary["p10"],
            "p90": summary["p90"],
            "spread": spread,
            "required_spread": threshold,
            "mean": summary["mean"],
            "reason": "" if spread > threshold else "inter-decile spread below requirement",
        }

    def report(self) -> Dict[str, object]:
        return {
            "length": self.length_summary(),
            "domains": self.domain_summary(),
            "dynamic_check": self.is_truly_dynamic(),
            "corr_length_sigma": self.length_sigma_correlation(),
            "corr_length_difficulty": self.length_difficulty_correlation(),
            "categories": self.category_summary(),
        }

    def write_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {
                    "summary": self.report(),
                    "records": [r.to_dict() for r in self.records],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return target


def threshold_sweep_report(
    rows: Iterable[Dict[str, object]],
) -> List[Dict[str, object]]:
    """Sort a calibration sweep by compression while keeping the metrics.

    The plan (section 25) selects "the threshold with the most compression that
    still satisfies the quality constraint" on the held-out calibration split.
    ``rows`` are caller-built dicts with at least ``threshold``,
    ``mean_kept`` and ``delta_pdm`` (plus optional ``ci_lower``); the helper only
    orders them so the choice is reproducible.
    """

    def key(row: Dict[str, object]):
        ci = row.get("ci_lower")
        near_lossless = ci is not None and float(ci) > -0.002
        return (
            0 if near_lossless else 1,
            float(row.get("mean_kept", float("inf"))),
            -float(row.get("delta_pdm", 0.0)),
        )

    return sorted((dict(r) for r in rows), key=key)
