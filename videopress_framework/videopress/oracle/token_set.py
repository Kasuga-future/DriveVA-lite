"""Token-level, set-level oracle search over future video tokens.

Motivation
----------
The 2026-09-18/20 future oracle dropped *normalized tiles*: a 3x4 grid per
future latent, ~30 tokens each.  Even the best tile subset was not
near-lossless (``combined keep=0.50`` gave dPDM ``-0.0182``), and the tile-level
single-drop signal was close to noise.  The conclusion report asked for a finer
instrument with three properties this module implements:

1. **token-level atoms** -- the atomic units are single tokens or small token
   groups, not 30-token tiles;
2. **set-level scoring** -- a candidate subset is scored *jointly*, because the
   tile study showed single-drop harm is not additive and cannot be summed;
3. **searches that can compose** -- best-of-N random masks, greedy forward
   selection, greedy backward elimination and beam search over token groups.

The module is evaluator agnostic.  Callers pass a ``score_many``/``score_fn``
that maps a token subset to a scalar (mean PDM, PDM harm, trajectory
displacement, ...).  That keeps the search itself unit-testable on CPU and lets
the GPU driver swap in the official NAVSIM evaluator.

Token coordinates
-----------------
Token sets are always expressed as **flat future offsets**
``0 <= token < num_latents * tokens_per_latent`` in storage order
(``future_latent_0`` first).  The JSON interchange format used by the runner
keeps the per-latent split::

    {"scene-token": {"future_latent_0": [0, 5, 7], "future_latent_1": [12]}}

``latent_local_mask`` / ``flat_from_latent_local`` convert between the two.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..utils.seed import stable_seed

__all__ = [
    "GROUP_MODES",
    "SearchStep",
    "SetScorer",
    "SetSearchResult",
    "beam_search",
    "build_token_groups",
    "entry_to_flat",
    "flat_from_latent_local",
    "greedy_backward_elimination",
    "greedy_forward_selection",
    "latent_local_mask",
    "mean_over_scenes",
    "per_scene_oracle",
    "random_search",
    "random_token_masks",
    "read_token_mask_json",
    "write_token_mask_json",
]


GROUP_MODES = ("token", "linear", "block", "random")


# ---------------------------------------------------------------------------
# shape helpers
# ---------------------------------------------------------------------------


def _validate_shape(tokens_per_latent: int, num_latents: int) -> tuple[int, int]:
    try:
        tpl = int(tokens_per_latent)
        n_lat = int(num_latents)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise TypeError("tokens_per_latent and num_latents must be integers") from exc
    if tpl <= 0:
        raise ValueError("tokens_per_latent must be positive")
    if n_lat <= 0:
        raise ValueError("num_latents must be positive")
    return tpl, n_lat


def _check_partition(groups: Sequence[Sequence[int]], total: int) -> None:
    seen: set[int] = set()
    for group in groups:
        if len(group) == 0:
            raise ValueError("token groups must be non-empty")
        for token in group:
            token = int(token)
            if token < 0 or token >= total:
                raise ValueError(f"token {token} is outside [0, {total})")
            if token in seen:
                raise ValueError(f"token {token} appears in more than one group")
            seen.add(token)
    if len(seen) != total:
        raise ValueError(f"token groups cover {len(seen)} of {total} tokens")


# ---------------------------------------------------------------------------
# grouping
# ---------------------------------------------------------------------------


def build_token_groups(
    tokens_per_latent: int,
    num_latents: int,
    *,
    mode: str = "linear",
    group_size: int | None = None,
    block_h: int | None = None,
    block_w: int | None = None,
    width: int | None = None,
    seed: int = 0,
) -> tuple[tuple[int, ...], ...]:
    """Partition all future tokens into atomic search groups.

    ``mode``:

    * ``token``  -- every token is its own group (true token-level search);
    * ``linear`` -- consecutive ``group_size`` tokens inside one latent
      (row-major order, last chunk may be shorter);
    * ``block``  -- ``block_h x block_w`` spatial blocks inside one latent;
      requires ``width`` (the latent grid width);
    * ``random`` -- seeded random partition into groups of ``group_size`` inside
      each latent (groups never span latents).
    """

    tpl, n_lat = _validate_shape(tokens_per_latent, num_latents)
    mode = str(mode).strip().lower()
    if mode not in GROUP_MODES:
        raise ValueError(f"unknown group mode {mode!r}; choose from {GROUP_MODES}")
    total = tpl * n_lat
    groups: list[tuple[int, ...]] = []

    for latent in range(n_lat):
        base = latent * tpl
        if mode == "token":
            if group_size not in (None, 1):
                raise ValueError("mode='token' requires group_size None or 1")
            groups.extend((base + local,) for local in range(tpl))
            continue

        if mode == "linear":
            size = 0 if group_size is None else int(group_size)
            if size <= 0:
                raise ValueError("mode='linear' requires a positive group_size")
            for start in range(0, tpl, size):
                groups.append(tuple(range(base + start, base + min(start + size, tpl))))
            continue

        if mode == "block":
            height_hint = None if width is None else int(width)
            bh = 0 if block_h is None else int(block_h)
            bw = 0 if block_w is None else int(block_w)
            if bh <= 0 or bw <= 0:
                raise ValueError("mode='block' requires positive block_h and block_w")
            if height_hint is None or height_hint <= 0:
                raise ValueError("mode='block' requires a positive width")
            if tpl % height_hint != 0:
                raise ValueError(
                    f"tokens_per_latent={tpl} is not divisible by width={height_hint}"
                )
            height = tpl // height_hint
            for y0 in range(0, height, bh):
                for x0 in range(0, height_hint, bw):
                    members = []
                    for y in range(y0, min(y0 + bh, height)):
                        for x in range(x0, min(x0 + bw, height_hint)):
                            members.append(base + y * height_hint + x)
                    groups.append(tuple(members))
            continue

        # mode == "random"
        size = 0 if group_size is None else int(group_size)
        if size <= 0:
            raise ValueError("mode='random' requires a positive group_size")
        rng = random.Random(stable_seed("future-token-group", int(seed), latent))
        locals_ = list(range(tpl))
        rng.shuffle(locals_)
        for start in range(0, tpl, size):
            chunk = sorted(locals_[start : start + size])
            groups.append(tuple(base + index for index in chunk))

    groups.sort(key=lambda group: group[0])
    _check_partition(groups, total)
    return tuple(groups)


# ---------------------------------------------------------------------------
# interchange format
# ---------------------------------------------------------------------------


def latent_local_mask(
    flat_indices: Iterable[int],
    *,
    tokens_per_latent: int,
    num_latents: int,
) -> dict[str, list[int]]:
    """Convert flat future offsets to the per-latent JSON mask format."""

    tpl, n_lat = _validate_shape(tokens_per_latent, num_latents)
    total = tpl * n_lat
    buckets: dict[int, set[int]] = {index: set() for index in range(n_lat)}
    for token in flat_indices:
        token = int(token)
        if token < 0 or token >= total:
            raise ValueError(f"token {token} is outside [0, {total})")
        latent, local = divmod(token, tpl)
        buckets[latent].add(local)
    return {
        f"future_latent_{index}": sorted(buckets[index]) for index in range(n_lat)
    }


def flat_from_latent_local(
    mask: Mapping[str, Sequence[int]],
    *,
    tokens_per_latent: int,
    num_latents: int,
) -> list[int]:
    """Inverse of :func:`latent_local_mask` (missing latents mean empty)."""

    tpl, n_lat = _validate_shape(tokens_per_latent, num_latents)
    if not isinstance(mask, Mapping):
        raise TypeError("mask must be a mapping of future_latent_i -> local indices")
    flat: set[int] = set()
    for key, values in mask.items():
        name = str(key)
        if not name.startswith("future_latent_"):
            raise ValueError(f"unknown future mask key {key!r}")
        try:
            latent = int(name.rsplit("_", 1)[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"unknown future mask key {key!r}") from exc
        if latent < 0 or latent >= n_lat:
            raise ValueError(f"future latent {latent} is outside [0, {n_lat})")
        if values is None:
            continue
        for value in values:
            local = int(value)
            if local < 0 or local >= tpl:
                raise ValueError(f"local token {local} is outside [0, {tpl})")
            flat.add(latent * tpl + local)
    return sorted(flat)


def entry_to_flat(
    entry: Any,
    *,
    tokens_per_latent: int,
    num_latents: int,
) -> list[int]:
    """Accept either a per-latent mapping or a flat future-offset list."""

    if isinstance(entry, Mapping):
        return flat_from_latent_local(
            entry, tokens_per_latent=tokens_per_latent, num_latents=num_latents
        )
    if isinstance(entry, Sequence) and not isinstance(entry, (str, bytes)):
        tpl, n_lat = _validate_shape(tokens_per_latent, num_latents)
        total = tpl * n_lat
        flat: set[int] = set()
        for value in entry:
            token = int(value)
            if token < 0 or token >= total:
                raise ValueError(f"token {token} is outside [0, {total})")
            flat.add(token)
        return sorted(flat)
    raise TypeError(
        "future token mask entry must be a mapping of future_latent_i or a list "
        "of flat future offsets"
    )


def read_token_mask_json(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError("future token mask JSON must map scene_token -> mask")
    return data


def write_token_mask_json(masks: Mapping[str, Any], path: str | Path) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(masks, indent=2, sort_keys=True), encoding="utf-8")
    return output


# ---------------------------------------------------------------------------
# set-level scoring
# ---------------------------------------------------------------------------


def _normalize_groups(groups: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    normalized = tuple(tuple(sorted(int(token) for token in group)) for group in groups)
    if not normalized:
        raise ValueError("groups must not be empty")
    for group in normalized:
        if not group:
            raise ValueError("groups must not contain empty entries")
    seen: set[int] = set()
    for group in normalized:
        for token in group:
            if token in seen:
                raise ValueError("token groups must be disjoint")
            seen.add(token)
    return normalized


class SetScorer:
    """Caching, batching wrapper around a set-level objective.

    ``score_many`` receives a list of token tuples and must return one score per
    tuple.  ``score_fn`` is the single-set fallback.  Repeated candidate sets are
    cached, so the search never pays twice for the same subset.
    """

    def __init__(
        self,
        score_many: Callable[[Sequence[tuple[int, ...]]], Sequence[float]] | None = None,
        *,
        score_fn: Callable[[Sequence[int]], float] | None = None,
    ):
        if score_many is None and score_fn is None:
            raise ValueError("SetScorer needs score_many or score_fn")
        self._score_many = score_many
        self._score_fn = score_fn
        self._cache: dict[tuple[int, ...], float] = {}
        self.n_unique_evaluations = 0
        self.n_requests = 0

    @staticmethod
    def _key(tokens: Iterable[int]) -> tuple[int, ...]:
        return tuple(sorted(int(token) for token in tokens))

    def evaluate(self, token_sets: Sequence[Sequence[int]]) -> list[float]:
        keys = [self._key(tokens) for tokens in token_sets]
        self.n_requests += len(keys)
        missing = list(dict.fromkeys(key for key in keys if key not in self._cache))
        if missing:
            self.n_unique_evaluations += len(missing)
            if self._score_many is not None:
                values = [float(value) for value in self._score_many(missing)]
            else:
                values = [float(self._score_fn(key)) for key in missing]  # type: ignore[misc]
            if len(values) != len(missing):
                raise ValueError(
                    "score_many returned "
                    f"{len(values)} values for {len(missing)} candidate sets"
                )
            for key, value in zip(missing, values):
                self._cache[key] = value
        return [self._cache[key] for key in keys]


@dataclass(frozen=True)
class SearchStep:
    """One recorded step of a set search."""

    round: int
    action: str
    group: int | None
    score: float
    n_selected: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": int(self.round),
            "action": self.action,
            "group": None if self.group is None else int(self.group),
            "score": float(self.score),
            "n_selected": int(self.n_selected),
        }


@dataclass
class SetSearchResult:
    """Outcome of a token-set search."""

    selected: tuple[int, ...]
    score: float
    steps: list[SearchStep] = field(default_factory=list)
    n_evaluations: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected": [int(token) for token in self.selected],
            "n_selected": len(self.selected),
            "score": float(self.score),
            "n_evaluations": int(self.n_evaluations),
            "metadata": dict(self.metadata),
            "steps": [step.to_dict() for step in self.steps],
        }


def _tokens_of_state(
    state: Iterable[int], groups: Sequence[tuple[int, ...]]
) -> tuple[int, ...]:
    tokens: list[int] = []
    for index in state:
        tokens.extend(groups[int(index)])
    return tuple(sorted(tokens))


def _argbest(scores: Sequence[float], maximize: bool) -> int:
    if not scores:
        raise ValueError("cannot rank an empty score list")
    best_index = 0
    best_value = scores[0]
    for index in range(1, len(scores)):
        value = scores[index]
        if maximize:
            if _is_better(value, best_value, True):
                best_index, best_value = index, value
        else:
            if _is_better(value, best_value, False):
                best_index, best_value = index, value
    return best_index


def _is_better(left: float, right: float, maximize: bool) -> bool:
    """NaN is always worst; ties keep the earlier/left value."""

    left_nan = not math.isfinite(left)
    right_nan = not math.isfinite(right)
    if left_nan and right_nan:
        return False
    if left_nan:
        return False
    if right_nan:
        return True
    return left > right if maximize else left < right


def _validate_initial(
    selected: Iterable[int], groups: Sequence[tuple[int, ...]]
) -> set[int]:
    state = {int(index) for index in selected}
    for index in state:
        if index < 0 or index >= len(groups):
            raise ValueError(f"group index {index} is outside [0, {len(groups)})")
    return state


def greedy_forward_selection(
    groups: Sequence[Sequence[int]],
    budget: int,
    scorer: SetScorer,
    *,
    maximize: bool = True,
    initial_groups: Iterable[int] = (),
    min_gain: float = 0.0,
    max_rounds: int | None = None,
) -> SetSearchResult:
    """Add the best-scoring group each round until the token budget is reached."""

    normalized = _normalize_groups(groups)
    budget = int(budget)
    if budget < 0:
        raise ValueError("budget must be non-negative")
    sizes = [len(group) for group in normalized]
    selected = _validate_initial(initial_groups, normalized)
    tokens = _tokens_of_state(selected, normalized)
    if len(tokens) > budget:
        raise ValueError("initial selection already exceeds budget")
    current = scorer.evaluate([tokens])[0]
    steps = [SearchStep(0, "init", None, current, len(tokens))]

    rounds = 0
    while len(tokens) < budget:
        if max_rounds is not None and rounds >= max_rounds:
            break
        candidates = [
            index
            for index in range(len(normalized))
            if index not in selected and len(tokens) + sizes[index] <= budget
        ]
        if not candidates:
            break
        candidate_sets = [
            _tokens_of_state(selected | {index}, normalized) for index in candidates
        ]
        scores = scorer.evaluate(candidate_sets)
        best = _argbest(scores, maximize)
        gain = scores[best] - current if maximize else current - scores[best]
        if not math.isfinite(gain) or gain < min_gain:
            break
        selected.add(candidates[best])
        tokens = candidate_sets[best]
        current = scores[best]
        rounds += 1
        steps.append(SearchStep(rounds, "add", candidates[best], current, len(tokens)))

    return SetSearchResult(
        selected=tuple(tokens),
        score=float(current),
        steps=steps,
        n_evaluations=scorer.n_requests,
        metadata={
            "search": "greedy_forward",
            "budget": budget,
            "maximize": bool(maximize),
            "n_groups": len(normalized),
            "budget_met": len(tokens) <= budget,
            "rounds": rounds,
        },
    )


def greedy_backward_elimination(
    groups: Sequence[Sequence[int]],
    budget: int,
    scorer: SetScorer,
    *,
    maximize: bool = True,
    initial_groups: Iterable[int] | None = None,
    max_harm: float = math.inf,
    max_rounds: int | None = None,
) -> SetSearchResult:
    """Start from the full set and remove the least harmful group each round."""

    normalized = _normalize_groups(groups)
    budget = int(budget)
    if budget < 0:
        raise ValueError("budget must be non-negative")
    sizes = [len(group) for group in normalized]
    selected = (
        set(range(len(normalized)))
        if initial_groups is None
        else _validate_initial(initial_groups, normalized)
    )
    tokens = _tokens_of_state(selected, normalized)
    current = scorer.evaluate([tokens])[0]
    steps = [SearchStep(0, "init", None, current, len(tokens))]

    rounds = 0
    while len(tokens) > budget and len(selected) > 1:
        if max_rounds is not None and rounds >= max_rounds:
            break
        candidates = sorted(selected)
        candidate_sets = [
            _tokens_of_state(selected - {index}, normalized) for index in candidates
        ]
        scores = scorer.evaluate(candidate_sets)
        best = _argbest(scores, maximize)
        harm = current - scores[best] if maximize else scores[best] - current
        if math.isfinite(harm) and harm > max_harm:
            break
        selected.discard(candidates[best])
        tokens = candidate_sets[best]
        current = scores[best]
        rounds += 1
        steps.append(
            SearchStep(rounds, "remove", candidates[best], current, len(tokens))
        )

    return SetSearchResult(
        selected=tuple(tokens),
        score=float(current),
        steps=steps,
        n_evaluations=scorer.n_requests,
        metadata={
            "search": "greedy_backward",
            "budget": budget,
            "maximize": bool(maximize),
            "n_groups": len(normalized),
            "budget_met": len(tokens) <= budget,
            "rounds": rounds,
        },
    )


def beam_search(
    groups: Sequence[Sequence[int]],
    budget: int,
    scorer: SetScorer,
    *,
    beam_width: int = 4,
    maximize: bool = True,
    initial_groups: Iterable[int] = (),
    max_rounds: int | None = None,
) -> SetSearchResult:
    """Beam search over token-group additions."""

    normalized = _normalize_groups(groups)
    budget = int(budget)
    if budget < 0:
        raise ValueError("budget must be non-negative")
    beam_width = int(beam_width)
    if beam_width <= 0:
        raise ValueError("beam_width must be positive")
    sizes = [len(group) for group in normalized]
    start = tuple(sorted(_validate_initial(initial_groups, normalized)))
    start_tokens = _tokens_of_state(start, normalized)
    if len(start_tokens) > budget:
        raise ValueError("initial selection already exceeds budget")
    start_score = scorer.evaluate([start_tokens])[0]
    frontier: list[tuple[tuple[int, ...], float, tuple[int, ...]]] = [
        (start, start_score, start_tokens)
    ]
    best_state, best_score, best_tokens = start, start_score, start_tokens
    steps = [SearchStep(0, "init", None, best_score, len(best_tokens))]

    depth = 0
    while frontier and len(best_tokens) < budget:
        if max_rounds is not None and depth >= max_rounds:
            break
        seen: set[tuple[int, ...]] = set()
        expansions: list[tuple[int, ...]] = []
        for state, _score, state_tokens in frontier:
            for index in range(len(normalized)):
                if index in state:
                    continue
                if len(state_tokens) + sizes[index] > budget:
                    continue
                new_state = tuple(sorted(state + (index,)))
                if new_state in seen:
                    continue
                seen.add(new_state)
                expansions.append(new_state)
        if not expansions:
            break
        exp_sets = [_tokens_of_state(state, normalized) for state in expansions]
        scores = scorer.evaluate(exp_sets)
        order = sorted(
            range(len(expansions)),
            key=lambda idx: _sort_key(scores[idx], maximize, len(exp_sets[idx])),
        )
        frontier = [
            (expansions[idx], scores[idx], exp_sets[idx]) for idx in order[:beam_width]
        ]
        depth += 1
        top_state, top_score, top_tokens = frontier[0]
        if _prefer(
            top_score, len(top_tokens), best_score, len(best_tokens), maximize
        ):
            best_state, best_score, best_tokens = top_state, top_score, top_tokens
        steps.append(
            SearchStep(depth, "expand", None, best_score, len(best_tokens))
        )

    return SetSearchResult(
        selected=tuple(best_tokens),
        score=float(best_score),
        steps=steps,
        n_evaluations=scorer.n_requests,
        metadata={
            "search": "beam",
            "budget": budget,
            "beam_width": beam_width,
            "maximize": bool(maximize),
            "n_groups": len(normalized),
            "rounds": depth,
            "budget_met": len(best_tokens) <= budget,
        },
    )


def _sort_key(
    score: float, maximize: bool, n_tokens: int = 0
) -> tuple[int, float, int]:
    if not math.isfinite(score):
        return (1, 0.0, -n_tokens)
    return (0, -score if maximize else score, -n_tokens)


def _prefer(
    score: float,
    n_tokens: int,
    best_score: float,
    best_tokens: int,
    maximize: bool,
) -> bool:
    """Strictly better score, or an exact tie with more tokens selected.

    PDM is discrete and largely binary, so equal panel means are common.  A tie
    must not freeze the beam at a smaller subset: the oracle has to spend its
    budget to stay a matched-budget comparison instead of "winning" by keeping
    nothing.
    """

    if _is_better(score, best_score, maximize):
        return True
    if not math.isfinite(score) or not math.isfinite(best_score):
        return False
    if score != best_score:
        return False
    return n_tokens > best_tokens


def random_search(
    groups: Sequence[Sequence[int]],
    budget: int,
    scorer: SetScorer,
    *,
    n_samples: int = 32,
    seed: int = 0,
    maximize: bool = True,
) -> SetSearchResult:
    """Best-of-N random token-group subsets (a matched-budget random control)."""

    normalized = _normalize_groups(groups)
    budget = int(budget)
    if budget < 0:
        raise ValueError("budget must be non-negative")
    n_samples = int(n_samples)
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    sizes = [len(group) for group in normalized]
    order = list(range(len(normalized)))
    rng = random.Random(stable_seed("future-token-random-search", int(seed), budget))
    candidates: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    attempts = 0
    while len(candidates) < n_samples and attempts < n_samples * 50:
        attempts += 1
        rng.shuffle(order)
        chosen: list[int] = []
        count = 0
        for index in order:
            if count + sizes[index] <= budget:
                chosen.append(index)
                count += sizes[index]
        key = tuple(sorted(chosen))
        if key in seen:
            continue
        seen.add(key)
        candidates.append(_tokens_of_state(key, normalized))
    scores = scorer.evaluate(candidates)
    best = _argbest(scores, maximize)
    return SetSearchResult(
        selected=tuple(candidates[best]),
        score=float(scores[best]),
        steps=[SearchStep(0, "init", None, float(scores[best]), len(candidates[best]))],
        n_evaluations=scorer.n_requests,
        metadata={
            "search": "random",
            "budget": budget,
            "n_samples": len(candidates),
            "seed": int(seed),
            "maximize": bool(maximize),
            "n_groups": len(normalized),
        },
    )


# ---------------------------------------------------------------------------
# per-scene best-of-N random masks
# ---------------------------------------------------------------------------


def random_token_masks(
    scene_tokens: Sequence[str],
    *,
    tokens_per_latent: int,
    num_latents: int,
    budget: int,
    n_samples: int,
    seed: int = 0,
) -> dict[int, dict[str, dict[str, list[int]]]]:
    """Per-scene, per-sample random token masks in the runner JSON format.

    Sample ``s`` gives scene ``t`` a deterministic random subset derived from
    ``(seed, s, t)``, so ``N`` of these files can be evaluated independently and
    the per-scene best-of-N oracle is the max over samples.
    """

    tpl, n_lat = _validate_shape(tokens_per_latent, num_latents)
    total = tpl * n_lat
    budget = int(budget)
    if budget < 0 or budget > total:
        raise ValueError(f"budget must be within [0, {total}]")
    n_samples = int(n_samples)
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    per_sample: dict[int, dict[str, dict[str, list[int]]]] = {}
    for sample in range(n_samples):
        masks: dict[str, dict[str, list[int]]] = {}
        for scene in scene_tokens:
            rng = random.Random(
                stable_seed("future-random-token-mask", int(seed), sample, str(scene))
            )
            chosen = rng.sample(range(total), budget)
            masks[str(scene)] = latent_local_mask(
                sorted(chosen), tokens_per_latent=tpl, num_latents=n_lat
            )
        per_sample[sample] = masks
    return per_sample


# ---------------------------------------------------------------------------
# per-scene aggregation
# ---------------------------------------------------------------------------


def mean_over_scenes(
    scores: Mapping[str, float],
    scenes: Sequence[str] | None = None,
) -> float:
    keys = list(scenes) if scenes is not None else sorted(scores)
    values = [float(scores[key]) for key in keys if key in scores]
    if not values:
        return math.nan
    return float(sum(values) / len(values))


def per_scene_oracle(
    candidate_scores: Mapping[str, Mapping[str, float]],
    *,
    higher_is_better: bool = True,
    scenes: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Best-of-N oracle: per scene take the best candidate score.

    ``candidate_scores`` maps ``candidate_id -> {scene_token: value}``.  A scene
    only enters the oracle when at least one candidate scored it; the returned
    ``n_scenes`` is that shared set so a per-scene max cannot be inflated by
    candidate-specific scene coverage.
    """

    if not candidate_scores:
        raise ValueError("candidate_scores must not be empty")
    if scenes is None:
        shared = set.intersection(
            *(set(values) for values in candidate_scores.values())
        )
        scene_list = sorted(shared)
    else:
        scene_list = list(scenes)
    if not scene_list:
        raise ValueError("no scene is scored by every candidate")

    winners: dict[str, str] = {}
    best_values: dict[str, float] = {}
    win_counts: dict[str, int] = {name: 0 for name in candidate_scores}
    for scene in scene_list:
        best_name = None
        best_value = None
        for name, values in candidate_scores.items():
            value = float(values[scene])
            if best_value is None or _is_better(value, best_value, higher_is_better):
                best_name, best_value = name, value
        winners[scene] = str(best_name)
        best_values[scene] = float(best_value)
        win_counts[str(best_name)] += 1

    return {
        "n_scenes": len(scene_list),
        "oracle_mean": mean_over_scenes(best_values),
        "per_candidate_mean": {
            name: mean_over_scenes(values, scene_list)
            for name, values in candidate_scores.items()
        },
        "candidate_win_counts": win_counts,
        "scene_winners": winners,
        "scene_best_values": best_values,
    }
