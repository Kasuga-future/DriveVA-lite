"""Runtime lifecycle and injection-point enums."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from enum import Enum
import time
from typing import Any, Iterator, Optional


class EvaluationMode(str, Enum):
    CAUSAL = "causal"
    PHYSICAL = "physical"

    @classmethod
    def parse(cls, value: "EvaluationMode | str") -> "EvaluationMode":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            raise ValueError(f"Unknown evaluation mode: {value}") from exc


class InjectionPoint(str, Enum):
    VIDEO_INPUT = "video_input"
    BLOCK_INPUT = "block_input"
    SELF_ATTN_KV = "self_attn_kv"
    SELF_ATTN_OUTPUT = "self_attn_output"

    @classmethod
    def parse(cls, value: "InjectionPoint | str") -> "InjectionPoint":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            raise ValueError(f"Unknown injection point: {value}") from exc


@dataclass(frozen=True)
class CompressionEventKey:
    """Stable identity for one press execution in a model forward."""

    scene_token: str
    diffusion_rank: int | None
    layer_idx: int | None
    injection_point: str


@dataclass
class CompressionEvent:
    key: CompressionEventKey
    result: Any
    context: Any = None


class VideoPressRuntime:
    """Owns press state and makes hook installation exception-safe."""

    def __init__(
        self,
        press=None,
        mode: EvaluationMode | str = EvaluationMode.CAUSAL,
        adapter=None,
        artifact_dir=None,
        score_cache=None,
        allow_sample_domain_override: bool = False,
    ):
        self.press = press
        self.mode = EvaluationMode.parse(mode)
        self.adapter = adapter
        self.artifact_dir = artifact_dir
        self.score_cache = score_cache
        self.allow_sample_domain_override = bool(allow_sample_domain_override)
        self.layout = None
        self.current_scene = None
        self.current_sample = None
        self.current_context = None
        self.current_diffusion_rank = None
        self.current_timestep = None
        # Per-forward round index and schedule resolution.  ``current_round_index``
        # is incremented whenever the diffusion timestep changes inside one
        # sample; ``resolve_scorer_layer`` can then map round -> source layer.
        self.current_round_index = -1
        self._round_last_timestep = None
        self.last_result = None
        self.events: list[CompressionEvent] = []
        self.artifacts: dict[str, Any] = {}
        self.selector_latency_ms = 0.0
        self._installed = False

    def enabled(self) -> bool:
        return self.press is not None

    def begin_sample(self, sample=None, layout=None) -> None:
        self.current_sample = sample
        self.current_scene = getattr(sample, "scene_token", None) if sample is not None else None
        self.current_diffusion_rank = getattr(sample, "diffusion_rank", None) if sample is not None else None
        self.current_timestep = getattr(sample, "timestep", None) if sample is not None else None
        self.current_round_index = -1
        self._round_last_timestep = None
        if layout is not None:
            self.layout = layout
        self.current_context = None
        self.last_result = None
        self.events = []
        self.artifacts = {}
        self.selector_latency_ms = 0.0
        scorer = getattr(self.press, "scorer", None)
        reset_observations = getattr(scorer, "reset_observations", None)
        if reset_observations is not None:
            reset_observations()
        persistence_store = getattr(self, "_cross_layer_selection_store", None)
        if persistence_store is not None:
            persistence_store.clear()

    def set_layout(self, layout) -> None:
        self.layout = layout

    def set_context(self, context) -> None:
        self.current_context = context
        self.layout = context.layout

    def note_model_timestep(self, timestep) -> None:
        """Track the 0-based flow-matching round index inside one sample.

        The official evaluator calls ``model_fn`` once per scheduler timestep.
        We only increment when the scalar timestep actually changes, so built-in
        CFG positive/negative calls with the same timestep stay in one round.
        """
        if timestep is None:
            return
        try:
            value = float(timestep)
        except (TypeError, ValueError):
            return
        if self._round_last_timestep is None or value != self._round_last_timestep:
            self.current_round_index += 1
            self._round_last_timestep = value

    def resolve_scorer_layer(self, default_layer=None):
        """Resolve the active press source layer for the current round.

        ``scorer.layer_schedule``, when present, is a list ordered by
        flow-matching round (first executed timestep = index 0).  Without it the
        historical static ``scorer.layer`` is returned unchanged.
        """
        scorer = getattr(self.press, "scorer", None) if self.press is not None else None
        schedule = getattr(scorer, "layer_schedule", None)
        if schedule:
            if not isinstance(schedule, (list, tuple)):
                schedule = [schedule]
            idx = 0 if self.current_round_index < 0 else int(self.current_round_index)
            idx = max(0, min(idx, len(schedule) - 1))
            return int(schedule[idx])
        if default_layer is not None:
            return default_layer
        return getattr(scorer, "layer", None)

    @staticmethod
    def _domain_label(domain: Any) -> str:
        if isinstance(domain, dict):
            domain = domain.get("name", "last_history")
        return str(getattr(domain, "name", domain))

    def resolve_domain_spec(self, sample=None) -> tuple[Any, Any, bool]:
        """Resolve the configured domain, with opt-in sample overrides only."""

        configured = getattr(self.press, "domain", None) or "last_history"
        resolved = configured
        overridden = False
        sample = self.current_sample if sample is None else sample
        metadata = getattr(sample, "metadata", {}) if sample is not None else {}
        if self.allow_sample_domain_override and isinstance(metadata, dict):
            sample_domain = metadata.get("domain")
            if sample_domain is not None:
                resolved = sample_domain
                overridden = self._domain_label(configured) != self._domain_label(resolved)
        return configured, resolved, overridden

    def score_key(self, ctx):
        """Build the disk-cache key used by probe/intervention execution."""

        from ..probes.score_cache import ScoreKey

        scene_tokens = ctx.metadata.get("scene_tokens") if isinstance(ctx.metadata, dict) else None
        if scene_tokens:
            unique_scenes = {str(token) for token in scene_tokens}
            if len(unique_scenes) > 1:
                raise NotImplementedError(
                    "ScoreCache V1 requires one scene per batch; split the batch before probe scoring"
                )

        scorer = getattr(self.press, "scorer", None)
        signature = scorer.signature() if scorer is not None and hasattr(scorer, "signature") else type(scorer).__name__
        model_name = ctx.metadata.get("model_name") if isinstance(ctx.metadata, dict) else None
        if model_name is not None:
            signature = f"{signature}:model={model_name}"
        candidate = ctx.domain.candidate_indices
        candidate_start = int(candidate.min().item()) if candidate.numel() else -1
        candidate_end = int(candidate.max().item()) + 1 if candidate.numel() else -1
        signature = (
            f"{signature}:domain={ctx.domain.name}:n={ctx.domain.n_candidate}:"
            f"range={candidate_start}-{candidate_end}"
        )
        layer_idx = ctx.layer_idx
        if layer_idx is None:
            layer_idx = getattr(scorer, "layer", None)
        return ScoreKey(
            scene_token=str(ctx.scene_token),
            diffusion_rank=ctx.diffusion_rank,
            layer_idx=layer_idx,
            scorer_signature=str(signature),
        )

    def execute_press(self, ctx):
        """Execute one press and record it as an event.

        Probe scorers use a detached context and, when a ``ScoreCache`` is
        configured, persist both scores and the selector ranking before the
        intervention pass.  The intervention never calls the scorer again.
        """

        if self.press is None:
            raise RuntimeError("cannot execute a press-less runtime")
        scorer = getattr(self.press, "scorer", None)
        requires_probe = bool(getattr(scorer, "requires_probe", False))
        injection_point = InjectionPoint.parse(getattr(self.press, "injection_point", InjectionPoint.VIDEO_INPUT))
        probe_mode = getattr(scorer, "probe_mode", "none")
        probe_mode = getattr(probe_mode, "value", probe_mode)
        requires_frozen_ranking = requires_probe or (
            injection_point is InjectionPoint.VIDEO_INPUT
            and probe_mode == "online"
        )
        if ctx.tokens.is_cuda:
            import torch

            torch.cuda.synchronize(ctx.tokens.device)
        started = time.perf_counter()
        if requires_frozen_ranking and hasattr(self.press, "apply_with_selection"):
            key = self.score_key(ctx)
            scores = None
            ranking = None
            cache = self.score_cache
            if cache is not None and cache.contains(key):
                scores = cache.load(key, map_location=ctx.tokens.device)
                ranking = cache.load_ranking(key, map_location=ctx.tokens.device)
                if ranking is None:
                    raise RuntimeError(f"score cache entry has no frozen ranking: {cache.path_for(key)}")
            else:
                probe_ctx = ctx.clone_for_probe() if requires_probe else ctx
                scores = self.press.score(probe_ctx)
                selection = self.press.select(probe_ctx, scores)
                ranking = self.press.ranking(scores)
                if cache is not None:
                    cache.save(key, scores, ranking=ranking, metadata={"selection": selection.metadata})
            if scores.device != ctx.tokens.device:
                scores = scores.to(ctx.tokens.device)
            if ranking is not None and ranking.device != ctx.tokens.device:
                ranking = ranking.to(ctx.tokens.device)
            selection = self.press.select(ctx, scores, cached_ranking=ranking)
            if ctx.tokens.is_cuda:
                import torch

                torch.cuda.synchronize(ctx.tokens.device)
            selector_latency_ms = (time.perf_counter() - started) * 1000.0
            result = self.press.apply_with_selection(ctx, scores.detach(), selection)
            if cache is not None:
                result.metadata.setdefault("score_cache", {})
                result.metadata["score_cache"].update(
                    {
                        "path": str(cache.path_for(key)),
                        "key": key.__dict__,
                        "digests": cache.digest(key),
                        "ranking_frozen": True,
                    }
                )
        elif (
            not requires_frozen_ranking
            and hasattr(self.press, "score")
            and hasattr(self.press, "select")
            and hasattr(self.press, "apply_with_selection")
        ):
            scores = self.press.score(ctx)
            selection = self.press.select(ctx, scores)
            if ctx.tokens.is_cuda:
                import torch

                torch.cuda.synchronize(ctx.tokens.device)
            selector_latency_ms = (time.perf_counter() - started) * 1000.0
            result = self.press.apply_with_selection(ctx, scores, selection)
        else:
            result = self.press.apply(ctx)
            if ctx.tokens.is_cuda:
                import torch

                torch.cuda.synchronize(ctx.tokens.device)
            selector_latency_ms = (time.perf_counter() - started) * 1000.0
        self._annotate_result_identity_and_selection(ctx, result)
        self.selector_latency_ms += selector_latency_ms
        result.metadata.setdefault("timing", {})
        result.metadata["timing"]["selector_latency_ms"] = selector_latency_ms
        self.record_result(ctx, result)
        return result

    def execute_persistent_selection(self, ctx, selection, *, source_layer: int):
        """Apply a previously scored selection without charging selector time."""

        if self.press is None or not hasattr(self.press, "apply_with_selection"):
            raise RuntimeError("active press cannot reuse a cross-layer selection")
        result = self.press.apply_with_selection(ctx, None, selection)
        result.metadata.update(
            {
                "cross_layer_persistent": True,
                "persistent_selection_reused": True,
                "selection_source_layer": int(source_layer),
                "selection_applied_layer": ctx.layer_idx,
            }
        )
        result.metadata.setdefault("timing", {})
        result.metadata["timing"]["selector_latency_ms"] = 0.0
        self._annotate_result_identity_and_selection(ctx, result)
        self.record_result(ctx, result)
        return result

    @staticmethod
    def _annotate_result_identity_and_selection(ctx, result) -> None:
        """Attach auditable scene/domain/selection facts to one result.

        The event journal is intentionally compact, so it must carry enough
        information to prove that an intervention stayed inside the declared
        candidate range.  This check is performed after selection and before
        the result is recorded; it does not alter the operator output.
        """

        metadata = getattr(result, "metadata", None)
        if not isinstance(metadata, dict):
            return

        # Context identity is copied into the result rather than inferred later
        # from the outer evaluator record.  This makes a single runtime event
        # self-describing when it is inspected independently.
        metadata.setdefault("scene_token", str(ctx.scene_token))
        if isinstance(getattr(ctx, "metadata", None), dict):
            for key in (
                "scene_tokens",
                "segment_scene_token",
                "segment_scene_name",
                "segment_frame_idx_start",
                "segment_frame_idx_end",
                "segment_frame_count",
                "segment_frame_idx_contiguous",
            ):
                if key in ctx.metadata:
                    metadata.setdefault(key, ctx.metadata[key])

        selection = getattr(result, "selection", None)
        keep = getattr(selection, "keep_global_indices", None)
        if keep is None:
            return
        if not hasattr(keep, "ndim") or keep.ndim != 2:
            metadata["selection_candidate_valid"] = False
            metadata["selection_candidate_unique"] = False
            return

        candidate_mask = ctx.domain.candidate_mask
        flat = keep.detach().reshape(-1).to(candidate_mask.device).long()
        in_bounds = True
        if flat.numel():
            in_bounds = bool(((flat >= 0) & (flat < candidate_mask.numel())).all().item())
        membership = bool(candidate_mask.index_select(0, flat).all().item()) if in_bounds and flat.numel() else True
        unique = True
        for row in keep.detach():
            if row.unique().numel() != row.numel():
                unique = False
                break
        metadata.update(
            {
                "selection_candidate_valid": bool(in_bounds and membership and unique),
                "selection_candidate_unique": bool(unique),
                "selected_global_count": int(keep.shape[1]),
                "selected_global_min": int(flat.min().item()) if flat.numel() else None,
                "selected_global_max": int(flat.max().item()) if flat.numel() else None,
            }
        )

    def record_result(self, ctx, result) -> CompressionEvent:
        """Journal an audit snapshot without retaining activation tensors."""

        point = getattr(self.press, "injection_point", InjectionPoint.VIDEO_INPUT)
        point = InjectionPoint.parse(point).value
        event_result = replace(
            result,
            output=None,
            # Q/K/V are live operator outputs, not audit artifacts. Retaining
            # them once per layer defeats the memory goal of persistent K/V
            # pruning. The returned result below remains untouched for the
            # active attention call.
            aux={},
        )
        event_context = replace(
            ctx,
            # A zero-width view preserves batch/sequence shape for audit code
            # without allocating storage or synchronizing the CUDA stream.
            tokens=ctx.tokens.new_empty((ctx.batch_size, ctx.layout.total_length, 0)),
            q=None,
            k=None,
            v=None,
            trajectory_pred=None,
        )
        event = CompressionEvent(
            key=CompressionEventKey(
                scene_token=str(ctx.scene_token),
                diffusion_rank=ctx.diffusion_rank,
                layer_idx=ctx.layer_idx,
                injection_point=point,
            ),
            result=event_result,
            context=event_context,
        )
        self.events.append(event)
        self.last_result = result
        return event

    def record_artifact(self, name: str, value: Any) -> None:
        self.artifacts[str(name)] = value

    def apply_video_input(self, **kwargs):
        if self.press is None:
            return kwargs["video_tokens"]
        if self.adapter is None or not hasattr(self.adapter, "apply_video_input"):
            raise RuntimeError("VIDEO_INPUT requires an adapter with apply_video_input()")
        return self.adapter.apply_video_input(self, **kwargs)

    def install(self, pipe) -> None:
        if self._installed:
            return
        if self.press is not None and hasattr(self.press, "prepare"):
            self.press.prepare(self)
        try:
            if self.adapter is not None:
                self.adapter.install_hooks(pipe, self)
            elif hasattr(pipe, "install_tokenpress_runtime"):
                pipe.install_tokenpress_runtime(self)
            else:
                self._previous_runtime = getattr(pipe, "tokenpress_runtime", None)
                pipe.tokenpress_runtime = self
        except Exception:
            if self.adapter is not None:
                self.adapter.remove_hooks(pipe, self)
            raise
        self._installed = True

    def remove(self, pipe) -> None:
        if not self._installed:
            return
        try:
            if self.adapter is not None:
                self.adapter.remove_hooks(pipe, self)
            elif hasattr(pipe, "remove_tokenpress_runtime"):
                pipe.remove_tokenpress_runtime(self)
            elif hasattr(self, "_previous_runtime"):
                pipe.tokenpress_runtime = self._previous_runtime
        finally:
            if self.press is not None and hasattr(self.press, "finalize"):
                self.press.finalize(self)
            self._installed = False

    @contextmanager
    def activate(self, pipe) -> Iterator["VideoPressRuntime"]:
        self.install(pipe)
        try:
            yield self
        finally:
            self.remove(pipe)
