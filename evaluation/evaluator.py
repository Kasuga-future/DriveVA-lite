"""Model-agnostic and backend-aware evaluation for VideoTokenPress."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import torch

from videopress.core.context import TokenContext
from videopress.core.budget import budget_stats
from videopress.core.domain import build_domain
from videopress.core.plan import build_execution_plan, validate_protocol
from videopress.core.runtime import EvaluationMode, VideoPressRuntime
from videopress.presses import NoPress, ScorerPress
from videopress.probes import ScoreCache
from videopress.scorers import RandomScorer

from .artifacts import ArtifactWriter, jsonable
from .efficiency import SynchronizedTimer
from .navsim_evaluator import validate_scene_order
from .statistics import aggregate_records, scene_level_paired_delta


@dataclass
class SceneSample:
    scene_token: str
    log_id: str
    timestamp: int
    tokens: torch.Tensor
    q: Optional[torch.Tensor] = None
    k: Optional[torch.Tensor] = None
    v: Optional[torch.Tensor] = None
    target_trajectory: Optional[torch.Tensor] = None
    frame_token: Optional[str] = None
    diffusion_rank: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalRecord:
    scene_token: str
    log_id: str
    press_name: str
    scorer: Optional[str]
    selector: Optional[str]
    operator: str
    domain: str
    K: int
    n_candidate: int
    n_history: int
    eligible_keep_ratio: Optional[float]
    history_keep_ratio: Optional[float]
    pdm: float
    trajectory_l2: Optional[float]
    endpoint_l2: Optional[float]
    latency_ms: float
    peak_memory_mb: float
    valid: bool
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    selector_latency_ms: float = float("nan")
    model_latency_ms: float = float("nan")
    e2e_latency_ms: float = float("nan")


def _metric_value(metrics: dict, key: str, default=float("nan")):
    value = metrics.get(key, default)
    return float(value) if value is not None else None


def _domain_label(spec: Any) -> str:
    if isinstance(spec, dict):
        spec = spec.get("name", "last_history")
    return str(getattr(spec, "name", spec))


def build_matched_random_press(method_press: ScorerPress, seed: int) -> ScorerPress:
    """Create an equal-budget control differing only in its scorer."""

    if not isinstance(method_press, ScorerPress):
        raise NotImplementedError("matched Random is currently defined for ScorerPress only")
    method_scorer = method_press.scorer
    return ScorerPress(
        scorer=RandomScorer(
            seed=seed,
            scope=(getattr(method_press, "random_scope", None) or getattr(method_scorer, "scope", "scene")),
            layer=getattr(method_scorer, "layer", None),
        ),
        selector=deepcopy(method_press.selector),
        operator=deepcopy(method_press.operator),
        budget=deepcopy(method_press.budget),
        domain=deepcopy(method_press.domain),
        injection_point=method_press.injection_point,
        random_scope=getattr(method_press, "random_scope", None),
    )


class Evaluator:
    def __init__(self, repo_root: str | Path | None = None):
        self.repo_root = Path(repo_root).resolve() if repo_root is not None else Path.cwd().resolve()

    def evaluate(
        self,
        samples: Iterable[SceneSample],
        press,
        layout,
        *,
        domain=None,
        mode: EvaluationMode | str = EvaluationMode.CAUSAL,
        output_dir: str | Path = "outputs/press_run",
        config: Optional[dict] = None,
        predict_fn: Optional[Callable] = None,
        metric_fn: Optional[Callable] = None,
        max_scenes: Optional[int] = None,
        fail_fast: bool = True,
        adapter=None,
        allow_sample_domain_override: bool = False,
    ) -> dict:
        """Evaluate a synthetic/model callback cohort.

        ``predict_fn`` is called as ``predict_fn(result, sample, ctx)``.  Its
        measured time is the model callback time; press and end-to-end timings
        are reported separately.  Real DriveVA/NAVSIM execution uses
        :meth:`evaluate_backend` below so its pipeline forward is measured in
        the same way for Full and compressed runs.
        """

        if press is None:
            press = NoPress()
        eval_mode = EvaluationMode.parse(mode)
        validate_protocol(press, eval_mode)
        ordered = validate_scene_order(list(samples))
        if max_scenes is not None:
            ordered = ordered[: int(max_scenes)]
        self._validate_cohort(ordered)
        output_path = Path(output_dir)
        writer = ArtifactWriter(output_path)
        config_snapshot = dict(config or {"press": press.describe(), "mode": eval_mode.value})
        writer.write_config(config_snapshot)
        writer.write_environment(self.repo_root)
        evaluation_config = config_snapshot.get("evaluation", {}) if isinstance(config_snapshot, dict) else {}
        cache_root = output_path / "artifacts" / "score_cache"
        runtime = VideoPressRuntime(
            press=press,
            mode=eval_mode,
            adapter=adapter,
            score_cache=ScoreCache(cache_root),
            allow_sample_domain_override=allow_sample_domain_override
            or bool(evaluation_config.get("allow_sample_domain_override", False)),
        )
        records = self._evaluate_ordered(
            ordered,
            press,
            layout,
            domain=domain,
            mode=eval_mode,
            writer=writer,
            runtime=runtime,
            predict_fn=predict_fn,
            metric_fn=metric_fn,
            fail_fast=fail_fast,
        )
        summary = aggregate_records(records)
        benchmark_config = config_snapshot.get("benchmark", {}) if isinstance(config_snapshot, dict) else {}
        backend_name = evaluation_config.get("backend") or benchmark_config.get("name") or "callback"
        summary.update({"press": press.describe(), "mode": eval_mode.value, "backend": str(backend_name)})
        writer.write_records(records)
        writer.write_tokens()

        random_result = None
        if bool(evaluation_config.get("random_baseline", False)):
            seeds = evaluation_config.get("random_seeds", [0])
            if not isinstance(press, ScorerPress):
                raise NotImplementedError("random_baseline requires a ScorerPress method")
            random_records = []
            random_summaries = []
            for seed in seeds:
                seed = int(seed)
                random_press = build_matched_random_press(press, seed)
                random_output = output_path / "random_baseline" / f"seed_{seed}"
                random_writer = ArtifactWriter(random_output)
                random_config = deepcopy(config_snapshot)
                random_config.setdefault("evaluation", {})["random_baseline"] = False
                random_config["press"] = random_press.describe()
                random_writer.write_config(random_config)
                random_writer.write_environment(self.repo_root)
                random_runtime = VideoPressRuntime(
                    press=random_press,
                    mode=eval_mode,
                    adapter=adapter,
                    score_cache=ScoreCache(random_output / "artifacts" / "score_cache"),
                    allow_sample_domain_override=runtime.allow_sample_domain_override,
                )
                current_records = self._evaluate_ordered(
                    ordered,
                    random_press,
                    layout,
                    domain=domain,
                    mode=eval_mode,
                    writer=random_writer,
                    runtime=random_runtime,
                    predict_fn=predict_fn,
                    metric_fn=metric_fn,
                    fail_fast=fail_fast,
                )
                random_writer.write_records(current_records)
                random_writer.write_tokens()
                random_records.append(current_records)
                random_summaries.append({"seed": seed, **aggregate_records(current_records)})
            random_result = self._random_comparison(records, random_records, random_summaries)
            summary["random_baseline"] = random_result
        writer.write_summary(summary)
        return {
            "records": records,
            "summary": summary,
            "output_dir": str(output_path.resolve()),
            "random_baseline": random_result,
        }

    def evaluate_backend(
        self,
        backend,
        press,
        *,
        mode: EvaluationMode | str = EvaluationMode.CAUSAL,
        output_dir: str | Path = "outputs/press_backend_run",
        config: Optional[dict] = None,
        fail_fast: bool = True,
    ) -> dict:
        """Evaluate a real backend whose prediction owns the model forward.

        ``DriveVANavsimBackend`` activates the runtime around the official
        pipeline.  Thus the model timer includes the actual DriveVA forward,
        while the event list contains every layer/step intervention instead of
        only the last one.
        """

        if press is None:
            press = NoPress()
        eval_mode = EvaluationMode.parse(mode)
        validate_protocol(press, eval_mode)
        ordered = validate_scene_order(backend.iter_samples())
        output_path = Path(output_dir)
        writer = ArtifactWriter(output_path)
        config_snapshot = dict(config or {"press": press.describe(), "mode": eval_mode.value})
        writer.write_config(config_snapshot)
        writer.write_environment(self.repo_root)
        evaluation_config = config_snapshot.get("evaluation", {}) if isinstance(config_snapshot, dict) else {}
        runtime = VideoPressRuntime(
            press=press,
            mode=eval_mode,
            adapter=getattr(backend, "adapter", None),
            score_cache=ScoreCache(output_path / "artifacts" / "score_cache"),
            allow_sample_domain_override=bool(evaluation_config.get("allow_sample_domain_override", False)),
        )
        records: list[EvalRecord] = []
        for sample in ordered:
            runtime.begin_sample(sample, getattr(backend, "layout", None))
            try:
                device = getattr(getattr(sample, "tokens", None), "device", None)
                if device is None:
                    device = getattr(backend, "device", None)
                e2e_timer = SynchronizedTimer(device)
                model_timer = SynchronizedTimer(device)
                e2e_timer.start()
                model_timer.start(reset_memory=False)
                prediction = backend.predict(sample, runtime)
                model_timing = model_timer.stop()
                e2e_timing = e2e_timer.stop()
                metrics = backend.evaluate(prediction, sample)
                if not isinstance(metrics, dict):
                    raise TypeError("backend.evaluate() must return a dictionary")
                event_result = runtime.last_result
                result_metadata = dict(event_result.metadata) if event_result is not None else {}
                if press.name not in {"noop", "none", "full"} and not runtime.events:
                    raise RuntimeError(
                        "backend forward completed without a VideoTokenPress event; "
                        "the configured injection point was not installed"
                    )
                if event_result is None and getattr(backend, "layout", None) is not None:
                    sample_tokens = getattr(sample, "tokens", None)
                    device_for_domain = getattr(sample_tokens, "device", "cpu")
                    base_spec = getattr(press, "domain", None) or "last_history"
                    if isinstance(base_spec, dict):
                        base_spec = base_spec.get("name", "last_history")
                    base_domain = build_domain(base_spec, backend.layout, device_for_domain)
                    result_metadata.update(
                        {
                            **budget_stats(backend.layout, base_domain, base_domain.n_candidate),
                            "configured_domain": _domain_label(getattr(press, "domain", None) or "last_history"),
                            "resolved_domain": base_domain.name,
                            "n_candidate": base_domain.n_candidate,
                        }
                    )
                domain_name = result_metadata.get("resolved_domain", getattr(press, "domain", "unknown"))
                n_candidate = int(result_metadata.get("n_candidate", 0))
                k = int(result_metadata.get("n_kept", result_metadata.get("selected_count", n_candidate)))
                if event_result is not None and event_result.selection is not None:
                    k = int(event_result.selection.K)
                metadata = {
                    **result_metadata,
                    "execution_plan": asdict(build_execution_plan(press, eval_mode)),
                    "timing": {
                        "selector_latency_ms": runtime.selector_latency_ms,
                        "model_latency_ms": model_timing.latency_ms,
                        "e2e_latency_ms": e2e_timing.latency_ms,
                        "peak_memory_mb": e2e_timing.peak_memory_mb,
                    },
                    "event_count": len(runtime.events),
                    "backend": getattr(backend, "name", type(backend).__name__),
                }
                records.append(
                    EvalRecord(
                        scene_token=sample.scene_token,
                        log_id=sample.log_id,
                        press_name=press.name,
                        scorer=getattr(getattr(press, "scorer", None), "name", None),
                        selector=getattr(getattr(press, "selector", None), "name", None),
                        operator=metadata.get("operator", "identity"),
                        domain=str(domain_name),
                        K=k,
                        n_candidate=n_candidate,
                        n_history=int(metadata.get("n_history", 0)),
                        eligible_keep_ratio=metadata.get("eligible_keep_ratio"),
                        history_keep_ratio=metadata.get("history_keep_ratio"),
                        pdm=_metric_value(metrics, "pdm", _metric_value(metrics, "score")),
                        trajectory_l2=_metric_value(metrics, "trajectory_l2"),
                        endpoint_l2=_metric_value(metrics, "endpoint_l2"),
                        latency_ms=e2e_timing.latency_ms,
                        peak_memory_mb=e2e_timing.peak_memory_mb,
                        valid=bool(metrics.get("valid", True)),
                        metadata=jsonable(metadata),
                        selector_latency_ms=runtime.selector_latency_ms,
                        model_latency_ms=model_timing.latency_ms,
                        e2e_latency_ms=e2e_timing.latency_ms,
                    )
                )
                for event in runtime.events:
                    if event.context is not None:
                        writer.add_result(event.context, event.result)
                writer.add_runtime_events(runtime)
            except Exception as exc:
                if fail_fast:
                    raise
                records.append(
                    EvalRecord(
                        scene_token=sample.scene_token,
                        log_id=sample.log_id,
                        press_name=getattr(press, "name", type(press).__name__),
                        scorer=None,
                        selector=None,
                        operator="error",
                        domain="unknown",
                        K=0,
                        n_candidate=0,
                        n_history=0,
                        eligible_keep_ratio=None,
                        history_keep_ratio=None,
                        pdm=float("nan"),
                        trajectory_l2=None,
                        endpoint_l2=None,
                        latency_ms=float("nan"),
                        peak_memory_mb=float("nan"),
                        valid=False,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
        summary = aggregate_records(records)
        summary.update({"press": press.describe(), "mode": eval_mode.value, "backend": getattr(backend, "name", type(backend).__name__)})
        writer.write_records(records)
        writer.write_tokens()
        if bool(evaluation_config.get("random_baseline", False)):
            if not isinstance(press, ScorerPress):
                raise NotImplementedError("random_baseline requires a ScorerPress method")
            random_records = []
            random_summaries = []
            for seed in evaluation_config.get("random_seeds", [0]):
                seed = int(seed)
                random_press = build_matched_random_press(press, seed)
                child_config = deepcopy(config_snapshot)
                child_config.setdefault("evaluation", {})["random_baseline"] = False
                child_config["press"] = random_press.describe()
                child = self.evaluate_backend(
                    backend,
                    random_press,
                    mode=eval_mode,
                    output_dir=output_path / "random_baseline" / f"seed_{seed}",
                    config=child_config,
                    fail_fast=fail_fast,
                )
                random_records.append(child["records"])
                random_summaries.append({"seed": seed, **aggregate_records(child["records"])})
            summary["random_baseline"] = self._random_comparison(records, random_records, random_summaries)
        writer.write_summary(summary)
        return {
            "records": records,
            "summary": summary,
            "output_dir": str(output_path.resolve()),
            "random_baseline": summary.get("random_baseline"),
        }

    def _evaluate_ordered(
        self,
        ordered,
        press,
        layout,
        *,
        domain,
        mode,
        writer,
        runtime,
        predict_fn,
        metric_fn,
        fail_fast,
    ):
        records: list[EvalRecord] = []
        for sample in ordered:
            runtime.begin_sample(sample, layout)
            try:
                configured_spec = getattr(press, "domain", None) or "last_history"
                resolved_spec = domain if domain is not None else configured_spec
                if domain is None and runtime.allow_sample_domain_override:
                    sample_metadata = getattr(sample, "metadata", {})
                    if isinstance(sample_metadata, dict) and sample_metadata.get("domain") is not None:
                        resolved_spec = sample_metadata["domain"]
                if isinstance(resolved_spec, dict):
                    resolved_spec = resolved_spec.get("name", "last_history")
                if hasattr(resolved_spec, "candidate_indices"):
                    domain_obj = resolved_spec
                    if domain_obj.total_length != layout.total_length:
                        raise ValueError("explicit TokenDomain does not match the evaluation layout")
                elif hasattr(resolved_spec, "build"):
                    domain_obj = resolved_spec.build(layout, sample.tokens.device)
                else:
                    domain_obj = build_domain(resolved_spec, layout, sample.tokens.device)
                configured_label = _domain_label(configured_spec)
                resolved_label = domain_obj.name
                context_metadata = dict(sample.metadata)
                context_metadata.setdefault("scene_tokens", [sample.scene_token] * sample.tokens.shape[0])
                context_metadata.setdefault("target_trajectory", sample.target_trajectory)
                context_metadata.update(
                    {
                        "configured_domain": configured_label,
                        "resolved_domain": resolved_label,
                        "domain_override": bool(configured_label != resolved_label),
                        "candidate_start": int(domain_obj.candidate_indices.min().item())
                        if domain_obj.n_candidate
                        else None,
                        "candidate_end": int(domain_obj.candidate_indices.max().item()) + 1
                        if domain_obj.n_candidate
                        else None,
                        "n_candidate": domain_obj.n_candidate,
                    }
                )
                ctx = TokenContext(
                    tokens=sample.tokens,
                    layout=layout,
                    domain=domain_obj,
                    scene_token=sample.scene_token,
                    frame_token=sample.frame_token,
                    log_id=sample.log_id,
                    timestamp=sample.timestamp,
                    diffusion_rank=sample.diffusion_rank,
                    q=sample.q,
                    k=sample.k,
                    v=sample.v,
                    metadata=context_metadata,
                )
                runtime.set_context(ctx)
                plan = build_execution_plan(press, mode)

                e2e_timer = SynchronizedTimer(sample.tokens.device)
                model_timer = SynchronizedTimer(sample.tokens.device)
                e2e_timer.start()
                result = runtime.execute_press(ctx)
                model_timer.start(reset_memory=False)
                prediction = self._call_predict(predict_fn, result, sample, ctx)
                model_timing = model_timer.stop()
                e2e_timing = e2e_timer.stop()
                metrics = self._call_metric(metric_fn, prediction, sample, ctx, result)
                if not isinstance(metrics, dict):
                    raise TypeError("metric_fn must return a dictionary")
                metadata = {
                    **result.metadata,
                    **context_metadata,
                    "execution_plan": asdict(plan),
                    "timing": {
                        "selector_latency_ms": runtime.selector_latency_ms,
                        "model_latency_ms": model_timing.latency_ms,
                        "e2e_latency_ms": e2e_timing.latency_ms,
                        "peak_memory_mb": e2e_timing.peak_memory_mb,
                    },
                    "event_count": len(runtime.events),
                }
                selection = result.selection
                k = selection.K if selection is not None else int(
                    result.metadata.get("n_kept", domain_obj.n_candidate)
                )
                records.append(
                    EvalRecord(
                        scene_token=sample.scene_token,
                        log_id=sample.log_id,
                        press_name=press.name,
                        scorer=getattr(getattr(press, "scorer", None), "name", None),
                        selector=getattr(getattr(press, "selector", None), "name", None),
                        operator=metadata.get("operator", "identity"),
                        domain=resolved_label,
                        K=int(k),
                        n_candidate=domain_obj.n_candidate,
                        n_history=int(result.metadata.get("n_history", layout.history_video.length)),
                        eligible_keep_ratio=result.metadata.get("eligible_keep_ratio"),
                        history_keep_ratio=result.metadata.get("history_keep_ratio"),
                        pdm=_metric_value(metrics, "pdm"),
                        trajectory_l2=_metric_value(metrics, "trajectory_l2"),
                        endpoint_l2=_metric_value(metrics, "endpoint_l2"),
                        latency_ms=e2e_timing.latency_ms,
                        peak_memory_mb=e2e_timing.peak_memory_mb,
                        valid=bool(metrics.get("valid", True)),
                        metadata=jsonable(metadata),
                        selector_latency_ms=runtime.selector_latency_ms,
                        model_latency_ms=model_timing.latency_ms,
                        e2e_latency_ms=e2e_timing.latency_ms,
                    )
                )
                writer.add_result(ctx, result)
                writer.add_runtime_events(runtime)
            except Exception as exc:
                if fail_fast:
                    raise
                records.append(
                    EvalRecord(
                        scene_token=sample.scene_token,
                        log_id=sample.log_id,
                        press_name=getattr(press, "name", type(press).__name__),
                        scorer=None,
                        selector=None,
                        operator="error",
                        domain="unknown",
                        K=0,
                        n_candidate=0,
                        n_history=0,
                        eligible_keep_ratio=None,
                        history_keep_ratio=None,
                        pdm=float("nan"),
                        trajectory_l2=None,
                        endpoint_l2=None,
                        latency_ms=float("nan"),
                        peak_memory_mb=float("nan"),
                        valid=False,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
        return records

    @staticmethod
    def _random_comparison(method_records, random_records, random_summaries):
        if not random_records:
            return {"seeds": [], "runs": [], "paired": None}
        scene_values: dict[str, list[float]] = {}
        for records in random_records:
            for record in records:
                value = float(record.pdm)
                if math.isfinite(value):
                    scene_values.setdefault(record.scene_token, []).append(value)
        averaged = [
            {"scene_token": scene, "pdm": sum(values) / len(values)}
            for scene, values in scene_values.items()
            if values
        ]
        method = [record for record in method_records if math.isfinite(float(record.pdm))]
        paired = None
        if method and averaged:
            paired = scene_level_paired_delta(method, averaged, field="pdm")
        return {
            "seeds": [item["seed"] for item in random_summaries],
            "runs": random_summaries,
            "paired": paired,
        }

    @staticmethod
    def _validate_cohort(samples: list[SceneSample]) -> None:
        seen = set()
        for sample in samples:
            key = (sample.log_id, sample.timestamp, sample.scene_token)
            if key in seen:
                raise ValueError(f"duplicate scene sample: {key}")
            seen.add(key)
            if sample.tokens.ndim != 3:
                raise ValueError(f"sample {sample.scene_token} tokens must be [B,N,D]")

    @staticmethod
    def _call_predict(fn, result, sample, ctx):
        if fn is None:
            return result.output
        return fn(result, sample, ctx)

    @staticmethod
    def _call_metric(fn, prediction, sample, ctx, result):
        if fn is None:
            return {"pdm": float("nan"), "valid": True}
        return fn(prediction, sample, ctx, result)
