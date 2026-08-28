"""Scene-order checks and an explicit official NAVSIM backend boundary."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class SceneOrderKey:
    log_id: str
    timestamp: int
    scene_token: str


def validate_scene_order(samples: Iterable) -> list:
    """Return samples in real log/timestamp order and reject duplicates."""

    ordered = sorted(samples, key=lambda sample: (sample.log_id, sample.timestamp, sample.scene_token))
    seen = set()
    for sample in ordered:
        key = (sample.log_id, sample.timestamp, sample.scene_token)
        if key in seen:
            raise ValueError(f"duplicate scene/timestamp sample: {key}")
        seen.add(key)
    return ordered


class DriveVANavsimBackend:
    """Run DriveVA inference and official NAVSIM PDM through one backend.

    The callbacks have intentionally explicit signatures:

    ``feature_builder(sample)``
        Builds the official DriveVA/NAVSIM model input.
    ``inference_fn(pipe, features, sample, runtime)``
        Runs the pipeline.  The backend activates the runtime around it.
    ``metric_fn(prediction, sample)``
        Optional official metric callback.  If omitted, the backend calls the
        vendored ``navsim.evaluate.pdm_score.pdm_score`` implementation and
        requires all of its components.

    There is no synthetic fallback in this class.  The lightweight synthetic
    evaluator remains a separate backend in ``evaluation.Evaluator``.
    """

    name = "official_navsim"

    def __init__(
        self,
        *,
        pipe,
        samples: Iterable,
        layout,
        feature_builder: Callable,
        inference_fn: Callable,
        metric_fn: Callable | None = None,
        metric_cache_loader=None,
        simulator=None,
        scorer=None,
        future_sampling=None,
        trajectory_converter: Callable | None = None,
        adapter=None,
        device=None,
        require_official: bool = True,
    ):
        if feature_builder is None:
            raise ValueError("official NAVSIM backend requires feature_builder")
        if inference_fn is None:
            raise ValueError("official NAVSIM backend requires inference_fn")
        if require_official and metric_fn is None:
            missing = [
                name
                for name, value in {
                    "metric_cache_loader": metric_cache_loader,
                    "simulator": simulator,
                    "scorer": scorer,
                    "future_sampling": future_sampling,
                }.items()
                if value is None
            ]
            if missing:
                raise ValueError(
                    "official NAVSIM PDM needs " + ", ".join(missing) + "; no synthetic fallback is allowed"
                )
        self.pipe = pipe
        self.samples = samples
        self.layout = layout
        self.feature_builder = feature_builder
        self.inference_fn = inference_fn
        self.metric_fn = metric_fn
        self.metric_cache_loader = metric_cache_loader
        self.simulator = simulator
        self.scorer = scorer
        self.future_sampling = future_sampling
        self.trajectory_converter = trajectory_converter
        self.adapter = adapter
        self.device = device or getattr(pipe, "device", None)
        if self.device is None:
            for model_name in ("dit", "dit2", "trajectory_encoder"):
                model = getattr(pipe, model_name, None)
                if model is None:
                    continue
                try:
                    self.device = next(model.parameters()).device
                    break
                except (AttributeError, StopIteration, TypeError):
                    continue
        self.require_official = bool(require_official)

    def iter_samples(self) -> list:
        values = self.samples() if callable(self.samples) else self.samples
        return validate_scene_order(list(values))

    def predict(self, sample, runtime):
        features = self.feature_builder(sample)
        with runtime.activate(self.pipe):
            return self.inference_fn(self.pipe, features, sample, runtime)

    def evaluate(self, prediction, sample) -> dict[str, Any]:
        if self.metric_fn is not None:
            metrics = self.metric_fn(prediction, sample)
            if not isinstance(metrics, dict):
                raise TypeError("official NAVSIM metric_fn must return a dictionary")
            return dict(metrics)

        from navsim.evaluate.pdm_score import pdm_score

        metric_cache = self.metric_cache_loader.get_from_token(sample.scene_token)
        trajectory = prediction
        if self.trajectory_converter is not None:
            trajectory = self.trajectory_converter(prediction, sample)
        result = pdm_score(
            metric_cache=metric_cache,
            model_trajectory=trajectory,
            future_sampling=self.future_sampling,
            simulator=self.simulator,
            scorer=self.scorer,
        )
        values = asdict(result) if is_dataclass(result) else {
            key: getattr(result, key)
            for key in dir(result)
            if not key.startswith("_") and not callable(getattr(result, key, None))
        }
        if "pdm" not in values:
            values["pdm"] = values.get("score")
        values["valid"] = bool(values.get("valid", True)) and values.get("pdm") is not None
        return values
