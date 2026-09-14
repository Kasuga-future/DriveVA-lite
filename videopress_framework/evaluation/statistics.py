"""Scene-level aggregation, suite tables and paired bootstrap utilities.

The evaluator deliberately writes one JSON object per scene.  This module is
the small, dependency-light reporting layer on top of those records: it keeps
the raw scene rows intact, adds run-level mean/std statistics, and then rolls
multiple rounds up by method.  The resulting dictionaries are JSON/CSV
friendly and are also consumed by :mod:`evaluation.visualization`.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


def _records_as_dict(records: Iterable) -> list[Mapping]:
    out = []
    for record in records:
        if isinstance(record, Mapping):
            out.append(record)
        else:
            out.append(vars(record))
    return out


def _finite_mean(values: Sequence) -> float:
    numeric_values = []
    for value in values:
        if value is None:
            continue
        try:
            numeric_values.append(float(value))
        except (TypeError, ValueError):
            continue
    numeric = np.asarray(numeric_values, dtype=np.float64)
    numeric = numeric[np.isfinite(numeric)]
    return float(numeric.mean()) if numeric.size else float("nan")


def _finite_std(values: Sequence) -> float:
    numeric_values = []
    for value in values:
        if value is None:
            continue
        try:
            numeric_values.append(float(value))
        except (TypeError, ValueError):
            continue
    numeric = np.asarray(numeric_values, dtype=np.float64)
    numeric = numeric[np.isfinite(numeric)]
    if not numeric.size:
        return float("nan")
    return float(numeric.std(ddof=1)) if numeric.size > 1 else 0.0


def _finite_max(values: Sequence) -> float:
    numeric_values = []
    for value in values:
        if value is None:
            continue
        try:
            numeric_values.append(float(value))
        except (TypeError, ValueError):
            continue
    numeric = np.asarray(numeric_values, dtype=np.float64)
    numeric = numeric[np.isfinite(numeric)]
    return float(numeric.max()) if numeric.size else float("nan")


def _finite_min(values: Sequence) -> float:
    numeric_values = []
    for value in values:
        if value is None:
            continue
        try:
            numeric_values.append(float(value))
        except (TypeError, ValueError):
            continue
    numeric = np.asarray(numeric_values, dtype=np.float64)
    numeric = numeric[np.isfinite(numeric)]
    return float(numeric.min()) if numeric.size else float("nan")


def _records_as_dicts(records: Iterable) -> list[dict[str, Any]]:
    return [dict(row) for row in _records_as_dict(records)]


def _first_value(rows: Sequence[Mapping], key: str, default=None):
    for row in rows:
        value = row.get(key, default)
        if value not in (None, ""):
            return value
    return default


def _numeric(row: Mapping, key: str, fallback: str | None = None):
    value = row.get(key)
    if value is None and fallback is not None:
        value = row.get(fallback)
    return value


def paired_bootstrap(method: Sequence[float], random: Sequence[float], n: int = 20_000, seed: int = 20260828) -> tuple[float, float]:
    method_arr = np.asarray(method, dtype=np.float64)
    random_arr = np.asarray(random, dtype=np.float64)
    if method_arr.shape != random_arr.shape or method_arr.ndim != 1:
        raise ValueError("method and random must be equal-length one-dimensional arrays")
    if method_arr.size == 0:
        raise ValueError("paired bootstrap needs at least one scene")
    delta = method_arr - random_arr
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, delta.size, size=(int(n), delta.size))
    means = delta[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def aggregate_records(records: Iterable) -> dict:
    rows = _records_as_dict(records)
    if not rows:
        return {"n_scenes": 0, "pdm": float("nan"), "trajectory_l2": float("nan"), "endpoint_l2": float("nan")}
    # Failed/invalid scenes remain counted in n_scenes, but their placeholder
    # scores (often zero) must not silently depress quality or timing means.
    metric_rows = [row for row in rows if bool(row.get("valid", True))]
    pdm = _finite_mean([row.get("pdm", float("nan")) for row in metric_rows])
    trajectory_l2 = _finite_mean([row.get("trajectory_l2", float("nan")) for row in metric_rows])
    endpoint_l2 = _finite_mean([row.get("endpoint_l2", float("nan")) for row in metric_rows])
    latency = _finite_mean([row.get("latency_ms", float("nan")) for row in metric_rows])
    selector_latency = _finite_mean([row.get("selector_latency_ms", float("nan")) for row in metric_rows])
    model_latency = _finite_mean([row.get("model_latency_ms", float("nan")) for row in metric_rows])
    e2e_latency = _finite_mean([row.get("e2e_latency_ms", row.get("latency_ms", float("nan"))) for row in metric_rows])
    memory = _finite_mean([row.get("peak_memory_mb", float("nan")) for row in metric_rows])
    eligible_ratio = _finite_mean([row.get("eligible_keep_ratio", float("nan")) for row in metric_rows])
    history_ratio = _finite_mean([row.get("history_keep_ratio", float("nan")) for row in metric_rows])
    effective_history_values = []
    for row in metric_rows:
        value = row.get("effective_history_keep_ratio")
        if value is None:
            value = row.get("metadata", {}).get("effective_history_keep_ratio")
        if isinstance(value, list):
            value = value[0] if value else None
        effective_history_values.append(value)
    effective_history_ratio = _finite_mean(effective_history_values)
    return {
        "n_scenes": len(rows),
        "valid_scenes": sum(bool(row.get("valid", True)) for row in rows),
        "pdm": pdm,
        "pdm_std": _finite_std([row.get("pdm", float("nan")) for row in metric_rows]),
        "trajectory_l2": trajectory_l2,
        "trajectory_l2_std": _finite_std([row.get("trajectory_l2", float("nan")) for row in metric_rows]),
        "endpoint_l2": endpoint_l2,
        "endpoint_l2_std": _finite_std([row.get("endpoint_l2", float("nan")) for row in metric_rows]),
        "latency_ms_mean": latency,
        "latency_ms_std": _finite_std([row.get("latency_ms", float("nan")) for row in metric_rows]),
        "selector_latency_ms_mean": selector_latency,
        "selector_latency_ms_std": _finite_std([row.get("selector_latency_ms", float("nan")) for row in metric_rows]),
        "model_latency_ms_mean": model_latency,
        "model_latency_ms_std": _finite_std([row.get("model_latency_ms", float("nan")) for row in metric_rows]),
        "e2e_latency_ms_mean": e2e_latency,
        "e2e_latency_ms_std": _finite_std(
            [row.get("e2e_latency_ms", row.get("latency_ms", float("nan"))) for row in metric_rows]
        ),
        "peak_memory_mb_mean": memory,
        "peak_memory_mb_std": _finite_std([row.get("peak_memory_mb", float("nan")) for row in metric_rows]),
        "peak_memory_mb_max": _finite_max([row.get("peak_memory_mb", float("nan")) for row in metric_rows]),
        "eligible_keep_ratio_mean": eligible_ratio,
        "eligible_keep_ratio_std": _finite_std([row.get("eligible_keep_ratio", float("nan")) for row in metric_rows]),
        "history_keep_ratio_mean": history_ratio,
        "history_keep_ratio_std": _finite_std([row.get("history_keep_ratio", float("nan")) for row in metric_rows]),
        "effective_history_keep_ratio_mean": effective_history_ratio,
        "effective_history_keep_ratio_std": _finite_std(effective_history_values),
        "K_mean": _finite_mean([row.get("K", float("nan")) for row in metric_rows]),
        "K_std": _finite_std([row.get("K", float("nan")) for row in metric_rows]),
        "dynamic_K_min": _finite_min(
            [row.get("metadata", {}).get("dynamic_n_kept_min", float("nan")) for row in metric_rows]
        ),
        "dynamic_K_max": _finite_max(
            [row.get("metadata", {}).get("dynamic_n_kept_max", float("nan")) for row in metric_rows]
        ),
        "n_candidate_mean": _finite_mean([row.get("n_candidate", float("nan")) for row in metric_rows]),
        "k_length_before_mean": _finite_mean(
            [row.get("metadata", {}).get("k_length_before", float("nan")) for row in metric_rows]
        ),
        "k_length_after_mean": _finite_mean(
            [
                row.get("metadata", {}).get(
                    "k_length_after_mean_across_steps",
                    row.get("metadata", {}).get("k_length_after", float("nan")),
                )
                for row in metric_rows
            ]
        ),
        "v_length_after_mean": _finite_mean(
            [
                row.get("metadata", {}).get(
                    "v_length_after_mean_across_steps",
                    row.get("metadata", {}).get("v_length_after", float("nan")),
                )
                for row in metric_rows
            ]
        ),
        "theoretical_attn_ratio_mean": _finite_mean(
            [
                row.get("metadata", {}).get(
                    "theoretical_attn_ratio_mean_across_steps",
                    row.get("metadata", {}).get("theoretical_attn_ratio", float("nan")),
                )
                for row in metric_rows
            ]
        ),
        "hidden_sequence_ratio_mean": _finite_mean(
            [
                row.get("metadata", {}).get(
                    "hidden_sequence_ratio_mean_across_steps",
                    row.get("metadata", {}).get("hidden_sequence_ratio", float("nan")),
                )
                for row in metric_rows
            ]
        ),
        "hidden_sequence_length_before_mean": _finite_mean(
            [row.get("metadata", {}).get("hidden_sequence_length_before", float("nan")) for row in metric_rows]
        ),
        "hidden_sequence_length_after_mean": _finite_mean(
            [
                row.get("metadata", {}).get(
                    "hidden_sequence_length_after_mean_across_steps",
                    row.get("metadata", {}).get("hidden_sequence_length_after", float("nan")),
                )
                for row in metric_rows
            ]
        ),
        "hidden_sequence_compressed_layer_count_mean": _finite_mean(
            [
                row.get("metadata", {}).get(
                    "hidden_sequence_compressed_layer_count", float("nan")
                )
                for row in metric_rows
            ]
        ),
    }


def scene_level_paired_delta(method_records: Iterable, random_records: Iterable, field: str = "pdm") -> dict:
    method = {row["scene_token"]: float(row[field]) for row in _records_as_dict(method_records)}
    random = {row["scene_token"]: float(row[field]) for row in _records_as_dict(random_records)}
    keys = sorted(set(method) & set(random))
    if not keys:
        raise ValueError("no paired scene records")
    values = np.asarray([method[key] - random[key] for key in keys], dtype=np.float64)
    low, high = paired_bootstrap(
        np.asarray([method[key] for key in keys]),
        np.asarray([random[key] for key in keys]),
    )
    return {"n_scenes": len(keys), "mean_delta": float(values.mean()), "ci95": [low, high]}


def load_records(run_dir: str | Path) -> list[dict[str, Any]]:
    """Read the scene records emitted by one evaluator run."""

    path = Path(run_dir) / "records.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"run has no records.jsonl: {path}")
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in {path}:{line_number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"record in {path}:{line_number} is not an object")
        rows.append(value)
    return rows


def _run_descriptor(run_dir: Path, metadata: Mapping | None = None) -> dict[str, Any]:
    metadata = dict(metadata or {})
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    press = summary.get("press", {})
    if not isinstance(press, Mapping):
        press = {}
    return {
        "method": metadata.get("method") or metadata.get("label") or press.get("name") or run_dir.name,
        "round": metadata.get("round"),
        "protocol": metadata.get("protocol") or summary.get("mode"),
        "backend": metadata.get("backend") or summary.get("backend"),
        "run_dir": str(run_dir),
        "press": press,
        "summary": summary,
    }


def summarize_run(run_dir: str | Path, metadata: Mapping | None = None) -> dict[str, Any]:
    """Return one flat statistics row for a run directory."""

    run_path = Path(run_dir)
    rows = load_records(run_path)
    descriptor = _run_descriptor(run_path, metadata)
    press = descriptor["press"]
    persistence = press.get("cross_layer_persistence", {}) if isinstance(press, Mapping) else {}
    if not isinstance(persistence, Mapping):
        persistence = {}
    first_record = rows[0] if rows else {}
    scorer = press.get("scorer", {}) if isinstance(press, Mapping) else {}
    selector = press.get("selector", {}) if isinstance(press, Mapping) else {}
    operator = press.get("operator", {}) if isinstance(press, Mapping) else {}
    if not isinstance(scorer, Mapping):
        scorer = {"name": scorer}
    if not isinstance(selector, Mapping):
        selector = {"name": selector}
    if not isinstance(operator, Mapping):
        operator = {"name": operator}
    aggregate = aggregate_records(rows)
    row = {
        "method": descriptor["method"],
        "round": descriptor["round"],
        "protocol": descriptor["protocol"],
        "backend": descriptor["backend"],
        "press": press.get("name") or first_record.get("press_name"),
        "scorer": scorer.get("name") or first_record.get("scorer"),
        "selector": selector.get("name") or first_record.get("selector"),
        "operator": operator.get("name") or first_record.get("operator"),
        "domain": press.get("domain") or first_record.get("domain"),
        "retention_policy": press.get("retention_policy") or first_record.get("retention_policy"),
        "persistence_mode": persistence.get("mode"),
        # Recorded so the intervention fingerprint can see it: two arms can share
        # a method name while differing only in whether cross-layer persistence
        # stops early (audit 2026-09-12, BUG-10 / BUG-13).
        "end_layer": persistence.get("end_layer"),
        "persistence_enabled": persistence.get("enabled"),
        # Keep the complete resolved intervention so selector thresholds,
        # scorer options and budget values participate in suite identity.
        "intervention_config": press,
        "run_dir": descriptor["run_dir"],
        **aggregate,
    }
    return row


def _intervention_fingerprint(row: Mapping) -> str:
    """Stable identity of WHAT an arm did, independent of its display name.

    ``aggregate_suite`` pools scene records by ``method`` name.  That name is
    built from scorer/mode/start-layer only, so two genuinely different arms
    (e.g. same scorer+layer but a different ``end_layer``, ``domain``, budget or
    retention policy) can collide.  This digest covers the fields that change the
    intervention, so a collision can be detected instead of silently merged.
    """

    payload = json.dumps(
        {
            "intervention_config": row.get("intervention_config"),
            "protocol": row.get("protocol"),
            "backend": row.get("backend"),
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _manifest_and_runs(manifest_or_root: str | Path) -> tuple[Path, list[dict[str, Any]]]:
    path = Path(manifest_or_root)
    if path.is_dir():
        path = path / "suite_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"suite manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping) or not isinstance(manifest.get("runs"), list):
        raise ValueError(f"suite manifest must contain a runs list: {path}")
    root = path.parent
    return root, [dict(item) for item in manifest["runs"]]


def aggregate_suite(manifest_or_root: str | Path) -> dict[str, Any]:
    """Aggregate all rounds listed in ``suite_manifest.json``.

    The return value has both ``run_rows`` (one row per method/round) and
    ``method_rows`` (all scene records pooled across rounds).  Pooling scene
    records rather than pooling run means gives each scene equal weight.
    """

    root, manifest_runs = _manifest_and_runs(manifest_or_root)
    run_rows = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    group_meta: dict[str, dict[str, Any]] = {}
    group_fingerprints: dict[str, str] = {}
    for item in manifest_runs:
        relative = item.get("output_dir") or item.get("run_dir")
        if relative is None:
            raise ValueError("each suite run needs output_dir")
        run_dir = Path(relative)
        if not run_dir.is_absolute():
            run_dir = root / run_dir
        metadata = {
            "method": item.get("method") or item.get("label"),
            "round": item.get("round"),
            "protocol": item.get("protocol"),
            "backend": item.get("backend"),
        }
        run_row = summarize_run(run_dir, metadata)
        run_rows.append(run_row)
        method = str(run_row["method"])
        # A method NAME is not a unique key for the intervention: it is built from
        # scorer/mode/layer only, so arms differing in `end_layer`, `domain`,
        # `budget` or `retention_policy` can share one name (audit 2026-09-12,
        # BUG-10).  Pooling those silently merges two different experiments and
        # then reports one arm's descriptor beside the other's records.
        fingerprint = _intervention_fingerprint(run_row)
        previous = group_fingerprints.get(method)
        if previous is not None and previous != fingerprint:
            raise ValueError(
                f"method {method!r} is used by two DIFFERENT interventions in this "
                f"suite (fingerprints {previous} vs {fingerprint}); refusing to pool "
                "them by name. Rename one arm or split the suite."
            )
        group_fingerprints[method] = fingerprint
        grouped.setdefault(method, []).extend(load_records(run_dir))
        group_meta.setdefault(method, run_row)

    method_rows = []
    for method in sorted(grouped):
        exemplar = group_meta[method]
        pooled = aggregate_records(grouped[method])
        row = {
            "method": method,
            "round": "all",
            "protocol": exemplar.get("protocol"),
            "backend": exemplar.get("backend"),
            "press": exemplar.get("press"),
            "scorer": exemplar.get("scorer"),
            "selector": exemplar.get("selector"),
            "operator": exemplar.get("operator"),
            "domain": exemplar.get("domain"),
            "retention_policy": exemplar.get("retention_policy"),
            "persistence_mode": exemplar.get("persistence_mode"),
            "end_layer": exemplar.get("end_layer"),
            "persistence_enabled": exemplar.get("persistence_enabled"),
            "intervention_config": exemplar.get("intervention_config"),
            "rounds": len({row.get("round") for row in run_rows if row.get("method") == method}),
            "run_dir": str(root),
            **pooled,
        }
        method_rows.append(row)
    return {
        "suite": json.loads((root / "suite_manifest.json").read_text(encoding="utf-8")),
        "run_rows": run_rows,
        "method_rows": method_rows,
    }


TABLE_COLUMNS = [
    "method",
    "round",
    "rounds",
    "protocol",
    "backend",
    "press",
    "scorer",
    "selector",
    "operator",
    "domain",
    "retention_policy",
    "persistence_mode",
    "n_scenes",
    "valid_scenes",
    "pdm",
    "pdm_std",
    "trajectory_l2",
    "trajectory_l2_std",
    "endpoint_l2",
    "endpoint_l2_std",
    "K_mean",
    "K_std",
    "dynamic_K_min",
    "dynamic_K_max",
    "n_candidate_mean",
    "eligible_keep_ratio_mean",
    "history_keep_ratio_mean",
    "effective_history_keep_ratio_mean",
    "theoretical_attn_ratio_mean",
    "hidden_sequence_ratio_mean",
    "hidden_sequence_length_before_mean",
    "hidden_sequence_length_after_mean",
    "hidden_sequence_compressed_layer_count_mean",
    "k_length_before_mean",
    "k_length_after_mean",
    "v_length_after_mean",
    "selector_latency_ms_mean",
    "selector_latency_ms_std",
    "model_latency_ms_mean",
    "model_latency_ms_std",
    "e2e_latency_ms_mean",
    "e2e_latency_ms_std",
    "peak_memory_mb_mean",
    "peak_memory_mb_std",
    "peak_memory_mb_max",
    "run_dir",
]


def _table_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and not np.isfinite(value):
        return ""
    return value


def _write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _table_value(row.get(column)) for column in TABLE_COLUMNS})


def _markdown_cell(value: Any) -> str:
    value = _table_value(value)
    if value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|")


def _write_markdown(path: Path, title: str, rows: Sequence[Mapping]) -> None:
    columns = [column for column in TABLE_COLUMNS if any(column in row for row in rows)]
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"# {title}\n\n")
        if not rows:
            handle.write("No rows.\n")
            return
        handle.write("| " + " | ".join(columns) + " |\n")
        handle.write("| " + " | ".join("---" for _ in columns) + " |\n")
        for row in rows:
            handle.write("| " + " | ".join(_markdown_cell(row.get(column)) for column in columns) + " |\n")


def write_suite_tables(summary: Mapping, output_dir: str | Path) -> dict[str, str]:
    """Write machine-readable and human-readable suite statistics tables."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    run_rows = list(summary.get("run_rows", []))
    method_rows = list(summary.get("method_rows", []))
    paths = {
        "run_csv": output / "run_statistics.csv",
        "method_csv": output / "method_statistics.csv",
        "run_markdown": output / "run_statistics.md",
        "method_markdown": output / "method_statistics.md",
    }
    _write_csv(paths["run_csv"], run_rows)
    _write_csv(paths["method_csv"], method_rows)
    _write_markdown(paths["run_markdown"], "Per-round VideoTokenPress statistics", run_rows)
    _write_markdown(paths["method_markdown"], "Pooled VideoTokenPress statistics", method_rows)
    return {key: str(value) for key, value in paths.items()}
