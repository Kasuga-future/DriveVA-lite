"""Static plots for a multi-round VideoTokenPress suite.

The plotting code intentionally accepts the plain dictionaries returned by
``statistics.aggregate_suite``.  It therefore works on a completed output
directory without importing a model or loading any tensors.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence


def _number(row: Mapping, key: str, default=float("nan")) -> float:
    value = row.get(key, default)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _short_label(value: Any, limit: int = 24) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _save(fig, path: Path) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return str(path)


def _method_order(rows: Sequence[Mapping]) -> list[Mapping]:
    # Keep protocol and method names stable across repeated runs.  Baselines
    # naturally sort first, which makes the plots easy to compare in reports.
    return sorted(rows, key=lambda row: (str(row.get("protocol", "")), str(row.get("method", ""))))


def generate_suite_visualizations(summary: Mapping, output_dir: str | Path) -> dict[str, str]:
    """Generate PNG plots and a compact visual summary table.

    Matplotlib is imported lazily so the evaluator itself remains usable in a
    minimal production environment.  A clear error is raised only when this
    reporting function is explicitly requested.
    """

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:  # pragma: no cover - exercised only without the optional dependency
        raise RuntimeError(
            "suite visualization requires matplotlib; install it in the evaluation environment"
        ) from exc

    rows = _method_order(list(summary.get("method_rows", [])))
    run_rows = list(summary.get("run_rows", []))
    if not rows:
        return {}
    labels = [_short_label(row.get("method", "unknown")) for row in rows]
    x = np.arange(len(rows), dtype=float)
    width = min(0.72, 0.72 / max(1, len(rows) / 12))
    colors = ["#4C78A8" if str(row.get("protocol")) == "causal" else "#F58518" for row in rows]
    paths: dict[str, str] = {}

    fig, ax = plt.subplots(figsize=(max(10.0, 0.7 * len(rows)), 5.6))
    pdm = np.asarray([_number(row, "pdm") for row in rows], dtype=float)
    pdm_std = np.nan_to_num(np.asarray([_number(row, "pdm_std", 0.0) for row in rows]), nan=0.0)
    ax.bar(x, pdm, yerr=pdm_std, capsize=3, color=colors, alpha=0.9)
    ax.set_ylabel("PDM proxy (mean ± scene std)")
    ax.set_title("VideoTokenPress multi-round quality")
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.25)
    paths["pdm_by_method"] = _save(fig, output / "pdm_by_method.png")

    fig, ax = plt.subplots(figsize=(max(10.0, 0.7 * len(rows)), 5.6))
    e2e = np.asarray([_number(row, "e2e_latency_ms_mean") for row in rows], dtype=float)
    model = np.asarray([_number(row, "model_latency_ms_mean") for row in rows], dtype=float)
    selector = np.asarray([_number(row, "selector_latency_ms_mean") for row in rows], dtype=float)
    bar_width = 0.24
    ax.bar(x - bar_width, np.nan_to_num(selector), bar_width, label="selector", color="#72B7B2")
    ax.bar(x, np.nan_to_num(model), bar_width, label="model callback", color="#54A24B")
    ax.bar(x + bar_width, np.nan_to_num(e2e), bar_width, label="end-to-end", color="#E45756")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Timing decomposition")
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.legend(loc="upper left", ncol=3)
    ax.grid(axis="y", alpha=0.25)
    paths["latency_by_method"] = _save(fig, output / "latency_by_method.png")

    fig, ax = plt.subplots(figsize=(max(10.0, 0.7 * len(rows)), 5.6))
    ratios = []
    ratio_labels = []
    for row in rows:
        if str(row.get("protocol")) == "physical":
            ratios.append(_number(row, "theoretical_attn_ratio_mean"))
            ratio_labels.append("K/V length ratio")
        else:
            ratios.append(_number(row, "eligible_keep_ratio_mean"))
            ratio_labels.append("eligible keep ratio")
    ratios = np.asarray(ratios, dtype=float)
    ax.bar(x, np.nan_to_num(ratios), color=colors, alpha=0.9)
    ax.set_ylabel("Retained ratio")
    ax.set_title("Compression ratio (lower means more compression)")
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    paths["compression_ratios"] = _save(fig, output / "compression_ratios.png")

    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    for row, color in zip(rows, colors):
        latency = _number(row, "e2e_latency_ms_mean")
        quality = _number(row, "pdm")
        if not (math.isfinite(latency) and math.isfinite(quality)):
            continue
        ax.scatter(latency, quality, s=70, color=color, edgecolor="white", linewidth=0.7)
        ax.annotate(
            _short_label(row.get("method", "unknown"), 20),
            (latency, quality),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("Mean end-to-end latency (ms)")
    ax.set_ylabel("PDM proxy")
    ax.set_title("Quality versus latency")
    ax.grid(alpha=0.25)
    paths["pdm_vs_latency"] = _save(fig, output / "pdm_vs_latency.png")

    fig, ax = plt.subplots(figsize=(max(10.0, 0.65 * len(rows)), 6.0))
    by_method: dict[str, list[Mapping]] = {}
    for row in run_rows:
        by_method.setdefault(str(row.get("method", "unknown")), []).append(row)
    color_by_method = {
        str(row.get("method")): color for row, color in zip(rows, colors)
    }
    for method in sorted(by_method):
        points = sorted(
            by_method[method],
            key=lambda row: (int(row["round"]) if str(row.get("round", "")).isdigit() else 0),
        )
        rounds = [row.get("round") for row in points]
        values = [_number(row, "pdm") for row in points]
        ax.plot(
            rounds,
            values,
            marker="o",
            linewidth=1.4,
            label=_short_label(method),
            color=color_by_method.get(method),
        )
    ax.set_xlabel("Round")
    ax.set_ylabel("PDM proxy")
    ax.set_title("Round-to-round stability")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=7, ncol=2)
    paths["round_stability"] = _save(fig, output / "round_stability.png")

    table_rows = []
    for row in rows:
        ratio = (
            _number(row, "theoretical_attn_ratio_mean")
            if str(row.get("protocol")) == "physical"
            else _number(row, "eligible_keep_ratio_mean")
        )
        table_rows.append(
            [
                _short_label(row.get("method", "unknown"), 18),
                str(row.get("protocol", "")),
                f"{_number(row, 'pdm'):.4f}" if math.isfinite(_number(row, "pdm")) else "",
                f"{_number(row, 'e2e_latency_ms_mean'):.3f}" if math.isfinite(_number(row, "e2e_latency_ms_mean")) else "",
                f"{ratio:.3f}" if math.isfinite(ratio) else "",
                f"{int(_number(row, 'valid_scenes', 0))}/{int(_number(row, 'n_scenes', 0))}",
            ]
        )
    fig, ax = plt.subplots(figsize=(11.5, max(3.2, 0.34 * len(table_rows) + 1.4)))
    ax.axis("off")
    table = ax.table(
        cellText=table_rows,
        colLabels=["method", "protocol", "PDM", "e2e ms", "retained", "valid/scenes"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.35)
    ax.set_title("Pooled suite statistics", pad=12)
    paths["summary_table"] = _save(fig, output / "summary_table.png")
    return paths

