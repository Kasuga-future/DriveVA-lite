"""Stable, inspectable experiment artifacts."""

from __future__ import annotations

import csv
from dataclasses import asdict, is_dataclass
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable

import numpy as np
import torch

from videopress.core.layout import decode_video_index_checked


def jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if hasattr(value, "value") and not isinstance(value, (str, bytes)):
        return jsonable(value.value)
    return value


def environment_snapshot(repo_root: str | Path | None = None) -> dict:
    root = Path(repo_root).resolve() if repo_root is not None else Path.cwd().resolve()
    git_commit = None
    try:
        git_commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        pass
    flash_attn = importlib.util.find_spec("flash_attn") is not None
    gpu = []
    if torch.cuda.is_available():
        gpu = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": gpu,
        "flash_attention": flash_attn,
        "repo_root": str(root),
        "git_commit": git_commit,
        "pid": os.getpid(),
    }


class ArtifactWriter:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "artifacts" / "scores").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "artifacts" / "masks").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "artifacts" / "mappings").mkdir(parents=True, exist_ok=True)
        self.token_rows: list[dict] = []
        self._event_counts: dict[str, int] = {}

    def write_config(self, config: dict) -> None:
        try:
            import yaml

            (self.output_dir / "config.yaml").write_text(yaml.safe_dump(jsonable(config), sort_keys=False), encoding="utf-8")
        except Exception:
            (self.output_dir / "config.json").write_text(json.dumps(jsonable(config), indent=2), encoding="utf-8")

    def write_environment(self, repo_root=None) -> None:
        (self.output_dir / "environment.json").write_text(
            json.dumps(jsonable(environment_snapshot(repo_root)), indent=2), encoding="utf-8"
        )

    @staticmethod
    def _slug(value: Any) -> str:
        text = str(value if value is not None else "none")
        text = "".join(char if char.isalnum() or char in "-_." else "_" for char in text)
        return text[:80] or "none"

    def _stem(self, ctx, result) -> str:
        scorer = result.metadata.get("scorer", {}) if isinstance(result.metadata, dict) else {}
        if isinstance(scorer, dict):
            scorer_name = scorer.get("name", "none")
        else:
            scorer_name = scorer
        base = "_".join(
            (
                self._slug(ctx.scene_token),
                f"rank_{self._slug(ctx.diffusion_rank)}",
                f"layer_{self._slug(ctx.layer_idx)}",
                self._slug(scorer_name),
            )
        )
        serial = self._event_counts.get(base, 0)
        self._event_counts[base] = serial + 1
        return f"{base}_{serial:03d}"

    def _write_score_tensor(self, stem: str, scores: torch.Tensor, metadata: dict) -> None:
        torch.save(
            {"scores": scores.detach().cpu(), "metadata": jsonable(metadata)},
            self.output_dir / "artifacts" / "scores" / f"{stem}.pt",
        )

    def _write_selection(self, ctx, result, stem: str) -> None:
        selection = result.selection
        selected = selection.keep_global_indices.detach().cpu()
        scores = result.scores.detach().cpu() if result.scores is not None else None
        if selected.ndim != 2:
            raise ValueError("selection keep_global_indices must be [B,K]")
        if scores is not None and scores.shape != (selected.shape[0], ctx.domain.n_candidate):
            raise ValueError("score shape does not match selection/domain")
        selected_sets = [set(row.tolist()) for row in selected]
        candidate = ctx.domain.candidate_indices.detach().cpu().tolist()
        mask = torch.zeros((selected.shape[0], len(candidate)), dtype=torch.bool)
        if selection.keep_candidate_indices.numel():
            mask.scatter_(1, selection.keep_candidate_indices.detach().cpu(), True)
        torch.save(
            {
                "candidate_indices": candidate,
                "keep_candidate_indices": selection.keep_candidate_indices.detach().cpu(),
                "keep_global_indices": selected,
                "drop_candidate_indices": selection.drop_candidate_indices.detach().cpu(),
                "mask": mask,
                "metadata": jsonable(selection.metadata),
            },
            self.output_dir / "artifacts" / "masks" / f"{stem}.pt",
        )
        score_by_batch = None if scores is None else scores
        scene_tokens = ctx.metadata.get("scene_tokens") if isinstance(ctx.metadata, dict) else None
        if scene_tokens is not None and (
            not isinstance(scene_tokens, (list, tuple)) or len(scene_tokens) != selected.shape[0]
        ):
            raise ValueError("ctx.metadata['scene_tokens'] must contain one value per batch item")
        for batch_index in range(selected.shape[0]):
            scene_token = scene_tokens[batch_index] if scene_tokens is not None else ctx.scene_token
            for local_index, global_index in enumerate(candidate):
                frame, y, x = decode_video_index_checked(int(global_index), ctx.layout)
                self.token_rows.append(
                    {
                        "scene_token": scene_token,
                        "batch_index": batch_index,
                        "layer": ctx.layer_idx,
                        "diffusion_rank": ctx.diffusion_rank,
                        "original_index": global_index,
                        "latent_frame": frame,
                        "spatial_y": y,
                        "spatial_x": x,
                        "score": None if score_by_batch is None else float(score_by_batch[batch_index, local_index]),
                        "selected": global_index in selected_sets[batch_index],
                        "configured_domain": result.metadata.get("configured_domain", ctx.metadata.get("configured_domain")),
                        "resolved_domain": result.metadata.get("resolved_domain", ctx.metadata.get("resolved_domain", ctx.domain.name)),
                    }
                )

    def _write_mapping(self, ctx, result, stem: str) -> None:
        mapping_path = self.output_dir / "artifacts" / "mappings" / f"{stem}.json"
        mapping_path.write_text(
            json.dumps(
                {
                    "scene_token": ctx.scene_token,
                    "layer": ctx.layer_idx,
                    "diffusion_rank": ctx.diffusion_rank,
                    "configured_domain": result.metadata.get("configured_domain", ctx.metadata.get("configured_domain")),
                    "resolved_domain": result.metadata.get("resolved_domain", ctx.metadata.get("resolved_domain", ctx.domain.name)),
                    "mapping": jsonable(result.mapping.to_dict()),
                    "metadata": jsonable(result.metadata),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def add_result(self, ctx, result) -> None:
        """Write selection, scores and mapping independently.

        A no-op result has selection/mapping but no scores; a merge result has
        mapping but no selection; both are valid artifact combinations.
        """

        if result.selection is None and result.scores is None and result.mapping is None:
            return
        stem = self._stem(ctx, result)
        if result.scores is not None:
            self._write_score_tensor(stem, result.scores, result.metadata)
        if result.selection is not None:
            self._write_selection(ctx, result, stem)
        if result.mapping is not None:
            self._write_mapping(ctx, result, stem)

    def add_tokens(self, ctx, result) -> None:
        """Backward-compatible alias for the independent result writer."""

        self.add_result(ctx, result)

    def add_runtime_events(self, runtime) -> None:
        events_path = self.output_dir / "events.jsonl"
        with events_path.open("a", encoding="utf-8") as handle:
            for event in runtime.events:
                handle.write(
                    json.dumps(
                        jsonable(
                            {
                                "key": event.key,
                                "metadata": event.result.metadata,
                                "mapping": event.result.mapping.to_dict() if event.result.mapping is not None else None,
                            }
                        ),
                        sort_keys=True,
                    )
                    + "\n"
                )

    def write_records(self, records: Iterable) -> None:
        rows = [jsonable(record) for record in records]
        (self.output_dir / "records.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
        )
        if rows:
            keys = sorted({key for row in rows for key in row})
            with (self.output_dir / "scenes.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)

    def write_tokens(self) -> None:
        if not self.token_rows:
            return
        parquet_path = self.output_dir / "tokens.parquet"
        try:
            import pandas as pd

            pd.DataFrame(self.token_rows).to_parquet(parquet_path, index=False)
        except Exception:
            (self.output_dir / "tokens.jsonl").write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in self.token_rows), encoding="utf-8"
            )

    def write_summary(self, summary: dict) -> None:
        (self.output_dir / "summary.json").write_text(json.dumps(jsonable(summary), indent=2), encoding="utf-8")
