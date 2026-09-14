#!/usr/bin/env python3
"""Validate and shard trajectory-gradient artifacts for LearnedCondition training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


EXPECTED_STEPS = (1000, 908, 716)
REQUIRED_ARTIFACT_KEYS = {
    "anchor_id",
    "artifact_status",
    "candidate_count",
    "candidate_positions",
    "candidate_tokens",
    "diffusion_timestep",
    "domain",
    "K",
    "objective_type",
    "rank_local_descending",
    "scores",
    "score_reduction",
    "topk_mask_candidate",
    "selected_indices",
}


def _rank_percentile(scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    order = scores.argsort(descending=True, stable=True)
    rank_position = torch.empty_like(order)
    rank_position[order] = torch.arange(scores.numel(), dtype=order.dtype)
    percentile = 1.0 - rank_position.float() / float(max(1, scores.numel() - 1))
    return rank_position, percentile


def _spearman(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.float() - left.float().mean()
    right = right.float() - right.float().mean()
    denom = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    return float("nan") if float(denom) == 0.0 else float(((left * right).sum() / denom).item())


def _selection_mask(
    scores: torch.Tensor,
    candidate_positions: torch.Tensor,
    k: int,
    retention_policy: str | None,
) -> torch.Tensor:
    """Rebuild the configured selection from scores for artifact validation."""

    mask = torch.zeros(scores.numel(), dtype=torch.bool)
    policy = retention_policy or "joint"
    if policy.startswith("drop_previous_keep_last_"):
        last_latent = int(candidate_positions[:, 0].max())
        eligible = torch.where(candidate_positions[:, 0] == last_latent)[0]
        if k > eligible.numel():
            raise RuntimeError(f"last-only K={k} exceeds last latent size {eligible.numel()}")
        order = scores.index_select(0, eligible).argsort(descending=True, stable=True)
        mask[eligible[order[:k]]] = True
    elif policy.startswith("per_latent_keep_"):
        latents = candidate_positions[:, 0].unique(sorted=True)
        if k % latents.numel():
            raise RuntimeError(f"per-latent K={k} is not divisible by latent count {latents.numel()}")
        quota = k // latents.numel()
        for latent in latents:
            eligible = torch.where(candidate_positions[:, 0] == latent)[0]
            order = scores.index_select(0, eligible).argsort(descending=True, stable=True)
            mask[eligible[order[:quota]]] = True
    else:
        mask[scores.argsort(descending=True, stable=True)[:k]] = True
    return mask


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=16)
    parser.add_argument("--split", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--inference-seed", type=int, default=0)
    args = parser.parse_args()

    manifest = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    args.output_root.mkdir(parents=True, exist_ok=True)
    samples = []
    token_diff_summaries = []
    validation_summaries = []
    dataset_spec = None
    for row in manifest:
        anchor_id = row["anchor_id"]
        anchor_root = args.artifact_root / anchor_id
        artifacts = {}
        for step in EXPECTED_STEPS:
            path = anchor_root / f"timestep_{step}.pt"
            if not path.is_file():
                raise FileNotFoundError(path)
            artifacts[step] = torch.load(path, map_location="cpu", weights_only=False)
            missing_keys = REQUIRED_ARTIFACT_KEYS - artifacts[step].keys()
            if missing_keys:
                raise RuntimeError(f"Missing artifact keys for {anchor_id} step={step}: {sorted(missing_keys)}")
            artifact_spec = {
                "candidate_count": int(artifacts[step]["candidate_count"]),
                "K": int(artifacts[step]["K"]),
                "domain": str(artifacts[step]["domain"]),
                "retention_policy": artifacts[step].get("retention_policy"),
            }
            if dataset_spec is None:
                dataset_spec = artifact_spec
            elif artifact_spec != dataset_spec:
                raise RuntimeError(
                    f"Inconsistent retention metadata for {anchor_id} step={step}: "
                    f"expected={dataset_spec!r} actual={artifact_spec!r}"
                )
            expected_metadata = {
                "anchor_id": str(anchor_id),
                "artifact_status": ["POC_ONLY", "TEST_DERIVED", "NOT_FOR_OFFICIAL_REPORTING"],
                "candidate_count": dataset_spec["candidate_count"],
                "diffusion_timestep": step,
                "domain": dataset_spec["domain"],
                "K": dataset_spec["K"],
                "objective_type": "detached_unit_trajectory_projection_v1",
                "score_reduction": "l2_embedding_then_batch_mean",
            }
            for key, expected in expected_metadata.items():
                if artifacts[step][key] != expected:
                    raise RuntimeError(
                        f"Invalid {key} for {anchor_id} step={step}: "
                        f"expected={expected!r} actual={artifacts[step][key]!r}"
                    )
        candidate_count = int(dataset_spec["candidate_count"])
        k = int(dataset_spec["K"])
        retention_policy = dataset_spec["retention_policy"]
        candidate_tokens = {step: artifacts[step]["candidate_tokens"].squeeze(0) for step in EXPECTED_STEPS}
        reference = candidate_tokens[EXPECTED_STEPS[0]].float()
        if reference.ndim != 2 or reference.shape[0] != candidate_count or not torch.isfinite(reference).all():
            raise RuntimeError(f"Invalid candidate tokens for {anchor_id}: {tuple(reference.shape)}")
        diffs = {}
        for step in EXPECTED_STEPS[1:]:
            delta = (reference - candidate_tokens[step].float()).abs()
            diffs[str(step)] = {"max_abs_diff": float(delta.max()), "mean_abs_diff": float(delta.mean())}
        token_diff_summaries.append({"anchor_id": anchor_id, "versus_step_1000": diffs})
        if any(value["max_abs_diff"] != 0.0 for value in diffs.values()):
            report = {
                "status": "STOP_PER_STEP_CANDIDATE_TOKENS_DIFFER",
                "anchor_id": anchor_id,
                "diffs": diffs,
            }
            (args.output_root / "candidate_token_difference.json").write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
            raise RuntimeError(json.dumps(report))

        positions = artifacts[EXPECTED_STEPS[0]]["candidate_positions"]
        if positions.shape != (candidate_count, 3):
            raise RuntimeError(f"Invalid candidate positions for {anchor_id}: {tuple(positions.shape)}")
        for step in EXPECTED_STEPS[1:]:
            if not torch.equal(positions, artifacts[step]["candidate_positions"]):
                raise RuntimeError(f"Candidate positions differ for {anchor_id} step={step}")

        scores_per_step = []
        rank_position_per_step = []
        rank_percentile_per_step = []
        topk_per_step = []
        for step in EXPECTED_STEPS:
            artifact = artifacts[step]
            scores = artifact["scores"].squeeze(0).float()
            if scores.shape != (candidate_count,) or not torch.isfinite(scores).all():
                raise RuntimeError(f"Invalid scores for {anchor_id} step={step}: {tuple(scores.shape)}")
            if float(scores.std()) == 0.0:
                raise RuntimeError(f"Constant scores for {anchor_id} step={step}")
            rank_position, percentile = _rank_percentile(scores)
            expected_order = scores.argsort(descending=True, stable=True)
            stored_order = artifact["rank_local_descending"].squeeze(0)
            if stored_order.shape != (candidate_count,) or not torch.equal(stored_order, expected_order):
                raise RuntimeError(f"Stored rank does not match score ordering for {anchor_id} step={step}")
            mask = artifact["topk_mask_candidate"].squeeze(0).bool()
            if mask.shape != (candidate_count,) or int(mask.sum()) != k:
                raise RuntimeError(f"Invalid Top{k} mask for {anchor_id} step={step}")
            expected_mask = _selection_mask(scores, positions, k, retention_policy)
            if not torch.equal(mask, expected_mask):
                raise RuntimeError(
                    f"Stored Top{k} mask does not match policy={retention_policy!r} "
                    f"for {anchor_id} step={step}"
                )
            target_shape = tuple(artifact["gradient_target_shape"])
            if len(target_shape) != 5:
                raise RuntimeError(f"Invalid gradient target shape for {anchor_id}: {target_shape}")
            video_width = int(target_shape[4])
            tokens_per_latent = int(target_shape[3]) * video_width
            candidate_global = (
                positions[:, 0].long() * tokens_per_latent
                + positions[:, 1].long() * video_width
                + positions[:, 2].long()
            )
            stored_selected = artifact["selected_indices"].squeeze(0).long()
            expected_selected = candidate_global[expected_mask]
            if stored_selected.numel() != k or not torch.equal(
                stored_selected.sort().values, expected_selected.sort().values
            ):
                raise RuntimeError(f"selected_indices do not match Top{k} mask for {anchor_id} step={step}")
            scores_per_step.append(scores)
            rank_position_per_step.append(rank_position)
            rank_percentile_per_step.append(percentile)
            topk_per_step.append(mask)

        normalized = torch.stack(rank_percentile_per_step)
        aggregate_score = normalized.mean(dim=0)
        aggregate_rank, _ = _rank_percentile(aggregate_score)
        aggregate_order = aggregate_score.argsort(descending=True, stable=True)
        aggregate_mask = _selection_mask(
            aggregate_score, positions, k, retention_policy
        )
        topks = torch.stack(topk_per_step)
        pair_diagnostics = {}
        for left_index, right_index in ((0, 1), (1, 2), (0, 2)):
            left_step, right_step = EXPECTED_STEPS[left_index], EXPECTED_STEPS[right_index]
            overlap = int((topks[left_index] & topks[right_index]).sum())
            union = int((topks[left_index] | topks[right_index]).sum())
            pair_diagnostics[f"{left_step}_{right_step}"] = {
                "topk_count": k,
                "topk_overlap": overlap,
                "jaccard": overlap / union,
                "spearman": _spearman(normalized[left_index], normalized[right_index]),
            }

        validation_summaries.append(
            {
                "anchor_id": anchor_id,
                "candidate_token_diffs_versus_step_1000": diffs,
                "candidate_positions_identical": True,
                "stored_ranks_match_scores": True,
                "stored_topk_masks_match_policy": True,
                "score_statistics": {
                    str(step): {
                        "min": float(scores.min()),
                        "max": float(scores.max()),
                        "mean": float(scores.mean()),
                        "std": float(scores.std()),
                        "topk_count": int(mask.sum()),
                    }
                    for step, scores, mask in zip(EXPECTED_STEPS, scores_per_step, topk_per_step)
                },
                "pair_diagnostics": pair_diagnostics,
            }
        )

        samples.append(
            {
                **row,
                "artifact_status": ["POC_ONLY", "TEST_DERIVED", "NOT_FOR_OFFICIAL_REPORTING"],
                "candidate_tokens": candidate_tokens[1000],
                "positions": positions,
                "teacher_scores_per_step": torch.stack(scores_per_step),
                "teacher_rank_position_per_step": torch.stack(rank_position_per_step),
                "teacher_rank_per_step": normalized,
                "teacher_topk_per_step": topks,
                "aggregate_teacher_score": aggregate_score,
                "aggregate_teacher_rank": aggregate_rank,
                "aggregate_topk_mask": aggregate_mask,
                "teacher_pair_diagnostics": pair_diagnostics,
            }
        )

    shard_paths = []
    for shard_index, start in enumerate(range(0, len(samples), args.shard_size)):
        path = args.output_root / f"shard_{shard_index:04d}.pt"
        torch.save(samples[start : start + args.shard_size], path)
        shard_paths.append(path.name)
    metadata = {
        "artifact_status": ["POC_ONLY", "TEST_DERIVED", "NOT_FOR_OFFICIAL_REPORTING"],
        "split": args.split,
        "sample_count": len(samples),
        "candidate_count": int(dataset_spec["candidate_count"]),
        "K": int(dataset_spec["K"]),
        "domain": dataset_spec["domain"],
        "retention_policy": dataset_spec["retention_policy"],
        "token_order": "[f][row][col]",
        "tokens_per_latent": int(samples[0]["positions"].shape[0] // 2)
        if samples and dataset_spec["domain"] in {"history", "all_history"}
        else int(dataset_spec["candidate_count"]),
        "teacher_method": "trajectory_projection_gradient_input",
        "teacher_objective_type": "detached_unit_trajectory_projection_v1",
        "gradient_input_reduction": "l2_embedding_then_batch_mean",
        "diffusion_timesteps": list(EXPECTED_STEPS),
        "checkpoint": args.checkpoint,
        "model_config": args.model_config,
        "inference_seed": args.inference_seed,
        "candidate_tokens_storage": "single_copy_step1000_verified_identical_across_steps",
        "candidate_token_diffs": token_diff_summaries,
        "validation": {
            "status": "PASS",
            "schema_valid": True,
            "candidate_tokens_identical_across_steps": True,
            "candidate_positions_identical_across_steps": True,
            "stored_ranks_match_scores": True,
            "stored_topk_masks_match_policy": True,
            "samples": validation_summaries,
        },
        "shards": shard_paths,
    }
    (args.output_root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
