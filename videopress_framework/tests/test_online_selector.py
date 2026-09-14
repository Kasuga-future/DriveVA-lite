from __future__ import annotations

import json
import math
import pickle
import sys
from pathlib import Path

import pytest
import torch


TRAIN_DIR = Path(__file__).resolve().parents[2] / "examples" / "wanvideo" / "driveva_train"
sys.path.insert(0, str(TRAIN_DIR))

from videopress.training.online_selector import (  # noqa: E402
    DynamicTokenSelector,
    counterfactual_group_bce,
    critical_box_token_mask,
    displacement_token_bce,
    gradient_input_scores,
    online_topk_labels,
    parse_horizon_weights,
    protected_topk_labels,
    selector_pairwise_ranking_loss,
    horizon_weighted_trajectory_displacement,
    selector_metrics,
    signed_removal_scores,
    signed_soft_keep_labels,
    spatial_counterfactual_probe,
)
from train_navsim_v1 import (  # noqa: E402
    DriveVANavsimTrainingModule,
    _dump_counterfactual_probe,
    _forbidden_manifest_tokens,
    _manifest_scene_tokens,
    _read_jsonl_manifest,
    _representative_frame_tokens,
    _trajectory_divergence,
)
from videopress.core.context import TokenContext  # noqa: E402
from videopress.core.domain import build_domain  # noqa: E402
from videopress.core.layout import build_driveva_layout  # noqa: E402
from videopress.scorers.learned_selector import (  # noqa: E402
    DynamicTokenSelector as RuntimeSelector,
    LearnedPlanningSelectorScorer,
)


def test_forbidden_manifest_tokens_unions_multiple_files(tmp_path: Path) -> None:
    first = tmp_path / "test.jsonl"
    second = tmp_path / "calibration.jsonl"
    first.write_text('{"scene_token":"a"}\n{"scene_token":"b"}\n', encoding="utf-8")
    second.write_text('{"scene_token":"b"}\n{"scene_token":"c"}\n', encoding="utf-8")

    tokens, summaries = _forbidden_manifest_tokens(f"{first},{second}")

    assert tokens == {"a", "b", "c"}
    assert [item["scenes"] for item in summaries] == [2, 2]


def test_gradient_input_teacher_and_topk_labels() -> None:
    tokens = torch.tensor([[[1.0, 2.0], [3.0, 1.0], [0.5, 0.5]]], requires_grad=True)
    weights = torch.tensor([[[1.0, 0.0], [0.0, 2.0], [0.1, 0.1]]])
    loss = (tokens * weights).sum()
    scores = gradient_input_scores(loss, tokens, retain_graph=False)
    assert torch.allclose(scores, torch.tensor([[1.0, 2.0, 0.1]]))
    labels = online_topk_labels(scores, keep_ratio=1 / 3)
    assert labels.tolist() == [[0.0, 1.0, 0.0]]


def test_protected_topk_preserves_budget_and_forces_critical_tokens() -> None:
    scores = torch.tensor([[0.9, 0.8, 0.7, 0.1]])
    protected = torch.tensor([[False, False, False, True]])
    labels = protected_topk_labels(scores, 0.5, protected)
    assert labels.tolist() == [[1.0, 0.0, 0.0, 1.0]]
    with pytest.raises(ValueError, match="exceeds fixed budget"):
        protected_topk_labels(scores, 0.5, torch.tensor([[True, True, True, False]]))


def test_critical_boxes_project_to_token_grid_with_validity_and_dilation() -> None:
    positions = torch.tensor(
        [[[0.0, 0.1, 0.1], [0.0, 0.5, 0.5], [0.0, 0.9, 0.9]]]
    )
    boxes = torch.tensor([[[0.45, 0.45, 0.55, 0.55], [0.0, 0.0, 0.2, 0.2]]])
    mask = critical_box_token_mask(
        positions, boxes, box_valid=torch.tensor([[True, False]])
    )
    assert mask.tolist() == [[False, True, False]]
    dilated = critical_box_token_mask(
        positions, boxes[:, :1], dilation=0.4
    )
    assert dilated.tolist() == [[True, True, True]]


def test_pairwise_ranking_loss_optimizes_topk_ordering() -> None:
    logits = torch.tensor([[-1.0, 1.0, 0.5]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.0, 0.0]])
    bad = selector_pairwise_ranking_loss(logits, targets)
    good = selector_pairwise_ranking_loss(
        torch.tensor([[2.0, 1.0, 0.5]]), targets
    )
    assert bad > good
    bad.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert logits.grad[0, 0] < 0


def test_selector_metrics_report_ranking_quality() -> None:
    metrics = selector_metrics(
        torch.tensor([[4.0, 3.0, 2.0, 1.0]]),
        torch.tensor([[1.0, 1.0, 0.0, 0.0]]),
    )
    assert metrics["selector_topk_overlap"] == 1.0
    assert metrics["selector_pairwise_accuracy"] == 1.0
    assert metrics["selector_ndcg_at_k"] == pytest.approx(1.0)


def test_long_horizon_schedule_and_weighted_displacement() -> None:
    schedule = parse_horizon_weights("3:5,1:2,2:3")
    assert schedule == [(1.0, 0.2), (2.0, 0.3), (3.0, 0.5)]
    baseline = torch.zeros(1, 6, 2)
    altered = torch.zeros_like(baseline)
    altered[0, 1, 0] = 1.0  # 1 s at 2 fps
    altered[0, 3, 0] = 2.0  # 2 s
    altered[0, 5, 0] = 3.0  # 3 s
    value, indices = horizon_weighted_trajectory_displacement(
        baseline, altered, schedule, target_fps=2
    )
    assert indices == [1, 3, 5]
    assert value.item() == pytest.approx(2.3)


def test_trajectory_divergence_exposes_preregistered_long_horizon_target() -> None:
    baseline = torch.zeros(1, 6, 2)
    altered = torch.zeros_like(baseline)
    altered[0, 1, 0], altered[0, 3, 0], altered[0, 5, 0] = 1, 2, 3
    result = _trajectory_divergence(
        baseline,
        altered,
        horizon_weights=parse_horizon_weights("1:0.2,2:0.3,3:0.5"),
        target_fps=2,
    )
    assert result["counterfactual_traj_disp_long_horizon"] == pytest.approx(2.3)
    assert result["counterfactual_traj_horizon_indices"] == [1, 3, 5]


def test_signed_removal_teacher_preserves_harmful_direction() -> None:
    tokens = torch.tensor([[[1.0, 2.0], [3.0, 1.0]]], requires_grad=True)
    weights = torch.tensor([[[-1.0, 0.0], [0.0, 2.0]]])
    loss = (tokens * weights).sum()
    scores = signed_removal_scores(loss, tokens, retain_graph=False)
    assert torch.allclose(scores, torch.tensor([[1.0, -2.0]]))
    labels = signed_soft_keep_labels(scores)
    assert labels[0, 0] > 0.5
    assert labels[0, 1] < 0.5


def test_spatial_counterfactual_probe_and_signed_group_loss() -> None:
    positions = torch.tensor(
        [[[1.0, 0.0, 0.0], [1.0, 0.2, 0.2], [1.0, 0.9, 0.9]]]
    )
    keep, membership = spatial_counterfactual_probe(
        positions, group_index=0, tile_h=2, tile_w=2
    )
    assert membership.tolist() == [[True, True, False]]
    assert keep.tolist() == [[0.0, 0.0, 1.0]]
    helpful_loss, helpful = counterfactual_group_bce(
        torch.tensor([[2.0, 2.0, -2.0]]),
        membership,
        torch.tensor(1.0),
        torch.tensor(1.2),
    )
    harmful_loss, harmful = counterfactual_group_bce(
        torch.tensor([[-2.0, -2.0, 2.0]]),
        membership,
        torch.tensor(1.0),
        torch.tensor(0.8),
    )
    assert helpful_loss < 0.2 and harmful_loss < 0.2
    assert helpful["counterfactual_helpful_target"] == 1.0
    assert harmful["counterfactual_helpful_target"] == 0.0


def _tile_grid_positions(tokens_per_tile: int, tile_h: int = 3, tile_w: int = 4) -> torch.Tensor:
    """Positions on a ``tile_h x tile_w`` grid, ``tokens_per_tile`` per tile."""
    groups = tile_h * tile_w
    rows = []
    for group in range(groups):
        row, col = divmod(group, tile_w)
        for offset in range(tokens_per_tile):
            sub_row, sub_col = divmod(offset, 2)
            rows.append(
                [
                    1.0,
                    (row + (sub_row + 0.5) / 2.0) / tile_h,
                    (col + (sub_col + 0.5) / 2.0) / tile_w,
                ]
            )
    return torch.tensor([rows], dtype=torch.float32)


def _tile_memberships(positions: torch.Tensor, tile_h: int = 3, tile_w: int = 4) -> torch.Tensor:
    masks = [
        spatial_counterfactual_probe(positions, group, tile_h=tile_h, tile_w=tile_w)[1][0]
        for group in range(tile_h * tile_w)
    ]
    return torch.stack(masks)


def test_counterfactual_group_bce_learns_a_synthetic_tile_signal() -> None:
    """P0-1 learnability gate.

    If the tile-level counterfactual loss and AdamW cannot fit a label that is a
    deterministic function of the very tokens the selector sees, then a negative
    result on real data says nothing about the data: the loop itself is broken.
    """
    torch.manual_seed(11)
    tile_h, tile_w = 3, 4
    groups = tile_h * tile_w
    per_tile = 4
    token_dim = 16
    positions = _tile_grid_positions(per_tile, tile_h, tile_w)
    memberships = _tile_memberships(positions, tile_h, tile_w)

    selector = DynamicTokenSelector(
        token_dim=token_dim, hidden_dim=32, position_dim=8, ego_dim=2, command_dim=3
    )
    optimizer = torch.optim.AdamW(selector.parameters(), lr=3e-3)
    direction = torch.randn(token_dim)
    direction = direction / direction.norm()
    ego = torch.zeros(1, 2)
    command = torch.nn.functional.one_hot(torch.tensor([1]), num_classes=3).float()

    def scene_batch() -> tuple[torch.Tensor, torch.Tensor]:
        base = torch.randn(1, groups * per_tile, token_dim)
        signed = torch.randint(0, 2, (groups,)).float() * 2.0 - 1.0
        base = base + signed.repeat_interleave(per_tile).unsqueeze(0).unsqueeze(-1) * 2.0 * direction
        # The label is a deterministic function of the tile's *mean* token.
        latent = (base.view(groups, per_tile, token_dim).mean(dim=1) @ direction)
        target = (latent > 0).float()
        delta = torch.where(target > 0.5, torch.full((groups,), 0.02), torch.full((groups,), -0.02))
        baseline = torch.ones(groups)
        return base, (baseline, baseline + delta)

    def balanced_accuracy(tokens: torch.Tensor, target: torch.Tensor) -> float:
        expanded = tokens[0].unsqueeze(0).expand(groups, -1, -1)
        logits = selector(expanded, positions.expand(groups, -1, -1), ego.expand(groups, -1), command.expand(groups, -1))
        predicted = torch.stack(
            [logits[row][memberships[row]].mean() for row in range(groups)]
        )
        keep = predicted >= 0
        positive = target >= 0.5
        recall_pos = (keep & positive).sum().float() / positive.sum().clamp_min(1)
        recall_neg = ((~keep) & (~positive)).sum().float() / (~positive).sum().clamp_min(1)
        return float(0.5 * (recall_pos + recall_neg))

    for _ in range(240):
        tokens, (baseline, masked) = scene_batch()
        expanded = tokens[0].unsqueeze(0).expand(groups, -1, -1)
        logits = selector(
            expanded,
            positions.expand(groups, -1, -1),
            ego.expand(groups, -1),
            command.expand(groups, -1),
        )
        loss, _ = counterfactual_group_bce(logits, memberships, baseline, masked)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.manual_seed(999)
    scores = []
    for _ in range(25):
        tokens, _ = scene_batch()
        latent = tokens.view(groups, per_tile, token_dim).mean(dim=1) @ direction
        scores.append(balanced_accuracy(tokens, (latent > 0).float()))
    mean_ba = sum(scores) / len(scores)
    assert mean_ba > 0.9, f"tile-level label is not learnable: balanced accuracy {mean_ba:.3f}"


def test_counterfactual_group_bce_is_blind_to_within_tile_token_ranking() -> None:
    """Review defect 3.4, stated as a test.

    ``counterfactual_group_bce`` supervises only the tile-mean logit.  Two probes
    with identical tile means but opposite within-tile token orderings therefore
    produce identical losses *and* identical per-token gradients, so no amount of
    training can teach the selector which token inside a tile matters.  That is
    exactly the ability token selection needs.
    """
    torch.manual_seed(3)
    tile_h, tile_w, per_tile = 2, 2, 4
    groups = tile_h * tile_w
    positions = _tile_grid_positions(per_tile, tile_h, tile_w)
    memberships = _tile_memberships(positions, tile_h, tile_w)

    scene = torch.randn(1, groups * per_tile)
    flipped = scene.clone()
    for group in range(groups):
        lo, hi = group * per_tile, (group + 1) * per_tile
        flipped[0, lo:hi] = flipped[0, lo:hi].flip(0)

    logits_a = scene.expand(groups, -1).clone().requires_grad_(True)
    logits_b = flipped.expand(groups, -1).clone().requires_grad_(True)
    baseline = torch.ones(groups)
    masked = baseline + 0.02
    loss_a, _ = counterfactual_group_bce(logits_a, memberships, baseline, masked)
    loss_b, _ = counterfactual_group_bce(logits_b, memberships, baseline, masked)
    assert torch.allclose(loss_a, loss_b), (
        "tile-mean supervision saw a difference the label cannot express"
    )

    loss_a.backward()
    loss_b.backward()
    grad_a = logits_a.grad
    grad_b = logits_b.grad
    for group in range(groups):
        lo, hi = group * per_tile, (group + 1) * per_tile
        chunk = grad_a[group, lo:hi]
        assert torch.allclose(chunk, chunk[0].expand_as(chunk), atol=1e-6), (
            "tile-mean supervision must give every token in a tile the same gradient"
        )
    assert torch.allclose(grad_a, grad_b, atol=1e-6), (
        "reordering tokens inside a tile must leave the gradient untouched"
    )

    # Fine-grained supervision (the P1 fix) produces token-dependent gradients,
    # so it *can* express within-tile ranking; the contrast is the control.
    fine_a = scene.expand(groups, -1).clone().requires_grad_(True)
    fine_b = flipped.expand(groups, -1).clone().requires_grad_(True)
    target = memberships.float()
    torch.nn.functional.binary_cross_entropy_with_logits(
        fine_a, target, reduction="none"
    ).mean().backward()
    torch.nn.functional.binary_cross_entropy_with_logits(
        fine_b, target, reduction="none"
    ).mean().backward()
    chunk = fine_a.grad[0, 0:per_tile]
    assert not torch.allclose(chunk, chunk[0].expand_as(chunk), atol=1e-6)
    assert not torch.allclose(fine_a.grad, fine_b.grad, atol=1e-6)


def test_counterfactual_group_bce_matches_legacy_weighted_mean_and_honours_dead_zone() -> None:
    """``abstain_eps=0`` must stay bit-compatible; ``abstain_eps>0`` must abstain."""
    torch.manual_seed(5)
    groups, per_tile = 6, 3
    positions = _tile_grid_positions(per_tile, 2, 3)
    tile_h, tile_w = 2, 3
    memberships = _tile_memberships(positions, tile_h, tile_w)
    logits = torch.randn(1, groups * per_tile).expand(groups, -1).clone()
    baseline = torch.ones(groups)
    deltas = torch.tensor([0.04, -0.03, 1e-5, -1e-6, 0.02, -0.05])
    masked = baseline + deltas

    loss, info = counterfactual_group_bce(logits, memberships, baseline, masked)
    group_logits = torch.stack(
        [logits[row][memberships[row]].mean() for row in range(logits.shape[0])]
    )
    target = (deltas >= 0).float()
    confidence = (deltas.abs() / 0.05).clamp(max=1.0)
    legacy = torch.nn.functional.binary_cross_entropy_with_logits(
        group_logits, target, weight=confidence.clamp_min(0.05), reduction="mean"
    )
    assert torch.allclose(loss, legacy, atol=1e-7)
    assert info["counterfactual_abstain_ratio"] == 0.0

    loss_dz, info_dz = counterfactual_group_bce(
        logits, memberships, baseline, masked, abstain_eps=1e-3
    )
    assert abs(info_dz["counterfactual_abstain_ratio"] - 2 / groups) < 1e-6
    assert loss_dz.item() != loss.item()
    # The dead zone must not silently zero every sample.
    all_inside = baseline + torch.full((groups,), 1e-6)
    loss_all, info_all = counterfactual_group_bce(
        logits, memberships, baseline, all_inside, abstain_eps=1e-3
    )
    assert abs(info_all["counterfactual_abstain_ratio"] - 1.0) < 1e-6
    assert torch.isfinite(loss_all)


def test_counterfactual_group_bce_learns_all_tiles_of_a_scene_and_ranks_them() -> None:
    """The P1 coverage fix, in loss form.

    Supervising only one tile per scene (defect 3.5) cannot teach within-scene
    ranking.  Stacking the memberships of all 12 tiles into one batch gives every
    tile its own label in the same step; this test asserts that the selector then
    learns both each tile's sign *and* the ordering of tiles inside a scene --
    the latter being the only capability token selection actually needs.
    """
    torch.manual_seed(23)
    tile_h, tile_w = 3, 4
    groups = tile_h * tile_w
    per_tile = 4
    token_dim = 16
    positions = _tile_grid_positions(per_tile, tile_h, tile_w)
    memberships = _tile_memberships(positions, tile_h, tile_w)

    selector = DynamicTokenSelector(token_dim=token_dim, hidden_dim=32, position_dim=8)
    optimizer = torch.optim.AdamW(selector.parameters(), lr=3e-3)
    direction = torch.randn(token_dim)
    direction = direction / direction.norm()
    ego = torch.zeros(1, 2)
    command = torch.nn.functional.one_hot(torch.tensor([1]), num_classes=3).float()

    def scene() -> tuple[torch.Tensor, torch.Tensor]:
        base = torch.randn(1, groups * per_tile, token_dim)
        tile_latent = torch.randn(groups)
        base = base + tile_latent.repeat_interleave(per_tile).unsqueeze(0).unsqueeze(-1) * 1.5 * direction
        latent = base.view(groups, per_tile, token_dim).mean(dim=1) @ direction
        return base, latent

    for _ in range(300):
        tokens, latent = scene()
        logits = selector(
            tokens[0].unsqueeze(0).expand(groups, -1, -1),
            positions.expand(groups, -1, -1),
            ego.expand(groups, -1),
            command.expand(groups, -1),
        )
        target = (latent > 0).float()
        delta = torch.where(target > 0.5, torch.full((groups,), 0.02), torch.full((groups,), -0.02))
        baseline = torch.ones(groups)
        loss, _ = counterfactual_group_bce(logits, memberships, baseline, baseline + delta)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.manual_seed(777)
    correct = total = 0
    concordant = pairs = 0
    for _ in range(20):
        tokens, latent = scene()
        logits = selector(
            tokens[0].unsqueeze(0).expand(groups, -1, -1),
            positions.expand(groups, -1, -1),
            ego.expand(groups, -1),
            command.expand(groups, -1),
        )
        predicted = torch.stack(
            [logits[row][memberships[row]].mean() for row in range(groups)]
        )
        correct += int(((predicted >= 0) == (latent > 0)).sum())
        total += groups
        high = int(torch.argmax(latent))
        low = int(torch.argmin(latent))
        pairs += 1
        concordant += int(predicted[high] > predicted[low])
    balanced_accuracy = correct / total
    assert balanced_accuracy > 0.9, f"per-tile sign not learned: {balanced_accuracy:.3f}"
    assert concordant >= 0.9 * pairs, f"within-scene ranking not learned: {concordant}/{pairs}"


def test_trajectory_divergence_measures_plan_displacement() -> None:
    baseline = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
    identical = _trajectory_divergence(baseline, baseline.clone())
    assert identical["counterfactual_traj_disp_mean"] == 0.0
    assert identical["counterfactual_traj_disp_relative"] == 0.0
    shifted = baseline + torch.tensor([[[0.0, 0.5, 0.0]]])
    moved = _trajectory_divergence(baseline, shifted)
    assert moved["counterfactual_traj_disp_mean"] > 0.0
    assert moved["counterfactual_traj_endpoint_disp"] > 0.0
    # A prefix must be excluded so only the predicted future is compared.
    prefixed_baseline = torch.cat([baseline, baseline], dim=1)
    prefixed_masked = torch.cat([baseline + 5.0, baseline], dim=1)
    assert _trajectory_divergence(prefixed_baseline, prefixed_masked)["counterfactual_traj_disp_mean"] > 0
    assert (
        _trajectory_divergence(prefixed_baseline, prefixed_masked, prefix_len=3)[
            "counterfactual_traj_disp_mean"
        ]
        == 0.0
    )
    assert _trajectory_divergence(None, baseline) == {}


def test_dump_counterfactual_probe_writes_one_shard_per_tile(tmp_path) -> None:
    tokens = torch.randn(1, 8, 4)
    positions = torch.rand(1, 8, 3)
    logits = torch.randn(1, 8)
    membership = torch.zeros(1, 8, dtype=torch.bool)
    membership[0, :4] = True
    written = []
    for tile in (0, 7):
        written.append(
            _dump_counterfactual_probe(
                tmp_path,
                rank=0,
                global_step=1,
                sample_token=["scene"],
                tokens=tokens,
                positions=positions,
                ego_state=torch.zeros(1, 2),
                command=torch.zeros(1, 3),
                selector_timestep=torch.tensor([1000.0]),
                logits=logits,
                membership=membership,
                group_index=tile,
                baseline_loss=torch.tensor([1.0]),
                masked_loss=torch.tensor([1.01]),
                relative_delta=0.01,
                helpful_target=1.0,
                confidence=0.2,
                trajectory_metrics={"counterfactual_traj_disp_mean": 0.5},
            )
        )
    assert len(set(written)) == 2, "each tile of a sweep must get its own shard"
    for path in written:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        assert payload["sample_token"] == "scene"
        assert payload["tokens"].dtype == torch.float16
        assert payload["counterfactual_traj_disp_mean"] == 0.5


class _SpecStub:
    """Minimal carrier so the trainer's pure spec helpers can be unit tested."""

    def __init__(self, scales=(), sweep_all=False, tile_h=3, tile_w=4):
        self.selector_counterfactual_scales = tuple(scales)
        self.selector_counterfactual_sweep_all = bool(sweep_all)
        self.selector_counterfactual_tile_h = tile_h
        self.selector_counterfactual_tile_w = tile_w


def test_counterfactual_probe_specs_cover_sweep_and_scales() -> None:
    stub = _SpecStub()
    # historical single-tile rotation, unchanged
    specs = DriveVANavsimTrainingModule._counterfactual_probe_specs(stub, 0, 2, 0, 12)
    assert specs == [{"tiles": (0,), "removal_count": 1, "anchor_tile": 0}]
    specs = DriveVANavsimTrainingModule._counterfactual_probe_specs(stub, 1, 2, 1, 12)
    assert specs[0]["tiles"] == (3,)
    # exhaustive tile sweep
    stub.selector_counterfactual_sweep_all = True
    specs = DriveVANavsimTrainingModule._counterfactual_probe_specs(stub, 0, 2, 0, 12)
    assert len(specs) == 12
    assert sorted(spec["tiles"][0] for spec in specs) == list(range(12))
    # perturbation-size sweep
    stub.selector_counterfactual_sweep_all = False
    stub.selector_counterfactual_scales = (1, 2, 4, 6, 12)
    specs = DriveVANavsimTrainingModule._counterfactual_probe_specs(stub, 0, 1, 0, 12)
    assert [spec["removal_count"] for spec in specs] == [1, 2, 4, 6, 12]
    assert specs[0]["tiles"] == (0,)
    assert specs[1]["tiles"] == (0, 1)
    assert specs[4]["tiles"] == tuple(range(12))
    # the rotating offset must reach every tile, so no geometry is systematically
    # favoured across a full rotation
    seen = set()
    for event in range(12):
        for spec in DriveVANavsimTrainingModule._counterfactual_probe_specs(stub, event, 1, 0, 12):
            seen |= set(spec["tiles"])
    assert seen == set(range(12))


def test_counterfactual_mask_unions_tiles_and_allows_full_frame_removal() -> None:
    stub = _SpecStub()
    positions = _tile_grid_positions(4, 3, 4)
    keep, membership = DriveVANavsimTrainingModule._counterfactual_mask(
        stub, positions, {"tiles": (0, 1), "removal_count": 2, "anchor_tile": 0}
    )
    assert int(membership.sum()) == 8
    assert int(keep.sum()) == 48 - 8
    # whole-frame removal is a legitimate probe: the pipeline scatters the kept
    # tokens back into a zero tensor of the original length
    keep_all, membership_all = DriveVANavsimTrainingModule._counterfactual_mask(
        stub, positions, {"tiles": tuple(range(12)), "removal_count": 12, "anchor_tile": 0}
    )
    assert int(membership_all.sum()) == 48
    assert float(keep_all.sum()) == 0.0


def test_displacement_token_bce_supervises_every_token_in_the_tile() -> None:
    """The magnitude teacher must not pool to the tile mean (defect 3.4)."""
    torch.manual_seed(31)
    positions = _tile_grid_positions(4, 3, 4)
    memberships = _tile_memberships(positions, 3, 4)
    logits = torch.zeros(12, 48, requires_grad=True)
    displacements = torch.linspace(0.001, 0.011, 12)
    loss, info = displacement_token_bce(
        logits, memberships, displacements, disp_scale=0.01
    )
    assert info["counterfactual_supervised_tokens"] == 48.0
    assert 0.0 < info["counterfactual_displacement_target_mean"] < 1.0
    # every token in a tile carries that tile's own target, not a shared mean
    assert abs(info["counterfactual_displacement_target_max"] - 1.0) < 1e-6
    loss.backward()
    for row in range(12):
        chunk = logits.grad[row][memberships[row]]
        assert torch.allclose(chunk, chunk[0].expand_as(chunk), atol=1e-6)
    # Gradient descent moves a logit by -grad, so a tile whose removal caused a
    # LARGER displacement must receive a more negative gradient (higher score).
    assert logits.grad[11][memberships[11]].mean() < logits.grad[0][memberships[0]].mean()

    # absolute mode: monotone and saturating
    _, big = displacement_token_bce(
        torch.zeros(12, 48), memberships, displacements * 100.0,
        disp_scale=0.01, normalize="absolute",
    )
    assert abs(big["counterfactual_displacement_target_mean"] - 1.0) < 1e-6
    _, exact = displacement_token_bce(
        torch.zeros(1, 48),
        memberships[0:1],
        torch.tensor([0.004]),
        disp_scale=0.01,
        normalize="absolute",
    )
    assert abs(exact["counterfactual_displacement_target_mean"] - 0.4) < 1e-6

    # scene mode: only the *relative* ordering inside the scene matters, so the
    # best tile is 1.0 and the worst is 0.0 regardless of absolute scale
    _, relative = displacement_token_bce(
        torch.zeros(12, 48), memberships, displacements, normalize="scene"
    )
    assert abs(relative["counterfactual_displacement_target_mean"] - 0.5) < 0.02
    assert abs(relative["counterfactual_displacement_target_max"] - 1.0) < 1e-6
    _, rescaled = displacement_token_bce(
        torch.zeros(12, 48), memberships, displacements * 7.0 + 3.0, normalize="scene"
    )
    assert abs(
        rescaled["counterfactual_displacement_target_mean"]
        - relative["counterfactual_displacement_target_mean"]
    ) < 1e-6, "scene normalisation must be invariant to affine rescaling"
    # a single probe has no within-scene ordering to learn
    _, single = displacement_token_bce(
        torch.zeros(1, 48), memberships[0:1], torch.tensor([0.004]), normalize="scene"
    )
    assert abs(single["counterfactual_displacement_target_mean"] - 0.4) < 1e-6


def test_displacement_token_bce_weighting_is_a_token_weighted_mean() -> None:
    logits = torch.zeros(2, 3)
    membership = torch.tensor([[True, False, False], [True, True, False]])
    displacements = torch.tensor([0.0, 1.0])
    loss, info = displacement_token_bce(
        logits,
        membership,
        displacements,
        normalize="absolute",
        sample_weight=torch.tensor([1.0, 3.0]),
    )
    # BCE(logit=0) is log(2) for every target. A correctly normalized weighted
    # mean remains log(2), even when rows supervise different token counts.
    assert torch.allclose(loss, torch.tensor(math.log(2.0)), atol=1e-6)
    assert abs(info["counterfactual_unweighted_bce"] - math.log(2.0)) < 1e-6


def test_displacement_teacher_can_learn_a_synthetic_relative_ordering() -> None:
    """Learnability gate for the displacement teacher (2026-09-11 P3 route).

    The real 3,768-scene run showed a flat loss, which is only interpretable if
    the objective is known to be learnable.  This control builds a scene whose
    tile displacements are an exact function of the tile's mean token and checks
    that the selector learns to rank the tiles inside a scene.
    """
    torch.manual_seed(5)
    groups, per_tile, token_dim = 12, 4, 32
    positions = _tile_grid_positions(per_tile, 3, 4)
    memberships = _tile_memberships(positions, 3, 4)
    selector = DynamicTokenSelector(token_dim=token_dim, hidden_dim=64, position_dim=16)
    optimizer = torch.optim.AdamW(selector.parameters(), lr=3e-3)
    direction = torch.randn(token_dim)
    direction = direction / direction.norm()
    ego = torch.zeros(1, 2)
    command = torch.nn.functional.one_hot(torch.tensor([1]), num_classes=3).float()

    def scene():
        tokens = torch.randn(1, groups * per_tile, token_dim)
        latent = tokens.view(groups, per_tile, token_dim).mean(dim=1) @ direction
        return tokens, latent

    first = last = None
    for step in range(400):
        tokens, latent = scene()
        logits = selector(
            tokens[0].unsqueeze(0).expand(groups, -1, -1),
            positions.expand(groups, -1, -1),
            ego.expand(groups, -1),
            command.expand(groups, -1),
        )
        loss, _ = displacement_token_bce(logits, memberships, latent, normalize="scene")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == 0:
            first = float(loss)
        last = float(loss)
    assert last < first - 0.03, f"displacement loss did not move: {first:.4f} -> {last:.4f}"

    torch.manual_seed(99)
    concordant = total = 0
    for _ in range(20):
        tokens, latent = scene()
        logits = selector(
            tokens[0].unsqueeze(0).expand(groups, -1, -1),
            positions.expand(groups, -1, -1),
            ego.expand(groups, -1),
            command.expand(groups, -1),
        )
        scores = torch.stack(
            [logits[row][memberships[row]].mean() for row in range(groups)]
        )
        for i in range(groups):
            for j in range(i + 1, groups):
                if latent[i] == latent[j]:
                    continue
                total += 1
                concordant += int((scores[i] - scores[j]) * (latent[i] - latent[j]) > 0)
    assert total > 0
    assert concordant / total > 0.8, f"within-scene ranking not learned: {concordant}/{total}"


def test_selector_can_overfit_a_fixed_dynamic_teacher() -> None:
    torch.manual_seed(7)
    selector = DynamicTokenSelector(token_dim=8, hidden_dim=24, position_dim=8)
    optimizer = torch.optim.AdamW(selector.parameters(), lr=3e-3)
    tokens = torch.randn(4, 20, 8)
    positions = torch.rand(4, 20, 3)
    ego = torch.randn(4, 2)
    command = torch.nn.functional.one_hot(torch.arange(4) % 3, num_classes=3).float()
    teacher_scores = tokens[..., 0] + 0.5 * positions[..., 1]
    labels = online_topk_labels(teacher_scores, keep_ratio=0.375)

    for _ in range(160):
        logits = selector(tokens, positions, ego, command)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    metrics = selector_metrics(selector(tokens, positions, ego, command), labels)
    assert metrics["selector_topk_overlap"] > 0.9


def test_condition_position_time_mode_ignores_token_content() -> None:
    torch.manual_seed(13)
    selector = DynamicTokenSelector(
        token_dim=8,
        hidden_dim=24,
        position_dim=8,
        feature_mode="condition_position_time",
    )
    positions = torch.rand(2, 12, 3)
    ego = torch.randn(2, 2)
    command = torch.nn.functional.one_hot(torch.arange(2), num_classes=3).float()
    first = selector(
        torch.randn(2, 12, 8), positions, ego, command, timestep=torch.tensor([1000, 716])
    )
    second = selector(
        torch.randn(2, 12, 8), positions, ego, command, timestep=torch.tensor([1000, 716])
    )
    torch.testing.assert_close(first, second)


def test_runtime_scorer_loads_training_checkpoint_and_scores_last_history(tmp_path) -> None:
    from safetensors.torch import save_file

    network = RuntimeSelector(token_dim=8, hidden_dim=256, position_dim=64)
    checkpoint = tmp_path / "selector.safetensors"
    save_file({f"selector.{key}": value for key, value in network.state_dict().items()}, checkpoint)
    layout = build_driveva_layout(f=3, h=2, w=2, num_cond_latents=2, traj_len=2, traj_prefix_len=1)
    tokens = torch.randn(1, layout.total_length, 8)
    domain = build_domain("last_history", layout, tokens.device)
    ctx = TokenContext(
        tokens=tokens,
        layout=layout,
        domain=domain,
        metadata={
            "selector_ego_state": [4.0, -0.2],
            "selector_command": [0.0, 1.0, 0.0],
        },
    )
    scorer = LearnedPlanningSelectorScorer(str(checkpoint), layer=15, token_dim=8)
    scores = scorer.score(ctx)
    assert scores.shape == (1, 4)
    assert torch.isfinite(scores).all()
    assert ((scores > 0) & (scores < 1)).all()


def test_runtime_scorer_preserves_per_sample_conditions_in_a_batch() -> None:
    layout = build_driveva_layout(
        f=3, h=2, w=2, num_cond_latents=2, traj_len=2, traj_prefix_len=1
    )
    tokens = torch.randn(2, layout.total_length, 8)
    ctx = TokenContext(
        tokens=tokens,
        layout=layout,
        domain=build_domain("last_history", layout, tokens.device),
        metadata={
            "selector_ego_state": [[1.0, 2.0], [3.0, 4.0]],
            "selector_command": [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        },
    )
    ego, command = LearnedPlanningSelectorScorer._conditions(ctx, torch.float32)
    assert ego.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert command.tolist() == [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]


def test_runtime_scorer_can_cache_features_before_compression_layer(tmp_path) -> None:
    from safetensors.torch import save_file

    network = RuntimeSelector(token_dim=8, hidden_dim=256, position_dim=64)
    checkpoint = tmp_path / "selector.safetensors"
    save_file(
        {f"selector.{key}": value for key, value in network.state_dict().items()},
        checkpoint,
    )
    layout = build_driveva_layout(
        f=3, h=2, w=2, num_cond_latents=2, traj_len=2, traj_prefix_len=1
    )
    domain = build_domain("last_history", layout, "cpu")

    def context(layer, diffusion, tokens):
        return TokenContext(
            tokens=tokens,
            layout=layout,
            domain=domain,
            scene_token="scene-a",
            diffusion_rank=diffusion,
            layer_idx=layer,
            metadata={"model_name": "dit"},
        )

    scorer = LearnedPlanningSelectorScorer(
        str(checkpoint), layer=2, feature_layer=1, token_dim=8
    )
    feature_ctx = context(1, 1000, torch.randn(1, layout.total_length, 8))
    scorer.observe(feature_ctx)
    expected = scorer._feature_scores[scorer._feature_key(feature_ctx)][0]
    source_ctx = context(2, 1000, torch.randn(1, layout.total_length, 8))
    actual = scorer.score(source_ctx)
    torch.testing.assert_close(actual, expected)
    assert source_ctx.metadata["score_diagnostics"]["feature_cache_reused"] is True

    with pytest.raises(RuntimeError, match="no cached feature-layer scores"):
        scorer.score(context(2, 716, torch.randn(1, layout.total_length, 8)))
    scorer.reset_observations()
    assert scorer._feature_scores == {}


def test_scene_manifest_selects_temporally_spread_windows(tmp_path) -> None:
    metadata = tmp_path / "scene.pkl"
    frames = [
        {"token": f"frame-{index}", "roadblock_ids": ["route"], "scene_token": "scene-a"}
        for index in range(20)
    ]
    with metadata.open("wb") as handle:
        pickle.dump(frames, handle)
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(
        json.dumps({"scene_token": "scene-a", "metadata_path": str(metadata)}) + "\n",
        encoding="utf-8",
    )

    rows = _read_jsonl_manifest(str(manifest))
    assert _manifest_scene_tokens(rows, str(manifest)) == ["scene-a"]
    selected = _representative_frame_tokens(
        rows,
        manifest_path=str(manifest),
        num_history_frames=3,
        num_future_frames=2,
        frame_interval=1,
        windows_per_scene=2,
    )
    assert selected == ["frame-7", "frame-12"]


def test_scene_manifest_rejects_duplicate_semantic_scenes(tmp_path) -> None:
    manifest = tmp_path / "duplicate.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps({"scene_token": "same", "metadata_path": "first.pkl"}),
                json.dumps({"scene_token": "same", "metadata_path": "second.pkl"}),
            ]
        ),
        encoding="utf-8",
    )
    rows = _read_jsonl_manifest(str(manifest))
    try:
        _manifest_scene_tokens(rows, str(manifest))
    except ValueError as exc:
        assert "duplicate scene_token" in str(exc)
    else:
        raise AssertionError("duplicate semantic scene tokens must be rejected")


# ---------------------------------------------------------------------------
# Learned-selector position convention (regression, 2026-09-12)
# ---------------------------------------------------------------------------
def _selector_context_for_domain(domain_name: str):
    """Minimal TokenContext over the REAL 2-latent history layout."""

    from videopress.core.domain import build_domain
    from videopress.core.layout import build_driveva_layout

    layout = build_driveva_layout(
        f=4, h=8, w=10, num_cond_latents=2, traj_len=2, traj_prefix_len=1
    )
    domain = build_domain(domain_name, layout, torch.device("cpu"))

    class _Ctx:
        pass

    ctx = _Ctx()
    ctx.layout = layout
    ctx.domain = domain
    ctx.batch_size = 2
    ctx.tokens = torch.zeros(2, layout.total_length, 8)
    return ctx, layout


def test_learned_selector_temporal_coordinate_matches_the_teacher() -> None:
    """The newest latent must carry t = 1.0, never t = 0.

    The training-time teacher writes ``t_idx = num_cond_latents - 1 -
    cf_latent_index`` with ``cf_latent_index`` counted BACK from the newest, so
    for the default two-latent case the newest latent gets ``t = 1.0`` (its own
    comment: "exactly the previous constant 1.0").  The newest latent is also
    the one stored SECOND, i.e. the deployed ``last_history`` candidate range.

    An earlier revision "simplified" this to a distance-back-from-newest
    reading, which silently INVERTED the coordinate for the deployed domain and
    changed every selection score.  This test pins the convention.
    """

    # The deployed domain covers only the newest latent.
    ctx, layout = _selector_context_for_domain("last_history")
    positions = LearnedPlanningSelectorScorer._positions(ctx, torch.float32)
    assert positions.shape == (2, ctx.domain.candidate_indices.numel(), 3)
    assert float(positions[..., 0].min()) == 1.0
    assert float(positions[..., 0].max()) == 1.0

    # Both-latent domain: older = 0, newest = 1, split at tokens_per_latent.
    ctx2, _ = _selector_context_for_domain("history")
    positions2 = LearnedPlanningSelectorScorer._positions(ctx2, torch.float32)
    t = positions2[0, :, 0]
    per_latent = int(ctx2.layout.tokens_per_latent)
    assert t.numel() == 2 * per_latent
    assert float(t[:per_latent].min()) == 0.0
    assert float(t[:per_latent].max()) == 0.0
    assert float(t[per_latent:].min()) == 1.0
    assert float(t[per_latent:].max()) == 1.0


def test_learned_selector_position_geometry_is_per_latent() -> None:
    """Each latent keeps its own full (y, x) grid; only t separates them."""

    ctx, layout = _selector_context_for_domain("history")
    positions = LearnedPlanningSelectorScorer._positions(ctx, torch.float32)[0]
    per_latent = int(layout.tokens_per_latent)
    for lo, hi in ((0, per_latent), (per_latent, 2 * per_latent)):
        y = positions[lo:hi, 1]
        x = positions[lo:hi, 2]
        assert float(y.min()) == 0.0 and float(y.max()) == 1.0
        assert float(x.min()) == 0.0 and float(x.max()) == 1.0


def test_learned_selector_rejects_non_history_domains() -> None:
    ctx, _ = _selector_context_for_domain("future_video")
    try:
        LearnedPlanningSelectorScorer._positions(ctx, torch.float32)
    except ValueError as exc:
        assert "history domain" in str(exc)
    else:
        raise AssertionError("a non-history domain must be rejected")
