"""Regression tests for the F3 / joint selector candidate-latent plumbing.

2026-09-21: ``--selector-candidate-latents`` lets the online-selector teacher
capture an explicit storage-coordinate latent range instead of the legacy single
newest history latent.  ``"2,3"`` trains the F3 future selector and ``"0,1,2,3"``
trains the joint history+future selector.

The load-bearing invariant is that the *training* temporal coordinate and the
*deployment* ``LearnedPlanningSelectorScorer._positions`` coordinate are the same
storage index, otherwise a checkpoint trained on future candidates would be
silently evaluated out of distribution.  These tests pin that contract down.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from videopress.core.domain import build_domain
from videopress.core.layout import build_driveva_layout
from videopress.scorers.learned_selector import LearnedPlanningSelectorScorer


def _ctx(layout, domain_name: str, batch_size: int = 1):
    domain = build_domain(domain_name, layout, "cpu")
    return SimpleNamespace(
        layout=layout,
        domain=domain,
        batch_size=batch_size,
        tokens=SimpleNamespace(device=torch.device("cpu")),
    )


def _temporal_coordinates(layout, domain_name, mode="storage"):
    ctx = _ctx(layout, domain_name)
    positions = LearnedPlanningSelectorScorer._positions(ctx, torch.float32, mode)
    return positions[0, :, 0]


def _expected_t(coordinates, tokens_per_latent):
    """Collapse the per-token t vector into the ordered list of latent indices."""
    collapsed = []
    for index in range(0, coordinates.numel(), tokens_per_latent):
        value = float(coordinates[index].item())
        assert torch.allclose(
            coordinates[index : index + tokens_per_latent],
            torch.full((tokens_per_latent,), value),
        ), "every token of one latent must share one temporal coordinate"
        collapsed.append(value)
    return collapsed


def test_future_domain_deployment_coordinates_are_storage_indices():
    # 4 latent frames, 2 conditioned history latents -> future latents 2 and 3.
    layout = build_driveva_layout(4, 2, 2, 2, 3, 1)
    per_latent = int(layout.tokens_per_latent)
    t = _temporal_coordinates(layout, "future_video", mode="storage")
    assert _expected_t(t, per_latent) == [2.0, 3.0]


def test_all_video_deployment_coordinates_are_storage_indices():
    layout = build_driveva_layout(4, 2, 2, 2, 3, 1)
    per_latent = int(layout.tokens_per_latent)
    t = _temporal_coordinates(layout, "all_video", mode="storage")
    assert _expected_t(t, per_latent) == [0.0, 1.0, 2.0, 3.0]


def test_history_domain_training_coordinate_is_unchanged():
    # The legacy candidate is the newest history latent, whose training-time
    # coordinate was the constant num_cond_latents - 1 = 1.
    layout = build_driveva_layout(4, 2, 2, 2, 3, 1)
    per_latent = int(layout.tokens_per_latent)
    t = _temporal_coordinates(layout, "history", mode="storage")
    matched = [value for value in _expected_t(t, per_latent) if value == 1.0]
    assert matched, "newest history latent must keep temporal coordinate 1.0"
    assert set(_expected_t(t, per_latent)) == {0.0, 1.0}


def test_candidate_range_token_counts_match_domains():
    """The capture range a mode maps to must equal the deployment domain size."""
    layout = build_driveva_layout(4, 2, 2, 2, 3, 1)
    per_latent = int(layout.tokens_per_latent)
    cases = {
        "2,3": ("future_video", 2 * per_latent),
        "0,1,2,3": ("all_video", 4 * per_latent),
    }
    for spec, (domain_name, expected_candidates) in cases.items():
        start, end = int(spec.split(",")[0]), int(spec.split(",")[-1]) + 1
        assert (end - start) * per_latent == expected_candidates
        domain = build_domain(domain_name, layout, "cpu")
        assert domain.candidate_indices.numel() == expected_candidates


def _trainer_module():
    """Import the training entry point the same way its launch script does.

    ``train_navsim_v1.py`` imports the sibling ``navsim_dataset`` module by bare
    name, so the train directory must be on ``sys.path`` (the launch script puts
    it there through PYTHONPATH).
    """
    import importlib
    import sys
    from pathlib import Path

    train_dir = (
        Path(__file__).resolve().parents[2] / "examples" / "wanvideo" / "driveva_train"
    )
    if str(train_dir) not in sys.path:
        sys.path.insert(0, str(train_dir))
    return importlib.import_module("examples.wanvideo.driveva_train.train_navsim_v1")


def test_selector_candidate_latents_defaults_to_legacy_mode():
    module = _trainer_module()
    args = module.parse_args(_minimal_train_argv())
    assert args.selector_candidate_latents == ""


def test_selector_candidate_latents_accepts_contiguous_ranges():
    module = _trainer_module()
    for spec in ("2,3", "0,1,2,3", "3"):
        args = module.parse_args(_minimal_train_argv() + ["--selector-candidate-latents", spec])
        assert args.selector_candidate_latents == spec


def test_selector_candidate_latents_rejects_bad_ranges():
    module = _trainer_module()
    for spec in ("1,3", "3,1", "-1,0", "2,2", "a"):
        with pytest.raises((ValueError, SystemExit)):
            module.main(_minimal_train_argv() + ["--selector-candidate-latents", spec])


def test_selector_candidate_latents_requires_gradient_abs_teacher():
    module = _trainer_module()
    with pytest.raises((ValueError, SystemExit)):
        module.main(
            _minimal_train_argv()
            + [
                "--selector-candidate-latents",
                "0,1,2,3",
                "--selector-teacher-mode",
                "displacement",
            ]
        )


def _minimal_train_argv():
    return [
        "--repo_root",
        "/tmp/repo",
        "--navsim_log_path",
        "/tmp/navsim",
        "--sensor_blobs_path",
        "/tmp/sensor",
        "--local_model_path",
        "/tmp/models",
        "--output_path",
        "/tmp/out",
    ]


# ---------------------------------------------------------------------------
# Deployment path for the F3 / joint checkpoints (2026-09-21)
#
# The plain (non ``all_video_history_only``) learned scorer had never been
# scored on ``future_video`` or ``all_video``; only the history-only variant had
# all_video coverage.  These are exactly the paths the F3 and joint evaluations
# take, so pin them without needing a GPU.
# ---------------------------------------------------------------------------


def test_plain_learned_scorer_scores_future_and_joint_domains(tmp_path):
    from safetensors.torch import save_file

    from videopress.core.context import TokenContext
    from videopress.training.online_selector import (
        DynamicTokenSelector as TrainingSelector,
    )

    network = TrainingSelector(token_dim=8, hidden_dim=256, position_dim=64)
    checkpoint = tmp_path / "selector.safetensors"
    save_file(
        {f"selector.{key}": value for key, value in network.state_dict().items()},
        checkpoint,
    )

    layout = build_driveva_layout(
        f=4, h=2, w=2, num_cond_latents=2, traj_len=2, traj_prefix_len=1
    )
    tokens = torch.randn(1, layout.total_length, 8)
    per_latent = int(layout.tokens_per_latent)

    scorer = LearnedPlanningSelectorScorer(
        str(checkpoint), layer=15, token_dim=8, future_position_mode="storage"
    )
    for domain_name, expected_candidates, expected_t in (
        ("future_video", 2 * per_latent, [2.0, 3.0]),
        ("all_video", 4 * per_latent, [0.0, 1.0, 2.0, 3.0]),
    ):
        ctx = TokenContext(
            tokens=tokens,
            layout=layout,
            domain=build_domain(domain_name, layout, "cpu"),
            scene_token="scene-a",
            diffusion_rank=1000,
            metadata={
                "selector_ego_state": [4.0, -0.2],
                "selector_command": [0.0, 1.0, 0.0],
            },
        )
        scores = scorer.score(ctx)
        assert scores.shape == (1, expected_candidates)
        assert torch.isfinite(scores).all()
        assert ((scores > 0) & (scores < 1)).all()
        assert "checkpoint" in ctx.metadata["score_diagnostics"]
        positions = LearnedPlanningSelectorScorer._positions(ctx, torch.float32)
        assert _expected_t(positions[0, :, 0], per_latent) == expected_t


# ---------------------------------------------------------------------------
# Compositional history+future selector (2026-09-21)
#
# "两个已训练 selector 的组合式同时压缩": one press over all_video, but each
# candidate is scored by the network trained for its own block.  The load-bearing
# property is exact fidelity -- the composed score vector must equal the history
# network's scores on the history slots and the future network's scores on the
# future slots, with no cross-contamination.
# ---------------------------------------------------------------------------


def _tiny_selector_checkpoint(tmp_path, name, seed):
    from safetensors.torch import save_file

    from videopress.training.online_selector import (
        DynamicTokenSelector as TrainingSelector,
    )

    torch.manual_seed(seed)
    network = TrainingSelector(token_dim=8, hidden_dim=256, position_dim=64)
    path = tmp_path / name
    save_file(
        {f"selector.{key}": value for key, value in network.state_dict().items()},
        path,
    )
    return str(path)


def _block_masks(layout, candidates):
    history_span = layout.history_video
    future_span = layout.future_video
    in_history = (candidates >= int(history_span.start)) & (
        candidates < int(history_span.end)
    )
    in_future = (candidates >= int(future_span.start)) & (
        candidates < int(future_span.end)
    )
    return in_history, in_future


def test_composed_selector_is_exactly_its_two_component_networks(tmp_path):
    from videopress.core.context import TokenContext
    from videopress.scorers.learned_selector import (
        ComposedLearnedPlanningSelectorScorer,
        LearnedPlanningSelectorScorer,
    )

    history_ckpt = _tiny_selector_checkpoint(tmp_path, "history.safetensors", 11)
    future_ckpt = _tiny_selector_checkpoint(tmp_path, "future.safetensors", 22)
    layout = build_driveva_layout(
        f=4, h=2, w=2, num_cond_latents=2, traj_len=2, traj_prefix_len=1
    )
    tokens = torch.randn(1, layout.total_length, 8)
    conditions = {
        "selector_ego_state": [4.0, -0.2],
        "selector_command": [0.0, 1.0, 0.0],
    }

    def context(domain_name):
        return TokenContext(
            tokens=tokens,
            layout=layout,
            domain=build_domain(domain_name, layout, "cpu"),
            scene_token="scene-a",
            diffusion_rank=1000,
            metadata=dict(conditions),
        )

    composed = ComposedLearnedPlanningSelectorScorer(
        history_checkpoint=history_ckpt,
        future_checkpoint=future_ckpt,
        layer=15,
        token_dim=8,
    )
    all_ctx = context("all_video")
    scores = composed.score(all_ctx)
    candidates = all_ctx.domain.candidate_indices
    in_history, in_future = _block_masks(layout, candidates)

    assert scores.shape == (1, int(candidates.numel()))
    assert int(in_history.sum()) == int(layout.history_video.length)
    assert int(in_future.sum()) == int(layout.future_video.length)
    assert torch.isfinite(scores).all()
    assert ((scores > 0) & (scores < 1)).all()

    history_reference = LearnedPlanningSelectorScorer(
        history_ckpt, layer=15, token_dim=8
    ).score(context("history"))
    future_reference = LearnedPlanningSelectorScorer(
        future_ckpt, layer=15, token_dim=8
    ).score(context("future_video"))

    # Each block must be reproduced bit-for-bit by its own trained network.
    torch.testing.assert_close(scores[:, in_history], history_reference)
    torch.testing.assert_close(scores[:, in_future], future_reference)
    # ... and the two networks must actually be different, otherwise the test
    # above would pass even if one checkpoint were used for both blocks.
    assert not torch.allclose(
        history_reference[:, : min(history_reference.shape[1], future_reference.shape[1])],
        future_reference[:, : min(history_reference.shape[1], future_reference.shape[1])],
    )
    diagnostics = all_ctx.metadata["score_diagnostics"]
    assert diagnostics["composed"] is True
    assert diagnostics["history_candidates"] == int(layout.history_video.length)
    assert diagnostics["future_candidates"] == int(layout.future_video.length)


def test_composed_selector_rejects_non_video_domains(tmp_path):
    from videopress.core.context import TokenContext
    from videopress.scorers.learned_selector import (
        ComposedLearnedPlanningSelectorScorer,
    )

    history_ckpt = _tiny_selector_checkpoint(tmp_path, "h.safetensors", 1)
    future_ckpt = _tiny_selector_checkpoint(tmp_path, "f.safetensors", 2)
    layout = build_driveva_layout(
        f=4, h=2, w=2, num_cond_latents=2, traj_len=2, traj_prefix_len=1
    )
    composed = ComposedLearnedPlanningSelectorScorer(
        history_checkpoint=history_ckpt,
        future_checkpoint=future_ckpt,
        layer=15,
        token_dim=8,
    )
    ctx = TokenContext(
        tokens=torch.randn(1, layout.total_length, 8),
        layout=layout,
        domain=build_domain("history", layout, "cpu"),
    )
    with pytest.raises(ValueError, match="requires an all_video/video domain"):
        composed.score(ctx)


def test_runner_exposes_the_composed_scorer_and_both_checkpoints():
    module = _runner_module()
    args = module.parse_args(
        [
            "--persistent-scorer",
            "composed_learned_planning_selector",
            "--domain",
            "all_video",
            "--persistent-history-selector-checkpoint",
            "/tmp/history.safetensors",
            "--persistent-future-selector-checkpoint",
            "/tmp/future.safetensors",
        ]
    )
    assert args.persistent_scorer == "composed_learned_planning_selector"
    assert args.persistent_history_selector_checkpoint is not None
    assert args.persistent_future_selector_checkpoint is not None


def _runner_module():
    import importlib
    import sys
    from pathlib import Path

    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    return importlib.import_module("run_official_navsim_press")
