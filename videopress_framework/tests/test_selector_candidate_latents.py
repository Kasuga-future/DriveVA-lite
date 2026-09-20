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
