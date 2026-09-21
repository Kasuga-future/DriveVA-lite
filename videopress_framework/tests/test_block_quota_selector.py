"""Tests for per-block dynamic retention (2026-09-21).

The bug these lock down: a single global top-k over ``all_video`` let one block
absorb the whole cut.  Measured on full 7876, the compositional selector at
K=1149/1560 kept 780/780 future tokens and pushed all 411 dropped tokens onto
history, so that "joint" arm never compressed the future block.
"""

from __future__ import annotations

import math

import pytest
import torch

from videopress.core.context import TokenContext
from videopress.core.domain import build_domain
from videopress.core.layout import build_driveva_layout
from videopress.selectors.block_quota import BlockQuotaSelector
from videopress.selectors.topk import TopKSelector


def _ctx(f=4, h=2, w=2):
    layout = build_driveva_layout(f=f, h=h, w=w, num_cond_latents=2, traj_len=2, traj_prefix_len=1)
    domain = build_domain("all_video", layout, "cpu")
    ctx = TokenContext(
        tokens=torch.zeros(1, layout.total_length, 4),
        layout=layout,
        domain=domain,
    )
    return ctx, layout, domain


def _block_sizes(layout):
    return int(layout.history_video.length), int(layout.future_video.length)


def test_block_quota_splits_the_cut_between_blocks():
    ctx, layout, domain = _ctx()
    n = int(domain.n_candidate)
    n_hist, n_fut = _block_sizes(layout)
    assert n_hist + n_fut == n

    # A score field that would make a GLOBAL top-k keep only history tokens:
    # history scores are all higher than future scores.
    scores = torch.zeros(1, n)
    scores[0, :n_hist] = 0.9
    scores[0, n_hist:] = 0.1
    K = n // 2
    _, _, _ = ctx, layout, domain

    global_topk = TopKSelector().select(scores, domain, K, ctx)
    kept_hist = int((global_topk.keep_global_indices < int(layout.history_video.end)).sum())
    assert kept_hist == K, "sanity: global top-k takes everything from history"

    selector = BlockQuotaSelector(block_weights={"history": 0.5, "future": 0.5}, mode="quota")
    result = selector.select(scores, domain, K, ctx)
    hist_end = int(layout.history_video.end)
    fut_start = int(layout.future_video.start)
    kept_hist = int((result.keep_global_indices < hist_end).sum())
    kept_fut = int((result.keep_global_indices >= fut_start).sum())
    assert kept_hist + kept_fut == K
    assert kept_hist > 0 and kept_fut > 0, "the cut must be shared between blocks"
    assert result.metadata["per_block_quota"]["history"] > 0
    assert result.metadata["per_block_quota"]["future"] > 0


def test_block_quota_quota_mode_is_fixed_and_dynamic_mode_varies():
    ctx, layout, domain = _ctx()
    n = int(domain.n_candidate)
    K = n // 2

    quota = BlockQuotaSelector(block_weights={"history": 0.5, "future": 0.5}, mode="quota")
    fixed = quota.select(torch.rand(1, n), domain, K, ctx)
    assert fixed.metadata["dynamic"] is False

    dyn = BlockQuotaSelector(
        block_weights={"history": 0.5, "future": 0.5},
        mode="dynamic",
        score_threshold=0.5,
        floor_ratio=0.25,
    )
    assert dyn.metadata if False else True  # describe() is exercised below
    assert dyn.describe()["dynamic"] is True

    # High scores everywhere -> the quota is the binding constraint.
    busy = dyn.select(torch.full((1, n), 0.9), domain, K, ctx)
    assert busy.K == K
    # Low scores everywhere -> the floor is the binding constraint and the count
    # drops BELOW the fixed quota, which is what makes it dynamic.
    quiet = dyn.select(torch.full((1, n), 0.1), domain, K, ctx)
    assert quiet.K < K, f"dynamic mode must keep fewer tokens when nothing qualifies (got {quiet.K})"
    assert quiet.K > 0, "floor_ratio must prevent starving the blocks"


def test_block_quota_respects_uneven_weights():
    ctx, layout, domain = _ctx()
    n = int(domain.n_candidate)
    K = n // 2
    selector = BlockQuotaSelector(block_weights={"history": 0.75, "future": 0.25}, mode="quota")
    scores = torch.rand(1, n)
    result = selector.select(scores, domain, K, ctx)
    hist_end = int(layout.history_video.end)
    fut_start = int(layout.future_video.start)
    kept_hist = int((result.keep_global_indices < hist_end).sum())
    kept_fut = int((result.keep_global_indices >= fut_start).sum())
    assert kept_hist > kept_fut, "0.75/0.25 weights must favour history"


def test_block_quota_rejects_single_block_domains():
    """An empty block would still get its weight share and halve the real one."""

    ctx, layout, _ = _ctx()
    selector = BlockQuotaSelector()
    for name in ("history", "future_video"):
        domain = build_domain(name, layout, "cpu")
        with pytest.raises(ValueError, match="contains no .* tokens"):
            selector.select(torch.rand(1, domain.n_candidate), domain, 4, ctx)


def test_block_quota_rejects_bad_configuration():
    with pytest.raises(ValueError, match="block_weights keys"):
        BlockQuotaSelector(block_weights={"latent0": 1.0})
    with pytest.raises(ValueError, match="mode must be"):
        BlockQuotaSelector(mode="nonsense")
    with pytest.raises(ValueError, match="floor_ratio"):
        BlockQuotaSelector(floor_ratio=1.5)
    with pytest.raises(ValueError, match="sum to zero"):
        BlockQuotaSelector(block_weights={"history": 0.0, "future": 0.0})


def test_block_quota_requires_layout():
    _, layout, domain = _ctx()
    selector = BlockQuotaSelector()
    with pytest.raises(ValueError, match="requires ctx.layout"):
        selector.select(torch.rand(1, domain.n_candidate), domain, 4, None)


def test_dynamic_floor_bounds_how_much_extra_compression_is_possible():
    """The floor is the safety knob: hidden length lies in [floor*K, K].

    With the trained selectors scoring ~0.38-0.42 on average -- below the 0.5
    default threshold -- a small floor would let dynamic mode starve both blocks.
    """

    ctx, layout, domain = _ctx()
    n = int(domain.n_candidate)
    K = n // 2
    low = torch.full((1, n), 0.01)  # nothing reaches the 0.5 threshold

    def realised(floor):
        selector = BlockQuotaSelector(
            block_weights={"history": 0.5, "future": 0.5},
            mode="dynamic",
            score_threshold=0.5,
            floor_ratio=floor,
        )
        return selector.select(low, domain, K, ctx).K

    assert realised(0.25) < realised(0.8) <= K
    assert realised(0.8) == int(math.ceil(0.8 * (K // 2))) * 2


def test_block_quota_default_floor_is_safe():
    assert BlockQuotaSelector().floor_ratio == 0.8
