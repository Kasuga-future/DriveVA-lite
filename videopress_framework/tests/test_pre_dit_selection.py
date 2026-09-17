from __future__ import annotations

from types import SimpleNamespace

import torch

import diffsynth.models.wan_video_dit as wan_dit
from diffsynth.models.wan_video_dit import WanModel
from diffsynth.pipelines.wan_video_new import model_fn_wan_video
from videopress.adapters.driveva import DriveVAAdapter
from videopress.core.domain import build_domain
from videopress.core.layout import build_driveva_layout
from videopress.core.plan import validate_protocol
from videopress.core.runtime import VideoPressRuntime
from videopress.factory import build_press


def _press(keep_ratio: float = 0.5):
    return build_press(
        {
            "name": "scorer_press",
            "injection_point": "block_input",
            "domain": "history",
            "scorer": {"name": "token_norm"},
            "selector": {"name": "topk"},
            "operator": {"name": "hidden_prune"},
            "budget": {
                "type": "ratio",
                "value": keep_ratio,
                "reference": "eligible",
            },
        }
    )


def test_block_input_protocol_and_full_sequence_round_trip() -> None:
    press = _press()
    validate_protocol(press, "physical")
    adapter = DriveVAAdapter()
    runtime = VideoPressRuntime(press=press, mode="physical", adapter=adapter)
    layout = build_driveva_layout(
        f=4, h=2, w=2, num_cond_latents=2, traj_len=3, traj_prefix_len=0
    )
    runtime.layout = layout
    runtime.begin_sample(SimpleNamespace(scene_token="scene-a"), layout=layout)

    class DummyDiT:
        blocks = [object(), object(), object()]

    class DummyPipe:
        dit = DummyDiT()

    pipe = DummyPipe()
    runtime.install(pipe)
    try:
        controller = pipe.dit._tokenpress_pre_dit_controller
        tokens = torch.arange(layout.total_length * 4, dtype=torch.float32).reshape(
            1, layout.total_length, 4
        )
        freqs = torch.randn(layout.total_length, 1, 6)
        t_mod = torch.randn(1, layout.total_length, 6, 4)
        short, short_freqs, short_t_mod = controller.begin_forward(
            tokens, freqs, t_mod, num_blocks=3
        )

        # 8 history candidates -> keep 4; every other token is protected.
        assert short.shape == (1, layout.total_length - 4, 4)
        assert short_freqs.shape[1] == short.shape[1]
        assert short_t_mod.shape[1] == short.shape[1]
        event = runtime.events[-1].result.metadata
        assert event["pre_dit"] is True
        assert event["selection_source_layer"] == -1
        assert event["hidden_sequence_compressed_layer_count"] == 3

        propagated = short + 1
        restored = controller.finish_forward(propagated)
        keep = runtime.last_result.mapping.output_to_input
        expected = torch.zeros_like(tokens)
        expected.scatter_(1, keep.unsqueeze(-1).expand_as(propagated), propagated)
        torch.testing.assert_close(restored, expected)
    finally:
        runtime.remove(pipe)
    assert not hasattr(pipe.dit, "_tokenpress_pre_dit_controller")


def test_block_input_rejects_kv_only_operator() -> None:
    press = build_press(
        {
            "name": "scorer_press",
            "injection_point": "block_input",
            "domain": "history",
            "scorer": {"name": "token_norm"},
            "selector": {"name": "topk"},
            "operator": {"name": "kv_prune"},
            "budget": {"type": "ratio", "value": 0.5, "reference": "eligible"},
        }
    )
    try:
        validate_protocol(press, "physical")
    except ValueError as exc:
        assert "hidden-token-compatible" in str(exc)
    else:
        raise AssertionError("KV-only operator was accepted at block_input")


def test_pre_dit_merge_matches_prune_length_and_keeps_every_token() -> None:
    """Hidden merge is the information-preserving counterpart of hidden prune.

    The merge arm is only a fair comparison if it costs exactly as much as the
    prune arm, i.e. it must emit the identical output sequence length while
    still carrying the dropped tokens' content.  It must also remain a total,
    reversible-in-coverage mapping so the pre-DiT controller can restore the
    original layout before the heads.
    """

    keep_ratio = 0.5
    layout = build_driveva_layout(4, 2, 2, 2, 3, 0)
    tokens = torch.arange(layout.total_length * 6, dtype=torch.float32).reshape(
        1, layout.total_length, 6
    )
    freqs = torch.randn(layout.total_length, 1, 8)
    t_mod = torch.randn(1, layout.total_length, 6, 6)

    def _run(press):
        adapter = DriveVAAdapter()
        runtime = VideoPressRuntime(press=press, mode="physical", adapter=adapter)
        runtime.layout = layout
        runtime.begin_sample(SimpleNamespace(scene_token="merge-scene"), layout)
        pipe = SimpleNamespace(dit=SimpleNamespace(blocks=[object()] * 2), dit2=None)
        runtime.install(pipe)
        try:
            controller = pipe.dit._tokenpress_pre_dit_controller
            short, short_freqs, short_t_mod = controller.begin_forward(
                tokens, freqs, t_mod, num_blocks=2
            )
            result = runtime.last_result
            restored = controller.finish_forward(short)
            return short, short_freqs, short_t_mod, result, restored
        finally:
            runtime.remove(pipe)

    merge_press = build_press(
        {
            "name": "similarity_merge",
            "injection_point": "block_input",
            "domain": "history",
            "feature": "random",
            "seed": 20260915,
            "budget": {"type": "ratio", "value": keep_ratio, "reference": "eligible"},
        }
    )
    validate_protocol(merge_press, "physical")
    prune_press = _press(keep_ratio)
    validate_protocol(prune_press, "physical")

    merged, merged_freqs, merged_t_mod, merge_result, restored = _run(merge_press)
    pruned, _, _, _, _ = _run(prune_press)

    assert merged.shape == pruned.shape, (
        "merge and prune arms must emit identical sequence lengths, got "
        f"{tuple(merged.shape)} vs {tuple(pruned.shape)}"
    )
    assert merged_freqs.shape[1] == merged.shape[1]
    assert merged_t_mod.shape[1] == merged.shape[1]
    assert merge_result.metadata["operator"] == "merge"
    assert merge_result.metadata["merge_multi_token_groups"] > 0

    mapping = merge_result.mapping
    # Merge is lossy in resolution but total in coverage: every input token
    # belongs to exactly one output group.
    assert mapping.input_to_output is not None
    assert int((mapping.input_to_output < 0).sum()) == 0
    assert int(mapping.input_to_output.max()) == merged.shape[1] - 1
    assert mapping.compressed_length == merged.shape[1]

    # Protected (non-candidate) tokens are singleton groups and must survive
    # verbatim at their own position.  The grouping is read back from the dense
    # tensor mapping rather than a Python group list.
    domain = build_domain("history", layout, "cpu")
    inverse = mapping.input_to_output[0]
    sizes = torch.bincount(inverse, minlength=merged.shape[1])
    assert int(sizes.min()) >= 1
    for out_pos in torch.nonzero(sizes == 1).flatten().tolist():
        source = int((inverse == out_pos).nonzero().flatten()[0])
        torch.testing.assert_close(merged[0, out_pos], tokens[0, source])

    # Round trip restores the original layout length and places each merged
    # group at its representative input position.
    assert restored.shape == tokens.shape
    for out_pos, input_index in enumerate(mapping.output_to_input[0].tolist()):
        torch.testing.assert_close(restored[0, input_index], merged[0, out_pos])
    assert domain.n_candidate == layout.history_video.length


def test_pre_dit_merge_with_unit_budget_is_the_identity() -> None:
    layout = build_driveva_layout(4, 2, 2, 2, 3, 0)
    tokens = torch.randn(1, layout.total_length, 6)
    freqs = torch.randn(layout.total_length, 1, 8)
    t_mod = torch.randn(1, layout.total_length, 6, 6)
    press = build_press(
        {
            "name": "similarity_merge",
            "injection_point": "block_input",
            "domain": "history",
            "feature": "random",
            "seed": 1,
            "budget": {"type": "ratio", "value": 1.0, "reference": "eligible"},
        }
    )
    adapter = DriveVAAdapter()
    runtime = VideoPressRuntime(press=press, mode="physical", adapter=adapter)
    runtime.layout = layout
    runtime.begin_sample(SimpleNamespace(scene_token="identity-scene"), layout)
    pipe = SimpleNamespace(dit=SimpleNamespace(blocks=[object()]), dit2=None)
    runtime.install(pipe)
    try:
        short, _, _ = pipe.dit._tokenpress_pre_dit_controller.begin_forward(
            tokens, freqs, t_mod, num_blocks=1
        )
        assert short.shape == tokens.shape
        torch.testing.assert_close(short, tokens)
    finally:
        runtime.remove(pipe)


def test_pre_dit_adaptive_selector_changes_token_count_between_inputs() -> None:
    press = build_press(
        {
            "name": "scorer_press",
            "injection_point": "block_input",
            "domain": "history",
            "scorer": {"name": "token_norm"},
            "selector": {
                "name": "adaptive_mass",
                "ratios": [0.5, 0.75, 1.0],
                "mass_thresholds": [0.60, 0.80],
                "gap_thresholds": [0.02, 0.01],
                "gap_window": 1,
            },
            "operator": {"name": "hidden_prune"},
            "budget": {"type": "ratio", "value": 1.0, "reference": "eligible"},
        }
    )
    adapter = DriveVAAdapter()
    runtime = VideoPressRuntime(press=press, mode="physical", adapter=adapter)
    layout = build_driveva_layout(4, 2, 2, 2, 3, 0)
    pipe = SimpleNamespace(dit=SimpleNamespace(blocks=[object()] * 2), dit2=None)
    runtime.install(pipe)
    controller = pipe.dit._tokenpress_pre_dit_controller
    freqs = torch.randn(layout.total_length, 1, 6)
    t_mod = torch.randn(1, layout.total_length, 6, 4)
    try:
        # Uniform candidate norms have no confident boundary -> retain all 8.
        uniform = torch.ones(1, layout.total_length, 4)
        runtime.begin_sample(SimpleNamespace(scene_token="uniform"), layout)
        short, _, _ = controller.begin_forward(uniform, freqs, t_mod, num_blocks=2)
        assert runtime.last_result.selection.K == 8
        controller.finish_forward(short)

        # Four high-norm candidates and four zero candidates give a clean 50% tier.
        concentrated = torch.zeros_like(uniform)
        concentrated[:, layout.history_video.start : layout.history_video.start + 4] = 10
        runtime.begin_sample(SimpleNamespace(scene_token="concentrated"), layout)
        short, _, _ = controller.begin_forward(
            concentrated, freqs, t_mod, num_blocks=2
        )
        assert runtime.last_result.selection.K == 4
        assert short.shape[1] == layout.total_length - 4
        controller.finish_forward(short)
    finally:
        runtime.remove(pipe)


def test_real_wan_model_fn_invokes_pre_dit_controller(monkeypatch) -> None:
    # The installed flash-attn package is CUDA-only; exercise the exact Wan
    # model_fn on CPU through PyTorch SDPA for this small integration test.
    monkeypatch.setattr(wan_dit, "FLASH_ATTN_3_AVAILABLE", False)
    monkeypatch.setattr(wan_dit, "FLASH_ATTN_2_AVAILABLE", False)
    monkeypatch.setattr(wan_dit, "SAGE_ATTN_AVAILABLE", False)
    dit = WanModel(
        dim=24,
        in_dim=4,
        ffn_dim=48,
        out_dim=4,
        text_dim=12,
        freq_dim=16,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=4,
        num_layers=2,
        has_image_input=False,
        require_vae_embedding=False,
        require_clip_embedding=False,
    ).eval()
    trajectory_head = torch.nn.Linear(24, 3)
    pipe = SimpleNamespace(dit=dit, dit2=None, model_fn=model_fn_wan_video)
    adapter = DriveVAAdapter()
    runtime = VideoPressRuntime(press=_press(), mode="physical", adapter=adapter)
    runtime.begin_sample(SimpleNamespace(scene_token="model-fn"))
    runtime.install(pipe)
    try:
        latents = torch.randn(1, 4, 4, 4, 4)
        output = pipe.model_fn(
            dit=dit,
            latents=latents,
            longcat_latents=torch.randn(1, 4, 2, 4, 4),
            timestep=torch.tensor([1000.0]),
            context=torch.randn(1, 5, 12),
            traj_tokens=torch.randn(1, 3, 24),
            trajectory_head=trajectory_head,
            return_traj_pred=True,
            pipe=pipe,
        )
    finally:
        runtime.remove(pipe)
    assert output["video"].shape == latents.shape
    assert output["traj"].shape == (1, 3, 3)
    assert len(runtime.events) == 1
    assert runtime.events[0].key.layer_idx == -1
    assert runtime.events[0].result.metadata["hidden_sequence_compressed_layer_count"] == 2


def test_counterfactual_teacher_physically_prunes_before_block_zero(
    monkeypatch,
) -> None:
    """The planning teacher must intervene at the deployment-time location."""
    monkeypatch.setattr(wan_dit, "FLASH_ATTN_3_AVAILABLE", False)
    monkeypatch.setattr(wan_dit, "FLASH_ATTN_2_AVAILABLE", False)
    monkeypatch.setattr(wan_dit, "SAGE_ATTN_AVAILABLE", False)
    dit = WanModel(
        dim=24,
        in_dim=4,
        ffn_dim=48,
        out_dim=4,
        text_dim=12,
        freq_dim=16,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=4,
        num_layers=2,
        has_image_input=False,
        require_vae_embedding=False,
        require_clip_embedding=False,
    ).eval()
    seen_lengths = []
    handle = dit.blocks[0].register_forward_pre_hook(
        lambda _module, args: seen_lengths.append(int(args[0].shape[1]))
    )
    latents = torch.randn(1, 4, 4, 4, 4)
    kwargs = dict(
        dit=dit,
        latents=latents,
        longcat_latents=torch.randn(1, 4, 2, 4, 4),
        timestep=torch.tensor([1000.0]),
        context=torch.randn(1, 5, 12),
        traj_tokens=torch.randn(1, 3, 24),
        trajectory_head=torch.nn.Linear(24, 3),
        return_traj_pred=True,
    )
    try:
        baseline = model_fn_wan_video(**kwargs)
        baseline_length = seen_lengths[-1]
        counterfactual = model_fn_wan_video(
            **kwargs,
            counterfactual_history_token_mask=torch.tensor(
                [[1.0, 0.0, 1.0, 0.0]]
            ),
            counterfactual_layer=-1,
            counterfactual_physical_prune=True,
            counterfactual_latent_index=0,
        )
        pruned_length = seen_lengths[-1]
    finally:
        handle.remove()

    assert pruned_length == baseline_length - 2
    # The deleted positions are restored after the DiT stack so existing video
    # and trajectory heads retain their public output contracts.
    assert counterfactual["video"].shape == baseline["video"].shape == latents.shape
    assert counterfactual["traj"].shape == baseline["traj"].shape == (1, 3, 3)


def test_hidden_merge_is_vectorised_and_matches_a_per_group_reference() -> None:
    """Lock in the segment-mean rewrite.

    The original operator issued one ``index_select`` + ``cat`` per output
    group.  A pre-DiT merge plan contains one group per protected token too, so
    a real 1569-token sequence produced ~1374 GPU ops and made merge 2.4x slower
    end to end than no-press.  The vectorised path must stay numerically
    identical, including for non-uniform group weights.
    """

    from videopress.operators.merge import (
        HiddenTokenMergeOperator,
        MergeGroup,
        MergePlan,
    )
    from videopress.core.context import TokenContext

    layout = build_driveva_layout(4, 2, 2, 2, 3, 0)
    domain = build_domain("history", layout, "cpu")
    torch.manual_seed(0)
    tokens = torch.randn(1, layout.total_length, 5)

    candidate = domain.candidate_indices.tolist()
    candidate_set = set(candidate)
    protected = [i for i in range(layout.total_length) if i not in candidate_set]
    groups = [MergeGroup((index,), (1.0,)) for index in protected]
    groups.append(MergeGroup((candidate[0], candidate[1], candidate[2]), (1.0, 2.0, 0.5)))
    groups.append(MergeGroup((candidate[3], candidate[4]), (3.0, 1.0)))
    groups.append(MergeGroup((candidate[5],), (1.0,)))
    groups.append(MergeGroup((candidate[6], candidate[7]), (1.0, 1.0)))
    groups.sort(key=lambda group: min(group.source_indices))
    plan = MergePlan(tuple(groups), len(groups))

    def reference_loop():
        outputs = []
        for group in plan.groups:
            indices = torch.tensor(group.source_indices, dtype=torch.long)
            values = tokens.index_select(1, indices)
            weights = torch.tensor(group.weights, dtype=values.dtype)
            outputs.append(
                (values * (weights / weights.sum()).view(1, -1, 1)).sum(dim=1, keepdim=True)
            )
        return torch.cat(outputs, dim=1)

    ctx = TokenContext(tokens=tokens, layout=layout, domain=domain)
    result = HiddenTokenMergeOperator().apply_plan(ctx, plan)
    torch.testing.assert_close(result.output, reference_loop())
    assert result.metadata["vectorised_merge"] is True
    assert result.metadata["merge_group_count"] == plan.output_length
    assert result.metadata["merge_multi_token_groups"] == 3
    assert result.metadata["merge_max_group_size"] == 3
    assert result.mapping.compressed_length == result.output.shape[1]
    inverse = result.mapping.input_to_output[0]
    assert int((inverse < 0).sum()) == 0
    assert int(inverse.max()) == plan.output_length - 1


def test_tensor_merge_plan_matches_the_legacy_greedy_plan() -> None:
    """A/B refactor equivalence.

    ``feature="greedy"`` keeps the original cosine farthest-point grouping.  Its
    tensor plan must reproduce the legacy list-of-MergeGroup operator bit for
    bit, otherwise the CPU optimisation silently changed the algorithm.
    """

    from videopress.core.budget import TokenBudget
    from videopress.core.context import TokenContext
    from videopress.operators.merge import HiddenTokenMergeOperator

    layout = build_driveva_layout(4, 3, 5, 2, 3, 0)
    domain = build_domain("last_history", layout, "cpu")
    torch.manual_seed(5)
    ctx = TokenContext(
        tokens=torch.randn(1, layout.total_length, 8), layout=layout, domain=domain
    )
    ctx.scene_token = "equiv"
    ctx.diffusion_rank = 1000
    k = 4

    press = build_press(
        {
            "name": "similarity_merge",
            "injection_point": "block_input",
            "domain": "last_history",
            "feature": "greedy",
            "budget": {"type": "ratio", "value": 1.0, "reference": "eligible"},
        }
    )
    legacy_plan = press.build_merge_plan(ctx, k)
    legacy = HiddenTokenMergeOperator().apply_plan(ctx, legacy_plan)

    tensor_plan = press.build_tensor_merge_plan(ctx, k)
    modern = HiddenTokenMergeOperator().apply_tensor_plan(ctx, tensor_plan)

    assert legacy.output.shape == modern.output.shape
    torch.testing.assert_close(legacy.output, modern.output)
    torch.testing.assert_close(
        legacy.mapping.output_to_input, modern.mapping.output_to_input
    )
    torch.testing.assert_close(
        legacy.mapping.input_to_output, modern.mapping.input_to_output
    )
    assert modern.metadata["tensor_merge_plan"] is True
    assert modern.metadata["merge_group_count"] == legacy.metadata["merge_group_count"]


def test_vectorised_kmeans_recovers_multidimensional_clusters() -> None:
    """K-means must use more than the dominant principal direction.

    The earlier ``feature="tokens"`` split projected candidates onto PC1 and cut
    the sorted axis into K contiguous slices; on a configuration with four
    clusters spread over two independent axes that mixes clusters.  The k-means
    grouping must recover the underlying neighbourhood structure.
    """

    from videopress.presses.merge_press import _vectorised_kmeans

    torch.manual_seed(0)
    n, d = 80, 16
    centers = torch.zeros(4, d)
    centers[0, 0] = 1.0
    centers[1, 0] = -1.0
    centers[2, 1] = 1.0
    centers[3, 1] = -1.0
    labels = torch.arange(n) % 4
    feats = centers[labels] + 0.05 * torch.randn(n, d)

    assignment = _vectorised_kmeans(feats, 4)
    counts = torch.bincount(assignment, minlength=4)
    assert int((counts > 0).sum()) == 4
    for group in range(4):
        assert torch.unique(labels[assignment == group]).numel() == 1

    # Deterministic: a second call yields bit-identical group ids.
    assert torch.equal(assignment, _vectorised_kmeans(feats, 4))


def test_vectorised_kmeans_is_exact_k_and_deterministic() -> None:
    """Exactly K non-empty groups keep merge length identical to top-K prune."""

    from videopress.presses.merge_press import _vectorised_kmeans

    torch.manual_seed(1)
    for n, k in [(390, 195), (390, 390), (390, 1), (17, 5), (6, 6)]:
        feats = torch.randn(n, 32)
        assignment = _vectorised_kmeans(feats, k)
        counts = torch.bincount(assignment, minlength=k)
        assert counts.numel() == k
        assert int((counts > 0).sum()) == k, f"n={n} k={k} counts={counts.tolist()}"
        assert torch.equal(assignment, _vectorised_kmeans(feats, k))


def test_pre_dit_kmeans_merge_emits_exact_k_and_matches_prune_length() -> None:
    """The k-means merge arm must cost exactly the same as top-K pruning."""

    from videopress.core.context import TokenContext
    from videopress.operators.merge import HiddenTokenMergeOperator

    layout = build_driveva_layout(4, 2, 2, 2, 3, 0)
    domain = build_domain("history", layout, "cpu")
    torch.manual_seed(3)
    ctx = TokenContext(
        tokens=torch.randn(1, layout.total_length, 8), layout=layout, domain=domain
    )
    ctx.scene_token = "kmeans-length"
    ctx.diffusion_rank = 1000

    n_protected = layout.total_length - int(domain.n_candidate)
    for keep_ratio in (1.0, 0.5, 0.25):
        k = max(1, int(round(domain.n_candidate * keep_ratio)))
        press = build_press(
            {
                "name": "similarity_merge",
                "injection_point": "block_input",
                "domain": "history",
                "feature": "kmeans",
                "budget": {
                    "type": "ratio",
                    "value": keep_ratio,
                    "reference": "eligible",
                },
            }
        )
        plan = press.build_tensor_merge_plan(ctx, k)
        assert plan.output_length == n_protected + k
        counts = torch.bincount(plan.input_to_output, minlength=plan.output_length)
        assert int((counts > 0).sum()) == plan.output_length
        result = HiddenTokenMergeOperator().apply_tensor_plan(ctx, plan)
        assert result.output.shape[1] == n_protected + k


def test_register_merge_emits_protected_plus_k_restores_and_is_trainable() -> None:
    """The learnable merge must be a trainable, layout-faithful compressor.

    Contract: output length is ``protected + K`` (same cost as top-K pruning),
    the controller's gather/scatter semantics are exact for protected tokens,
    and a trajectory loss on the short sequence must reach the bottleneck
    parameters -- otherwise training-time compression is impossible.
    """

    from videopress.core.context import TokenContext

    layout = build_driveva_layout(4, 2, 2, 2, 3, 0)
    domain = build_domain("last_history", layout, "cpu")
    n_candidate = int(domain.n_candidate)
    k = n_candidate // 2
    torch.manual_seed(7)
    tokens = torch.randn(1, layout.total_length, 8)
    ctx = TokenContext(tokens=tokens, layout=layout, domain=domain)
    ctx.scene_token = "register-merge"
    ctx.diffusion_rank = 1000

    press = build_press(
        {
            "name": "register_merge",
            "injection_point": "block_input",
            "domain": "last_history",
            "num_key_tokens": k,
            "hidden_dim": 8,
            "attn_dim": 8,
            "num_heads": 2,
            "budget": {"type": "ratio", "value": 0.5, "reference": "eligible"},
        }
    )
    validate_protocol(press, "physical")
    result = press.apply(ctx)

    expected_length = layout.total_length - n_candidate + k
    assert result.output.shape == (1, expected_length, 8)
    assert result.mapping.compressed_length == expected_length
    keep = result.mapping.output_to_input
    assert keep.shape == (1, expected_length)

    # Controller semantics: the short sequence is scattered back on `keep`.
    restored = result.output.new_zeros((1, layout.total_length, 8))
    restored.scatter_(1, keep.unsqueeze(-1).expand_as(result.output), result.output)
    protected = torch.where(domain.protected_mask)[0]
    torch.testing.assert_close(restored[0, protected], tokens[0, protected])

    # input_to_output covers every input and points at valid short positions.
    ito = result.mapping.input_to_output
    assert ito.shape == (1, layout.total_length)
    assert int(ito.min()) >= 0
    assert int(ito.max()) < expected_length

    # Differentiable end to end.
    loss = result.output.float().pow(2).mean()
    loss.backward()
    grads = [p.grad for p in press.module.parameters() if p.grad is not None]
    assert grads, "register_merge produced no gradients"
    assert any(float(g.abs().sum()) > 0 for g in grads)


def test_register_merge_trains_through_real_wan_model_fn(monkeypatch) -> None:
    """Training-time physical compression must backprop into the bottleneck.

    ``training_loss`` calls ``self.model_fn``, so the pre-DiT controller used at
    inference is the same hook a trainer sees.  This test installs a learnable
    merge press on a real (tiny) WanModel, runs the public ``model_fn`` forward,
    and checks that a trajectory loss reaches the RegisterBottleneck parameters.
    """

    monkeypatch.setattr(wan_dit, "FLASH_ATTN_3_AVAILABLE", False)
    monkeypatch.setattr(wan_dit, "FLASH_ATTN_2_AVAILABLE", False)
    monkeypatch.setattr(wan_dit, "SAGE_ATTN_AVAILABLE", False)
    dit = WanModel(
        dim=24,
        in_dim=4,
        ffn_dim=48,
        out_dim=4,
        text_dim=12,
        freq_dim=16,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=4,
        num_layers=2,
        has_image_input=False,
        require_vae_embedding=False,
        require_clip_embedding=False,
    )
    press = build_press(
        {
            "name": "register_merge",
            "injection_point": "block_input",
            "domain": "last_history",
            "num_key_tokens": 2,
            "hidden_dim": 24,
            "attn_dim": 24,
            "num_heads": 4,
            "budget": {"type": "ratio", "value": 0.5, "reference": "eligible"},
        }
    )
    adapter = DriveVAAdapter()
    runtime = VideoPressRuntime(press=press, mode="physical", adapter=adapter)
    pipe = SimpleNamespace(dit=dit, dit2=None, model_fn=model_fn_wan_video)
    runtime.begin_sample(SimpleNamespace(scene_token="trainable-register"))
    runtime.install(pipe)
    try:
        output = pipe.model_fn(
            dit=dit,
            latents=torch.randn(1, 4, 4, 4, 4),
            longcat_latents=torch.randn(1, 4, 2, 4, 4),
            timestep=torch.tensor([1000.0]),
            context=torch.randn(1, 5, 12),
            traj_tokens=torch.randn(1, 3, 24),
            trajectory_head=torch.nn.Linear(24, 3),
            return_traj_pred=True,
            pipe=pipe,
        )
        loss = output["traj"].float().pow(2).mean()
        loss.backward()
    finally:
        runtime.remove(pipe)

    grads = [p.grad for p in press.module.parameters() if p.grad is not None]
    assert grads, "no gradient reached the RegisterBottleneck"
    assert any(float(g.abs().sum()) > 0 for g in grads)
    assert runtime.events, "pre-DiT compression event was not recorded"
    assert runtime.events[0].key.layer_idx == -1
