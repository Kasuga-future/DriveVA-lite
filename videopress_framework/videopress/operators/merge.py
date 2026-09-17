"""Mapping-preserving token merge primitives."""

from __future__ import annotations

from dataclasses import dataclass
import torch

from ..core.result import OperatorResult, TokenMapping
from ..core.registry import register_operator
from .base import TokenOperator


@dataclass(frozen=True)
class MergeGroup:
    source_indices: tuple[int, ...]
    weights: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.source_indices:
            raise ValueError("merge group cannot be empty")
        if len(self.source_indices) != len(self.weights):
            raise ValueError("source_indices and weights must have equal length")
        if any(weight <= 0 for weight in self.weights):
            raise ValueError("merge weights must be positive")


@dataclass(frozen=True)
class MergePlan:
    groups: tuple[MergeGroup, ...]
    output_length: int

    def __post_init__(self) -> None:
        if self.output_length != len(self.groups):
            raise ValueError("output_length must equal the number of groups")


def _validate_plan(plan: MergePlan, input_length: int) -> None:
    seen: list[int] = []
    for group in plan.groups:
        if any(index < 0 or index >= input_length for index in group.source_indices):
            raise IndexError("merge plan contains an input index outside the sequence")
        seen.extend(group.source_indices)
    if sorted(seen) != list(range(input_length)):
        raise ValueError("merge plan must cover every input token exactly once")


@dataclass(frozen=True)
class TensorMergePlan:
    """Sync-free, object-free merge plan.

    The list-of-``MergeGroup`` representation costs one Python object per output
    group.  A pre-DiT merge plan contains one group per *protected* token as
    well, so a 1568-token sequence at 50% candidate compression produced 1373
    objects, 4119 ``__post_init__`` calls and two O(L log L) validations per
    forward.  This representation keeps the same information in four tensors so
    the whole merge is three tensor ops and one O(L) check.

    ``input_to_output[i]`` is the output position of input token ``i`` and
    ``output_to_input[g]`` is that group's representative (lowest-index) input
    token.  Output positions are ordered by ascending representative, which is
    what the pre-DiT controller needs when it gathers RoPE frequencies with the
    same index tensor.
    """

    input_to_output: torch.Tensor  # [L] long
    output_to_input: torch.Tensor  # [G] long, ascending
    weights: torch.Tensor  # [L] float, normalised so each group sums to 1
    group_sizes: torch.Tensor  # [G] long
    original_length: int
    output_length: int

    def __post_init__(self) -> None:
        if self.input_to_output.ndim != 1 or self.output_to_input.ndim != 1:
            raise ValueError("tensor merge plan indices must be one-dimensional")
        if self.weights.ndim != 1 or self.group_sizes.ndim != 1:
            raise ValueError("tensor merge plan weights/sizes must be one-dimensional")
        if self.input_to_output.numel() != self.original_length:
            raise ValueError("input_to_output length must equal original_length")
        if self.output_to_input.numel() != self.output_length:
            raise ValueError("output_to_input length must equal output_length")
        if self.output_length != self.group_sizes.numel():
            raise ValueError("group_sizes length must equal output_length")


def validate_tensor_plan(plan: TensorMergePlan) -> None:
    """O(L) structural check with a single device sync on the happy path.

    Replaces the legacy ``sorted(seen) != list(range(L))`` coverage test, which
    sorted the full token index list twice per forward.  All predicates are
    folded into one boolean tensor so the normal path costs one sync instead of
    the ~200 the greedy grouping used to cost.
    """

    index = plan.input_to_output
    reps = plan.output_to_input
    counts = torch.bincount(index, minlength=plan.output_length)
    # ``~torch.equal(...)`` would be bitwise-not on a Python bool (== -2, always
    # truthy), which silently forced every call down the slow path.
    sizes_disagree = not torch.equal(plan.group_sizes, counts)
    bad = (
        ((index < 0) | (index >= plan.output_length)).any()
        | (counts == 0).any()
        | (counts.numel() != plan.output_length)
        | (reps < 0).any()
        | (reps >= plan.original_length).any()
        | (reps[1:] <= reps[:-1]).any()
        | sizes_disagree
    )
    if not bool(bad.item()):
        return
    # Slow path only: produce an actionable message.
    if bool(((index < 0) | (index >= plan.output_length)).any().item()):
        raise IndexError("tensor merge plan maps a token outside the output range")
    if counts.numel() != plan.output_length or bool((counts == 0).any().item()):
        raise ValueError("tensor merge plan does not cover every output group")
    if bool((reps < 0).any().item()) or bool((reps >= plan.original_length).any().item()):
        raise IndexError("tensor merge plan representative is outside the sequence")
    if bool((reps[1:] <= reps[:-1]).any().item()):
        raise ValueError("tensor merge plan representatives must be strictly ascending")
    raise ValueError("tensor merge plan group_sizes disagree with the mapping")


def _apply_tensor_plan_to_sequence(
    tensor: torch.Tensor, plan: TensorMergePlan, *, sequence_dim: int
) -> torch.Tensor:
    """Segment-mean a ``[..., L, ...]`` tensor along ``sequence_dim``."""

    moved = tensor.movedim(sequence_dim, 1)
    weighted = moved * plan.weights.view(1, -1, *((1,) * (moved.ndim - 2))).to(
        moved.dtype
    )
    shape = list(moved.shape)
    shape[1] = plan.output_length
    output = moved.new_zeros(shape)
    output.index_add_(1, plan.input_to_output, weighted)
    return output.movedim(1, sequence_dim)


def _merge_qkv_sequence(x: torch.Tensor, plan: MergePlan) -> torch.Tensor:
    """Merge ``[B,H,L,D]`` K/V values according to a shared plan."""

    _validate_plan(plan, int(x.shape[2]))
    outputs = []
    for group in plan.groups:
        indices = torch.tensor(group.source_indices, dtype=torch.long, device=x.device)
        values = x.index_select(2, indices)
        weights = torch.tensor(group.weights, dtype=x.dtype, device=x.device)
        weights = weights / weights.sum()
        outputs.append((values * weights.view(1, 1, -1, 1)).sum(dim=2, keepdim=True))
    return torch.cat(outputs, dim=2) if outputs else x[:, :, :0]


@register_operator("merge")
class HiddenTokenMergeOperator(TokenOperator):
    """Hidden-token merge primitive kept separate from DriveVA K/V merge.

    ``block_input_compatible`` makes this the information-preserving counterpart
    of ``HiddenPruneOperator``: instead of gathering only the kept tokens, every
    candidate is averaged into one of ``K`` groups, so the compressed sequence
    still carries the dropped tokens' content.  ``output_to_input`` is the
    representative (lowest-index) member of each group, which is what the
    DriveVA pre-DiT controller uses to restore the original token layout.
    """

    name = "merge"
    preserves_sequence_length = False
    physical_compression = True
    driveva_compatible = True
    block_input_compatible = True

    def apply_plan(self, ctx, plan: MergePlan) -> OperatorResult:
        _validate_plan(plan, ctx.layout.total_length)
        device = ctx.tokens.device
        # Fully vectorised segment-mean.  The previous implementation issued one
        # ``index_select`` + one ``cat`` per output group; a pre-DiT merge plan
        # has one group per *protected* token as well, so a 1569-token sequence
        # at 50% candidate compression produced ~1374 Python-level GPU ops and
        # made the merge arm 2.4x slower end to end than no-press.  Flattening
        # the plan to (source_index, group_id) turns the whole merge into a
        # single index_select + index_add + divide.
        source_index: list[int] = []
        group_id: list[int] = []
        token_weight: list[float] = []
        representative: list[int] = []
        source_groups: list[list[int]] = []
        group_sizes: list[int] = []
        for output_index, group in enumerate(plan.groups):
            indices = [int(index) for index in group.source_indices]
            weights = [float(weight) for weight in group.weights]
            total = sum(weights)
            source_index.extend(indices)
            group_id.extend([output_index] * len(indices))
            token_weight.extend([weight / total for weight in weights])
            representative.append(indices[0])
            source_groups.append(indices)
            group_sizes.append(len(indices))

        batch, total_length, hidden_dim = ctx.tokens.shape
        num_groups = len(plan.groups)
        if num_groups == 0:
            empty = ctx.tokens[:, :0]
            mapping = TokenMapping(
                output_to_input=torch.empty(
                    (batch, 0), dtype=torch.long, device=device
                ),
                input_to_output=None,
                original_length=int(total_length),
                compressed_length=0,
                source_groups=[],
            )
            return OperatorResult(
                output=empty,
                mapping=mapping,
                metadata={"operator": self.name, "source_groups": []},
            )

        index_t = torch.tensor(source_index, dtype=torch.long, device=device)
        group_t = torch.tensor(group_id, dtype=torch.long, device=device)
        weight_t = torch.tensor(
            token_weight, dtype=ctx.tokens.dtype, device=device
        ).view(1, -1, 1)
        reordered = ctx.tokens.index_select(1, index_t) * weight_t
        output = ctx.tokens.new_zeros((batch, num_groups, hidden_dim))
        output.index_add_(1, group_t, reordered)

        representative_t = torch.tensor(
            representative, dtype=torch.long, device=device
        )
        output_to_input = representative_t.unsqueeze(0).expand(batch, -1).clone()
        # Every input token maps to exactly one output group, so the inverse
        # mapping is well defined and lets audits verify the merge is lossless
        # in *coverage* even though it is lossy in resolution.
        input_to_output = torch.empty(
            (batch, int(total_length)), dtype=torch.long, device=device
        )
        input_to_output[:, index_t] = group_t.unsqueeze(0).expand(batch, -1)
        mapping = TokenMapping(
            output_to_input=output_to_input,
            input_to_output=input_to_output,
            original_length=int(total_length),
            compressed_length=num_groups,
            source_groups=source_groups,
        )
        multi_token = sum(1 for size in group_sizes if size > 1)
        return OperatorResult(
            output=output,
            mapping=mapping,
            metadata={
                "operator": self.name,
                "source_groups": source_groups,
                "hidden_length_before": int(total_length),
                "hidden_length_after": num_groups,
                "hidden_length_ratio": float(
                    num_groups / max(int(total_length), 1)
                ),
                "merge_group_count": num_groups,
                "merge_multi_token_groups": int(multi_token),
                "merge_max_group_size": int(max(group_sizes)),
                "vectorised_merge": True,
                "representative_only": True,
                "block_input": True,
            },
        )

    def apply(self, ctx, selection) -> OperatorResult:
        if isinstance(selection, TensorMergePlan):
            return self.apply_tensor_plan(ctx, selection)
        if not isinstance(selection, MergePlan):
            raise TypeError("MergeOperator.apply expects a MergePlan")
        return self.apply_plan(ctx, selection)

    def apply_tensor_plan(self, ctx, plan: TensorMergePlan) -> OperatorResult:
        """Vectorised hidden-path merge: three tensor ops, no Python objects."""

        if int(plan.original_length) != int(ctx.layout.total_length):
            raise ValueError("tensor merge plan length differs from the active layout")
        validate_tensor_plan(plan)
        output = _apply_tensor_plan_to_sequence(ctx.tokens, plan, sequence_dim=1)
        batch = int(ctx.tokens.shape[0])
        mapping = TokenMapping(
            output_to_input=plan.output_to_input.unsqueeze(0).expand(batch, -1).clone(),
            input_to_output=plan.input_to_output.unsqueeze(0).expand(batch, -1).clone(),
            original_length=int(plan.original_length),
            compressed_length=int(plan.output_length),
            # Deliberately not materialised: a nested list of 1373 groups per
            # event is what made the legacy path CPU-bound and bloated the
            # event journal.  ``input_to_output`` carries the same information.
            source_groups=None,
        )
        return OperatorResult(
            output=output,
            mapping=mapping,
            metadata=self._tensor_metadata(plan),
        )

    def _tensor_metadata(self, plan: TensorMergePlan) -> dict:
        sizes = plan.group_sizes
        multi = int((sizes > 1).sum().item())
        return {
            "operator": self.name,
            "hidden_length_before": int(plan.original_length),
            "hidden_length_after": int(plan.output_length),
            "hidden_length_ratio": float(
                plan.output_length / max(int(plan.original_length), 1)
            ),
            "merge_group_count": int(plan.output_length),
            "merge_multi_token_groups": multi,
            "merge_max_group_size": int(sizes.max().item()) if sizes.numel() else 0,
            "tensor_merge_plan": True,
            "representative_only": True,
            "block_input": True,
        }


@register_operator("kv_merge")
class KVMergeOperator(TokenOperator):
    """DriveVA-compatible physical merge at the post-RoPE K/V boundary."""

    name = "kv_merge"
    preserves_sequence_length = False
    physical_compression = True
    driveva_compatible = True

    def apply_plan(self, ctx, plan: MergePlan) -> OperatorResult:
        if ctx.k is None or ctx.v is None:
            raise ValueError("KVMergeOperator requires ctx.k and ctx.v")
        if ctx.k.ndim != 4 or ctx.v.ndim != 4:
            raise ValueError("KVMergeOperator expects canonical [B,H,L,D] k/v")
        if ctx.k.shape != ctx.v.shape:
            raise ValueError("k and v shapes differ")
        if ctx.k.shape[0] != ctx.batch_size or ctx.k.shape[2] != ctx.layout.total_length:
            raise ValueError("k sequence does not match TokenContext layout")
        _validate_plan(plan, int(ctx.k.shape[2]))
        k_new = _merge_qkv_sequence(ctx.k, plan)
        v_new = _merge_qkv_sequence(ctx.v, plan)
        b, _, original_length, _ = ctx.k.shape
        output_length = int(k_new.shape[2])
        representatives = torch.tensor(
            [group.source_indices[0] for group in plan.groups],
            dtype=torch.long,
            device=ctx.k.device,
        ).expand(b, -1).clone()
        input_to_output = torch.full(
            (b, original_length), -1, dtype=torch.long, device=ctx.k.device
        )
        for output_index, group in enumerate(plan.groups):
            source = torch.tensor(group.source_indices, dtype=torch.long, device=ctx.k.device)
            input_to_output[:, source] = output_index
        mapping = TokenMapping(
            output_to_input=representatives,
            input_to_output=input_to_output,
            original_length=original_length,
            compressed_length=output_length,
            source_groups=[list(group.source_indices) for group in plan.groups],
        )
        metadata = {
            "operator": self.name,
            "q_length": int(ctx.q.shape[2]) if ctx.q is not None and ctx.q.ndim == 4 else original_length,
            "k_length_before": original_length,
            "k_length_after": output_length,
            "v_length_after": int(v_new.shape[2]),
            "theoretical_attn_ratio": float(output_length / original_length),
            "post_rope": True,
            "mapping_complete": True,
        }
        return OperatorResult(
            output=ctx.tokens,
            metadata=metadata,
            mapping=mapping,
            aux={"q": ctx.q, "k": k_new, "v": v_new},
        )

    def apply(self, ctx, selection) -> OperatorResult:
        if isinstance(selection, TensorMergePlan):
            return self.apply_tensor_plan(ctx, selection)
        if not isinstance(selection, MergePlan):
            raise TypeError("KVMergeOperator.apply expects a MergePlan")
        return self.apply_plan(ctx, selection)

    def apply_tensor_plan(self, ctx, plan: TensorMergePlan) -> OperatorResult:
        """Vectorised K/V merge sharing the same sync-free plan."""

        if ctx.k is None or ctx.v is None:
            raise ValueError("KVMergeOperator requires ctx.k and ctx.v")
        if ctx.k.ndim != 4 or ctx.v.ndim != 4 or ctx.k.shape != ctx.v.shape:
            raise ValueError("KVMergeOperator expects canonical matching [B,H,L,D] k/v")
        if int(ctx.k.shape[2]) != int(plan.original_length):
            raise ValueError("tensor merge plan length differs from the active k/v")
        validate_tensor_plan(plan)
        k_new = _apply_tensor_plan_to_sequence(ctx.k, plan, sequence_dim=2)
        v_new = _apply_tensor_plan_to_sequence(ctx.v, plan, sequence_dim=2)
        batch, _, original_length, _ = ctx.k.shape
        mapping = TokenMapping(
            output_to_input=plan.output_to_input.unsqueeze(0).expand(batch, -1).clone(),
            input_to_output=plan.input_to_output.unsqueeze(0).expand(batch, -1).clone(),
            original_length=int(original_length),
            compressed_length=int(plan.output_length),
            source_groups=None,
        )
        metadata = {
            "operator": self.name,
            "k_length_before": int(original_length),
            "k_length_after": int(plan.output_length),
            "v_length_after": int(v_new.shape[2]),
            "theoretical_attn_ratio": float(
                plan.output_length / max(int(original_length), 1)
            ),
            "post_rope": True,
            "mapping_complete": True,
            "tensor_merge_plan": True,
        }
        return OperatorResult(
            output=ctx.tokens, metadata=metadata, mapping=mapping,
            aux={"q": ctx.q, "k": k_new, "v": v_new},
        )


# Existing imports keep working while the class name makes the distinction
# explicit to new callers.
MergeOperator = HiddenTokenMergeOperator
