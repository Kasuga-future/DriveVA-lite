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
    """Hidden-token merge primitive kept separate from DriveVA K/V merge."""

    name = "merge"
    preserves_sequence_length = False
    physical_compression = True

    def apply_plan(self, ctx, plan: MergePlan) -> OperatorResult:
        _validate_plan(plan, ctx.layout.total_length)
        outputs = []
        source_groups = []
        for group in plan.groups:
            indices = torch.tensor(group.source_indices, dtype=torch.long, device=ctx.tokens.device)
            values = ctx.tokens.index_select(1, indices)
            weights = torch.tensor(group.weights, dtype=values.dtype, device=values.device)
            merged = (values * (weights / weights.sum()).view(1, -1, 1)).sum(dim=1, keepdim=True)
            outputs.append(merged)
            source_groups.append(list(group.source_indices))
        output = torch.cat(outputs, dim=1) if outputs else ctx.tokens[:, :0]
        representative = torch.tensor(
            [group.source_indices[0] for group in plan.groups], dtype=torch.long, device=ctx.tokens.device
        ).expand(ctx.batch_size, -1).clone()
        mapping = TokenMapping(
            output_to_input=representative,
            input_to_output=None,
            original_length=ctx.layout.total_length,
            compressed_length=plan.output_length,
            source_groups=source_groups,
        )
        return OperatorResult(
            output=output,
            mapping=mapping,
            metadata={"operator": self.name, "source_groups": source_groups},
        )

    def apply(self, ctx, selection) -> OperatorResult:
        if not isinstance(selection, MergePlan):
            raise TypeError("MergeOperator.apply expects a MergePlan")
        return self.apply_plan(ctx, selection)


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
        if not isinstance(selection, MergePlan):
            raise TypeError("KVMergeOperator.apply expects a MergePlan")
        return self.apply_plan(ctx, selection)


# Existing imports keep working while the class name makes the distinction
# explicit to new callers.
MergeOperator = HiddenTokenMergeOperator
