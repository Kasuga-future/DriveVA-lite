from __future__ import annotations

import torch
import torch.nn.functional as F

from ..core.budget import budget_stats, resolve_budget
from ..core.registry import register_press
from ..core.result import CompressionResult
from ..core.runtime import InjectionPoint
from ..operators.merge import KVMergeOperator, MergeGroup, MergePlan
from .base import BaseVideoPress


@register_press("similarity_merge")
class SimilarityMergePress(BaseVideoPress):
    name = "similarity_merge"
    injection_point = InjectionPoint.SELF_ATTN_KV

    def __init__(self, budget, domain=None, feature="tokens"):
        self.budget = budget
        self.domain = domain
        self.feature = feature
        self.operator = KVMergeOperator()

    def build_merge_plan(self, ctx, K: int) -> MergePlan:
        if ctx.batch_size != 1:
            raise NotImplementedError("SimilarityMergePress V1 supports batch_size=1 only")
        candidate = ctx.domain.candidate_indices.tolist()
        protected = torch.where(ctx.domain.protected_mask)[0].tolist()
        if K > len(candidate):
            raise ValueError("merge K exceeds candidate count")
        if K == 0 and candidate:
            raise ValueError("SimilarityMerge requires K > 0 when candidates exist")
        if not candidate:
            candidate_groups = []
        elif K == len(candidate):
            candidate_groups = [[index] for index in candidate]
        else:
            if self.feature == "k" and ctx.k is not None:
                features = ctx.k.index_select(2, ctx.domain.candidate_indices).mean(dim=1)[0]
            else:
                features = ctx.candidate_tokens()[0]
            features = F.normalize(features.float(), dim=-1)
            # Select deterministic anchors with farthest-point-like greedy cosine
            # coverage, then assign each token to its closest anchor.
            anchor_positions = [0]
            while len(anchor_positions) < K:
                chosen = torch.tensor(anchor_positions, device=features.device)
                similarity = features.index_select(0, torch.arange(len(candidate), device=features.device)) @ features.index_select(0, chosen).T
                min_similarity = similarity.max(dim=1).values
                min_similarity[chosen] = 1.0
                anchor_positions.append(int(torch.argmin(min_similarity).item()))
            anchors = features[torch.tensor(anchor_positions, device=features.device)]
            assignment = (features @ anchors.T).argmax(dim=-1).tolist()
            candidate_groups = [[] for _ in range(K)]
            for position, group_index in enumerate(assignment):
                candidate_groups[group_index].append(candidate[position])
            candidate_groups = [group for group in candidate_groups if group]
            if len(candidate_groups) != K:
                # This should be rare for tied features; preserve exact K by
                # splitting the largest groups deterministically.
                while len(candidate_groups) < K:
                    largest = max(range(len(candidate_groups)), key=lambda i: len(candidate_groups[i]))
                    group = candidate_groups.pop(largest)
                    split = max(1, len(group) // 2)
                    candidate_groups.extend([group[:split], group[split:]])
                candidate_groups = candidate_groups[:K]
            candidate_groups.sort(key=lambda group: min(group))
        groups = [MergeGroup((index,), (1.0,)) for index in protected]
        groups.extend(
            MergeGroup(tuple(group), tuple(1.0 for _ in group)) for group in candidate_groups
        )
        groups.sort(key=lambda group: min(group.source_indices))
        return MergePlan(tuple(groups), len(groups))

    def apply(self, ctx) -> CompressionResult:
        k = resolve_budget(self.budget, ctx.layout, ctx.domain)
        plan = self.build_merge_plan(ctx, k)
        operator_result = self.operator.apply_plan(ctx, plan)
        metadata = {
            "press": self.name,
            "domain": ctx.domain.name,
            "operator": "kv_merge",
            "feature": self.feature,
            "injection_point": self.injection_point.value,
            **budget_stats(ctx.layout, ctx.domain, k),
            **operator_result.metadata,
        }
        return CompressionResult(
            output=operator_result.output,
            scores=None,
            selection=None,
            metadata=metadata,
            mapping=operator_result.mapping,
            aux=operator_result.aux,
        )
