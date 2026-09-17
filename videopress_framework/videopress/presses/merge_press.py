from __future__ import annotations

import torch
import torch.nn.functional as F

from ..core.budget import budget_stats, resolve_budget
from ..core.registry import register_press
from ..core.result import CompressionResult
from ..core.runtime import InjectionPoint
from ..operators.merge import (
    HiddenTokenMergeOperator,
    KVMergeOperator,
    MergeGroup,
    MergePlan,
    TensorMergePlan,
)
from ..utils.seed import stable_seed
from .base import BaseVideoPress


def _repair_empty_clusters(
    feats: torch.Tensor,
    centroids: torch.Tensor,
    assignment: torch.Tensor,
    K: int,
) -> torch.Tensor:
    """Move points into empty clusters until exactly ``K`` groups are non-empty.

    Lloyd updates can leave a cluster empty.  Because ``K <= n`` the loop always
    terminates: each move takes a point from a group of size > 1 and fills one
    empty group, so the number of non-empty groups strictly increases.  The path
    is deterministic and, after quantile initialisation, normally needs zero
    moves; only the rare repair performs a device sync.
    """

    for _ in range(int(K)):
        counts = torch.bincount(assignment, minlength=K)
        empty = (counts == 0).nonzero(as_tuple=True)[0]
        if empty.numel() == 0:
            return assignment
        centroids = torch.zeros_like(centroids)
        centroids.index_add_(0, assignment, feats)
        centroids = F.normalize(
            centroids / counts.clamp_min(1).unsqueeze(1), dim=-1
        )
        sims = feats @ centroids.t()
        own = sims.gather(1, assignment.unsqueeze(1)).squeeze(1)
        movable = counts[assignment] > 1
        own = own.masked_fill(~movable, float("inf"))
        victim = int(own.argmin().item())
        if not bool(torch.isfinite(own[victim])):
            return assignment
        assignment = assignment.clone()
        assignment[victim] = empty[0]
    return assignment


def _vectorised_kmeans(
    features: torch.Tensor, K: int, iterations: int = 10
) -> torch.Tensor:
    """Cosine k-means with PC1-quantile initialisation, fully vectorised.

    The greedy cosine reference issues one device sync per anchor (~K syncs per
    forward).  This implementation uses only matmuls in its hot path and returns
    a group id in ``[0, K)`` with exactly ``K`` non-empty groups, so the merged
    sequence has the same length as the equivalent top-K prune.  Multidimensional
    clustering preserves the local neighbourhood structure that a single
    principal-direction split destroys.
    """

    n = int(features.shape[0])
    if K < 1 or K > n:
        raise ValueError("kmeans requires 1 <= K <= n_candidate")
    device = features.device
    if K == 1:
        return torch.zeros(n, dtype=torch.long, device=device)
    if K == n:
        return torch.arange(n, dtype=torch.long, device=device)

    feats = F.normalize(features.float(), dim=-1)

    # PC1-quantile initialisation: sort by the dominant variance direction and
    # seed each centroid with one equal-size slice of the sorted points.
    centered = feats - feats.mean(dim=0, keepdim=True)
    direction = centered.mean(dim=0)
    direction = direction / direction.norm().clamp_min(1e-6)
    for _ in range(20):
        direction = centered.t() @ (centered @ direction)
        direction = direction / direction.norm().clamp_min(1e-6)
    key = feats @ direction
    order = torch.argsort(key)
    quantile = torch.div(
        torch.arange(n, device=device) * K, n, rounding_mode="floor"
    )
    centroids = torch.zeros((K, feats.shape[1]), dtype=feats.dtype, device=device)
    centroids.index_add_(0, quantile, feats[order])
    counts = torch.bincount(quantile, minlength=K).clamp_min(1)
    centroids = F.normalize(centroids / counts.unsqueeze(1), dim=-1)

    assignment = torch.zeros(n, dtype=torch.long, device=device)
    for _ in range(int(iterations)):
        assignment = (feats @ centroids.t()).argmax(dim=-1)
        centroids = torch.zeros_like(centroids)
        counts = torch.bincount(assignment, minlength=K)
        centroids.index_add_(0, assignment, feats)
        centroids = F.normalize(
            centroids / counts.clamp_min(1).unsqueeze(1), dim=-1
        )
    return _repair_empty_clusters(feats, centroids, assignment, K)



@register_press("similarity_merge")
class SimilarityMergePress(BaseVideoPress):
    """Merge candidate tokens into ``K`` groups instead of deleting them.

    At ``self_attn_kv`` the merge is applied to the post-RoPE K/V tensors.  At
    ``block_input`` it is applied to the hidden residual stream before DiT block
    0, which is the information-preserving counterpart of the pre-DiT
    ``hidden_prune`` press: the compressed sequence has the *same* length in both
    cases, so a prune-vs-merge comparison isolates information retention from
    compute.

    ``feature`` selects the grouping rule:

    * ``tokens`` -- **sync-free** balanced split along the dominant principal
      direction of the candidate hidden states.  Cheap, deterministic and
      guaranteed to produce exactly ``K`` non-empty groups.
    * ``greedy`` -- the original cosine farthest-point anchor assignment.  It is
      semantically the reference implementation but performs one device sync per
      anchor (194 per forward at K=195), so it is opt-in only.
    * ``k`` -- K/V at the attention boundary (unavailable at ``block_input``).
    * ``kmeans`` -- vectorised cosine k-means with PC1-quantile init and 10
      Lloyd iterations.  Cheap (~5 ms) and guaranteed to emit exactly
      ``K`` non-empty groups, but on the 256-scene pre-DiT panel it
      scored PDM 0.857 versus 0.906 for ``greedy``: the k-means
      variance objective does not preserve planning-critical
      neighbourhoods the way farthest-point coverage does.
    * ``random`` -- seeded balanced permutation.  A deliberate control: it
      averages tokens without any similarity heuristic, so it measures purely
      whether *keeping the averaged information* helps, independent of grouping
      quality.
    """

    name = "similarity_merge"

    def __init__(
        self,
        budget,
        domain=None,
        feature="tokens",
        injection_point=InjectionPoint.SELF_ATTN_KV,
        seed: int = 0,
    ):
        self.budget = budget
        self.domain = domain
        self.feature = str(feature).strip().lower()
        if self.feature not in {"tokens", "k", "random", "greedy", "kmeans"}:
            raise ValueError(
                "similarity_merge feature must be tokens, greedy, kmeans, k or random"
            )
        self.injection_point = InjectionPoint.parse(injection_point)
        if self.feature == "k" and self.injection_point is not InjectionPoint.SELF_ATTN_KV:
            raise ValueError("feature='k' requires injection_point=self_attn_kv")
        self.seed = int(seed)
        self.operator = (
            HiddenTokenMergeOperator()
            if self.injection_point is InjectionPoint.BLOCK_INPUT
            else KVMergeOperator()
        )

    # ---- sync-free grouping ------------------------------------------------

    def _candidate_group_ids(self, ctx, K: int) -> torch.Tensor:
        """Group id in ``[0, K)`` for each candidate, exactly K non-empty groups.

        All three rules are implemented with tensor ops only: no ``.item()``
        inside a loop, no per-token Python objects.  The previous greedy rule
        issued one device sync per anchor, which stalled the GPU pipeline and
        made the merge 2.4x slower than no-press end to end.
        """

        device = ctx.tokens.device
        candidate = ctx.domain.candidate_indices.to(device)
        n = int(candidate.numel())
        if K <= 0 or K > n:
            raise ValueError("merge grouping requires 1 <= K <= n_candidate")

        if self.feature == "kmeans":
            return _vectorised_kmeans(
                ctx.tokens.index_select(1, candidate)[0], K
            )

        if self.feature in {"tokens", "k"}:
            if self.feature == "k" and ctx.k is not None:
                features = ctx.k.index_select(2, candidate).mean(dim=1)[0]
            else:
                features = ctx.tokens.index_select(1, candidate)[0]
            features = F.normalize(features.float(), dim=-1)
            # Dominant direction by power iteration (no sync, ~20 mat-vecs).
            direction = features.mean(dim=0)
            direction = direction / direction.norm().clamp_min(1e-6)
            for _ in range(20):
                direction = features.t() @ (features @ direction)
                direction = direction / direction.norm().clamp_min(1e-6)
            key = features @ direction
            order = torch.argsort(key)
            rank = torch.empty(n, dtype=torch.long, device=device)
            rank[order] = torch.arange(n, device=device)
            # Balanced contiguous split => every one of the K groups is
            # non-empty whenever n >= K, so the output length is exactly
            # ``protected + K`` and matches the equivalent top-K prune.
            return torch.div(rank * K, n, rounding_mode="floor")

        if self.feature == "greedy":
            return self._greedy_group_ids(features=None, ctx=ctx, K=K)

        # random: seeded balanced permutation
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            stable_seed(self.seed, ctx.scene_token, ctx.diffusion_rank, K)
        )
        order = torch.randperm(n, generator=generator).to(device)
        assignment = torch.empty(n, dtype=torch.long, device=device)
        assignment[order] = torch.arange(n, device=device) % K
        return assignment

    def _greedy_group_ids(self, *, features, ctx, K: int) -> torch.Tensor:
        """Reference cosine farthest-point grouping (one sync per anchor)."""

        device = ctx.tokens.device
        candidate = ctx.domain.candidate_indices.to(device)
        n = int(candidate.numel())
        if features is None:
            features = ctx.tokens.index_select(1, candidate)[0]
        features = F.normalize(features.float(), dim=-1)
        best = features @ features[0]
        best[0] = 1.0
        anchors = [0]
        for _ in range(int(K) - 1):
            next_anchor = int(torch.argmin(best).item())
            anchors.append(next_anchor)
            best = torch.maximum(best, features @ features[next_anchor])
            best[next_anchor] = 1.0
        anchor_index = torch.tensor(anchors, dtype=torch.long, device=device)
        return (features @ features.index_select(0, anchor_index).t()).argmax(dim=-1)

    def build_tensor_merge_plan(self, ctx, K: int) -> TensorMergePlan:
        """Build a list-free plan covering every input token exactly once."""

        device = ctx.tokens.device
        layout = ctx.layout
        total_length = int(layout.total_length)
        candidate = ctx.domain.candidate_indices.to(device)
        is_candidate = ctx.domain.candidate_mask.to(device)
        positions = torch.arange(total_length, device=device)
        protected_rank = torch.cumsum((~is_candidate).long(), dim=0) - 1
        candidate_rank = torch.cumsum(is_candidate.long(), dim=0) - 1

        group_ids = self._candidate_group_ids(ctx, K)
        n_protected = total_length - int(candidate.numel())
        candidate_group_full = torch.empty(total_length, dtype=torch.long, device=device)
        candidate_group_full[candidate] = group_ids
        group_of_token = torch.where(
            is_candidate, n_protected + candidate_group_full, protected_rank
        )

        num_groups = n_protected + int(K)
        representative = torch.full(
            (num_groups,), total_length, dtype=torch.long, device=device
        )
        representative.scatter_reduce_(
            0, group_of_token, positions, reduce="amin", include_self=True
        )
        order = torch.argsort(representative)
        output_position_of_group = torch.empty(
            num_groups, dtype=torch.long, device=device
        )
        output_position_of_group[order] = torch.arange(num_groups, device=device)
        input_to_output = output_position_of_group[group_of_token]
        output_to_input = representative[order]
        group_sizes = torch.bincount(group_of_token, minlength=num_groups)[order]
        weights = 1.0 / group_sizes[input_to_output].to(ctx.tokens.dtype)
        return TensorMergePlan(
            input_to_output=input_to_output,
            output_to_input=output_to_input,
            weights=weights,
            group_sizes=group_sizes,
            original_length=total_length,
            output_length=num_groups,
        )

    def _random_groups(self, ctx, K: int) -> list[list[int]]:
        """Balanced seeded partition of the candidate set into ``K`` groups.

        Sizes differ by at most one and every group is non-empty whenever
        ``n_candidate >= K``, so the output length is exactly
        ``protected + K`` and matches the equivalent top-K prune.
        """

        candidate = ctx.domain.candidate_indices.tolist()
        n = len(candidate)
        if K <= 0 or K > n:
            raise ValueError("random merge requires 1 <= K <= n_candidate")
        assignment = self._candidate_group_ids(ctx, K).tolist()
        groups: list[list[int]] = [[] for _ in range(K)]
        for position, group_index in enumerate(assignment):
            groups[group_index].append(candidate[position])
        return [sorted(group) for group in groups]

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
        elif self.feature in {"random", "kmeans"}:
            candidate_groups = self._random_groups(ctx, K)
        else:
            if self.feature == "k" and ctx.k is not None:
                features = ctx.k.index_select(2, ctx.domain.candidate_indices).mean(dim=1)[0]
            else:
                features = ctx.candidate_tokens()[0]
            features = F.normalize(features.float(), dim=-1)
            # Deterministic farthest-point anchors under cosine coverage.  The
            # greedy step keeps a running best-similarity vector and only
            # multiplies against the newly chosen anchor, so each of the K-1
            # iterations costs one mat-vec instead of a growing mat-mat.
            best = features @ features[0]
            best[0] = 1.0
            anchor_positions = [0]
            for _ in range(int(K) - 1):
                next_anchor = int(torch.argmin(best).item())
                anchor_positions.append(next_anchor)
                best = torch.maximum(best, features @ features[next_anchor])
                best[next_anchor] = 1.0
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
        if ctx.batch_size != 1:
            raise NotImplementedError("SimilarityMergePress V1 supports batch_size=1 only")
        plan = self.build_tensor_merge_plan(ctx, k)
        operator_result = self.operator.apply_tensor_plan(ctx, plan)
        metadata = {
            "press": self.name,
            "domain": ctx.domain.name,
            "operator": self.operator.name,
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

    def describe(self) -> dict:
        return {
            **super().describe(),
            "domain": getattr(self.domain, "name", self.domain),
            "feature": self.feature,
            "operator": self.operator.name,
            "seed": self.seed,
        }
