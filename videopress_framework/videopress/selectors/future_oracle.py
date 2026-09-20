"""Future-token oracle selectors for counterfactual experiments.

The oracle is built in stages:

1. ``future_fixed_tiles`` physically drops one deterministic tile from one
   future latent.  Running the official evaluator for every tile gives a
   scene-level leave-one-tile-out PDM curve.
2. ``oracle_future_mask`` reads a JSON mapping produced from those curves and
   keeps the highest-harm tiles for each scene, still applying a real physical
   KV/hidden-sequence prune.
3. ``oracle_future_token_mask`` applies the finer **token-level** mask the
   2026-09-20 conclusion asked for: the JSON stores explicit local token
   indices per future latent, so a set-level search (best-of-N random, greedy,
   beam) can be evaluated through the same physical path.

These selectors intentionally do not learn anything.  They are the measurement
path for the future oracle upper bound.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from ..core.registry import register_selector
from ..core.result import SelectionResult
from .base import TokenSelector


def _tile_membership(
    height: int,
    width: int,
    tile_h: int,
    tile_w: int,
    tile_id: int,
) -> torch.Tensor:
    """Return local token indices belonging to one normalized tile.

    This reproduces ``spatial_counterfactual_probe`` from the training teacher:
    normalized ``y,x`` are multiplied by ``tile_h,tile_w`` and floored.
    """

    height = int(height)
    width = int(width)
    tile_h = int(tile_h)
    tile_w = int(tile_w)
    tile_id = int(tile_id)
    if height <= 0 or width <= 0 or tile_h <= 0 or tile_w <= 0:
        raise ValueError("height, width, tile_h and tile_w must be positive")
    groups = tile_h * tile_w
    if not 0 <= tile_id < groups:
        raise ValueError(f"tile_id must be in [0, {groups - 1}]")
    local = torch.arange(height * width, dtype=torch.long)
    y = torch.div(local, width, rounding_mode="floor")
    x = local.remainder(width)
    norm_y = y.float() / max(height - 1, 1)
    norm_x = x.float() / max(width - 1, 1)
    tile_y = torch.floor(norm_y * tile_h).long().clamp(0, tile_h - 1)
    tile_x = torch.floor(norm_x * tile_w).long().clamp(0, tile_w - 1)
    membership = tile_y * tile_w + tile_x == tile_id
    if not bool(membership.any()):
        raise ValueError(f"tile {tile_id} contains no tokens")
    return local[membership]


@register_selector("future_fixed_tiles")
class FutureFixedTileSelector(TokenSelector):
    """Keep all future tokens except selected tiles in one future latent."""

    name = "future_fixed_tiles"

    def __init__(
        self,
        latent_index: int,
        drop_tiles: Sequence[int] = (),
        keep_tiles: Sequence[int] | None = None,
        tile_h: int = 3,
        tile_w: int = 4,
        require_num_future_latents: int = 2,
    ):
        self.latent_index = int(latent_index)
        self.drop_tiles = tuple(int(value) for value in drop_tiles)
        self.keep_tiles = (
            None if keep_tiles is None else tuple(int(value) for value in keep_tiles)
        )
        self.tile_h = int(tile_h)
        self.tile_w = int(tile_w)
        self.require_num_future_latents = int(require_num_future_latents)
        if self.latent_index < 0:
            raise ValueError("latent_index must be non-negative")
        if self.keep_tiles is not None and self.drop_tiles:
            raise ValueError("keep_tiles and drop_tiles are mutually exclusive")
        if not self.drop_tiles and self.keep_tiles is None:
            raise ValueError("future_fixed_tiles requires keep_tiles or drop_tiles")
        groups = self.tile_h * self.tile_w
        for value in (*self.drop_tiles, *(self.keep_tiles or ())):
            if not 0 <= value < groups:
                raise ValueError(f"tile id {value} is outside [0, {groups})")

    def _target_frame(self, layout):
        num_future = int(layout.video_f) - int(layout.num_cond_latents)
        if num_future != self.require_num_future_latents:
            raise ValueError(
                f"expected {self.require_num_future_latents} future latents, got {num_future}"
            )
        if self.latent_index >= num_future:
            raise ValueError(f"latent_index {self.latent_index} is outside future")
        return layout.frame_range(int(layout.num_cond_latents) + self.latent_index)

    def select(self, scores, domain, K: int | None = None, ctx=None) -> SelectionResult:
        if ctx is None:
            raise ValueError("future_fixed_tiles requires a TokenContext")
        if domain.name != "future_video":
            raise ValueError("future_fixed_tiles requires domain=future_video")
        if scores.shape != (ctx.batch_size, domain.n_candidate):
            raise ValueError("scores must have shape [B,N_candidate]")
        frame = self._target_frame(ctx.layout)
        per_latent = int(ctx.layout.tokens_per_latent)
        local_keep = torch.ones(per_latent, dtype=torch.bool)
        if self.keep_tiles is not None:
            local_keep[:] = False
            for tile_id in self.keep_tiles:
                local_keep[_tile_membership(
                    ctx.layout.video_h, ctx.layout.video_w,
                    self.tile_h, self.tile_w, tile_id
                )] = True
        else:
            for tile_id in self.drop_tiles:
                local_keep[_tile_membership(
                    ctx.layout.video_h, ctx.layout.video_w,
                    self.tile_h, self.tile_w, tile_id
                )] = False
        candidate = domain.candidate_indices.to(scores.device)
        selected = []
        for index in range(domain.n_candidate):
            global_index = int(candidate[index].item())
            if frame.start <= global_index < frame.end:
                if bool(local_keep[global_index - frame.start]):
                    selected.append(index)
            else:
                selected.append(index)
        selected_tensor = torch.tensor(selected, device=scores.device, dtype=torch.long)
        all_local = torch.arange(domain.n_candidate, device=scores.device)
        keep_local = selected_tensor.unsqueeze(0).expand(ctx.batch_size, -1)
        drop_mask = torch.ones(domain.n_candidate, dtype=torch.bool, device=scores.device)
        drop_mask[selected_tensor] = False
        drop_local = all_local[drop_mask].unsqueeze(0).expand(ctx.batch_size, -1)
        keep_global = candidate[keep_local]
        return SelectionResult(
            keep_candidate_indices=keep_local,
            drop_candidate_indices=drop_local,
            keep_global_indices=keep_global,
            K=int(keep_local.shape[1]),
            metadata={
                "selector": self.name,
                "latent_index": self.latent_index,
                "drop_tiles": list(self.drop_tiles),
                "keep_tiles": None if self.keep_tiles is None else list(self.keep_tiles),
                "tile_h": self.tile_h,
                "tile_w": self.tile_w,
            },
        )

    def describe(self) -> dict:
        return {
            "name": self.name,
            "latent_index": self.latent_index,
            "drop_tiles": list(self.drop_tiles),
            "keep_tiles": None if self.keep_tiles is None else list(self.keep_tiles),
            "tile_h": self.tile_h,
            "tile_w": self.tile_w,
        }


class _FutureMaskSelector(TokenSelector):
    """Shared physical application of a per-scene future keep-mask.

    Subclasses convert one JSON entry into a boolean keep vector over the
    concatenated future latents (storage order, ``future_latent_0`` first); this
    class turns that into a real :class:`SelectionResult` on the
    ``future_video`` candidate domain.  Rows of one batch are padded to a common
    K, exactly like the tile-level oracle did, so a different per-scene K stays
    representable.
    """

    require_num_future_latents = 2

    def __init__(self, require_num_future_latents: int = 2):
        self.require_num_future_latents = int(require_num_future_latents)
        if self.require_num_future_latents <= 0:
            raise ValueError("require_num_future_latents must be positive")

    # -- subclass hooks ----------------------------------------------------
    def _entry_for_scene(self, scene_token: str) -> Any:
        """Per-scene entry, falling back to the ``"*"`` broadcast entry.

        A searched oracle pattern is shared by every scene (that is what a
        deployable selector would emit), so a mask JSON may carry a single
        ``"*"`` entry instead of one entry per scene.  An explicit scene entry
        always wins, and a missing scene with no ``"*"`` fallback is still a
        loud error rather than a silent empty mask.
        """

        entry = self.masks.get(scene_token)
        if entry is None:
            entry = self.masks.get("*")
        if entry is None:
            raise KeyError(
                f"{self.name} JSON has no entry for scene {scene_token!r} and no "
                "'*' broadcast entry"
            )
        return entry

    def _row_local_keep(self, layout, entry) -> torch.Tensor:
        raise NotImplementedError

    def _selector_metadata(self) -> dict:
        return {"selector": self.name}

    # -- shared implementation --------------------------------------------
    def _num_future_latents(self, layout) -> int:
        num_future = int(layout.video_f) - int(layout.num_cond_latents)
        if num_future != self.require_num_future_latents:
            raise ValueError(
                f"expected {self.require_num_future_latents} future latents, "
                f"got {num_future}"
            )
        return num_future

    def select(self, scores, domain, K: int | None = None, ctx=None) -> SelectionResult:
        if ctx is None:
            raise ValueError(f"{self.name} requires a TokenContext")
        if domain.name != "future_video":
            raise ValueError(f"{self.name} requires domain=future_video")
        if scores.shape != (ctx.batch_size, domain.n_candidate):
            raise ValueError("scores must have shape [B,N_candidate]")
        candidate = domain.candidate_indices.to(scores.device)
        rows = []
        for batch_index in range(ctx.batch_size):
            scene_token = str(ctx.scene_token)
            entry = self._entry_for_scene(scene_token)
            local_keep = self._row_local_keep(ctx.layout, entry).to(scores.device)
            selected = []
            for index in range(domain.n_candidate):
                global_index = int(candidate[index].item())
                future_offset = global_index - int(ctx.layout.future_video.start)
                if future_offset < 0 or future_offset >= int(local_keep.numel()):
                    raise RuntimeError("candidate is not a future token")
                if bool(local_keep[future_offset]):
                    selected.append(index)
            rows.append(torch.tensor(selected, device=scores.device, dtype=torch.long))
        counts = [int(row.numel()) for row in rows]
        actual_k = max(counts) if counts else 0
        keep_rows = []
        for row in rows:
            if row.numel() < actual_k:
                fill = torch.arange(
                    domain.n_candidate, device=scores.device, dtype=torch.long
                )
                mask = torch.ones(domain.n_candidate, dtype=torch.bool, device=scores.device)
                mask[row] = False
                extra = fill[mask][: actual_k - row.numel()]
                row = torch.cat([row, extra])
            keep_rows.append(torch.sort(row)[0])
        keep_local = (
            torch.stack(keep_rows)
            if keep_rows
            else torch.empty((0, 0), dtype=torch.long, device=scores.device)
        )
        keep_mask = torch.zeros(
            (ctx.batch_size, domain.n_candidate), dtype=torch.bool, device=scores.device
        )
        keep_mask.scatter_(1, keep_local, True)
        all_local = torch.arange(domain.n_candidate, device=scores.device).expand(
            ctx.batch_size, -1
        )
        drop_local = all_local[~keep_mask].reshape(
            ctx.batch_size, domain.n_candidate - actual_k
        )
        metadata = dict(self._selector_metadata())
        metadata["per_scene_keep"] = counts
        return SelectionResult(
            keep_candidate_indices=keep_local,
            drop_candidate_indices=drop_local,
            keep_global_indices=candidate[keep_local],
            K=actual_k,
            metadata=metadata,
        )


@register_selector("oracle_future_mask")
class OracleFutureMaskSelector(_FutureMaskSelector):
    """Apply a precomputed per-scene future tile oracle mask."""

    name = "oracle_future_mask"

    def __init__(
        self,
        path: str,
        tile_h: int = 3,
        tile_w: int = 4,
        require_num_future_latents: int = 2,
    ):
        super().__init__(require_num_future_latents=require_num_future_latents)
        self.path = str(Path(path).expanduser().resolve())
        data = json.loads(Path(self.path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise TypeError("oracle future mask JSON must map scene_token -> mask")
        self.masks = data
        self.tile_h = int(tile_h)
        self.tile_w = int(tile_w)

    def _row_local_keep(self, layout, entry):
        num_future = self._num_future_latents(layout)
        pieces = []
        for latent_index in range(num_future):
            key = f"future_latent_{latent_index}"
            tile_ids = entry.get(key)
            if tile_ids is None:
                raise ValueError(f"oracle mask entry missing {key!r}")
            local_keep = torch.zeros(int(layout.tokens_per_latent), dtype=torch.bool)
            for tile_id in tile_ids:
                local_keep[
                    _tile_membership(
                        layout.video_h,
                        layout.video_w,
                        self.tile_h,
                        self.tile_w,
                        int(tile_id),
                    )
                ] = True
            pieces.append(local_keep)
        return torch.cat(pieces)

    def _selector_metadata(self) -> dict:
        return {
            "selector": self.name,
            "oracle_path": self.path,
            "tile_h": self.tile_h,
            "tile_w": self.tile_w,
        }

    def describe(self) -> dict:
        return {
            "name": self.name,
            "path": self.path,
            "tile_h": self.tile_h,
            "tile_w": self.tile_w,
        }


@register_selector("oracle_future_token_mask")
class OracleFutureTokenMaskSelector(_FutureMaskSelector):
    """Apply a precomputed per-scene **token-level** future oracle mask.

    JSON format (scene token -> per-latent local token indices)::

        {
          "scene-A": {
            "future_latent_0": [0, 5, 17],
            "future_latent_1": [1, 2]
          }
        }

    Missing ``future_latent_i`` keys mean "keep nothing from that latent".  A
    scene entry may also be a flat list of future offsets (storage order,
    ``latent * tokens_per_latent + local``), which is what
    :func:`videopress.oracle.token_set.latent_local_mask` writes for single
    latent panels.  Indices are validated against the runtime layout, so a mask
    built for a different grid fails loudly instead of silently mis-selecting.
    """

    name = "oracle_future_token_mask"

    def __init__(
        self,
        path: str,
        require_num_future_latents: int = 2,
    ):
        super().__init__(require_num_future_latents=require_num_future_latents)
        self.path = str(Path(path).expanduser().resolve())
        data = json.loads(Path(self.path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise TypeError("oracle future token mask JSON must map scene_token -> mask")
        self.masks = data

    def _row_local_keep(self, layout, entry):
        num_future = self._num_future_latents(layout)
        tokens_per_latent = int(layout.tokens_per_latent)
        local_keep = torch.zeros(
            tokens_per_latent * num_future, dtype=torch.bool
        )
        if isinstance(entry, Mapping):
            for latent_index in range(num_future):
                key = f"future_latent_{latent_index}"
                values = entry.get(key)
                if values is None:
                    continue
                if isinstance(values, (str, bytes)) or not isinstance(
                    values, Sequence
                ):
                    raise TypeError(f"{key} must be a list of local token indices")
                for value in values:
                    local = int(value)
                    if local < 0 or local >= tokens_per_latent:
                        raise ValueError(
                            f"{key} token {local} is outside "
                            f"[0, {tokens_per_latent})"
                        )
                    local_keep[latent_index * tokens_per_latent + local] = True
            return local_keep
        if isinstance(entry, Sequence) and not isinstance(entry, (str, bytes)):
            total = tokens_per_latent * num_future
            for value in entry:
                token = int(value)
                if token < 0 or token >= total:
                    raise ValueError(
                        f"flat future token {token} is outside [0, {total})"
                    )
                local_keep[token] = True
            return local_keep
        raise TypeError(
            "future token mask entry must be a mapping of future_latent_i or a "
            "list of flat future offsets"
        )

    def _selector_metadata(self) -> dict:
        return {"selector": self.name, "oracle_path": self.path}

    def describe(self) -> dict:
        return {"name": self.name, "path": self.path}
