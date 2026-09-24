"""Cross-layer selection persistence for physical token compression.

The source layer owns scoring and selection.  Deeper layers reuse the exact
same global token indices, while still gathering their freshly computed K/V
tensors.  State is scoped to one runtime sample and additionally keyed by the
active DiT and diffusion timestep to prevent accidental cross-call reuse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CrossLayerPersistence:
    """Cross-layer reuse of one source-layer token selection.

    ``enabled`` is the ONLY switch that means "reuse the selection in later
    blocks":

    * ``enabled=False`` -- the source-layer selection is applied **once** (a
      one-shot K/V prune at the source layer) and is never reused.  This is the
      explicit "cross-layer persistence off" configuration.
    * ``enabled=True`` -- the selection is stored and reused by every block in
      ``[source_layer + 1, end_layer]`` (``end_layer=None`` means through the
      last block).

    ``enabled=True`` together with ``end_layer == source_layer`` used to be
    accepted and silently turned persistence off while the runtime still
    stamped ``cross_layer_persistent: True`` -- a run that was really a one-shot
    prune was labelled as persistent (audit 2026-09-12, BUG-13).  That
    combination is now rejected by :meth:`validate_for`; use ``enabled=False``
    to request a one-shot prune explicitly.
    """

    enabled: bool = False
    end_layer: int | None = None
    mode: str = "kv_only"

    def __post_init__(self) -> None:
        if self.end_layer is not None and int(self.end_layer) < 0:
            raise ValueError("cross-layer persistence end_layer must be non-negative")
        object.__setattr__(self, "enabled", bool(self.enabled))
        object.__setattr__(
            self,
            "end_layer",
            None if self.end_layer is None else int(self.end_layer),
        )
        mode = str(self.mode).strip().lower()
        if mode not in {"kv_only", "hidden_sequence"}:
            raise ValueError(
                "cross-layer persistence mode must be kv_only or hidden_sequence"
            )
        object.__setattr__(self, "mode", mode)

    @classmethod
    def from_config(cls, value: Any) -> "CrossLayerPersistence":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, bool):
            return cls(enabled=value)
        if not isinstance(value, dict):
            raise TypeError("cross_layer_persistence must be a bool or mapping")
        section = dict(value)
        enabled = section.pop("enabled", True)
        end_layer = section.pop("end_layer", None)
        mode = section.pop("mode", "kv_only")
        if section:
            raise ValueError(
                "unknown cross_layer_persistence fields: "
                + ", ".join(sorted(section))
            )
        return cls(enabled=enabled, end_layer=end_layer, mode=mode)

    def persists_beyond(self, source_layer: int) -> bool:
        """True only when the stored selection is actually reused later.

        With ``enabled=False`` the intervention is a one-shot prune: the
        source-layer K/V selection is applied once and no later block reuses it.
        """
        if not self.enabled:
            return False
        if self.end_layer is not None and int(self.end_layer) <= int(source_layer):
            return False
        return True

    def validate_for(self, source_layer: int) -> None:
        """Reject configurations that would silently change the intervention."""
        if not self.enabled:
            return
        if self.end_layer is None:
            return
        end_layer = int(self.end_layer)
        source_layer = int(source_layer)
        if end_layer == source_layer:
            raise ValueError(
                "cross_layer_persistence enabled=True with end_layer == scorer.layer "
                f"({source_layer}) persists nothing: the source-layer selection would "
                "be applied once and never reused, yet the run used to be stamped "
                "cross_layer_persistent=True (audit 2026-09-12, BUG-13). Set "
                "cross_layer_persistence.enabled=false for an explicit one-shot prune, "
                "or leave end_layer unset to persist through the last block."
            )
        if end_layer < source_layer:
            raise ValueError(
                "cross-layer persistence end_layer cannot precede scorer.layer"
            )

    def includes(self, source_layer: int, current_layer: int) -> bool:
        if not self.enabled or int(current_layer) < int(source_layer):
            return False
        return self.end_layer is None or int(current_layer) <= self.end_layer

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "end_layer": self.end_layer,
            "mode": self.mode,
        }


@dataclass(frozen=True)
class PersistentSelectionRecord:
    selection: Any
    source_layer: int
    scene_token: str
    diffusion_rank: int | None
    model_name: str
    layout_length: int
    domain_name: str
    batch_size: int


class CrossLayerSelectionStore:
    """Store one source-layer selection per model/timestep within a sample."""

    def __init__(self) -> None:
        self._records: dict[tuple[str, int | None], PersistentSelectionRecord] = {}

    def clear(self) -> None:
        self._records.clear()

    @staticmethod
    def _key(model_name: str, diffusion_rank: int | None) -> tuple[str, int | None]:
        return str(model_name), None if diffusion_rank is None else int(diffusion_rank)

    def remember(self, ctx, model_name: str, source_layer: int, selection) -> None:
        key = self._key(model_name, ctx.diffusion_rank)
        self._records[key] = PersistentSelectionRecord(
            selection=selection,
            source_layer=int(source_layer),
            scene_token=str(ctx.scene_token),
            diffusion_rank=None if ctx.diffusion_rank is None else int(ctx.diffusion_rank),
            model_name=str(model_name),
            layout_length=int(ctx.layout.total_length),
            domain_name=str(ctx.domain.name),
            batch_size=int(ctx.batch_size),
        )

    def recall(self, ctx, model_name: str, source_layer: int) -> PersistentSelectionRecord:
        key = self._key(model_name, ctx.diffusion_rank)
        record = self._records.get(key)
        if record is None:
            raise RuntimeError(
                "persistent K/V selection is unavailable before its source layer "
                f"(model={model_name}, timestep={ctx.diffusion_rank}, source_layer={source_layer})"
            )
        expected = (
            str(ctx.scene_token),
            int(ctx.layout.total_length),
            str(ctx.domain.name),
            int(ctx.batch_size),
            int(source_layer),
        )
        actual = (
            record.scene_token,
            record.layout_length,
            record.domain_name,
            record.batch_size,
            record.source_layer,
        )
        if actual != expected:
            raise RuntimeError(
                "persistent K/V selection context mismatch: "
                f"expected={expected}, cached={actual}"
            )
        return record


class HiddenSequencePersistenceController:
    """Physically shorten the residual sequence after the source block.

    The source block still performs the configured post-RoPE K/V pruning. Once
    it has selected tokens, this controller gathers the residual stream, RoPE
    frequencies, and per-token timestep modulation for all downstream blocks.
    Before the model head, dropped conditioned-history positions are restored
    as zeros; those positions are overwritten by clean history latents by the
    surrounding DriveVA denoising loop.
    """

    def __init__(self, runtime, model_name: str, persistence: CrossLayerPersistence):
        if persistence.mode != "hidden_sequence":
            raise ValueError("hidden controller requires mode=hidden_sequence")
        self.runtime = runtime
        self.model_name = str(model_name)
        self.persistence = persistence
        self._active = False
        self._keep = None
        self._original_length = 0
        self._original_freqs = None
        self._original_t_mod = None
        self._num_blocks = 0

    @staticmethod
    def _gather_sequence(tensor, keep):
        import torch

        if tensor.shape[0] != keep.shape[0]:
            raise ValueError("batch dimension differs from persistent selection")
        view_shape = (keep.shape[0], keep.shape[1]) + (1,) * (tensor.ndim - 2)
        expand_shape = (keep.shape[0], keep.shape[1]) + tuple(tensor.shape[2:])
        index = keep.view(view_shape).expand(expand_shape)
        return torch.gather(tensor, dim=1, index=index)

    def begin_forward(self, x, freqs, t_mod, *, num_blocks: int) -> None:
        # Reject the one end_layer setting that silently corrupts the forward
        # pass.  The controller shortens the residual stream after the source
        # layer and RESTORES it (scattering kept rows into a zero tensor) at
        # `end_layer`.  When `end_layer` is a strict interior layer, every block
        # after it runs full-length self-attention over a sequence whose dropped
        # history slots are ZERO vectors -- so every kept token, including the
        # trajectory tokens that produce PDM, mixes with zeros.  Such a run is
        # not a valid compression measurement (audit 2026-09-12, BUG-11).
        #
        # The two legitimate settings are untouched:
        #   * end_layer is None          -> restore immediately before the head,
        #                                   which is position-wise, so the zeros
        #                                   only occupy dropped slots (harmless);
        # A source-layer-only K/V prune is represented by enabled=False and
        # never installs this hidden-sequence controller.
        end_layer = self.persistence.end_layer
        source_layer = self.runtime.resolve_scorer_layer()
        if source_layer is None:
            source_layer = getattr(self.runtime.press.scorer, "layer")
        source_layer = int(source_layer)
        last_layer = int(num_blocks) - 1
        if end_layer is not None and source_layer < int(end_layer) < last_layer:
            raise ValueError(
                "hidden_sequence persistence with an interior end_layer would feed "
                f"zero-filled history slots into layers {int(end_layer) + 1}..{last_layer} "
                f"(source_layer={source_layer}, end_layer={int(end_layer)}, "
                f"last_layer={last_layer}); this is not a valid compression setting. "
                "Use end_layer=None to compress through the last block, or "
                "end_layer == source_layer to disable compression."
            )
        self._active = False
        self._keep = None
        self._original_length = int(x.shape[1])
        self._original_freqs = freqs
        self._original_t_mod = t_mod
        self._num_blocks = int(num_blocks)

    def _restore(self, x):
        if not self._active or self._keep is None:
            return x
        keep = self._keep.to(x.device)
        restored = x.new_zeros((x.shape[0], self._original_length, x.shape[2]))
        restored.scatter_(1, keep.unsqueeze(-1).expand_as(x), x)
        self._active = False
        self._keep = None
        return restored

    def after_block(self, block_idx: int, x, freqs, t_mod):
        source_layer = self.runtime.resolve_scorer_layer()
        if source_layer is None:
            source_layer = getattr(self.runtime.press.scorer, "layer")
        source_layer = int(source_layer)
        block_idx = int(block_idx)
        if self._active:
            end_layer = self.persistence.end_layer
            if end_layer is not None and block_idx >= int(end_layer):
                original_freqs = self._original_freqs
                original_t_mod = self._original_t_mod
                x = self._restore(x)
                self._original_freqs = None
                self._original_t_mod = None
                return x, original_freqs, original_t_mod
            return x, freqs, t_mod
        if block_idx != source_layer:
            return x, freqs, t_mod
        if self.persistence.end_layer == source_layer:
            return x, freqs, t_mod

        ctx = self.runtime.current_context
        if ctx is None or int(ctx.layer_idx) != source_layer:
            raise RuntimeError(
                "hidden-sequence persistence did not observe its source-layer selection"
            )
        record = self.runtime._cross_layer_selection_store.recall(
            ctx, self.model_name, source_layer
        )
        protected = ctx.domain.protected_mask.nonzero(as_tuple=False).flatten()
        import torch

        keep_rows = [
            torch.sort(torch.cat([protected, row.to(protected.device).long()]))[0]
            for row in record.selection.keep_global_indices
        ]
        keep = torch.stack(keep_rows).to(x.device)
        if int(x.shape[1]) != self._original_length:
            raise ValueError("source block output was already sequence-compressed")
        x = self._gather_sequence(x, keep)

        if freqs.ndim == 3:
            freqs = freqs.unsqueeze(0).expand(x.shape[0], -1, -1, -1)
        freqs = self._gather_sequence(freqs, keep)
        if t_mod.ndim == 4:
            t_mod = self._gather_sequence(t_mod, keep)
        elif t_mod.ndim != 3:
            raise ValueError(f"unsupported t_mod rank for hidden pruning: {t_mod.ndim}")

        self._active = True
        self._keep = keep
        last_layer = (
            self._num_blocks - 1
            if self.persistence.end_layer is None
            else min(int(self.persistence.end_layer), self._num_blocks - 1)
        )
        metadata = self.runtime.last_result.metadata
        metadata.update(
            {
                "cross_layer_persistence_mode": "hidden_sequence",
                "hidden_sequence_length_before": self._original_length,
                "hidden_sequence_length_after": int(x.shape[1]),
                "hidden_sequence_ratio": float(x.shape[1] / self._original_length),
                "hidden_sequence_first_compressed_layer": source_layer + 1,
                "hidden_sequence_last_compressed_layer": last_layer,
                "hidden_sequence_compressed_layer_count": max(
                    0, last_layer - source_layer
                ),
            }
        )
        return x, freqs, t_mod

    def finish_forward(self, x):
        x = self._restore(x)
        self._original_freqs = None
        self._original_t_mod = None
        return x
