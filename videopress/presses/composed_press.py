from __future__ import annotations

from ..core.result import CompressionResult
from .base import BaseVideoPress


class ComposedPress(BaseVideoPress):
    name = "composed"

    def __init__(self, presses):
        self.presses = list(presses)
        if not self.presses:
            raise ValueError("ComposedPress needs at least one press")

    def prepare(self, runtime) -> None:
        for press in self.presses:
            press.prepare(runtime)

    def apply(self, ctx) -> CompressionResult:
        current = ctx
        history = []
        final = None
        for press in self.presses:
            final = press.apply(current)
            history.append(final.metadata)
            # A shorter merged sequence no longer satisfies the original layout;
            # composition is therefore intentionally limited to same-length V1 ops.
            if final.output.shape[1] != current.layout.total_length:
                if press is not self.presses[-1]:
                    raise ValueError("a shortening press must be last in ComposedPress V1")
                break
            current = type(current)(
                tokens=final.output,
                layout=current.layout,
                domain=current.domain,
                scene_token=current.scene_token,
                frame_token=current.frame_token,
                log_id=current.log_id,
                timestamp=current.timestamp,
                timestep=current.timestep,
                diffusion_rank=current.diffusion_rank,
                layer_idx=current.layer_idx,
                q=current.q,
                k=current.k,
                v=current.v,
                trajectory_pred=current.trajectory_pred,
                metadata={**current.metadata, **final.metadata},
            )
        assert final is not None
        final.metadata["composed_history"] = history
        return final

    def finalize(self, runtime) -> None:
        for press in reversed(self.presses):
            press.finalize(runtime)
