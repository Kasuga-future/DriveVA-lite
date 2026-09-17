"""Token domains: the only token positions eligible for intervention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .layout import TokenLayout, TokenRange, get_last_history_range


@dataclass(frozen=True)
class TokenDomain:
    name: str
    candidate_mask: torch.Tensor
    protected_mask: torch.Tensor
    candidate_indices: torch.Tensor

    def __post_init__(self) -> None:
        if self.candidate_mask.ndim != 1 or self.protected_mask.ndim != 1:
            raise ValueError("domain masks must be one-dimensional")
        if self.candidate_mask.dtype != torch.bool or self.protected_mask.dtype != torch.bool:
            raise TypeError("domain masks must have bool dtype")
        if self.candidate_mask.shape != self.protected_mask.shape:
            raise ValueError("candidate/protected masks must have equal length")
        if self.candidate_indices.ndim != 1:
            raise ValueError("candidate_indices must be one-dimensional")
        if self.candidate_indices.device != self.candidate_mask.device:
            raise ValueError("domain tensors must be on the same device")
        if torch.any(self.candidate_mask & self.protected_mask):
            raise ValueError("candidate and protected masks overlap")
        expected = torch.where(self.candidate_mask)[0]
        if not torch.equal(expected, self.candidate_indices):
            raise ValueError("candidate_indices must match candidate_mask in sorted order")

    @property
    def total_length(self) -> int:
        return int(self.candidate_mask.numel())

    @property
    def n_candidate(self) -> int:
        return int(self.candidate_indices.numel())

    @property
    def n_protected(self) -> int:
        return int(self.protected_mask.sum().item())


class DomainBuilder:
    name = "domain"

    def build(self, layout: TokenLayout, device: torch.device | str) -> TokenDomain:
        raise NotImplementedError


def _range_domain(layout: TokenLayout, name: str, token_range: TokenRange, device: torch.device | str) -> TokenDomain:
    candidate = torch.zeros(layout.total_length, dtype=torch.bool, device=device)
    candidate[token_range.start : token_range.end] = True
    protected = ~candidate
    return TokenDomain(
        name=name,
        candidate_mask=candidate,
        protected_mask=protected,
        candidate_indices=torch.where(candidate)[0],
    )


class LastHistoryDomain(DomainBuilder):
    name = "last_history"

    def build(self, layout: TokenLayout, device: torch.device | str) -> TokenDomain:
        return _range_domain(layout, self.name, get_last_history_range(layout), device)


class HistoryDomain(DomainBuilder):
    name = "history"

    def build(self, layout: TokenLayout, device: torch.device | str) -> TokenDomain:
        return _range_domain(layout, self.name, layout.history_video, device)


class AllVideoDomain(DomainBuilder):
    name = "all_video"

    def build(self, layout: TokenLayout, device: torch.device | str) -> TokenDomain:
        return _range_domain(layout, self.name, layout.video, device)


class FutureVideoDomain(DomainBuilder):
    name = "future_video"

    def build(self, layout: TokenLayout, device: torch.device | str) -> TokenDomain:
        return _range_domain(layout, self.name, layout.future_video, device)


class HistoryLatentDomain(DomainBuilder):
    def __init__(self, latent_index: int) -> None:
        self.latent_index = int(latent_index)
        self.name = f"history_latent_{self.latent_index}"

    def build(self, layout: TokenLayout, device: torch.device | str) -> TokenDomain:
        if self.latent_index < 0 or self.latent_index >= layout.num_cond_latents:
            raise ValueError(f"latent_index {self.latent_index} is outside history")
        return _range_domain(layout, self.name, layout.frame_range(self.latent_index), device)


class FutureLatentDomain(DomainBuilder):
    """One future VAE latent in storage order (0 = nearest future).

    DriveVA stores video latents in temporal order: history frames first, then
    future frames.  ``future_latent_0`` is therefore the nearest future latent
    (the first generated frame after history) and ``future_latent_1`` is the
    next one.  The underlying global frame index is
    ``layout.num_cond_latents + latent_index``.
    """

    def __init__(self, latent_index: int) -> None:
        self.latent_index = int(latent_index)
        self.name = f"future_latent_{self.latent_index}"

    def build(self, layout: TokenLayout, device: torch.device | str) -> TokenDomain:
        if self.latent_index < 0:
            raise ValueError(f"future latent_index {self.latent_index} is negative")
        frame_index = int(layout.num_cond_latents) + self.latent_index
        if frame_index >= int(layout.video_f):
            raise ValueError(
                f"future latent_index {self.latent_index} is outside future "
                f"(num_cond_latents={layout.num_cond_latents}, video_f={layout.video_f})"
            )
        return _range_domain(layout, self.name, layout.frame_range(frame_index), device)


def build_domain(
    name: str | DomainBuilder,
    layout: TokenLayout,
    device: torch.device | str,
    **kwargs,
) -> TokenDomain:
    if isinstance(name, DomainBuilder):
        return name.build(layout, device)
    if hasattr(name, "build") and not isinstance(name, str):
        return name.build(layout, device)
    key = str(name).strip().lower()
    builders = {
        "last_history": LastHistoryDomain,
        "history": HistoryDomain,
        "all_history": HistoryDomain,
        "all_video": AllVideoDomain,
        "video": AllVideoDomain,
        "future_video": FutureVideoDomain,
    }
    if key.startswith("history_latent_"):
        return HistoryLatentDomain(int(key.rsplit("_", 1)[1])).build(layout, device)
    if key.startswith("future_latent_"):
        return FutureLatentDomain(int(key.rsplit("_", 1)[1])).build(layout, device)
    if key not in builders:
        raise ValueError(f"Unknown token domain: {name}")
    return builders[key]().build(layout, device)
