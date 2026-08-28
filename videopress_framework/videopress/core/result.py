"""Results and reversible mappings emitted by operators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch


@dataclass
class SelectionResult:
    keep_candidate_indices: torch.Tensor
    drop_candidate_indices: torch.Tensor
    keep_global_indices: torch.Tensor
    K: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "keep_candidate_indices",
            "drop_candidate_indices",
            "keep_global_indices",
        ):
            value = getattr(self, name)
            if value.ndim != 2:
                raise ValueError(f"{name} must have shape [B,K]")
        if self.keep_candidate_indices.shape != self.keep_global_indices.shape:
            raise ValueError("local and global keep indices must have equal shape")
        if self.K != self.keep_global_indices.shape[1]:
            raise ValueError("K does not match keep index shape")

    @property
    def batch_size(self) -> int:
        return int(self.keep_global_indices.shape[0])


@dataclass
class TokenMapping:
    output_to_input: Any
    input_to_output: Optional[torch.Tensor]
    original_length: int
    compressed_length: int
    source_groups: Optional[list[list[int]]] = None

    def __post_init__(self) -> None:
        if self.original_length < 0 or self.compressed_length < 0:
            raise ValueError("mapping lengths must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if torch.is_tensor(value):
                return value.detach().cpu().tolist()
            return value

        return {
            "output_to_input": convert(self.output_to_input),
            "input_to_output": convert(self.input_to_output),
            "original_length": self.original_length,
            "compressed_length": self.compressed_length,
            "source_groups": self.source_groups,
        }

    @classmethod
    def identity(cls, length: int, batch_size: int = 1, device: torch.device | str = "cpu") -> "TokenMapping":
        indices = torch.arange(length, device=device, dtype=torch.long).expand(batch_size, -1).clone()
        return cls(indices, indices.clone(), length, length)


@dataclass
class OperatorResult:
    output: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)
    mapping: Optional[TokenMapping] = None
    aux: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompressionResult:
    output: torch.Tensor
    scores: Optional[torch.Tensor]
    selection: Optional[SelectionResult]
    metadata: dict[str, Any] = field(default_factory=dict)
    mapping: Optional[TokenMapping] = None
    aux: dict[str, Any] = field(default_factory=dict)

