from __future__ import annotations

from typing import Any


class TokenOperator:
    name = "operator"
    preserves_sequence_length = True
    physical_compression = False
    driveva_compatible = False

    def apply(self, ctx, selection):
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "preserves_sequence_length": self.preserves_sequence_length,
            "physical_compression": self.physical_compression,
            "driveva_compatible": self.driveva_compatible,
        }
