from __future__ import annotations

from typing import Any


class TokenSelector:
    name = "selector"

    def select(self, scores, domain, K: int, ctx=None):
        raise NotImplementedError

    def select_from_order(self, order, scores, domain, K: int, ctx=None):
        raise NotImplementedError(
            f"{type(self).__name__} does not support applying a frozen ranking"
        )

    def describe(self) -> dict[str, Any]:
        return {"name": self.name}
