from __future__ import annotations

from typing import Any


class TokenScorer:
    requires_probe = False
    requires_grad = False
    probe_mode = "none"
    name = "scorer"

    def score(self, ctx):
        raise NotImplementedError

    def signature(self) -> str:
        return self.name

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "requires_probe": self.requires_probe, "probe_mode": self.probe_mode}
