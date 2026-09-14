"""Deterministic seeds independent of process/rank ordering."""

from __future__ import annotations

import hashlib


def stable_seed(*parts: object, modulo: int = 2**63 - 1) -> int:
    payload = "\x1f".join("<none>" if part is None else str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little", signed=False) % modulo


def stable_hash(*parts: object) -> int:
    """Compatibility name used by the design document."""

    return stable_seed(*parts)
