"""Small registries used by config-driven experiment construction."""

from __future__ import annotations

from typing import Any, Callable


PRESS_REGISTRY: dict[str, type] = {}
SCORER_REGISTRY: dict[str, type] = {}
SELECTOR_REGISTRY: dict[str, type] = {}
OPERATOR_REGISTRY: dict[str, type] = {}


def _register(registry: dict[str, type], name: str) -> Callable:
    key = str(name).strip().lower()

    def decorator(cls: type) -> type:
        if key in registry and registry[key] is not cls:
            raise KeyError(f"Registry entry already exists: {key}")
        registry[key] = cls
        return cls

    return decorator


def register_press(name: str) -> Callable:
    return _register(PRESS_REGISTRY, name)


def register_scorer(name: str) -> Callable:
    return _register(SCORER_REGISTRY, name)


def register_selector(name: str) -> Callable:
    return _register(SELECTOR_REGISTRY, name)


def register_operator(name: str) -> Callable:
    return _register(OPERATOR_REGISTRY, name)


def get_registered(registry: dict[str, type], name: str) -> type:
    key = str(name).strip().lower()
    try:
        return registry[key]
    except KeyError as exc:
        available = ", ".join(sorted(registry))
        raise KeyError(f"Unknown registry entry {name!r}; available: {available}") from exc
