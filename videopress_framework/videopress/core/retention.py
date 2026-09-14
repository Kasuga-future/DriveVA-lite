"""Named retention policies for the two DriveVA history latents."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HistoryRetentionPolicy:
    name: str
    selector: str
    selector_mode: str | None
    ratio: float
    budget_reference: str
    description: str

    def press_overrides(self) -> dict[str, Any]:
        selector: dict[str, Any] = {"name": self.selector}
        if self.selector_mode is not None:
            selector["mode"] = self.selector_mode
        return {
            "domain": "history",
            "selector": selector,
            "budget": {
                "type": "ratio",
                "value": self.ratio,
                "reference": self.budget_reference,
            },
        }


HISTORY_RETENTION_POLICIES: dict[str, HistoryRetentionPolicy] = {
    "drop_previous_keep_last_100": HistoryRetentionPolicy(
        name="drop_previous_keep_last_100",
        selector="history_topk",
        selector_mode="last_only",
        ratio=1.0,
        budget_reference="last_history",
        description="drop the previous history latent and keep all tokens in the last history latent",
    ),
    "drop_previous_keep_last_50": HistoryRetentionPolicy(
        name="drop_previous_keep_last_50",
        selector="history_topk",
        selector_mode="last_only",
        ratio=0.5,
        budget_reference="last_history",
        description="drop the previous history latent and keep 50% of the last history latent",
    ),
    "joint_keep_50": HistoryRetentionPolicy(
        name="joint_keep_50",
        selector="topk",
        selector_mode=None,
        ratio=0.5,
        budget_reference="eligible",
        description="rank both history latents together and keep 50% globally",
    ),
    "joint_keep_25": HistoryRetentionPolicy(
        name="joint_keep_25",
        selector="topk",
        selector_mode=None,
        ratio=0.25,
        budget_reference="eligible",
        description="rank both history latents together and keep 25% globally",
    ),
    "per_latent_keep_50": HistoryRetentionPolicy(
        name="per_latent_keep_50",
        selector="history_topk",
        selector_mode="per_latent",
        ratio=0.5,
        budget_reference="each_history",
        description="rank each history latent independently and keep 50% from each",
    ),
    "per_latent_keep_25": HistoryRetentionPolicy(
        name="per_latent_keep_25",
        selector="history_topk",
        selector_mode="per_latent",
        ratio=0.25,
        budget_reference="each_history",
        description="rank each history latent independently and keep 25% from each",
    ),
}


def get_history_retention_policy(name: str) -> HistoryRetentionPolicy:
    key = str(name).strip().lower()
    try:
        return HISTORY_RETENTION_POLICIES[key]
    except KeyError as exc:
        choices = ", ".join(HISTORY_RETENTION_POLICIES)
        raise ValueError(f"unknown history retention policy {name!r}; choose one of: {choices}") from exc


def apply_history_retention_policy(
    press_config: dict[str, Any], policy_name: str
) -> dict[str, Any]:
    """Return a press config with one authoritative retention policy applied."""

    config = deepcopy(press_config)
    policy = get_history_retention_policy(policy_name)
    config.update(policy.press_overrides())
    config["retention_policy"] = policy.name
    return config
