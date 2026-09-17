"""Execution plan that makes probe/intervention separation explicit."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .runtime import EvaluationMode, InjectionPoint


class ProbeMode(str, Enum):
    NONE = "none"
    ONLINE = "online"
    FORWARD_PROBE = "forward_probe"
    BACKWARD_PROBE = "backward_probe"


@dataclass(frozen=True)
class PressExecutionPlan:
    probe_required: bool
    probe_mode: ProbeMode
    intervention_pass: bool
    physical_mode: bool


def _operators(press):
    if hasattr(press, "presses"):
        for child in press.presses:
            yield from _operators(child)
        return
    operator = getattr(press, "operator", None)
    if operator is not None:
        yield operator


def validate_protocol(press, mode: EvaluationMode | str) -> None:
    """Reject semantic protocol combinations before a model forward."""

    eval_mode = EvaluationMode.parse(mode)
    if press is None:
        return
    if getattr(press, "name", "") in {"noop", "none", "full"}:
        return
    point = InjectionPoint.parse(getattr(press, "injection_point", InjectionPoint.VIDEO_INPUT))
    if point is InjectionPoint.SELF_ATTN_OUTPUT:
        raise NotImplementedError(f"{point.value} is declared but not implemented")
    operators = list(_operators(press))
    if not operators:
        raise ValueError(f"press {getattr(press, 'name', type(press).__name__)} has no operator")
    has_physical = any(bool(getattr(op, "physical_compression", False)) for op in operators)
    if eval_mode is EvaluationMode.CAUSAL:
        if point is InjectionPoint.BLOCK_INPUT:
            raise NotImplementedError(
                "block_input is implemented only for physical hidden-token pruning"
            )
        if has_physical:
            raise ValueError("causal mode cannot use a physical compression operator")
        if point is not InjectionPoint.VIDEO_INPUT:
            raise ValueError("causal mode requires injection_point=video_input")
        if any(not bool(getattr(op, "preserves_sequence_length", False)) for op in operators):
            raise ValueError("causal mode requires sequence-length-preserving operators")
    elif eval_mode is EvaluationMode.PHYSICAL:
        if point not in {InjectionPoint.SELF_ATTN_KV, InjectionPoint.BLOCK_INPUT}:
            raise ValueError(
                "physical mode requires injection_point=self_attn_kv or block_input"
            )
        if not has_physical:
            raise ValueError("physical mode requires a real sequence-compression operator")
        if any(not bool(getattr(op, "driveva_compatible", False)) for op in operators):
            raise ValueError("physical mode operator is not DriveVA K/V compatible")
        if point is InjectionPoint.BLOCK_INPUT and any(
            not bool(getattr(op, "block_input_compatible", False)) for op in operators
        ):
            raise ValueError("block_input requires a hidden-token-compatible operator")
    if point is InjectionPoint.VIDEO_INPUT and any(
        not bool(getattr(op, "preserves_sequence_length", False)) for op in operators
    ):
        raise ValueError("VIDEO_INPUT V1 requires sequence-length-preserving operators")


def build_execution_plan(press, mode: EvaluationMode | str | None = None) -> PressExecutionPlan:
    if mode is not None:
        validate_protocol(press, mode)
    scorer = getattr(press, "scorer", None)
    injection = InjectionPoint.parse(getattr(press, "injection_point", InjectionPoint.VIDEO_INPUT))
    raw_probe = getattr(scorer, "probe_mode", None)
    if raw_probe is None:
        raw_probe = ProbeMode.BACKWARD_PROBE if getattr(scorer, "requires_probe", False) else ProbeMode.NONE
    probe_value = getattr(raw_probe, "value", raw_probe)
    requires_probe = bool(getattr(scorer, "requires_probe", False)) or (
        injection is InjectionPoint.VIDEO_INPUT and probe_value == "online"
    )
    if requires_probe and probe_value == "none":
        raw_probe = ProbeMode.BACKWARD_PROBE
        probe_value = raw_probe.value
    if requires_probe and probe_value == "online":
        raw_probe = ProbeMode.FORWARD_PROBE
    probe_mode = raw_probe if isinstance(raw_probe, ProbeMode) else ProbeMode(str(raw_probe))
    physical = mode is not None and EvaluationMode.parse(mode) is EvaluationMode.PHYSICAL
    return PressExecutionPlan(
        probe_required=requires_probe,
        probe_mode=probe_mode,
        intervention_pass=True,
        physical_mode=physical,
    )
