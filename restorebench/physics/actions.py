# ABOUTME: Implements canonical enumeration, applicability, and isolated application of Q-V actions.
# ABOUTME: Enforces atomic generator, shunt-step, and transformer-tap moves in stable order.
from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any, Literal, TypeAlias

import pandas as pd
from annotated_types import Ge, Le
from pydantic import BaseModel, TypeAdapter

from restorebench.schemas.actions import (
    Action,
    GenVoltageSetpointAction,
    ShuntStepAction,
    TapAdjustmentAction,
)
from restorebench.schemas.errors import InvalidActionError
from restorebench.schemas.topology import ApplicabilityResult


ACTION_POLICY_VERSION = "qv-atomic-v1"
VM_MIN = 0.95
VM_MAX = 1.05
VM_STEP = 0.01

GeneratorQStatus = Literal[
    "PV_CONTROLLABLE",
    "Q_LIMITED_UPPER",
    "Q_LIMITED_LOWER",
    "UNKNOWN",
]
QContext: TypeAlias = Mapping[int, GeneratorQStatus]

_ACTION_ADAPTER: TypeAdapter[Action] = TypeAdapter(Action)


def _schema_int_bounds(model: type[BaseModel], field_name: str) -> tuple[int, int]:
    """Read a field's inclusive integer bounds from the action schema itself."""
    lower: int | None = None
    upper: int | None = None
    for constraint in model.model_fields[field_name].metadata:
        if isinstance(constraint, Ge):
            lower = int(constraint.ge)  # type: ignore[arg-type]
        elif isinstance(constraint, Le):
            upper = int(constraint.le)  # type: ignore[arg-type]
    if lower is None or upper is None:
        raise ValueError(f"{model.__name__}.{field_name} has no inclusive integer bounds")
    return lower, upper


_TAP_SCHEMA_BOUNDS = _schema_int_bounds(TapAdjustmentAction, "new_tap_pos")


def enumerate_legal_qv_actions(
    net: Any,
    q_context: QContext | None = None,
) -> list[Action]:
    """Return every runtime-legal atomic action in stable family/id/target order."""
    context = q_context or {}
    actions: list[Action] = []

    for gen_id, row in net.gen.sort_index().iterrows():
        if not _in_service(row):
            continue
        current = float(row["vm_pu"])
        status = context.get(int(gen_id), "UNKNOWN")
        for target in _generator_targets(current):
            action = GenVoltageSetpointAction(
                type="GEN_V_SETPOINT",
                gen_id=int(gen_id),
                new_vm_pu=target,
            )
            if get_qv_action_applicability(net, action, {int(gen_id): status}).applicable:
                actions.append(action)

    for shunt_id, row in net.shunt.sort_index().iterrows():
        if not _in_service(row):
            continue
        current_step = _integer_or_none(row.get("step"))
        max_step = _integer_or_none(row.get("max_step"))
        if current_step not in {0, 1} or max_step != 1:
            continue
        action = ShuntStepAction(
            type="SHUNT_STEP",
            shunt_id=int(shunt_id),
            new_step=1 - current_step,
        )
        if get_qv_action_applicability(net, action, context).applicable:
            actions.append(action)

    for trafo_id, row in net.trafo.sort_index().iterrows():
        if not _in_service(row):
            continue
        current = _integer_or_none(row.get("tap_pos"))
        tap_min = _integer_or_none(row.get("tap_min"))
        tap_max = _integer_or_none(row.get("tap_max"))
        if current is None or tap_min is None or tap_max is None:
            continue
        schema_min, schema_max = _TAP_SCHEMA_BOUNDS
        for target in (current - 1, current + 1):
            if not tap_min <= target <= tap_max:
                continue
            # A net may declare wider tap bounds than the action schema accepts; those
            # targets are not runtime-legal, so skip them instead of failing construction.
            if not schema_min <= target <= schema_max:
                continue
            action = TapAdjustmentAction(
                type="TAP_ADJUSTMENT",
                trafo_id=int(trafo_id),
                new_tap_pos=target,
            )
            if get_qv_action_applicability(net, action, context).applicable:
                actions.append(action)

    return actions


def get_qv_action_applicability(
    net: Any,
    action: Action,
    q_context: QContext | None = None,
) -> ApplicabilityResult:
    """Check canonical runtime applicability without mutating the network."""
    parsed = _ACTION_ADAPTER.validate_python(action)
    context = q_context or {}

    if parsed.type == "GEN_V_SETPOINT":
        return _check_gen_voltage(net, parsed, context)
    if parsed.type == "SHUNT_STEP":
        return _check_shunt_step(net, parsed)
    return _check_tap_adjustment(net, parsed)


def apply_qv_action(
    net: Any,
    action: Action,
    q_context: QContext | None = None,
) -> Any:
    """Apply one applicable action to a deep copy and return that isolated state."""
    parsed = _ACTION_ADAPTER.validate_python(action)
    applicability = get_qv_action_applicability(net, parsed, q_context)
    if not applicability.applicable:
        raise InvalidActionError(parsed, applicability.reason or "action is not applicable")

    changed = copy.deepcopy(net)
    if parsed.type == "GEN_V_SETPOINT":
        changed.gen.at[parsed.gen_id, "vm_pu"] = parsed.new_vm_pu
    elif parsed.type == "SHUNT_STEP":
        changed.shunt.at[parsed.shunt_id, "step"] = parsed.new_step
    else:
        changed.trafo.at[parsed.trafo_id, "tap_pos"] = parsed.new_tap_pos
    return changed


def _check_gen_voltage(
    net: Any,
    action: GenVoltageSetpointAction,
    q_context: QContext,
) -> ApplicabilityResult:
    gen_id = action.gen_id
    if gen_id not in net.gen.index:
        return _not_applicable(action, f"gen {gen_id} does not exist")
    if not bool(net.gen.at[gen_id, "in_service"]):
        return _not_applicable(action, f"gen {gen_id} is out of service")

    current = float(net.gen.at[gen_id, "vm_pu"])
    target = float(action.new_vm_pu)
    if current < VM_MIN - 1e-10 or current > VM_MAX + 1e-10:
        return _not_applicable(
            action,
            f"gen {gen_id} current voltage setpoint {current:.10g} is outside [{VM_MIN}, {VM_MAX}]",
        )
    if math.isclose(target, current, rel_tol=0.0, abs_tol=1e-10):
        return _not_applicable(action, f"gen {gen_id} already at that voltage setpoint - no-op")
    if not _matches_any(target, _generator_targets(current)):
        return _not_applicable(
            action,
            f"generator voltage target must be one clipped atomic step from {current:.10g}",
        )

    status = q_context.get(gen_id, "UNKNOWN")
    if status == "Q_LIMITED_UPPER" and target > current:
        return _not_applicable(
            action,
            f"gen {gen_id} is Q-saturated at its upper limit; a setpoint increase is unavailable",
        )
    if status == "Q_LIMITED_LOWER" and target < current:
        return _not_applicable(
            action,
            f"gen {gen_id} is Q-saturated at its lower limit; a setpoint decrease is unavailable",
        )
    return _applicable(action)


def _check_shunt_step(net: Any, action: ShuntStepAction) -> ApplicabilityResult:
    shunt_id = action.shunt_id
    if shunt_id not in net.shunt.index:
        return _not_applicable(action, f"shunt {shunt_id} does not exist")
    if not bool(net.shunt.at[shunt_id, "in_service"]):
        return _not_applicable(action, f"shunt {shunt_id} is structurally unavailable")

    current = _integer_or_none(net.shunt.at[shunt_id, "step"])
    max_step = _integer_or_none(net.shunt.at[shunt_id, "max_step"])
    if current not in {0, 1} or max_step != 1:
        return _not_applicable(action, f"shunt {shunt_id} is not an available single-step device")
    if action.new_step == current:
        return _not_applicable(action, f"shunt {shunt_id} already at step {current} - no-op")
    if action.new_step != 1 - current:
        return _not_applicable(action, f"shunt {shunt_id} target must toggle step {current} to {1 - current}")
    return _applicable(action)


def _check_tap_adjustment(net: Any, action: TapAdjustmentAction) -> ApplicabilityResult:
    trafo_id = action.trafo_id
    if trafo_id not in net.trafo.index:
        return _not_applicable(action, f"trafo {trafo_id} does not exist")
    if not bool(net.trafo.at[trafo_id, "in_service"]):
        return _not_applicable(action, f"trafo {trafo_id} is out of service")

    current = _integer_or_none(net.trafo.at[trafo_id, "tap_pos"])
    tap_min = _integer_or_none(net.trafo.at[trafo_id, "tap_min"])
    tap_max = _integer_or_none(net.trafo.at[trafo_id, "tap_max"])
    if current is None or tap_min is None or tap_max is None:
        return _not_applicable(action, f"trafo {trafo_id} is not tappable")
    if not tap_min <= action.new_tap_pos <= tap_max:
        return _not_applicable(
            action,
            f"tap_pos {action.new_tap_pos} outside [{tap_min}, {tap_max}] for trafo {trafo_id}",
        )
    if action.new_tap_pos == current:
        return _not_applicable(action, f"trafo {trafo_id} already at tap position {current} - no-op")
    if abs(action.new_tap_pos - current) != 1:
        return _not_applicable(action, f"trafo {trafo_id} target must be one atomic tap step from {current}")
    return _applicable(action)


def _generator_targets(current: float) -> tuple[float, ...]:
    if not math.isfinite(current) or current < VM_MIN - 1e-10 or current > VM_MAX + 1e-10:
        return ()
    targets = {
        round(max(VM_MIN, current - VM_STEP), 10),
        round(min(VM_MAX, current + VM_STEP), 10),
    }
    return tuple(sorted(target for target in targets if not math.isclose(target, current, abs_tol=1e-10)))


def _matches_any(value: float, expected: tuple[float, ...]) -> bool:
    return any(math.isclose(value, candidate, rel_tol=0.0, abs_tol=1e-10) for candidate in expected)


def _integer_or_none(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    numeric = float(value)
    if not numeric.is_integer():
        return None
    return int(numeric)


def _in_service(row: Any) -> bool:
    return bool(row.get("in_service", True))


def _applicable(action: Action) -> ApplicabilityResult:
    return ApplicabilityResult(action=action, applicable=True, reason=None)


def _not_applicable(action: Action, reason: str) -> ApplicabilityResult:
    return ApplicabilityResult(action=action, applicable=False, reason=reason)
