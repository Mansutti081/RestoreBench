# ABOUTME: Defines the locked action and maneuver data contracts.
# ABOUTME: Keeps the action space limited to the three atomic Q-V variants in the schema spec.
import json
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field


DiagnosedCause = Literal[
    "REACTIVE_DEFICIT",
    "CORRIDOR_OVERSTRESS",
    "BAD_SETPOINTS",
    "QLIM_INSTABILITY",
    "WEAK_SLACK",
]


class GenVoltageSetpointAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["GEN_V_SETPOINT"]
    gen_id: int
    new_vm_pu: Annotated[float, Field(ge=0.95, le=1.05)]


class ShuntStepAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["SHUNT_STEP"]
    shunt_id: int
    new_step: Annotated[int, Field(strict=True, ge=0, le=1)]


class TapAdjustmentAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["TAP_ADJUSTMENT"]
    trafo_id: int
    new_tap_pos: Annotated[int, Field(ge=-2, le=2)]


def _decode_stringified_action(value: Any) -> Any:
    """Decode an action the model handed back as a JSON string instead of an object.

    Bedrock models serialize a `oneOf` object property as a JSON string: `propose_maneuver`
    comes back with `"action": "{\\"type\\": \\"GEN_V_SETPOINT\\", ...}"`. The decision inside
    is well-formed and in-vocabulary, so rejecting it would score a transport quirk as a model
    failure and would penalize exactly the tool-using configurations. Decoding is the only step
    taken here: the discriminated union and every field bound still validate the result.
    """
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value  # not JSON at all; let the union report the real error


Action = Annotated[
    Union[
        GenVoltageSetpointAction,
        ShuntStepAction,
        TapAdjustmentAction,
    ],
    Field(discriminator="type"),
    BeforeValidator(_decode_stringified_action),
]


class Maneuver(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Action
    diagnosed_cause: DiagnosedCause | None
    rationale: str


class ManeuverSequence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # The orchestrator enforces MANEUVER_BUDGET; the schema intentionally has no length cap.
    maneuvers: list[Maneuver]
    reconstruction_summary: str | None
