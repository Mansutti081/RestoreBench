# ABOUTME: Defines iteration feedback and opaque sandbox-handle schemas.
# ABOUTME: Keeps grids represented only by UUID handles across schema boundaries.
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from restorebench.schemas.actions import Maneuver
from restorebench.schemas.power_flow import NRDiagnostics


class FailedManeuverAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["PREVIEW_DIVERGED", "PREVIEW_INVALID"]
    maneuver: Maneuver
    diagnostics: NRDiagnostics | None
    detail: str | None


class AcceptedManeuver(BaseModel):
    """A committed maneuver together with what it did to the distance from solvability.

    The agent's conversation is rebuilt every iteration while the grid keeps every accepted
    maneuver, so the maneuver alone tells it what it did but never whether that helped.
    Either overstress is None when the state carried no nose diagnostics, which is not the
    same as no change and must stay distinguishable from it.
    """

    model_config = ConfigDict(extra="forbid")

    maneuver: Maneuver
    overstress_before: float | None
    overstress_after: float | None


class FailureFeedback(BaseModel):
    model_config = ConfigDict(extra="forbid")

    iteration: int
    kind: Literal[
        "STILL_DIVERGED",
        "SOLVED_INFEASIBLE",
        "INVALID_ACTION",
        "MALFORMED_OUTPUT",
        # The provider or our own caps stopped the call: distinct from a model that answered
        # with the wrong shape, and the two must stay separable in the results.
        "LLM_FAILURE",
        "PREVIEW_DIVERGED",
        "PREVIEW_INVALID",
    ]
    diagnostics: NRDiagnostics | None
    detail: str | None
    maneuver: Maneuver | None


class SandboxNet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sandbox_id: UUID
    scenario_request_id: UUID
