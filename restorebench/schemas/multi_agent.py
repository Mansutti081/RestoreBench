# ABOUTME: Defines multi-agent handoff schemas for analyst and executor roles.
# ABOUTME: The contracts carry structured maneuvers and tool results only.

from pydantic import BaseModel, ConfigDict

from restorebench.schemas.actions import DiagnosedCause, Maneuver
from restorebench.schemas.power_flow import PowerFlowResult
from restorebench.schemas.topology import ApplicabilityResult


class AnalystAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    diagnosed_cause: DiagnosedCause
    proposed_maneuver: Maneuver
    rationale: str


class ExecutorReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    maneuver: Maneuver
    applicability: ApplicabilityResult
    pf_result: PowerFlowResult
