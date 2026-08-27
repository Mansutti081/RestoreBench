# ABOUTME: Defines trace, citation, and top-level resolution response schemas.
# ABOUTME: Enforces UTC-aware timestamps and status/convergence consistency.
from datetime import datetime, timezone
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from restorebench.schemas.actions import Maneuver
from restorebench.schemas.feedback import FailureFeedback
from restorebench.schemas.power_flow import OperatingWarning, QualityResult


ScenarioId = Annotated[str, Field(pattern=r"^S\d{4}$")]
VersionId = Annotated[str, Field(min_length=1)]
RESULT_SCHEMA_VERSION = "resolution-response-v3"


def _ensure_utc_datetime(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime must be timezone-aware")
    return value.astimezone(timezone.utc)


class TraceEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: datetime
    phase: Literal["baseline", "diagnosis", "maneuver", "solve", "response"]
    event_name: str
    duration_ms: float
    payload: dict[str, Any]

    @field_validator("timestamp")
    @classmethod
    def timestamp_must_be_utc_aware(cls, value: datetime) -> datetime:
        return _ensure_utc_datetime(value)


class ReasoningEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    iteration: int
    role: str
    text: str


class ExecutionTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: UUID
    events: list[TraceEvent]
    n_llm_calls: int
    total_llm_tokens_in: int
    total_llm_tokens_out: int
    # Bedrock's billed total across every call. Not necessarily in + out: cache tokens are
    # counted here too. Reasoning tokens are not separable — they live inside the output count.
    total_llm_tokens: int = 0
    n_tool_calls: int
    n_power_flows: int
    reasoning: list[ReasoningEntry] = Field(default_factory=list)


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    marker_id: str
    solver_field_path: str
    resolved_value: float | str
    formatting: str | None


class ResolutionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: UUID
    scenario_id: ScenarioId
    configuration: Literal[1, 2, 3]
    llm_assignment: dict[str, str]
    # Which of the N repetitions of this scenario this run is. Not a random seed — see
    # OrchestratorConfig.repetition_index. Runs are paired across configurations by
    # scenario_id, never by this index: two repetitions with the same index share nothing.
    repetition_index: Annotated[int, Field(ge=0)]
    dataset_version: VersionId | None = None
    solver_version: VersionId | None = None
    action_policy_version: VersionId | None = None
    ranking_policy_version: VersionId | None = None
    result_schema_version: VersionId | None = None
    status: Literal["SUCCESS", "BUDGET_EXHAUSTED", "TIMEOUT", "LLM_FAILURE", "TOOL_FAILURE"]
    maneuvers: list[Maneuver]
    n_maneuvers: int
    converged: bool
    quality: QualityResult | None
    final_warnings: list[OperatingWarning] = Field(default_factory=list)
    diagnosis_rationale: str | None
    citations: list[Citation] = Field(default_factory=list)
    failure_feedback: list[FailureFeedback]
    trace: ExecutionTrace
    total_runtime_seconds: float
    started_at: datetime
    completed_at: datetime

    @field_validator("started_at", "completed_at")
    @classmethod
    def datetimes_must_be_utc_aware(cls, value: datetime) -> datetime:
        return _ensure_utc_datetime(value)

    @model_validator(mode="after")
    def converged_matches_status(self) -> "ResolutionResponse":
        expected = self.status == "SUCCESS"
        if self.converged != expected:
            raise ValueError('converged must equal (status == "SUCCESS")')
        version_stamp = (
            self.dataset_version,
            self.solver_version,
            self.action_policy_version,
            self.ranking_policy_version,
            self.result_schema_version,
        )
        if any(value is None for value in version_stamp) and any(
            value is not None for value in version_stamp
        ):
            raise ValueError("version stamp must provide all five versions or none")
        return self
