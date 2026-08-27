# ABOUTME: Defines typed contracts for shared trajectory, scheduling, probing, and feasibility primitives.
# ABOUTME: Keeps PandaPower states opaque while making policy, status, and accounting data explicit.
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

ACTIVE_BALANCE_POLICY_VERSION = "base-anchored-bounded-participation-v1"


class CurationLoadWeight(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    load_id: StrictInt
    weight: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)


class GeneratorParticipation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    gen_id: StrictInt
    factor: float = Field(gt=0.0, allow_inf_nan=False)


class ActiveBalancePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    participation: tuple[GeneratorParticipation, ...]
    policy_version: str = ACTIVE_BALANCE_POLICY_VERSION

    @model_validator(mode="after")
    def participation_is_stably_ordered(self) -> "ActiveBalancePolicy":
        ids = [item.gen_id for item in self.participation]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            raise ValueError("participation must contain unique generator IDs in ascending order")
        return self


class CurationTrajectoryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    stress: float = Field(ge=0.0, allow_inf_nan=False)
    ordered_load_weights: tuple[CurationLoadWeight, ...]


class DiagnosticTrajectoryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lambda_value: float = Field(gt=0.0, allow_inf_nan=False)


class GeneratorDispatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    gen_id: int
    reference_p_mw: float
    scheduled_p_mw: float
    delta_p_mw: float
    participation_factor: float
    at_active_bound: bool


class ActiveBalanceResult(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    net: Any
    status: Literal["SCHEDULED", "ACTIVE_HEADROOM_EXHAUSTED"]
    requested_delta_mw: float
    allocated_delta_mw: float
    unallocated_delta_mw: float
    generator_dispatch: tuple[GeneratorDispatch, ...]
    policy_version: str


class TrajectoryState(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    net: Any
    trajectory_type: Literal["CURATION", "DIAGNOSTIC"]
    coordinate: float
    requested_load_delta_mw: float
    active_balance: ActiveBalanceResult
    trajectory_policy_version: str


class SolverAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    attempt_number: int = Field(ge=1, le=2)
    tolerance_mva: float = Field(gt=0.0)
    status: Literal["SOLVED", "NO_SOLUTION"]
    iterations: int = Field(ge=0)
    elapsed_ms: float = Field(ge=0.0)
    error_message: str | None = None


class LockedProbeResult(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    status: Literal["SOLVED", "NO_SOLUTION"]
    solved_net: Any | None
    attempts: tuple[SolverAttempt, ...]
    logical_probe_count: Literal[1] = 1
    solver_attempt_count: int = Field(ge=1, le=2)
    tolerance_used_mva: float
    recovery_used: bool
    elapsed_ms: float = Field(ge=0.0)
    error_message: str | None = None
    policy_version: str

    @model_validator(mode="after")
    def solved_state_matches_status(self) -> "LockedProbeResult":
        if (self.status == "SOLVED") != (self.solved_net is not None):
            raise ValueError("solved_net must be present exactly when status is SOLVED")
        if self.solver_attempt_count != len(self.attempts):
            raise ValueError("solver_attempt_count must equal the number of attempts")
        return self


class FeasibilityFailureReason(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: Literal[
        "GEN_P_LIMIT",
        "GEN_Q_LIMIT",
        "EXT_GRID_P_LIMIT",
        "EXT_GRID_Q_LIMIT",
        "DISCONNECTED",
        "LOAD_UNENERGIZED",
        "HARD_VOLTAGE_ENVELOPE",
    ]
    element_id: int | None = None
    detail: str


class VoltageEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    min_vm_pu: float
    max_vm_pu: float
    low_bus_ids: tuple[int, ...]
    high_bus_ids: tuple[int, ...]
    hard_envelope_ok: bool
    runtime_quality_ok: bool


class GeneratorQState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    gen_id: int
    status: Literal[
        "PV_CONTROLLABLE",
        "Q_LIMITED_UPPER",
        "Q_LIMITED_LOWER",
        "UNKNOWN",
    ]
    q_mvar: float | None
    min_q_mvar: float
    max_q_mvar: float
    lower_headroom_mvar: float | None
    upper_headroom_mvar: float | None
    near_limit: bool
    policy_version: str


class SlackResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ext_grid_id: int
    p_mw: float
    q_mvar: float
    p_within_limits: bool
    q_within_limits: bool


class SolvedFeasibility(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    feasible: bool
    generator_p_within_limits: bool
    generator_q_within_limits: bool
    external_grid_within_limits: bool
    connected: bool
    loads_energized: bool
    voltage: VoltageEnvelope
    generator_q_status: tuple[GeneratorQState, ...]
    slack_results: tuple[SlackResult, ...]
    q_limited_gen_ids: tuple[int, ...]
    failure_reasons: tuple[FeasibilityFailureReason, ...]
    policy_version: str


class QLimitEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    reference_name: str
    generator_q_status: tuple[GeneratorQState, ...]
    q_limited_gen_ids: tuple[int, ...]
    newly_q_limited_gen_ids: tuple[int, ...]
    policy_version: str


class BoundaryFeasibilityPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    stop_on_solved_infeasibility: bool = True
    policy_version: str = "scan-first-boundary-feasibility-v1"


class BoundaryProbeRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    coordinate: float
    logical_result: Literal["SOLVED", "NO_SOLUTION", "NOT_RUN"]
    probe_status: Literal["SOLVED", "NO_SOLUTION", "INFEASIBLE"]
    feasible: bool | None
    infeasibility_codes: tuple[str, ...] = ()
    logical_probe_count: int = Field(ge=0, le=1)
    solver_attempt_count: int = Field(ge=0, le=2)
    phase: Literal["COARSE", "REFINEMENT"]


class BoundaryMeasurement(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal[
        "BOUNDARY_FOUND",
        "OBSERVED_NON_MONOTONIC",
        "NO_TRANSITION",
        "ACTIVE_HEADROOM_EXHAUSTED",
        "SLACK_INFEASIBLE",
        "SOLVED_STATE_INFEASIBLE",
    ]
    highest_solved: float | None
    lowest_unsolved: float | None
    records: tuple[BoundaryProbeRecord, ...]
    logical_probe_count: int = Field(ge=0)
    solver_attempt_count: int = Field(ge=0)
    policy_version: str


class ElectricalDistancePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    common_mva_base: float = Field(gt=0.0, allow_inf_nan=False)
    minimum_edge_weight_pu: float = Field(gt=0.0, allow_inf_nan=False)
    policy_version: str = "impedance-weighted-graph-distance-v1"


class ElectricalDistanceResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    method: Literal["IMPEDANCE_WEIGHTED_GRAPH_DISTANCE"]
    source_bus_ids: tuple[int, ...]
    distances_pu: dict[int, float | None]
    unreachable_bus_ids: tuple[int, ...]
    policy_version: str


class StateFingerprint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    value: str
    algorithm: Literal["sha256"] = "sha256"
    policy_version: str
