# ABOUTME: Defines canonical power-flow result and diagnostic schemas.
# ABOUTME: These are pure transport contracts with no solver or pandapower dependency.
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from restorebench.schemas.physics import GeneratorQState, SlackResult, SolvedFeasibility


class NRDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    iterations_attempted: int
    worst_bus: int
    lowest_vm_pu: float
    lowest_vm_bus: int
    gens_at_q_limit: list[int]
    max_mismatch_mw: float | None = None
    max_mismatch_mvar: float | None = None
    overstress: float | None = None
    error_message: str
    diagnostics_source: Literal["local_nose", "ppc_last_iterate"]


class QualitySymptom(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["V_LOW", "V_HIGH"]
    element_id: int
    value: float
    threshold: float


class OperatingWarning(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["GEN_Q_NEAR_LIMIT", "GEN_Q_LIMIT"]
    gen_id: int
    q_mvar: float
    nearest_bound: float
    voltage_control_status: Literal[
        "PV_CONTROLLABLE",
        "Q_LIMITED_UPPER",
        "Q_LIMITED_LOWER",
        "UNKNOWN",
    ]


class QualityResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clean: bool
    n_buses_out_of_band: int
    worst_vm_pu: float
    worst_vm_bus: int
    symptoms: list[QualitySymptom]


class PowerFlowResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    converged: bool
    iterations: int
    tolerance_used: float
    runtime_ms: float
    error_message: str | None = None
    diagnostics: NRDiagnostics | None = None
    quality: QualityResult | None = None
    warnings: list[OperatingWarning] = Field(default_factory=list)
    solver_attempt_count: int = 1
    recovery_used: bool = False
    feasibility: SolvedFeasibility | None = None
    generator_q_status: list[GeneratorQState] = Field(default_factory=list)
    slack: SlackResult | None = None


class ConvergenceResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    converged: bool
    iterations: int
    tolerance_used: float
    diagnostics: NRDiagnostics | None = None
    solver_attempt_count: int = 1
    recovery_used: bool = False
