# ABOUTME: Defines topology summaries and action applicability contracts.
# ABOUTME: Keeps grid structure serializable without pandapower/DataFrame fields.
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from restorebench.schemas.actions import Action


class BusInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bus_id: int
    name: str | None
    vn_kv: float
    in_service: bool


class LineInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    line_id: int
    from_bus: int
    to_bus: int
    in_service: bool


class TrafoInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trafo_id: int
    hv_bus: int
    lv_bus: int
    tap_pos: int | None
    tap_min: int | None
    tap_max: int | None
    in_service: bool


class GenInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gen_id: int
    bus: int
    p_mw: float
    vm_pu: float
    min_p_mw: float
    max_p_mw: float
    min_q_mvar: float
    max_q_mvar: float
    dispatchable: bool
    voltage_control_status: (
        Literal[
            "PV_CONTROLLABLE",
            "Q_LIMITED_UPPER",
            "Q_LIMITED_LOWER",
            "UNKNOWN",
        ]
        | None
    )
    in_service: bool

    @model_validator(mode="after")
    def dispatchable_matches_bounds(self) -> "GenInfo":
        expected = self.min_p_mw < self.max_p_mw
        if self.dispatchable != expected:
            raise ValueError("dispatchable must equal (min_p_mw < max_p_mw)")
        return self


class ExtGridInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ext_grid_id: int
    bus: int
    min_p_mw: float
    max_p_mw: float
    min_q_mvar: float
    max_q_mvar: float
    in_service: bool


class LoadInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    load_id: int
    bus: int
    p_mw: float
    q_mvar: float
    in_service: bool


class ShuntInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shunt_id: int
    bus: int
    q_mvar: float
    type: Literal["capacitor", "reactor"]
    step: int
    max_step: int
    in_service: bool

    @model_validator(mode="after")
    def type_matches_sign(self) -> "ShuntInfo":
        expected = "capacitor" if self.q_mvar < 0 else "reactor"
        if self.type != expected:
            raise ValueError("type must equal ('capacitor' if q_mvar < 0 else 'reactor')")
        return self


class TopologySummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_buses: int
    n_lines: int
    n_trafos: int
    n_gens: int
    n_ext_grids: int
    n_loads: int
    n_shunts: int
    buses: list[BusInfo]
    lines: list[LineInfo]
    trafos: list[TrafoInfo]
    gens: list[GenInfo]
    ext_grids: list[ExtGridInfo]
    loads: list[LoadInfo]
    shunts: list[ShuntInfo]
    slack_bus: int


class ApplicabilityResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Action
    applicable: bool
    reason: str | None
