# ABOUTME: Verifies topology and applicability schemas for the four-action space.
# ABOUTME: Covers round trips, extra rejection, and dispatchable bound consistency.
import pytest
from pydantic import ValidationError

from restorebench.schemas.topology import (
    ApplicabilityResult,
    BusInfo,
    ExtGridInfo,
    GenInfo,
    LineInfo,
    LoadInfo,
    ShuntInfo,
    TopologySummary,
    TrafoInfo,
)


def round_trip(model):
    assert type(model).model_validate(model.model_dump()) == model
    assert type(model).model_validate_json(model.model_dump_json()) == model


def gen_info(**overrides) -> GenInfo:
    data = {
        "gen_id": 1,
        "bus": 4,
        "p_mw": 50.0,
        "vm_pu": 1.01,
        "min_p_mw": 0.0,
        "max_p_mw": 100.0,
        "min_q_mvar": -40.0,
        "max_q_mvar": 40.0,
        "dispatchable": True,
        "voltage_control_status": "PV_CONTROLLABLE",
        "in_service": True,
    }
    data.update(overrides)
    return GenInfo(**data)


def topology_summary() -> TopologySummary:
    return TopologySummary(
        n_buses=1,
        n_lines=1,
        n_trafos=1,
        n_gens=1,
        n_ext_grids=1,
        n_loads=1,
        n_shunts=1,
        buses=[BusInfo(bus_id=4, name=None, vn_kv=138.0, in_service=True)],
        lines=[LineInfo(line_id=0, from_bus=4, to_bus=5, in_service=True)],
        trafos=[TrafoInfo(trafo_id=0, hv_bus=4, lv_bus=5, tap_pos=0, tap_min=-2, tap_max=2, in_service=True)],
        gens=[gen_info()],
        ext_grids=[
            ExtGridInfo(
                ext_grid_id=0,
                bus=4,
                min_p_mw=-805.0,
                max_p_mw=805.0,
                min_q_mvar=-300.0,
                max_q_mvar=300.0,
                in_service=True,
            )
        ],
        loads=[LoadInfo(load_id=0, bus=4, p_mw=10.0, q_mvar=4.0, in_service=True)],
        shunts=[
            ShuntInfo(
                shunt_id=0,
                bus=4,
                q_mvar=-20.0,
                type="capacitor",
                step=1,
                max_step=1,
                in_service=True,
            )
        ],
        slack_bus=4,
    )


def test_topology_models_round_trip():
    models = [
        BusInfo(bus_id=4, name="bus 4", vn_kv=138.0, in_service=True),
        LineInfo(line_id=0, from_bus=4, to_bus=5, in_service=True),
        TrafoInfo(trafo_id=0, hv_bus=4, lv_bus=5, tap_pos=None, tap_min=None, tap_max=None, in_service=True),
        gen_info(),
        ExtGridInfo(
            ext_grid_id=0,
            bus=4,
            min_p_mw=-805.0,
            max_p_mw=805.0,
            min_q_mvar=-300.0,
            max_q_mvar=300.0,
            in_service=True,
        ),
        LoadInfo(load_id=0, bus=4, p_mw=10.0, q_mvar=4.0, in_service=True),
        ShuntInfo(
            shunt_id=0,
            bus=4,
            q_mvar=40.0,
            type="reactor",
            step=0,
            max_step=1,
            in_service=False,
        ),
        topology_summary(),
        ApplicabilityResult(
            action={"type": "GEN_V_SETPOINT", "gen_id": 1, "new_vm_pu": 1.02},
            applicable=True,
            reason=None,
        ),
    ]

    for model in models:
        round_trip(model)


def test_gen_dispatchable_must_match_active_power_bounds():
    with pytest.raises(ValidationError):
        gen_info(dispatchable=False)

    condenser = gen_info(min_p_mw=0.0, max_p_mw=0.0, dispatchable=False)
    assert condenser.dispatchable is False


def test_topology_models_reject_extra_fields():
    with pytest.raises(ValidationError):
        BusInfo(bus_id=1, name=None, vn_kv=138.0, in_service=True, extra=True)
