# ABOUTME: Verifies solved-state feasibility, explicit slack checks, and directional Q-limit evidence.
# ABOUTME: Prevents failed or unconstrained result tables from being presented as physical evidence.
from __future__ import annotations

from typing import Any

import pandapower as pp
import pytest

from restorebench.physics.feasibility import (
    compare_q_limit_evidence,
    evaluate_solved_feasibility,
)
from restorebench.physics.solver import solve_locked_probe


def _solved_net() -> Any:
    net = pp.create_empty_network()
    slack = pp.create_bus(net, vn_kv=110.0)
    load_bus = pp.create_bus(net, vn_kv=110.0)
    pp.create_ext_grid(
        net,
        bus=slack,
        vm_pu=1.0,
        min_p_mw=-100.0,
        max_p_mw=100.0,
        min_q_mvar=-100.0,
        max_q_mvar=100.0,
    )
    pp.create_line_from_parameters(
        net,
        from_bus=slack,
        to_bus=load_bus,
        length_km=1.0,
        r_ohm_per_km=0.1,
        x_ohm_per_km=0.2,
        c_nf_per_km=0.0,
        max_i_ka=1.0,
    )
    pp.create_gen(
        net,
        bus=load_bus,
        p_mw=5.0,
        vm_pu=1.0,
        min_p_mw=0.0,
        max_p_mw=20.0,
        min_q_mvar=-10.0,
        max_q_mvar=10.0,
        index=3,
    )
    pp.create_gen(
        net,
        bus=load_bus,
        p_mw=3.0,
        vm_pu=1.0,
        min_p_mw=0.0,
        max_p_mw=20.0,
        min_q_mvar=-8.0,
        max_q_mvar=8.0,
        index=7,
    )
    pp.create_load(net, bus=load_bus, p_mw=10.0, q_mvar=4.0)
    result = solve_locked_probe(net)
    assert result.status == "SOLVED"
    return result.solved_net


def test_feasibility_checks_external_grid_q_even_when_generators_are_constrained() -> None:
    solved = _solved_net()
    solved.ext_grid.at[0, "max_q_mvar"] = 1.0
    solved.res_ext_grid.at[0, "q_mvar"] = 2.0

    result = evaluate_solved_feasibility(solved)

    assert result.feasible is False
    assert result.external_grid_within_limits is False
    assert any(reason.code == "EXT_GRID_Q_LIMIT" for reason in result.failure_reasons)


def test_feasibility_separates_hard_envelope_from_runtime_quality() -> None:
    solved = _solved_net()
    solved.res_bus.loc[:, "vm_pu"] = 1.0
    solved.res_bus.at[1, "vm_pu"] = 0.93

    result = evaluate_solved_feasibility(solved)

    assert result.voltage.hard_envelope_ok is True
    assert result.voltage.runtime_quality_ok is False
    assert result.feasible is True

    solved.res_bus.at[1, "vm_pu"] = 0.89
    outside_hard = evaluate_solved_feasibility(solved)
    assert outside_hard.voltage.hard_envelope_ok is False
    assert outside_hard.feasible is False
    assert any(reason.code == "HARD_VOLTAGE_ENVELOPE" for reason in outside_hard.failure_reasons)


def test_q_status_is_directional_and_newly_limited_uses_a_named_valid_reference() -> None:
    reference = _solved_net()
    current = _solved_net()
    reference.res_gen.at[3, "q_mvar"] = 0.0
    reference.res_gen.at[7, "q_mvar"] = float(reference.gen.at[7, "min_q_mvar"])
    current.res_gen.at[3, "q_mvar"] = float(current.gen.at[3, "max_q_mvar"])
    current.res_gen.at[7, "q_mvar"] = float(current.gen.at[7, "min_q_mvar"])

    evidence = compare_q_limit_evidence(
        current,
        reference_net=reference,
        reference_name="unstressed-profile",
    )

    by_id = {item.gen_id: item for item in evidence.generator_q_status}
    assert by_id[3].status == "Q_LIMITED_UPPER"
    assert by_id[3].upper_headroom_mvar == pytest.approx(0.0)
    assert by_id[7].status == "Q_LIMITED_LOWER"
    assert by_id[7].lower_headroom_mvar == pytest.approx(0.0)
    assert evidence.q_limited_gen_ids == (3, 7)
    assert evidence.newly_q_limited_gen_ids == (3,)
    assert evidence.reference_name == "unstressed-profile"


def test_q_status_refuses_failed_result_tables() -> None:
    failed = _solved_net()
    failed.converged = False

    with pytest.raises(ValueError, match="valid convergent"):
        evaluate_solved_feasibility(failed)
    with pytest.raises(ValueError, match="valid convergent"):
        compare_q_limit_evidence(
            failed,
            reference_net=_solved_net(),
            reference_name="valid-reference",
        )
