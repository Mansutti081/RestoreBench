# ABOUTME: Verifies the locked lightweight AC probe and its one-retry accounting contract.
# ABOUTME: Ensures failed attempts remain isolated and never launch public retreat diagnostics.
from __future__ import annotations

import copy
from typing import Any

import pandapower as pp
import pytest
from pandapower.auxiliary import LoadflowNotConverged

from restorebench.physics import solver
from restorebench.physics.solver import solve_locked_probe
from restorebench.tools.power_flow import run_ac_pf


def _solvable_net() -> Any:
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
    pp.create_load(net, bus=load_bus, p_mw=10.0, q_mvar=3.0)
    return net


def test_locked_probe_primary_success_uses_one_isolated_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    source = {"marker": "caller"}
    seen: list[Any] = []

    def succeed(net: Any, tolerance_mva: float) -> None:
        seen.append(net)
        assert tolerance_mva == solver.PRIMARY_TOLERANCE_MVA
        net["marker"] = "solved"
        net["_ppc"] = {"iterations": 4}
        net["converged"] = True

    monkeypatch.setattr(solver, "_run_locked_pf", succeed)

    result = solve_locked_probe(source)

    assert result.status == "SOLVED"
    assert result.solver_attempt_count == 1
    assert result.recovery_used is False
    assert result.tolerance_used_mva == solver.PRIMARY_TOLERANCE_MVA
    assert result.solved_net["marker"] == "solved"
    assert source == {"marker": "caller"}
    assert seen[0] is not source


def test_locked_probe_recovery_uses_a_fresh_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    source = {"marker": "caller"}
    seen: list[Any] = []

    def recover(net: Any, tolerance_mva: float) -> None:
        seen.append(net)
        assert net["marker"] == "caller"
        if tolerance_mva == solver.PRIMARY_TOLERANCE_MVA:
            net["marker"] = "failed-attempt"
            raise LoadflowNotConverged("primary failed")
        net["marker"] = "recovered"
        net["_ppc"] = {"iterations": 6}
        net["converged"] = True

    monkeypatch.setattr(solver, "_run_locked_pf", recover)

    result = solve_locked_probe(source)

    assert result.status == "SOLVED"
    assert result.solver_attempt_count == 2
    assert result.recovery_used is True
    assert result.tolerance_used_mva == solver.RECOVERY_TOLERANCE_MVA
    assert result.solved_net["marker"] == "recovered"
    assert seen[0] is not seen[1]


def test_locked_probe_two_attempt_failure_has_no_solved_evidence_or_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[float] = []

    def fail(_net: Any, tolerance_mva: float) -> None:
        calls.append(tolerance_mva)
        raise LoadflowNotConverged(f"failed at {tolerance_mva}")

    monkeypatch.setattr(solver, "_run_locked_pf", fail)

    result = solve_locked_probe({"marker": "caller"})

    assert result.status == "NO_SOLUTION"
    assert result.solved_net is None
    assert result.solver_attempt_count == 2
    assert result.recovery_used is True
    assert calls == [solver.PRIMARY_TOLERANCE_MVA, solver.RECOVERY_TOLERANCE_MVA]
    assert "diagnostics" not in type(result).model_fields


def test_locked_probe_verdict_matches_public_run_ac_pf_and_preserves_input() -> None:
    source = _solvable_net()
    before = copy.deepcopy(source)

    probe = solve_locked_probe(source)
    public = run_ac_pf(source)

    assert (probe.status == "SOLVED") is public.converged
    assert probe.solver_attempt_count == public.solver_attempt_count
    assert source.load.equals(before.load)
    assert source.ext_grid.equals(before.ext_grid)
