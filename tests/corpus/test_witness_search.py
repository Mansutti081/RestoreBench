# ABOUTME: Verifies deterministic ranker-independent Q-V witness search and descriptors.
# ABOUTME: Covers exhaustive direct checks, exact shallow BFS, state deduplication, and budget limits.
from __future__ import annotations

import builtins
import runpy
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any

import pytest

from restorebench.schemas.actions import GenVoltageSetpointAction
from restorebench.schemas.physics import (
    FeasibilityFailureReason,
    SolvedFeasibility,
    StateFingerprint,
    VoltageEnvelope,
)
from restorebench.schemas.power_flow import PowerFlowResult
from restorebench.corpus import witness_search
from restorebench.corpus.witness_search import (
    StateEvaluation,
    WitnessSearchPolicy,
    search_curation_witness,
)


def _action(gen_id: int) -> GenVoltageSetpointAction:
    return GenVoltageSetpointAction(
        type="GEN_V_SETPOINT",
        gen_id=gen_id,
        new_vm_pu=1.01,
    )


def test_witness_module_import_fails_on_any_ranker_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def reject_ranker(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        lowered = name.lower()
        if "ranker" in lowered or "control_influence" in lowered:
            raise AssertionError(f"witness search attempted forbidden dependency {name}")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", reject_ranker)

    runpy.run_path(witness_search.__file__, run_name="__witness_ranker_guard__")


def _terminal_pf() -> PowerFlowResult:
    return PowerFlowResult(
        converged=True,
        iterations=4,
        tolerance_used=1e-8,
        runtime_ms=1.0,
    )


def _outside_envelope_feasibility(
    **updates: bool,
) -> SolvedFeasibility:
    payload: dict[str, Any] = {
        "feasible": False,
        "generator_p_within_limits": True,
        "generator_q_within_limits": True,
        "external_grid_within_limits": True,
        "connected": True,
        "loads_energized": True,
        "voltage": VoltageEnvelope(
            min_vm_pu=0.79,
            max_vm_pu=1.04,
            low_bus_ids=(44,),
            high_bus_ids=(),
            hard_envelope_ok=False,
            runtime_quality_ok=False,
        ),
        "generator_q_status": (),
        "slack_results": (),
        "q_limited_gen_ids": (),
        "failure_reasons": (
            FeasibilityFailureReason(
                code="HARD_VOLTAGE_ENVELOPE",
                detail="outside the hard envelope",
            ),
        ),
        "policy_version": "test-feasibility",
    }
    payload.update(updates)
    return SolvedFeasibility.model_validate(payload)


def _install_solved_probe(
    monkeypatch: pytest.MonkeyPatch,
    feasibility: SolvedFeasibility,
) -> None:
    monkeypatch.setattr(
        witness_search,
        "solve_locked_probe",
        lambda _net: SimpleNamespace(
            status="SOLVED",
            solved_net=object(),
            attempts=(SimpleNamespace(iterations=4),),
            tolerance_used_mva=1e-8,
            elapsed_ms=1.0,
            solver_attempt_count=1,
            recovery_used=False,
        ),
    )
    monkeypatch.setattr(
        witness_search,
        "evaluate_solved_feasibility",
        lambda _net: feasibility,
    )


def test_convergent_witness_terminal_accepts_voltage_outside_hard_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_solved_probe(
        monkeypatch,
        _outside_envelope_feasibility(),
    )

    evaluation = witness_search._evaluate_state(
        object(),
        WitnessSearchPolicy(),
    )

    assert evaluation.status == "TERMINAL"
    assert evaluation.terminal_pf is not None
    assert evaluation.terminal_pf.feasibility is not None
    assert evaluation.terminal_pf.feasibility.voltage.min_vm_pu == 0.79
    assert evaluation.terminal_pf.feasibility.voltage.max_vm_pu == 1.04
    assert evaluation.terminal_pf.feasibility.voltage.hard_envelope_ok is False


@pytest.mark.parametrize(
    "invalid_field",
    [
        "generator_p_within_limits",
        "generator_q_within_limits",
        "external_grid_within_limits",
        "connected",
        "loads_energized",
    ],
)
def test_convergent_witness_terminal_keeps_non_voltage_gates_hard(
    monkeypatch: pytest.MonkeyPatch,
    invalid_field: str,
) -> None:
    _install_solved_probe(
        monkeypatch,
        _outside_envelope_feasibility(**{invalid_field: False}),
    )

    evaluation = witness_search._evaluate_state(
        object(),
        WitnessSearchPolicy(),
    )

    assert evaluation.status == "INFEASIBLE"
    assert evaluation.terminal_pf is None


def _install_graph(
    monkeypatch: pytest.MonkeyPatch,
    *,
    transitions: dict[int, list[tuple[int, int]]],
    terminal_states: set[int],
) -> list[int]:
    evaluated: list[int] = []

    def enumerate_actions(net: int, _context: Any = None):
        return [_action(action_id) for action_id, _target in transitions.get(net, [])]

    def apply_action(net: int, action: Any, _context: Any = None):
        return next(target for action_id, target in transitions.get(net, []) if action_id == action.gen_id)

    def evaluate(net: int, _policy: Any) -> StateEvaluation:
        evaluated.append(net)
        terminal = net in terminal_states
        return StateEvaluation(
            status="TERMINAL" if terminal else "DIVERGED",
            q_context={},
            terminal_pf=_terminal_pf() if terminal else None,
            score=float(net),
            logical_probe_count=1,
            solver_attempt_count=1,
        )

    monkeypatch.setattr(
        witness_search,
        "enumerate_legal_qv_actions",
        enumerate_actions,
    )
    monkeypatch.setattr(witness_search, "apply_qv_action", apply_action)
    monkeypatch.setattr(
        witness_search,
        "state_fingerprint",
        lambda net: StateFingerprint(
            value=f"{net:064x}",
            policy_version="test",
        ),
    )
    monkeypatch.setattr(witness_search, "_evaluate_state", evaluate)
    return evaluated


def test_direct_search_evaluates_every_first_action_and_marks_exact_minimum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluated = _install_graph(
        monkeypatch,
        transitions={0: [(2, 1), (3, 2)]},
        terminal_states={1},
    )

    result = search_curation_witness(
        0,
        scenario_id="S0001",
        policy=WitnessSearchPolicy(
            maneuver_budget=10,
            bfs_max_depth=3,
            frontier_budget=100,
            beam_width=10,
        ),
    )

    assert evaluated == [0, 1, 2]
    assert result.resolution_regime == "DIRECT"
    assert result.direct_restorer_available is True
    assert result.witness_length == 1
    assert result.witness_optimality == "EXACT_MINIMUM"
    assert result.witness.maneuvers == (_action(2),)


def test_shallow_bfs_proves_sequential_minimum_and_deduplicates_states(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluated = _install_graph(
        monkeypatch,
        transitions={
            0: [(2, 1), (3, 1)],
            1: [(4, 2)],
        },
        terminal_states={2},
    )

    result = search_curation_witness(
        0,
        scenario_id="S0002",
        policy=WitnessSearchPolicy(
            maneuver_budget=10,
            bfs_max_depth=3,
            frontier_budget=100,
            beam_width=10,
        ),
    )

    assert evaluated == [0, 1, 2]
    assert result.resolution_regime == "SEQUENTIAL"
    assert result.direct_restorer_available is False
    assert result.witness_length == 2
    assert result.witness_optimality == "EXACT_MINIMUM"
    assert len(result.witness.state_hashes) == 3


def test_search_can_repair_a_converged_infeasible_intermediate_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transitions = {0: [(2, 1)], 1: [(3, 2)]}

    monkeypatch.setattr(
        witness_search,
        "enumerate_legal_qv_actions",
        lambda net, _context=None: [_action(action_id) for action_id, _target in transitions.get(net, [])],
    )
    monkeypatch.setattr(
        witness_search,
        "apply_qv_action",
        lambda net, action, _context=None: next(
            target for action_id, target in transitions[net] if action_id == action.gen_id
        ),
    )
    monkeypatch.setattr(
        witness_search,
        "state_fingerprint",
        lambda net: StateFingerprint(
            value=f"{net:064x}",
            policy_version="test",
        ),
    )

    def evaluate(net: int, _policy: Any) -> StateEvaluation:
        status = {0: "DIVERGED", 1: "INFEASIBLE", 2: "TERMINAL"}[net]
        return StateEvaluation(
            status=status,
            q_context={},
            terminal_pf=_terminal_pf() if status == "TERMINAL" else None,
            score=float(net),
            logical_probe_count=1,
            solver_attempt_count=1,
            diagnostics_complete=True,
        )

    monkeypatch.setattr(witness_search, "_evaluate_state", evaluate)

    result = search_curation_witness(
        0,
        scenario_id="S0003",
        policy=WitnessSearchPolicy(
            maneuver_budget=2,
            bfs_max_depth=2,
            frontier_budget=100,
            beam_width=10,
        ),
    )

    assert result.resolution_regime == "SEQUENTIAL"
    assert result.witness_length == 2
    assert result.witness_optimality == "EXACT_MINIMUM"


def test_search_rejects_target_without_witness_inside_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_graph(
        monkeypatch,
        transitions={0: [(2, 1)], 1: [(3, 2)], 2: [(4, 3)]},
        terminal_states={3},
    )

    with pytest.raises(ValueError, match="within maneuver budget"):
        search_curation_witness(
            0,
            scenario_id="S0003",
            policy=WitnessSearchPolicy(
                maneuver_budget=2,
                bfs_max_depth=2,
                frontier_budget=100,
                beam_width=10,
            ),
        )


def test_direct_search_does_not_diagnose_unpromoted_failed_children(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    diagnostic_calls: list[int] = []

    monkeypatch.setattr(
        witness_search,
        "enumerate_legal_qv_actions",
        lambda net, _context=None: [_action(2), _action(3)] if net == 0 else [],
    )
    monkeypatch.setattr(
        witness_search,
        "apply_qv_action",
        lambda _net, action, _context=None: action.gen_id - 1,
    )
    monkeypatch.setattr(
        witness_search,
        "state_fingerprint",
        lambda net: StateFingerprint(
            value=f"{net:064x}",
            policy_version="test",
        ),
    )

    def evaluate(net: int, _policy: Any) -> StateEvaluation:
        terminal = net == 1
        return StateEvaluation(
            status="TERMINAL" if terminal else "DIVERGED",
            q_context={},
            terminal_pf=_terminal_pf() if terminal else None,
            score=float("-inf"),
            logical_probe_count=1,
            solver_attempt_count=1,
            diagnostics_complete=terminal,
        )

    def diagnose(net: int, _policy: Any):
        diagnostic_calls.append(net)
        return ({}, float(net), 2, 3)

    monkeypatch.setattr(witness_search, "_evaluate_state", evaluate)
    monkeypatch.setattr(witness_search, "_diagnose_q_context", diagnose)

    result = search_curation_witness(
        0,
        scenario_id="S0004",
        policy=WitnessSearchPolicy(
            maneuver_budget=10,
            bfs_max_depth=3,
            frontier_budget=100,
            beam_width=10,
        ),
    )

    assert result.resolution_regime == "DIRECT"
    assert diagnostic_calls == [0]


def test_budget_exhaustion_does_not_diagnose_dead_end_children(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    diagnostic_calls: list[int] = []
    monkeypatch.setattr(
        witness_search,
        "enumerate_legal_qv_actions",
        lambda net, _context=None: [_action(2), _action(3)] if net == 0 else [],
    )
    monkeypatch.setattr(
        witness_search,
        "apply_qv_action",
        lambda _net, action, _context=None: action.gen_id - 1,
    )
    monkeypatch.setattr(
        witness_search,
        "state_fingerprint",
        lambda net: StateFingerprint(
            value=f"{net:064x}",
            policy_version="test",
        ),
    )
    monkeypatch.setattr(
        witness_search,
        "_evaluate_state",
        lambda _net, _policy: StateEvaluation(
            status="DIVERGED",
            q_context={},
            terminal_pf=None,
            score=float("-inf"),
            logical_probe_count=1,
            solver_attempt_count=1,
            diagnostics_complete=False,
        ),
    )

    def diagnose(net: int, _policy: Any):
        diagnostic_calls.append(net)
        return ({}, float(net), 2, 3)

    monkeypatch.setattr(witness_search, "_diagnose_q_context", diagnose)

    with pytest.raises(ValueError, match="within maneuver budget"):
        search_curation_witness(
            0,
            scenario_id="S0005",
            policy=WitnessSearchPolicy(
                maneuver_budget=1,
                bfs_max_depth=1,
                frontier_budget=100,
                beam_width=10,
            ),
        )

    assert diagnostic_calls == [0]


def test_beam_extension_prefilters_deterministically_and_returns_upper_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluated = _install_graph(
        monkeypatch,
        transitions={
            0: [(2, 1), (3, 2), (4, 3)],
            1: [(5, 4)],
            2: [(6, 5)],
            3: [(7, 6)],
        },
        terminal_states={4},
    )

    result = search_curation_witness(
        0,
        scenario_id="S0006",
        policy=WitnessSearchPolicy(
            maneuver_budget=3,
            bfs_max_depth=1,
            frontier_budget=1,
            beam_width=1,
        ),
    )

    assert evaluated == [0, 1, 2, 3, 4]
    assert result.resolution_regime == "SEQUENTIAL"
    assert result.witness_optimality == "UPPER_BOUND"
    assert result.witness_length == 2


def test_beam_node_evaluates_prefiltered_actions_as_one_parallel_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arrivals = 0
    all_arrived = threading.Event()
    arrivals_lock = threading.Lock()
    transitions = {
        0: [(2, 1)],
        1: [(3, 2), (4, 3), (5, 4)],
    }

    monkeypatch.setattr(
        witness_search,
        "enumerate_legal_qv_actions",
        lambda net, _context=None: [_action(action_id) for action_id, _target in transitions.get(net, [])],
    )
    monkeypatch.setattr(
        witness_search,
        "apply_qv_action",
        lambda net, action, _context=None: next(
            target for action_id, target in transitions.get(net, []) if action_id == action.gen_id
        ),
    )
    monkeypatch.setattr(
        witness_search,
        "state_fingerprint",
        lambda net: StateFingerprint(
            value=f"{net:064x}",
            policy_version="test",
        ),
    )

    def evaluate(net: int, _policy: Any) -> StateEvaluation:
        nonlocal arrivals
        if net in {2, 3, 4}:
            with arrivals_lock:
                arrivals += 1
                if arrivals == 3:
                    all_arrived.set()
            assert all_arrived.wait(timeout=1.0)
        terminal = net == 2
        return StateEvaluation(
            status="TERMINAL" if terminal else "DIVERGED",
            q_context={},
            terminal_pf=_terminal_pf() if terminal else None,
            score=float(net),
            logical_probe_count=1,
            solver_attempt_count=1,
            diagnostics_complete=True,
        )

    monkeypatch.setattr(witness_search, "_evaluate_state", evaluate)

    with ThreadPoolExecutor(max_workers=3) as executor:
        result = search_curation_witness(
            0,
            scenario_id="S0097",
            policy=WitnessSearchPolicy(
                maneuver_budget=2,
                bfs_max_depth=1,
                frontier_budget=1,
                beam_width=1,
                beam_branching_width=3,
            ),
            executor=executor,
        )

    assert arrivals == 3
    assert result.witness.maneuvers == (_action(2), _action(3))
    assert result.witness_optimality == "UPPER_BOUND"


def test_parallel_exhaustive_layer_replays_out_of_order_results_deterministically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completion_order: list[int] = []
    monkeypatch.setattr(
        witness_search,
        "enumerate_legal_qv_actions",
        lambda net, _context=None: [_action(2), _action(3)] if net == 0 else [],
    )
    monkeypatch.setattr(
        witness_search,
        "apply_qv_action",
        lambda _net, action, _context=None: action.gen_id - 1,
    )
    monkeypatch.setattr(
        witness_search,
        "state_fingerprint",
        lambda net: StateFingerprint(
            value=f"{net:064x}",
            policy_version="test",
        ),
    )

    def evaluate(net: int, _policy: Any) -> StateEvaluation:
        if net == 1:
            time.sleep(0.05)
        completion_order.append(net)
        terminal = net == 1
        return StateEvaluation(
            status="TERMINAL" if terminal else "DIVERGED",
            q_context={},
            terminal_pf=_terminal_pf() if terminal else None,
            score=float(net),
            logical_probe_count=1,
            solver_attempt_count=1,
            diagnostics_complete=True,
        )

    monkeypatch.setattr(witness_search, "_evaluate_state", evaluate)

    with ThreadPoolExecutor(max_workers=2) as executor:
        result = search_curation_witness(
            0,
            scenario_id="S0098",
            policy=WitnessSearchPolicy(),
            executor=executor,
        )

    assert completion_order == [0, 2, 1]
    assert result.witness.maneuvers == (_action(2),)
    assert result.logical_probe_count == 3
    assert result.solver_attempt_count == 3
    assert result.expanded_state_count == 3


def test_parallel_worker_failure_identifies_the_exact_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        witness_search,
        "enumerate_legal_qv_actions",
        lambda net, _context=None: [_action(2), _action(3)] if net == 0 else [],
    )
    monkeypatch.setattr(
        witness_search,
        "apply_qv_action",
        lambda _net, action, _context=None: action.gen_id - 1,
    )
    monkeypatch.setattr(
        witness_search,
        "state_fingerprint",
        lambda net: StateFingerprint(
            value=f"{net:064x}",
            policy_version="test",
        ),
    )

    def evaluate(net: int, _policy: Any) -> StateEvaluation:
        if net == 1:
            raise RuntimeError("synthetic solver failure")
        return StateEvaluation(
            status="DIVERGED",
            q_context={},
            terminal_pf=None,
            score=float("-inf"),
            logical_probe_count=1,
            solver_attempt_count=1,
            diagnostics_complete=True,
        )

    monkeypatch.setattr(witness_search, "_evaluate_state", evaluate)

    with (
        ThreadPoolExecutor(max_workers=2) as executor,
        pytest.raises(
            RuntimeError,
            match=r"S0099.*depth 1.*GEN_V_SETPOINT.*gen_id.*2",
        ),
    ):
        search_curation_witness(
            0,
            scenario_id="S0099",
            policy=WitnessSearchPolicy(),
            executor=executor,
        )
