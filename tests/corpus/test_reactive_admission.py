# ABOUTME: Verifies target failure, alternative-init exclusion, and reactive-deficit evidence gates.
# ABOUTME: Uses only valid solved states for Q, voltage, and external-grid evidence.
from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

import pandapower as pp
import pytest

from restorebench.physics.solver import solve_locked_probe
from restorebench.physics.fingerprint import state_fingerprint
from restorebench.schemas.dataset import PocketRecipe
from restorebench.corpus import reactive_admission
from restorebench.corpus.reactive_admission import (
    ReactiveDeficitThresholds,
    admit_target_candidate,
    audit_alternative_initialization,
    build_qv_evidence,
    evaluate_q_unlimited_counterfactual,
)


def _net() -> Any:
    net = pp.create_empty_network(sn_mva=100.0)
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
    pp.create_load(net, bus=load_bus, p_mw=10.0, q_mvar=4.0, index=4)
    return net


def _pocket() -> PocketRecipe:
    return PocketRecipe.model_validate(
        {
            "anchor_bus": 1,
            "distance_method": "IMPEDANCE_WEIGHTED_GRAPH_DISTANCE",
            "loads": [
                {
                    "load_id": 4,
                    "base_p_mw": 10.0,
                    "base_q_mvar": 4.0,
                    "weight": 1.0,
                }
            ],
            "vector_hash": "a" * 64,
            "policy_version": "pocket-v1",
        }
    )


def _thresholds() -> ReactiveDeficitThresholds:
    return ReactiveDeficitThresholds(
        minimum_voltage_deterioration_pu=0.01,
        weak_bus_band_pu=0.02,
        minimum_q_headroom_reduction_mvar=1.0,
        material_q_violation_floor_mvar=5.0,
        material_q_violation_fraction=0.05,
        maximum_weak_region_distance_pu=0.1,
    )


def test_alternative_initialization_audit_records_primary_success() -> None:
    audit = audit_alternative_initialization(_net(), init_policy="flat")

    assert audit.primary_status == "SOLVED"
    assert audit.recovery_status == "NOT_RUN"
    assert audit.converged_without_action is True
    assert audit.solver_attempt_count == 1


def test_q_unlimited_counterfactual_requires_material_q_violation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _net()

    def fake_unlimited(net: Any) -> Any:
        solved = copy.deepcopy(net)
        solved.converged = True
        solved.res_gen = solved.gen[["p_mw"]].copy()
        solved.res_gen["q_mvar"] = 20.0
        solved.res_ext_grid = solved.ext_grid[["bus"]].copy()
        solved.res_ext_grid["p_mw"] = 5.0
        solved.res_ext_grid["q_mvar"] = 2.0
        solved.res_bus = solved.bus[["vn_kv"]].copy()
        solved.res_bus["vm_pu"] = 1.0
        return solved

    monkeypatch.setattr(
        reactive_admission,
        "_solve_q_unlimited",
        fake_unlimited,
    )

    result = evaluate_q_unlimited_counterfactual(source, thresholds=_thresholds())

    assert result.converged is True
    assert result.material_violation_gen_ids == (3,)
    assert result.max_violation_mvar == pytest.approx(10.0)
    assert result.ext_grid_feasible is True

    monkeypatch.setattr(
        reactive_admission,
        "_solve_q_unlimited",
        lambda net: copy.deepcopy(fake_unlimited(net)),
    )
    source.gen.at[3, "max_q_mvar"] = 19.0
    with pytest.raises(ValueError, match="material"):
        evaluate_q_unlimited_counterfactual(source, thresholds=_thresholds())


def test_qv_evidence_requires_voltage_and_upper_q_headroom_deterioration() -> None:
    base_probe = solve_locked_probe(_net())
    assert base_probe.status == "SOLVED"
    base = base_probe.solved_net
    evidence = copy.deepcopy(base)
    base.res_gen.at[3, "q_mvar"] = 0.0
    evidence.res_gen.at[3, "q_mvar"] = float(evidence.gen.at[3, "max_q_mvar"])
    base.res_bus.loc[:, "vm_pu"] = 1.0
    evidence.res_bus.loc[:, "vm_pu"] = 1.0
    evidence.res_bus.at[1, "vm_pu"] = 0.95

    result = build_qv_evidence(
        base_solution=base,
        evidence_solution=evidence,
        evidence_stress=1.1,
        pocket=_pocket(),
        thresholds=_thresholds(),
    )

    assert result.weak_bus_ids == (1,)
    assert result.q_limited_gen_ids == (3,)
    assert result.newly_q_limited_gen_ids == (3,)
    assert result.q_headroom_reduction_mvar == pytest.approx(10.0)

    insufficient = copy.deepcopy(evidence)
    insufficient.res_bus.loc[:, "vm_pu"] = 0.995
    with pytest.raises(ValueError, match="voltage deterioration"):
        build_qv_evidence(
            base_solution=base,
            evidence_solution=insufficient,
            evidence_stress=1.1,
            pocket=_pocket(),
            thresholds=_thresholds(),
        )


def test_qv_evidence_records_remote_weak_region_without_rejecting_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_probe = solve_locked_probe(_net())
    assert base_probe.status == "SOLVED"
    base = base_probe.solved_net
    evidence = copy.deepcopy(base)
    base.res_gen.at[3, "q_mvar"] = 0.0
    evidence.res_gen.at[3, "q_mvar"] = float(
        evidence.gen.at[3, "max_q_mvar"]
    )
    base.res_bus.loc[:, "vm_pu"] = 1.0
    evidence.res_bus.loc[:, "vm_pu"] = 1.0
    evidence.res_bus.at[1, "vm_pu"] = 0.95
    monkeypatch.setattr(
        reactive_admission,
        "impedance_weighted_graph_distances",
        lambda *_args, **_kwargs: SimpleNamespace(
            distances_pu={0: 0.0, 1: 0.4}
        ),
    )

    result = build_qv_evidence(
        base_solution=base,
        evidence_solution=evidence,
        evidence_stress=1.1,
        pocket=_pocket(),
        thresholds=_thresholds(),
    )

    assert result.weak_region_min_distance_pu == pytest.approx(0.4)
    assert result.weak_region_local is False


def test_target_admission_rejects_any_locked_or_alternative_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = SimpleNamespace(
        profile_id="OP-test",
        state_hash="b" * 64,
        net=_net(),
    )
    family = SimpleNamespace(
        pocket=_pocket(),
        active_policy=SimpleNamespace(),
        base_solution=SimpleNamespace(),
        last_convergent_solution=SimpleNamespace(),
        last_convergent_state=SimpleNamespace(coordinate=1.0),
    )
    target_state = SimpleNamespace(
        net=_net(),
        active_balance=SimpleNamespace(status="SCHEDULED"),
    )
    solved_probe = SimpleNamespace(
        status="SOLVED",
        attempts=(SimpleNamespace(status="SOLVED"),),
        solver_attempt_count=1,
    )
    monkeypatch.setattr(
        reactive_admission,
        "build_target_state",
        lambda *_args, **_kwargs: target_state,
    )
    monkeypatch.setattr(
        reactive_admission,
        "solve_locked_probe",
        lambda _net: solved_probe,
    )

    with pytest.raises(ValueError, match="locked target"):
        admit_target_candidate(
            profile,
            family,
            target_stress=1.1,
            thresholds=_thresholds(),
        )


def test_target_admission_probes_the_public_storage_normalized_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_net = _net()
    raw_net.gen.at[3, "p_mw"] = float.fromhex("0x1.27cd16f7da3c6p+2")
    normalized_net = pp.from_json(pp.to_json(raw_net))
    raw_hash = state_fingerprint(raw_net).value
    normalized_hash = state_fingerprint(normalized_net).value
    assert raw_hash != normalized_hash

    profile = SimpleNamespace(
        profile_id="OP-test",
        state_hash="b" * 64,
        net=_net(),
    )
    family = SimpleNamespace(
        pocket=_pocket(),
        active_policy=SimpleNamespace(),
        base_solution=SimpleNamespace(),
        last_convergent_solution=SimpleNamespace(),
        last_convergent_state=SimpleNamespace(coordinate=1.0),
    )
    target_state = SimpleNamespace(
        net=raw_net,
        active_balance=SimpleNamespace(status="SCHEDULED", net=raw_net),
    )
    monkeypatch.setattr(
        reactive_admission,
        "build_target_state",
        lambda *_args, **_kwargs: copy.deepcopy(target_state),
    )

    def assert_normalized_probe(net: Any) -> Any:
        assert state_fingerprint(net).value == normalized_hash
        raise RuntimeError("normalized target reached the first admission gate")

    monkeypatch.setattr(
        reactive_admission,
        "solve_locked_probe",
        assert_normalized_probe,
    )

    with pytest.raises(RuntimeError, match="normalized target"):
        admit_target_candidate(
            profile,
            family,
            target_stress=1.1,
            thresholds=_thresholds(),
        )
