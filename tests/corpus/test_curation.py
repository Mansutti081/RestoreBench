# ABOUTME: Verifies dataset families consume shared trajectory, balance, and boundary primitives.
# ABOUTME: Covers deterministic family identity, active policy, and fresh target reconstruction.
from __future__ import annotations

import copy
from types import SimpleNamespace

import pandapower as pp
import pandas as pd
import pytest

from restorebench.schemas.dataset import PocketRecipe
from restorebench.corpus import curation
from restorebench.corpus.curation import (
    CurationScanPolicy,
    build_active_policy,
    build_curation_recipe,
    build_target_state,
    measure_curation_family,
    scenario_family_id,
)
from restorebench.corpus.operating_profiles import OperatingProfileCandidate


def _net():
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
        p_mw=30.0,
        vm_pu=1.0,
        min_p_mw=10.0,
        max_p_mw=80.0,
        min_q_mvar=-40.0,
        max_q_mvar=40.0,
        index=2,
    )
    pp.create_load(net, bus=load_bus, p_mw=20.0, q_mvar=8.0, index=3)
    pp.create_load(net, bus=load_bus, p_mw=10.0, q_mvar=4.0, index=8)
    return net


def _pocket() -> PocketRecipe:
    return PocketRecipe.model_validate(
        {
            "anchor_bus": 1,
            "distance_method": "IMPEDANCE_WEIGHTED_GRAPH_DISTANCE",
            "loads": [
                {
                    "load_id": 3,
                    "base_p_mw": 20.0,
                    "base_q_mvar": 8.0,
                    "weight": 1.0,
                },
                {
                    "load_id": 8,
                    "base_p_mw": 10.0,
                    "base_q_mvar": 4.0,
                    "weight": 0.5,
                },
            ],
            "vector_hash": "a" * 64,
            "policy_version": "pocket-v1",
        }
    )


def _profile() -> OperatingProfileCandidate:
    return OperatingProfileCandidate(
        profile_id="OP-0001",
        state_hash="b" * 64,
        modifications=(),
        net=_net(),
    )


def test_active_policy_and_family_id_are_base_anchored_and_stable() -> None:
    profile = _profile()
    active_policy = build_active_policy(profile.net)

    assert [(item.gen_id, item.factor) for item in active_policy.participation] == [
        (2, 30.0)
    ]
    first = scenario_family_id(
        profile_id=profile.profile_id,
        pocket=_pocket(),
        active_policy=active_policy,
    )
    second = scenario_family_id(
        profile_id=profile.profile_id,
        pocket=_pocket(),
        active_policy=active_policy,
    )
    assert first == second
    assert first.startswith("F-")


def test_measure_family_scans_then_rebuilds_last_convergent_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _profile()
    calls: list[float] = []

    def fake_measure(state_builder, **_kwargs):
        for coordinate in (0.0, 1.0, 2.0):
            calls.append(state_builder(coordinate).coordinate)
        return SimpleNamespace(
            status="BOUNDARY_FOUND",
            highest_solved=1.0,
            lowest_unsolved=2.0,
            records=(
                SimpleNamespace(
                    coordinate=0.0,
                    phase="COARSE",
                    probe_status="SOLVED",
                    logical_result="SOLVED",
                    feasible=True,
                ),
                SimpleNamespace(
                    coordinate=1.0,
                    phase="COARSE",
                    probe_status="SOLVED",
                    logical_result="SOLVED",
                    feasible=True,
                ),
                SimpleNamespace(
                    coordinate=2.0,
                    phase="COARSE",
                    probe_status="NO_SOLUTION",
                    logical_result="NO_SOLUTION",
                    feasible=None,
                ),
            ),
            logical_probe_count=3,
            solver_attempt_count=4,
            policy_version="boundary-v1",
        )

    def fake_probe(net):
        solved = copy.deepcopy(net)
        solved.converged = True
        return SimpleNamespace(
            status="SOLVED",
            solved_net=solved,
            solver_attempt_count=1,
        )

    monkeypatch.setattr(curation, "measure_boundary", fake_measure)
    monkeypatch.setattr(curation, "solve_locked_probe", fake_probe)
    monkeypatch.setattr(
        curation,
        "evaluate_solved_feasibility",
        lambda _net: SimpleNamespace(feasible=True, failure_reasons=()),
    )

    result = measure_curation_family(
        profile,
        _pocket(),
        scan_policy=CurationScanPolicy(
            coarse_coordinates=(0.0, 1.0, 2.0),
            refinement_resolution=1.0,
        ),
    )

    assert calls == [0.0, 1.0, 2.0]
    assert result.boundary.lower == 1.0
    assert result.boundary.upper == 2.0
    assert result.monotonicity.status == "OBSERVED_MONOTONIC"
    assert result.last_convergent_state.coordinate == 1.0


def test_measure_family_keeps_solver_boundary_separate_from_valid_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _profile()

    def fake_measure(_state_builder, **_kwargs):
        return SimpleNamespace(
            status="BOUNDARY_FOUND",
            highest_solved=1.75,
            lowest_unsolved=2.0,
            records=(
                SimpleNamespace(
                    coordinate=0.0,
                    phase="COARSE",
                    probe_status="SOLVED",
                    logical_result="SOLVED",
                    feasible=True,
                ),
                SimpleNamespace(
                    coordinate=1.0,
                    phase="COARSE",
                    probe_status="SOLVED",
                    logical_result="SOLVED",
                    feasible=True,
                ),
                SimpleNamespace(
                    coordinate=1.5,
                    phase="COARSE",
                    probe_status="INFEASIBLE",
                    logical_result="SOLVED",
                    feasible=False,
                ),
                SimpleNamespace(
                    coordinate=2.0,
                    phase="COARSE",
                    probe_status="NO_SOLUTION",
                    logical_result="NO_SOLUTION",
                    feasible=None,
                ),
            ),
            logical_probe_count=4,
            solver_attempt_count=5,
            policy_version="boundary-v1",
        )

    def fake_probe(net):
        solved = copy.deepcopy(net)
        solved.converged = True
        return SimpleNamespace(
            status="SOLVED",
            solved_net=solved,
            solver_attempt_count=1,
        )

    monkeypatch.setattr(curation, "measure_boundary", fake_measure)
    monkeypatch.setattr(curation, "solve_locked_probe", fake_probe)
    monkeypatch.setattr(
        curation,
        "evaluate_solved_feasibility",
        lambda _net: SimpleNamespace(feasible=True, failure_reasons=()),
    )

    result = measure_curation_family(
        profile,
        _pocket(),
        scan_policy=CurationScanPolicy(
            coarse_coordinates=(0.0, 1.0, 1.5, 2.0),
            refinement_resolution=0.25,
        ),
    )

    assert result.boundary.lower == 1.75
    assert result.last_convergent_state.coordinate == 1.0
    assert result.monotonicity.probe_statuses == (
        "SOLVED",
        "SOLVED",
        "SOLVED",
        "NO_SOLUTION",
    )


def test_target_and_recipe_are_reconstructed_from_immutable_profile() -> None:
    profile = _profile()
    active_policy = build_active_policy(profile.net)
    original = copy.deepcopy(profile.net)

    target = build_target_state(
        profile.net,
        pocket=_pocket(),
        target_stress=1.3,
        active_policy=active_policy,
    )
    recipe = build_curation_recipe(
        profile,
        pocket=_pocket(),
        target_stress=1.3,
        active_policy=active_policy,
    )

    assert target.coordinate == 1.3
    assert target.net.load.at[3, "p_mw"] == pytest.approx(46.0)
    assert target.net.load.at[3, "q_mvar"] == pytest.approx(18.4)
    assert target.net.load.at[8, "p_mw"] == pytest.approx(16.5)
    assert recipe.target_stress == 1.3
    assert recipe.operating_profile_id == profile.profile_id
    assert len(recipe.recipe_hash) == 64
    pd.testing.assert_frame_equal(profile.net.load, original.load, check_exact=True)
    pd.testing.assert_frame_equal(profile.net.gen, original.gen, check_exact=True)
