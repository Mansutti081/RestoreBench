# ABOUTME: Locks the private reactive-deficit label and public dataset-manifest boundary.
# ABOUTME: Rejects legacy outage, lambda-nose, difficulty, and cause-label contracts.
from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from restorebench.schemas.actions import GenVoltageSetpointAction
from restorebench.schemas.dataset import (
    CurationWitness,
    DatasetManifest,
    EvaluationManifest,
    Scenario,
    ScenarioLabel,
)


SHA = "a" * 64


def _label_payload() -> dict:
    return {
        "scenario_id": "S0001",
        "scenario_class": "REACTIVE_DEFICIT",
        "scenario_family_id": "family-001",
        "leakage_group_id": "leakage-001",
        "recipe": {
            "operating_profile_id": "profile-001",
            "base_state_hash": SHA,
            "pocket": {
                "anchor_bus": 12,
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
                        "weight": 0.25,
                    },
                ],
                "vector_hash": SHA,
                "policy_version": "pocket-v1",
            },
            "active_schedule": {
                "generators": [
                    {"gen_id": 2, "base_p_mw": 30.0},
                    {"gen_id": 7, "base_p_mw": 40.0},
                ],
                "participation_factors": [1.0, 3.0],
                "policy_version": "active-v1",
            },
            "target_stress": 1.3,
            "recipe_hash": SHA,
        },
        "convergence_boundary": {
            "lower": 1.1,
            "upper": 1.2,
            "resolution": 0.1,
            "capped": False,
        },
        "monotonicity": {
            "status": "OBSERVED_MONOTONIC",
            "probe_coordinates": [0.0, 1.0, 1.2],
            "probe_statuses": ["SOLVED", "SOLVED", "NO_SOLUTION"],
            "policy_version": "scan-v1",
        },
        "qv_evidence": {
            "evidence_stress": 1.1,
            "weak_bus_ids": [12, 13],
            "weak_region_min_distance_pu": 0.04,
            "weak_region_local": True,
            "min_vm_pu": 0.93,
            "q_near_limit_gen_ids": [2],
            "q_limited_gen_ids": [2],
            "newly_q_limited_gen_ids": [2],
            "q_headroom_reduction_mvar": 12.0,
            "generator_p_feasible": True,
            "ext_grid_p_mw": 5.0,
            "ext_grid_q_mvar": 20.0,
            "hard_voltage_envelope_passed": True,
            "thresholds_version": "qv-thresholds-v1",
        },
        "q_unlimited_counterfactual": {
            "converged": True,
            "material_violation_gen_ids": [2],
            "max_violation_mvar": 10.0,
            "ext_grid_feasible": True,
            "policy_version": "q-unlimited-v1",
        },
        "alternative_init_audit": {
            "init_policy": "flat",
            "primary_status": "NO_SOLUTION",
            "recovery_status": "NO_SOLUTION",
            "converged_without_action": False,
            "solver_attempt_count": 2,
        },
        "resolvable_within_budget": True,
        "resolution_regime": "DIRECT",
        "direct_restorer_available": True,
        "witness_length": 1,
        "witness_optimality": "EXACT_MINIMUM",
        "target_depth": {
            "stress_offset": 0.1,
            "relative_offset": 1 / 12,
            "policy_version": "target-depth-v1",
        },
        "memory_split": "memory_population",
        "generation_metadata": {
            "generator_version": "reactive-deficit-generator-v1",
            "validator_version": "reactive-deficit-validator-v1",
            "python_version": "3.11.9",
            "pandapower_version": "3.4.0",
            "seed": 42,
            "solver_settings": {
                "algorithm": "nr",
                "enforce_q_lims": True,
                "init": "dc",
                "max_iteration": 30,
                "primary_tolerance_mva": 1e-8,
                "recovery_tolerance_mva": 1e-6,
                "check_connectivity": True,
            },
            "shared_policy_versions": {
                "active_balance": "active-v1",
                "action": "action-v1",
                "solver_probe": "solver-v1",
                "feasibility": "feasibility-v1",
                "electrical_distance": "distance-v1",
                "fingerprint": "fingerprint-v1",
            },
            "curation_policy_versions": {
                "augmentation": "augmentation-v1",
                "operating_profile": "profile-v1",
                "pocket_weighting": "pocket-v1",
                "load_stress": "stress-v1",
                "alternative_init": "flat-v1",
                "monotonicity_scan": "scan-v1",
                "qv_thresholds": "qv-thresholds-v1",
                "witness_search": "witness-v1",
                "composition": "composition-v1",
                "split": "split-v1",
            },
        },
    }


def test_private_label_round_trips_and_rejects_every_v0_field() -> None:
    payload = _label_payload()
    label = ScenarioLabel.model_validate(payload)

    assert ScenarioLabel.model_validate_json(label.model_dump_json()) == label
    for field, value in (
        ("pre_weakening", {"type": "none", "element_id": None, "factor": None}),
        ("cause_hint", "REACTIVE_DEFICIT"),
        ("lambda_nose", 1.2),
        ("lambda_scenario", 1.3),
        ("epsilon", 0.1),
        ("difficulty", "EASY"),
        ("load_pattern", {"type": "pocket", "region": [12]}),
    ):
        stale = deepcopy(payload)
        stale[field] = value
        with pytest.raises(ValidationError):
            ScenarioLabel.model_validate(stale)


def test_resolution_regime_and_target_depth_are_cross_validated() -> None:
    direct = _label_payload()

    sequential = deepcopy(direct)
    sequential.update(
        {
            "resolution_regime": "SEQUENTIAL",
            "direct_restorer_available": False,
            "witness_length": 3,
            "witness_optimality": "UPPER_BOUND",
        }
    )
    assert ScenarioLabel.model_validate(sequential).witness_length == 3

    inconsistent = deepcopy(direct)
    inconsistent["witness_length"] = 2
    with pytest.raises(ValidationError, match="DIRECT"):
        ScenarioLabel.model_validate(inconsistent)

    wrong_depth = deepcopy(direct)
    wrong_depth["target_depth"]["stress_offset"] = 0.2
    with pytest.raises(ValidationError, match="target depth"):
        ScenarioLabel.model_validate(wrong_depth)


def test_recipe_requires_stable_order_and_matching_participation_lengths() -> None:
    unordered = _label_payload()
    unordered["recipe"]["pocket"]["loads"].reverse()
    with pytest.raises(ValidationError, match="load IDs"):
        ScenarioLabel.model_validate(unordered)

    mismatched = _label_payload()
    mismatched["recipe"]["active_schedule"]["participation_factors"] = [1.0]
    with pytest.raises(ValidationError, match="participation"):
        ScenarioLabel.model_validate(mismatched)


def test_public_manifest_and_runtime_scenario_cannot_carry_private_fields() -> None:
    manifest = DatasetManifest.model_validate(
        {
            "dataset_version": "reactive-deficit-v0",
            "base_network_hash": SHA,
            "scenario_count": 1,
            "scenarios": [
                {
                    "scenario_id": "S0001",
                    "full_artifact_hash": SHA,
                    "lean_artifact_hash": SHA,
                    "card_artifact_hash": SHA,
                }
            ],
            "split_manifest_hash": SHA,
            "policy_hashes": {"shared": SHA},
            "environment": {"python": "3.11.9", "pandapower": "3.4.0"},
        }
    )
    evaluation = EvaluationManifest.model_validate(
        {
            "dataset_version": manifest.dataset_version,
            "scenarios": [{"scenario_id": "S0001", "memory_split": "memory_population"}],
        }
    )
    scenario = Scenario(
        scenario_id="S0001",
        full_net_path="/staging/full/S0001.json",
        card_path="/staging/llm/S0001.md",
        memory_split="memory_population",
    )

    assert manifest.scenario_count == 1
    assert evaluation.scenarios[0].memory_split == "memory_population"
    assert "label" not in Scenario.model_fields
    for model, payload in (
        (Scenario, {**scenario.model_dump(), "label": _label_payload()}),
        (
            DatasetManifest,
            {**manifest.model_dump(), "private_labels": [_label_payload()]},
        ),
    ):
        with pytest.raises(ValidationError):
            model.model_validate(payload)


def test_private_witness_requires_consistent_lengths_and_atomic_actions() -> None:
    witness = CurationWitness.model_validate(
        {
            "scenario_id": "S0001",
            "maneuvers": [
                GenVoltageSetpointAction(
                    type="GEN_V_SETPOINT",
                    gen_id=2,
                    new_vm_pu=1.01,
                )
            ],
            "state_hashes": [SHA, "b" * 64],
            "terminal_pf": {
                "converged": True,
                "iterations": 4,
                "tolerance_used": 1e-8,
                "runtime_ms": 1.0,
            },
            "search_policy_version": "witness-v1",
        }
    )
    assert len(witness.state_hashes) == len(witness.maneuvers) + 1

    bad = witness.model_dump()
    bad["state_hashes"] = [SHA]
    with pytest.raises(ValidationError, match="state_hashes"):
        CurationWitness.model_validate(bad)
