# ABOUTME: Tests deterministic aggregation, stratification joins, and paired statistics.
# ABOUTME: Uses canned ResolutionResponses and tmp_path result files only.
import json
from types import SimpleNamespace

import pytest

from restorebench.analysis import aggregate
from restorebench.eval import store
from restorebench.llm import models
from restorebench.physics.actions import ACTION_POLICY_VERSION
from restorebench.physics.policies import RANKING_POLICY_VERSION, SOLVER_PROBE_POLICY_VERSION
from restorebench.schemas.response import RESULT_SCHEMA_VERSION, ResolutionResponse

from builders import maneuver, quality, response


TARGET_DATASET_VERSION = "reactive-deficit-v1"
VERSION_FIELDS = {
    "dataset_version",
    "solver_version",
    "action_policy_version",
    "ranking_policy_version",
    "result_schema_version",
}


def _response(
    scenario_id: str,
    *,
    configuration: int = 2,
    repetition_index: int = 0,
    status: str = "SUCCESS",
    runtime: float = 10.0,
    dataset_version: str = TARGET_DATASET_VERSION,
):
    converged = status == "SUCCESS"
    run = response(
        scenario_id=scenario_id,
        configuration=configuration,
        status=status,
        converged=converged,
        result_quality=quality() if converged else None,
        maneuvers=[maneuver({"type": "GEN_V_SETPOINT", "gen_id": 10, "new_vm_pu": 1.02})] if converged else [],
    )
    assignment = (
        {"single_agent": models.GLM_5}
        if configuration in {1, 2, 4}
        else {"analyst": models.GLM_5, "executor": models.GLM_5, "orchestrator": models.GLM_5}
    )
    payload = run.model_dump()
    payload.update(
        {
            "dataset_version": dataset_version,
            "solver_version": SOLVER_PROBE_POLICY_VERSION,
            "action_policy_version": ACTION_POLICY_VERSION,
            "ranking_policy_version": RANKING_POLICY_VERSION,
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "llm_assignment": assignment,
            "repetition_index": repetition_index,
            "total_runtime_seconds": runtime,
        }
    )
    return ResolutionResponse.model_validate(payload)


def _legacy_response(scenario_id: str) -> ResolutionResponse:
    payload = _response(scenario_id).model_dump(exclude=VERSION_FIELDS)
    return ResolutionResponse.model_validate(payload)


def _write_target_manifest(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "dataset_version": TARGET_DATASET_VERSION,
                "base_network_hash": "a" * 64,
                "scenario_count": 1,
                "scenarios": [
                    {
                        "scenario_id": "S0001",
                        "full_artifact_hash": "b" * 64,
                        "lean_artifact_hash": "c" * 64,
                        "card_artifact_hash": "d" * 64,
                    }
                ],
                "split_manifest_hash": "e" * 64,
                "policy_hashes": {"shared": "f" * 64},
                "environment": {"python": "3.11", "pandapower": "3.4"},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_stratification_join_splits_by_resolution_regime_and_sums_to_whole(monkeypatch, tmp_path) -> None:
    labels = {
        "S0001": _label(profile="OP-A", regime="DIRECT", length=1, optimality="EXACT_MINIMUM"),
        "S0002": _label(profile="OP-A", regime="SEQUENTIAL", length=3, optimality="UPPER_BOUND"),
        "S0003": _label(profile="OP-B", regime="SEQUENTIAL", length=2, optimality="EXACT_MINIMUM"),
    }
    monkeypatch.setattr(aggregate, "_load_private_labels", lambda _path: labels)
    runs = [_response("S0001"), _response("S0002", status="TIMEOUT"), _response("S0003")]

    strata = aggregate.build_strata(
        runs,
        dataset_manifest_path=_write_target_manifest(tmp_path),
    )
    rows = aggregate.success_by_stratum(runs, strata, "resolution_regime")
    whole = aggregate.success_by_stratum(runs, strata, None)

    assert [(row.key, row.numerator, row.denominator) for row in rows] == [
        ("DIRECT", 1, 1),
        ("SEQUENTIAL", 1, 2),
    ]
    assert sum(row.denominator for row in rows) == whole[0].denominator


def test_stratification_unknown_scenario_is_a_hard_error(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(aggregate, "_load_private_labels", lambda _path: {})
    runs = [_response("S9999")]

    with pytest.raises(ValueError, match="unknown scenario_id S9999"):
        aggregate.build_strata(
            runs,
                dataset_manifest_path=_write_target_manifest(tmp_path),
        )


def test_historical_result_loads_for_provider_metrics_but_pairing_fails_closed(tmp_path) -> None:
    historical = _legacy_response("S0042")
    path = tmp_path / "historical.json"
    path.write_text(
        json.dumps(historical.model_dump(mode="json", exclude=VERSION_FIELDS)),
        encoding="utf-8",
    )

    loaded = store.load_response(path)

    assert loaded.total_runtime_seconds == 10.0
    assert loaded.trace.total_llm_tokens_in == 10
    with pytest.raises(aggregate.LegacyResultNotComparableError):
        aggregate.paired_scenario_means(
            [loaded],
            control_config=2,
            treatment_config=4,
            metric="success_rate",
        )


@pytest.mark.parametrize(
    ("field", "incompatible"),
    [
        ("dataset_version", "reactive-deficit-v0"),
        ("solver_version", "other-solver"),
        ("action_policy_version", "other-actions"),
        ("ranking_policy_version", "other-ranking"),
        ("result_schema_version", "other-schema"),
    ],
)
def test_target_stratification_rejects_each_incompatible_version(
    monkeypatch,
    tmp_path,
    field,
    incompatible,
) -> None:
    labels = {
        "S0001": _label(
            profile="OP-A",
            regime="DIRECT",
            length=1,
            optimality="EXACT_MINIMUM",
        )
    }
    monkeypatch.setattr(aggregate, "_load_private_labels", lambda _path: labels)
    run = _response("S0001").model_copy(update={field: incompatible})

    with pytest.raises(aggregate.ResultVersionMismatchError, match=field):
        aggregate.build_strata(
            [run],
                dataset_manifest_path=_write_target_manifest(tmp_path),
        )


def test_pairing_rejects_raw_two_hundred_fifty_vs_two_hundred_fifty_shape() -> None:
    with pytest.raises(ValueError, match="expected 50 paired scenario means"):
        aggregate.compare_paired_values([1.0] * 250, [0.8] * 250, metric_kind="proportion", higher_is_better=True)


def test_comparison_verdict_blocks_ranking_when_not_significant() -> None:
    control = [1.0 if index % 2 == 0 else 0.0 for index in range(50)]
    treatment = [0.0 if index % 2 == 0 else 1.0 for index in range(50)]

    result = aggregate.compare_paired_values(
        control, treatment, metric_kind="continuous", n_comparisons=1, higher_is_better=True
    )

    assert result.verdict == "not significantly different"
    assert result.adjusted_p_value >= 0.05


def test_deterministic_table_output_is_byte_identical() -> None:
    rows = [
        {"metric": "success_rate", "group": "HARD", "value": 0.5, "denominator": 2},
        {"metric": "success_rate", "group": "EASY", "value": 1.0, "denominator": 1},
    ]

    first = aggregate.render_table(rows)
    second = aggregate.render_table(list(reversed(rows)))

    assert first.encode("utf-8") == second.encode("utf-8")
    assert first == "denominator,group,metric,value\n1,EASY,success_rate,1.000000\n2,HARD,success_rate,0.500000\n"


def test_load_results_reads_through_store(tmp_path) -> None:
    saved = store.save_response(_response("S0001"), tmp_path / "cells")

    loaded = aggregate.load_results(tmp_path)

    assert [run.request_id for run in loaded] == [store.load_response(saved).request_id]


def _label(*, profile: str, regime: str, length: int, optimality: str):
    return SimpleNamespace(
        recipe=SimpleNamespace(operating_profile_id=profile),
        resolution_regime=regime,
        witness_length=length,
        witness_optimality=optimality,
        generation_metadata=SimpleNamespace(
            shared_policy_versions=SimpleNamespace(
                solver_probe=SOLVER_PROBE_POLICY_VERSION,
                action=ACTION_POLICY_VERSION,
            )
        ),
    )


def test_verdict_respects_metric_orientation_for_lower_is_better() -> None:
    # mean_time / mean_maneuvers: a positive delta means the treatment is SLOWER/WORSE.
    control = [10.0 + 0.01 * i for i in range(50)]
    treatment = [value + 5.0 for value in control]  # consistently slower

    worse = aggregate.compare_paired_values(
        control, treatment, metric_kind="continuous", n_comparisons=1, higher_is_better=False
    )
    better = aggregate.compare_paired_values(
        control, treatment, metric_kind="continuous", n_comparisons=1, higher_is_better=True
    )

    assert worse.delta_mean > 0
    assert worse.verdict == "control better"
    assert better.verdict == "treatment better"


def test_metric_orientation_table_covers_every_metric() -> None:
    assert aggregate.METRIC_ORIENTATION == {
        "success_rate": True,
        "clean_rate": True,
        "mean_maneuvers": False,
        "mean_time": False,
    }


def test_zero_success_scenario_is_dropped_with_visibility_not_a_crash() -> None:
    # A scenario with 0/5 successes in one arm has no defined mean_maneuvers: the pair
    # is dropped and reported, instead of aborting the whole 50-pair comparison.
    runs = []
    for index in range(50):
        scenario_id = f"S{index:04d}"
        for repetition in range(5):
            for configuration, ok in ((2, index != 7), (3, True)):
                runs.append(
                    _response(
                        scenario_id=scenario_id,
                        configuration=configuration,
                        repetition_index=repetition,
                        status="SUCCESS" if ok else "BUDGET_EXHAUSTED",
                    )
                )

    pairs = aggregate.paired_scenario_means(runs, control_config=2, treatment_config=3, metric="mean_maneuvers")

    assert pairs.dropped_scenario_ids == ("S0007",)
    assert len(pairs.control) == 49
    assert "S0007" not in pairs.scenario_ids


def test_every_headline_metric_is_stratified(monkeypatch, tmp_path) -> None:
    # Plan 14: "Report every metric stratified" — not just the success rate.
    labels = {
        "S0001": _label(profile="OP-A", regime="DIRECT", length=1, optimality="EXACT_MINIMUM"),
        "S0002": _label(profile="OP-B", regime="SEQUENTIAL", length=3, optimality="UPPER_BOUND"),
    }
    monkeypatch.setattr(aggregate, "_load_private_labels", lambda _path: labels)
    runs = [
        _response("S0001", configuration=2, status="SUCCESS"),
        _response("S0001", configuration=2, status="BUDGET_EXHAUSTED"),
        _response("S0002", configuration=2, status="SUCCESS", runtime=30.0),
    ]
    strata = aggregate.build_strata(
        runs,
        dataset_manifest_path=_write_target_manifest(tmp_path),
    )

    rows = aggregate.metrics_by_stratum(runs, strata, "resolution_regime")

    by_key = {row.key: row for row in rows}
    assert set(by_key) == {"DIRECT", "SEQUENTIAL"}
    direct = by_key["DIRECT"]
    assert direct.success_rate.value == 0.5
    assert direct.success_rate.denominator == 2
    assert direct.mean_maneuvers.denominator == 1  # SUCCESS runs only
    assert direct.time.denominator == 2
    assert direct.quality.clean_rate.denominator == 1
    sequential = by_key["SEQUENTIAL"]
    assert sequential.success_rate.value == 1.0
    assert sequential.time.mean_seconds == 30.0

