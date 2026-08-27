# ABOUTME: Tests figure data preparation for the headline analysis figures.
# ABOUTME: Avoids importing matplotlib so default test runs do not need the analysis dependency group.
from restorebench.analysis import figures
from restorebench.physics.actions import ACTION_POLICY_VERSION
from restorebench.physics.policies import RANKING_POLICY_VERSION, SOLVER_PROBE_POLICY_VERSION
from restorebench.schemas.response import RESULT_SCHEMA_VERSION, ResolutionResponse

from builders import maneuver, quality, response


TARGET_DATASET_VERSION = "reactive-deficit-v1"


def _response(scenario_id: str, *, configuration: int, repetition_index: int, status: str):
    converged = status == "SUCCESS"
    run = response(
        scenario_id=scenario_id,
        configuration=configuration,
        status=status,
        converged=converged,
        result_quality=quality() if converged else None,
        maneuvers=[maneuver({"type": "GEN_V_SETPOINT", "gen_id": 10, "new_vm_pu": 1.02})] if converged else [],
    )
    payload = run.model_dump()
    payload.update(
        {
            "dataset_version": TARGET_DATASET_VERSION,
            "solver_version": SOLVER_PROBE_POLICY_VERSION,
            "action_policy_version": ACTION_POLICY_VERSION,
            "ranking_policy_version": RANKING_POLICY_VERSION,
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "llm_assignment": {"single_agent": "model-a"},
            "repetition_index": repetition_index,
        }
    )
    return ResolutionResponse.model_validate(payload)


def test_architecture_ladder_data_groups_by_model_config_and_repetition() -> None:
    runs = [
        _response("S0001", configuration=2, repetition_index=0, status="SUCCESS"),
        _response("S0002", configuration=2, repetition_index=0, status="TIMEOUT"),
        _response("S0001", configuration=2, repetition_index=1, status="SUCCESS"),
        _response("S0002", configuration=2, repetition_index=1, status="SUCCESS"),
    ]

    [point] = figures.architecture_ladder_data(runs)

    assert point.model_id == "model-a"
    assert point.configuration == 2
    assert point.mean_success_rate == 0.75
    assert point.n_repetitions == 2
