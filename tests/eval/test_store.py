# ABOUTME: Verifies on-disk ResolutionResponse storage and resume keys for the eval harness.
# ABOUTME: Uses synthetic responses only; no solver, LLM, or Bedrock calls.
import pytest

from restorebench.eval import store
from restorebench.llm import models

from builders import maneuver, quality, response


def _response(
    scenario_id: str = "S0121",
    *,
    configuration: int = 3,
    repetition_index: int = 0,
    model_id: str = models.GLM_5,
):
    base = response(
        scenario_id=scenario_id,
        configuration=configuration,
        result_quality=quality(),
        maneuvers=[maneuver({"type": "GEN_V_SETPOINT", "gen_id": 10, "new_vm_pu": 1.02})],
    )
    if configuration in {1, 2, 4}:
        assignment = {"single_agent": model_id}
    else:
        assignment = {"analyst": model_id, "executor": model_id, "orchestrator": model_id}
    return base.model_copy(update={"llm_assignment": assignment, "repetition_index": repetition_index})


def test_save_load_save_round_trip_is_byte_faithful(tmp_path) -> None:
    results_dir = tmp_path / "cells"
    saved = store.save_response(_response(), results_dir)
    first_bytes = saved.read_bytes()

    loaded = store.load_response(saved)
    saved_again = store.save_response(loaded, results_dir)

    assert saved_again == saved
    assert saved.read_bytes() == first_bytes


def test_atomic_write_leaves_no_partial_result_on_replace_failure(monkeypatch, tmp_path) -> None:
    def fail_replace(_tmp_path, _target_path):
        raise RuntimeError("simulated rename failure")

    monkeypatch.setattr(store, "_replace_temp", fail_replace)
    results_dir = tmp_path / "cells"

    with pytest.raises(RuntimeError, match="simulated"):
        store.save_response(_response(), results_dir)

    assert not (results_dir / "S0121__config3__glm-5__rep0.json").exists()
    assert not list(results_dir.glob("*.tmp"))


def test_resume_helpers_list_only_missing_cells_runs(tmp_path) -> None:
    results_dir = tmp_path / "cells"
    scenario_ids = [f"S000{i}" for i in range(1, 6)]
    for scenario_id in scenario_ids[:3]:
        store.save_response(_response(scenario_id=scenario_id), results_dir)

    missing = store.missing_runs(
        results_dir=results_dir,
        scenario_ids=scenario_ids,
        model_ids=[models.GLM_5],
        configurations=[3],
        repetitions=[0],
    )

    assert store.is_done(
        results_dir=results_dir,
        scenario_id="S0001",
        model_id=models.GLM_5,
        configuration=3,
        repetition=0,
    )
    assert [(key.scenario_id, key.configuration, key.repetition) for key in missing] == [
        ("S0004", 3, 0),
        ("S0005", 3, 0),
    ]
