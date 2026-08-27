# ABOUTME: Verifies the Phase-A/Phase-B eval harness with fake resolve/bootstrap seams.
# ABOUTME: No Bedrock, no real solver loop; tests checkpointing, resume, split discipline, and wiring.
from __future__ import annotations


import pytest

from restorebench.eval import harness
from restorebench.llm import models
from restorebench.schemas.config import OrchestratorConfig

from builders import maneuver, quality, response


def _response(
    scenario_id: str,
    configuration: int,
    model_id: str,
    *,
    repetition_index: int = 0,
    status: str = "SUCCESS",
):
    converged = status == "SUCCESS"
    assignment = (
        {"single_agent": model_id}
        if configuration in {1, 2, 4}
        else {"analyst": model_id, "executor": model_id, "orchestrator": model_id}
    )
    return response(
        scenario_id=scenario_id,
        configuration=configuration,
        status=status,
        converged=converged,
        result_quality=quality() if converged else None,
        maneuvers=[maneuver({"type": "GEN_V_SETPOINT", "gen_id": 10, "new_vm_pu": 1.02})] if converged else [],
    ).model_copy(update={"llm_assignment": assignment, "repetition_index": repetition_index})


def _model_from_config(config: OrchestratorConfig) -> str:
    assignment = config.LLM_ASSIGNMENT
    return assignment.single_agent or assignment.analyst or models.GLM_5


def test_cells_cli_configuration_override_replaces_default(monkeypatch) -> None:
    seen = {}

    def fake_run_evaluation(models_arg, **kwargs):
        seen["models"] = models_arg
        seen.update(kwargs)
        return harness.HarnessSummary(n_planned=0, n_run=0, n_skipped=0, result_paths=())

    monkeypatch.setattr(harness, "run_evaluation", fake_run_evaluation)

    assert harness.main(["--configuration", "2", "--model", models.GLM_5]) == 0

    assert seen["models"] == (models.GLM_5,)
    assert seen["configurations"] == (2,)


def test_parser_rejects_models_outside_the_suite() -> None:
    with pytest.raises(SystemExit):
        harness._parser().parse_args(["phase-b", "--model", "not-a-benchmark-model"])


