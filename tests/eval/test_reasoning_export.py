# ABOUTME: Verifies Markdown export of saved run reasoning traces.
# ABOUTME: Pins the HOWTO transcript format without reading raw JSON by hand.

from restorebench.eval import reasoning_export, store
from restorebench.llm import models
from restorebench.schemas.response import ReasoningEntry

from builders import maneuver, quality, response


def _response_with_reasoning(entries: list[ReasoningEntry], *, n_maneuvers: int = 2):
    maneuvers = [
        maneuver({"type": "GEN_V_SETPOINT", "gen_id": 10, "new_vm_pu": 1.02})
        for _ in range(n_maneuvers)
    ]
    base = response(
        scenario_id="S0121",
        configuration=3,
        result_quality=quality(),
        maneuvers=maneuvers,
    )
    trace = base.trace.model_copy(update={"reasoning": entries})
    return base.model_copy(
        update={
            "llm_assignment": {
                "analyst": models.GLM_5,
                "executor": models.GLM_5,
                "orchestrator": models.GLM_5,
            },
            "trace": trace,
            "n_maneuvers": n_maneuvers,
        }
    )


def test_reasoning_export_matches_howto_transcript_format(tmp_path) -> None:
    run_id = "S0121__config3__glm-5__rep0"
    saved = store.save_response(
        _response_with_reasoning(
            [
                ReasoningEntry(iteration=0, role="analyst", text="Bus 90 sits at 0.71 pu."),
                ReasoningEntry(iteration=0, role="executor", text="Raising gen 12 is within bounds."),
                ReasoningEntry(iteration=1, role="analyst", text="The first move shifted the weak bus."),
            ]
        ),
        tmp_path / "cells",
    )
    assert saved.name == f"{run_id}.json"

    [exported] = reasoning_export.export_reasoning([run_id], results_dir=tmp_path)

    assert exported.read_text(encoding="utf-8") == (
        "# S0121 · multi-agent · glm-5 · rep 0 · SUCCESS in 2 maneuvers\n"
        "\n"
        "## Iteration 1\n"
        "**analyst** — Bus 90 sits at 0.71 pu.\n"
        "**executor** — Raising gen 12 is within bounds.\n"
        "\n"
        "## Iteration 2\n"
        "**analyst** — The first move shifted the weak bus.\n"
    )


def test_reasoning_export_empty_reasoning_writes_header_only(tmp_path) -> None:
    run_id = "S0121__config3__glm-5__rep0"
    store.save_response(_response_with_reasoning([], n_maneuvers=0), tmp_path / "cells")

    [exported] = reasoning_export.export_reasoning([run_id], results_dir=tmp_path)

    assert exported.read_text(encoding="utf-8") == (
        "# S0121 · multi-agent · glm-5 · rep 0 · SUCCESS in 0 maneuvers\n"
    )


