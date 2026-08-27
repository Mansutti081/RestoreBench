# ABOUTME: Verifies every agent prompt states that accepted maneuvers accumulate into one plan.
# ABOUTME: Locks the progress semantics of overstress so a non-converging maneuver can still be kept.
from __future__ import annotations

import pytest

from restorebench.agents.prompt_fragments import COMPOSITION_AND_PROGRESS
from restorebench.schemas.config import LLMAssignment, OrchestratorConfig

MODEL_ID = "anthropic.claude-3-5-sonnet-20241022-v2:0"


def _config(configuration: int = 2) -> OrchestratorConfig:
    single = MODEL_ID if configuration in {1, 2, 4} else None
    role = None if configuration in {1, 2, 4} else MODEL_ID
    return OrchestratorConfig(
        CONFIGURATION=configuration,
        MANEUVER_BUDGET=10,
        MAX_RUNTIME_SECONDS=120,
        LLM_ASSIGNMENT=LLMAssignment(
            single_agent=single, analyst=role, executor=role, orchestrator=role
        ),
    )


def test_fragment_states_that_accepted_maneuvers_persist() -> None:
    """The agent restarts its conversation each iteration; the grid does not restart with it."""
    assert "persist" in COMPOSITION_AND_PROGRESS
    assert "cumulative" in COMPOSITION_AND_PROGRESS


def test_fragment_defines_overstress_direction_and_keeps_partial_progress() -> None:
    """Without this the only signal is STILL_DIVERGED, which reads as 'that move was wrong'."""
    assert "overstress" in COMPOSITION_AND_PROGRESS
    assert "lower" in COMPOSITION_AND_PROGRESS
    assert "still diverges" in COMPOSITION_AND_PROGRESS


@pytest.mark.parametrize(
    "prompt_factory",
    [
        pytest.param(
            lambda: __import__(
                "restorebench.agents.single_agent", fromlist=["_system_prompt"]
            )._system_prompt(_config(2)),
            id="single_agent",
        ),
        pytest.param(
            lambda: __import__(
                "restorebench.agents.analyst", fromlist=["_system_prompt"]
            )._system_prompt(_config(3)),
            id="analyst",
        ),
        pytest.param(
            lambda: __import__(
                "restorebench.agents.executor", fromlist=["_system_prompt"]
            )._system_prompt(_config(3)),
            id="executor",
        ),
        pytest.param(
            lambda: __import__(
                "restorebench.agents.executor", fromlist=["_orchestrator_system_prompt"]
            )._orchestrator_system_prompt(_config(3)),
            id="orchestrator_agent",
        ),
    ],
)
def test_iterative_agents_carry_the_composition_policy(prompt_factory) -> None:
    assert COMPOSITION_AND_PROGRESS in prompt_factory()


def test_chatbot_states_accumulation_without_claiming_iterative_feedback() -> None:
    """Case 1 gets no diagnostics between maneuvers, so it takes the accumulation half only."""
    from restorebench.agents import baseline_chatbot

    prompt = baseline_chatbot._system_prompt(_config(1))
    assert "cumulative" in prompt
    # The old wording asked for alternatives ranked by promise, which is the wrong shape for a
    # case that needs several maneuvers stacked on one another.
    assert "most-promising first" not in prompt
    assert COMPOSITION_AND_PROGRESS not in prompt
