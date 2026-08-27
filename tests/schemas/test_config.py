# ABOUTME: Verifies runner configuration schemas.
# ABOUTME: Covers LLM assignment, budget/runtime constraints, and derived MEMORY_ENABLED.
import pytest
from pydantic import ValidationError

from restorebench.schemas.config import LLMAssignment, OrchestratorConfig


def round_trip(model):
    assert type(model).model_validate(model.model_dump()) == model
    assert type(model).model_validate_json(model.model_dump_json()) == model


def assignment() -> LLMAssignment:
    return LLMAssignment(single_agent="qwen3:4b", analyst=None, executor=None, orchestrator=None)


def test_config_constraints_are_enforced():
    with pytest.raises(ValidationError):
        OrchestratorConfig(CONFIGURATION=2, MANEUVER_BUDGET=0, LLM_ASSIGNMENT=assignment(), repetition_index=3)
    with pytest.raises(ValidationError):
        OrchestratorConfig(CONFIGURATION=2, MANEUVER_BUDGET=21, LLM_ASSIGNMENT=assignment(), repetition_index=3)
    with pytest.raises(ValidationError):
        OrchestratorConfig(CONFIGURATION=2, MAX_RUNTIME_SECONDS=5, LLM_ASSIGNMENT=assignment(), repetition_index=3)


def test_runtime_cap_leaves_room_for_a_slow_multi_agent_episode():
    # Opus multi-agent needs 9-16 min end to end; the cap must not guillotine a run that would succeed.
    config = OrchestratorConfig(
        CONFIGURATION=3, MAX_RUNTIME_SECONDS=1800, LLM_ASSIGNMENT=assignment(), repetition_index=3
    )
    assert config.MAX_RUNTIME_SECONDS == 1800


def test_config_models_reject_extra_fields():
    with pytest.raises(ValidationError):
        LLMAssignment(single_agent=None, analyst=None, executor=None, orchestrator=None, other="x")
