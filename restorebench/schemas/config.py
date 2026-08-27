# ABOUTME: Defines orchestrator configuration schemas.
# ABOUTME: Selects the agent architecture and bounds one resolution episode.
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class LLMAssignment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    single_agent: str | None
    analyst: str | None
    executor: str | None
    orchestrator: str | None


class OrchestratorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    CONFIGURATION: Literal[1, 2, 3] = 2
    MANEUVER_BUDGET: Annotated[int, Field(ge=1, le=20)] = 10
    # Opus multi-agent measured 9-16 min end to end; a 300s/900s window guillotined runs that would
    # have converged. Default and cap raised so a slow episode finishes instead of recording TIMEOUT.
    MAX_RUNTIME_SECONDS: Annotated[int, Field(ge=10, le=3600)] = 1200
    LLM_ASSIGNMENT: LLMAssignment
    # Which of the N repetitions of this scenario this run is. Not a random seed: no model in
    # the suite is deterministic and Bedrock's Converse API exposes no seed parameter, so a run
    # cannot be reproduced from a number. It labels the run for resume-from-disk and for keeping
    # result files from colliding.
    repetition_index: Annotated[int, Field(ge=0)] = 0
