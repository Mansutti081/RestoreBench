# ABOUTME: Defines structured exception types used by schema-facing tools.
# ABOUTME: These are raised errors, not Pydantic serialization contracts.
from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from restorebench.schemas.actions import Action
from restorebench.schemas.power_flow import PowerFlowResult

if TYPE_CHECKING:
    from collections.abc import Sequence

    from restorebench.llm.providers import LLMResponse

_ExcT = TypeVar("_ExcT", bound=BaseException)

# Attribute an agent stamps on the exception it raises so the tokens it already spent are not lost.
LLM_RESPONSES_ATTR = "llm_responses"


def attach_llm_responses(exc: _ExcT, responses: Sequence["LLMResponse"]) -> _ExcT:
    """Carry the LLM responses an agent produced on the exception it raises.

    A failing agent has already billed AWS for every call it made. Without this the orchestrator's
    failure paths record zero tokens, understating the cost of the configurations that fail most.
    """
    setattr(exc, LLM_RESPONSES_ATTR, tuple(responses))
    return exc


def llm_responses_of(exc: BaseException) -> tuple["LLMResponse", ...]:
    return tuple(getattr(exc, LLM_RESPONSES_ATTR, ()))


class InvalidActionError(Exception):
    def __init__(self, action: Action, reason: str) -> None:
        self.action = action
        self.reason = reason
        super().__init__(f"Invalid action: {reason}")


class PowerFlowDivergenceError(Exception):
    def __init__(self, pf_result: PowerFlowResult) -> None:
        self.pf_result = pf_result
        super().__init__("Power flow did not converge")


class CorpusIntegrityError(AssertionError):
    """Raised when a frozen benchmark scenario no longer diverges on load."""


class ToolFailureError(Exception):
    def __init__(self, tool_name: str, underlying_exception: str) -> None:
        self.tool_name = tool_name
        self.underlying_exception = underlying_exception
        super().__init__(f"{tool_name} failed: {underlying_exception}")


class LLMFailureError(Exception):
    def __init__(self, model_id: str, underlying_exception: str) -> None:
        self.model_id = model_id
        self.underlying_exception = underlying_exception
        super().__init__(f"LLM call failed for {model_id}: {underlying_exception}")
