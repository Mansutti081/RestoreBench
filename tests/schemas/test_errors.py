# ABOUTME: Verifies structured schema error types are exceptions, not Pydantic models.
# ABOUTME: Covers attributes and readable string forms.
from pydantic import BaseModel

from restorebench.schemas.actions import GenVoltageSetpointAction
from restorebench.llm.models import OPUS_4_6
from restorebench.schemas.errors import InvalidActionError, LLMFailureError, PowerFlowDivergenceError, ToolFailureError
from restorebench.schemas.power_flow import PowerFlowResult


def test_error_types_are_exceptions_not_base_models():
    for error_type in (InvalidActionError, PowerFlowDivergenceError, ToolFailureError, LLMFailureError):
        assert issubclass(error_type, Exception)
        assert not issubclass(error_type, BaseModel)


def test_invalid_action_error_carries_action_and_reason():
    action = GenVoltageSetpointAction(type="GEN_V_SETPOINT", gen_id=1, new_vm_pu=1.02)
    exc = InvalidActionError(action=action, reason="gen 1 is Q_LIMITED_UPPER")

    assert exc.action == action
    assert exc.reason == "gen 1 is Q_LIMITED_UPPER"
    assert "Q_LIMITED_UPPER" in str(exc)


def test_power_flow_and_tool_errors_carry_structured_attributes():
    pf_result = PowerFlowResult(
        converged=False,
        iterations=30,
        tolerance_used=1e-8,
        runtime_ms=50.0,
        error_message="failed",
        diagnostics=None,
        quality=None,
        warnings=[],
    )
    pf_exc = PowerFlowDivergenceError(pf_result=pf_result)
    tool_exc = ToolFailureError(tool_name="PowerFlowServer", underlying_exception="boom")

    assert pf_exc.pf_result == pf_result
    assert "did not converge" in str(pf_exc)
    assert tool_exc.tool_name == "PowerFlowServer"
    assert tool_exc.underlying_exception == "boom"
    assert "PowerFlowServer" in str(tool_exc)


def test_llm_failure_error_carries_model_id_and_underlying_exception():
    exc = LLMFailureError(model_id=OPUS_4_6, underlying_exception="timeout")

    assert exc.model_id == OPUS_4_6
    assert exc.underlying_exception == "timeout"
    assert OPUS_4_6 in str(exc)
    assert "timeout" in str(exc)
