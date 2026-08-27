# ABOUTME: Tests the Bedrock benchmark runner that turns LLM tool calls into attempts.
# ABOUTME: Uses real scenario cards and mocked Bedrock clients without scoring or network calls.
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from restorebench.llm import providers
from restorebench.schemas.actions import ManeuverSequence
from restorebench.schemas.errors import LLMFailureError

from restorebench.llm.models import BENCHMARK_MODELS


MODEL_ID = "qwen.qwen3-32b-v1:0"


def _maneuver(index: int = 0) -> dict[str, Any]:
    return {
        "action": {"type": "GEN_V_SETPOINT", "gen_id": index, "new_vm_pu": 1.05},
        "diagnosed_cause": "REACTIVE_DEFICIT",
        "rationale": f"raise generator {index} voltage",
    }


def _bedrock_response(
    tool_input: dict[str, Any] | None,
    *,
    reasoning: bool = True,
    stop_reason: str | None = None,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    if reasoning:
        content.append({"reasoningContent": {"reasoningText": {"text": "reasoned grid diagnosis"}}})
    if tool_input is not None:
        content.append({"toolUse": {"toolUseId": "toolu_1", "name": "propose_maneuvers", "input": tool_input}})
    return {
        "output": {"message": {"content": content}},
        "usage": {"inputTokens": 101, "outputTokens": 202},
        "stopReason": stop_reason or ("tool_use" if tool_input is not None else "end_turn"),
    }


def _mock_bedrock(monkeypatch: pytest.MonkeyPatch, result: dict[str, Any] | None = None) -> Mock:
    client = Mock()
    client.converse.return_value = result or _bedrock_response(
        {"maneuvers": [_maneuver()], "reconstruction_summary": "read the card"}
    )
    monkeypatch.setattr(providers, "_bedrock_client", lambda: client)
    return client


def test_tool_schema_is_generated_from_maneuver_sequence() -> None:
    from restorebench.scoring.run_benchmark import PROPOSE_MANEUVERS_TOOL_NAME, maneuver_sequence_tool_config

    tool_config = maneuver_sequence_tool_config()
    tool_spec = tool_config["tools"][0]["toolSpec"]

    assert tool_spec["name"] == PROPOSE_MANEUVERS_TOOL_NAME
    assert tool_spec["inputSchema"]["json"] == ManeuverSequence.model_json_schema()


def test_valid_tool_response_writes_truncated_attempt_and_run_record(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from restorebench.scoring.run_benchmark import MANEUVER_BUDGET, run_one

    _mock_bedrock(
        monkeypatch,
        _bedrock_response(
            {
                "maneuvers": [_maneuver(index) for index in range(MANEUVER_BUDGET + 1)],
                "reconstruction_summary": "read the card",
            }
        ),
    )

    result = run_one("S0008", MODEL_ID, repeat_index=0, out_dir=tmp_path)

    attempt = json.loads(result.attempt_path.read_text(encoding="utf-8"))
    record = json.loads(result.record_path.read_text(encoding="utf-8"))
    assert result.skipped is False
    assert result.validation_error is None
    assert attempt == {
        "scenario_id": "S0008",
        "maneuvers": [_maneuver(index)["action"] for index in range(MANEUVER_BUDGET)],
        "source": MODEL_ID,
    }
    assert record["model_id"] == MODEL_ID
    assert record["tokens_in"] == 101
    assert record["tokens_out"] == 202
    assert record["latency_seconds"] >= 0
    assert record["reasoning_present"] is True
    assert record["truncated"] is False
    assert record["validation_error"] is None
    assert record["raw_response"]["stop_reason"] == "tool_use"
    assert "Scenario Card" in record["prompt"][1]["content"]


def test_invalid_tool_response_is_recorded_and_attempt_is_scoreable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from restorebench.scoring.run_benchmark import run_one

    _mock_bedrock(
        monkeypatch,
        _bedrock_response({"maneuvers": [{"action": {"type": "LINE_SWITCH"}}], "reconstruction_summary": None}),
    )

    result = run_one("S0008", MODEL_ID, repeat_index=0, out_dir=tmp_path)

    attempt = json.loads(result.attempt_path.read_text(encoding="utf-8"))
    record = json.loads(result.record_path.read_text(encoding="utf-8"))
    assert result.validation_error is not None
    assert attempt == {"scenario_id": "S0008", "maneuvers": [], "source": MODEL_ID}
    assert record["validation_error"] == result.validation_error
    assert record["truncated"] is False
    assert record["raw_response"]["tool_use"]["input"]["maneuvers"][0]["action"]["type"] == "LINE_SWITCH"


def test_max_tokens_stop_reason_is_recorded_as_truncated_not_validation_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from restorebench.scoring.run_benchmark import run_one

    _mock_bedrock(monkeypatch, _bedrock_response(None, stop_reason="max_tokens"))

    result = run_one("S0008", MODEL_ID, repeat_index=0, out_dir=tmp_path)

    attempt = json.loads(result.attempt_path.read_text(encoding="utf-8"))
    record = json.loads(result.record_path.read_text(encoding="utf-8"))
    assert result.validation_error is None
    assert attempt == {"scenario_id": "S0008", "maneuvers": [], "source": MODEL_ID}
    assert record["truncated"] is True
    assert record["validation_error"] is None
    assert record["raw_response"]["stop_reason"] == "max_tokens"


def test_resume_skips_existing_attempt_unless_force(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from restorebench.scoring.run_benchmark import attempt_path, run_one

    existing_path = attempt_path(tmp_path, MODEL_ID, "S0008", 0)
    existing_path.parent.mkdir(parents=True)
    existing_path.write_text('{"scenario_id":"S0008","maneuvers":[],"source":"cached"}\n', encoding="utf-8")
    client = _mock_bedrock(monkeypatch)

    skipped = run_one("S0008", MODEL_ID, repeat_index=0, out_dir=tmp_path)
    forced = run_one("S0008", MODEL_ID, repeat_index=0, out_dir=tmp_path, force=True)

    assert skipped.skipped is True
    assert forced.skipped is False
    assert client.converse.call_count == 1
    assert json.loads(existing_path.read_text(encoding="utf-8"))["source"] == MODEL_ID


def test_checkpoint_is_written_before_next_call(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from restorebench.scoring.run_benchmark import attempt_path, run_many

    client = Mock()

    def converse_side_effect(**_kwargs: Any) -> dict[str, Any]:
        if client.converse.call_count == 2:
            assert attempt_path(tmp_path, MODEL_ID, "S0008", 0).exists()
        return _bedrock_response({"maneuvers": [_maneuver()], "reconstruction_summary": "read the card"})

    client.converse.side_effect = converse_side_effect
    monkeypatch.setattr(providers, "_bedrock_client", lambda: client)

    results = run_many(["S0008", "S0014"], MODEL_ID, repeats=1, out_dir=tmp_path)

    assert [result.scenario_id for result in results] == ["S0008", "S0014"]
    assert client.converse.call_count == 2


def test_bedrock_call_uses_tool_config_thinking_and_temperature(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from restorebench.scoring.run_benchmark import run_one

    client = _mock_bedrock(monkeypatch)

    run_one("S0008", MODEL_ID, repeat_index=0, out_dir=tmp_path)

    kwargs = client.converse.call_args.kwargs
    assert kwargs["toolConfig"]["tools"][0]["toolSpec"]["inputSchema"]["json"] == ManeuverSequence.model_json_schema()
    assert kwargs["additionalModelRequestFields"] == {"reasoning_effort": "high"}
    assert kwargs["inferenceConfig"]["temperature"] == 1.0
    assert kwargs["inferenceConfig"]["maxTokens"] == 8192


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("RESTOREBENCH_LLM_INTEGRATION") != "1",
    reason="real Bedrock benchmark integration is opt-in and not part of pre-push",
)
@pytest.mark.parametrize("model_id", BENCHMARK_MODELS)
def test_live_bedrock_runner_returns_valid_tool_use_for_all_models_when_operator_opts_in(
    tmp_path: Path,
    model_id: str,
) -> None:
    from restorebench.scoring.run_benchmark import run_one

    try:
        result = run_one("S0001", model_id, repeat_index=0, out_dir=tmp_path, force=True)
    except LLMFailureError as exc:
        pytest.skip(f"Bedrock endpoint unreachable or unavailable: {exc}")

    record = json.loads(result.record_path.read_text(encoding="utf-8"))
    assert record["raw_response"]["tool_use"] is not None
    ManeuverSequence.model_validate(record["raw_response"]["tool_use"]["input"])
    assert record["reasoning_present"] is True
    assert record["raw_response"]["stop_reason"] != "max_tokens"


def test_max_token_headroom_is_generous_for_gpt_oss() -> None:
    from restorebench.scoring.run_benchmark import max_tokens_for_model

    assert max_tokens_for_model(MODEL_ID) == 8192
    assert max_tokens_for_model("openai.gpt-oss-120b-1:0") == 16384
