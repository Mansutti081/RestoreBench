# ABOUTME: Serves OpenAI models over the Responses API behind the LLM facade.
# ABOUTME: Translates the Converse-shaped blocks the agents build, and echoes output items back.
from __future__ import annotations

import json
import time
from threading import Lock
from typing import TYPE_CHECKING, Any

import openai

from restorebench.llm import models
from restorebench.schemas.errors import LLMFailureError

if TYPE_CHECKING:  # pragma: no cover - import cycle at runtime, types only
    from restorebench.llm.providers import ChatMessage, LLMResponse


_CLIENT: openai.OpenAI | None = None
_CLIENT_LOCK = Lock()

# The Responses API is the only endpoint where these models combine function tools with
# reasoning: Chat Completions rejects the pair with a 400 (verified live on this family).
# Reasoning is opt-in per model id; unlisted ids ignore the thinking flag instead of sending a
# field the model may reject.
REASONING: dict[str, dict[str, Any]] = {
    models.OPENAI_SOL: {"effort": "high"},
}

# Reasoning-first models reject sampling parameters; temperature is only ever sent for ids known
# to accept it (none of the current suite does).
ACCEPTS_SAMPLING: frozenset[str] = frozenset()


def call(
    model_id: str,
    messages: list["ChatMessage"],
    *,
    temperature: float,
    max_tokens: int,
    thinking: bool,
    tools: dict[str, Any] | None,
    raise_on_truncation: bool = True,
) -> "LLMResponse":
    """Run one Responses API turn and return the shared LLMResponse."""
    from restorebench.llm.providers import LLMResponse, ToolUse

    instructions, items = _to_input(messages)
    request: dict[str, Any] = {
        "model": model_id,
        "input": items,
        "max_output_tokens": max_tokens,
    }
    if instructions:
        request["instructions"] = instructions
    if model_id in ACCEPTS_SAMPLING:
        request["temperature"] = temperature
    if thinking and model_id in REASONING:
        request["reasoning"] = REASONING[model_id]
    if tools is not None:
        request["tools"] = _tools(tools)

    start = time.monotonic()
    try:
        response = _client().responses.create(**request)
    except openai.OpenAIError as exc:
        # After the SDK's own retries: surface as the shared failure type so the orchestrator
        # records LLM_FAILURE instead of crashing the sweep.
        raise LLMFailureError(model_id=model_id, underlying_exception=repr(exc)) from exc
    latency_seconds = time.monotonic() - start

    incomplete = getattr(response, "incomplete_details", None)
    if (
        response.status == "incomplete"
        and getattr(incomplete, "reason", None) == "max_output_tokens"
        and raise_on_truncation
    ):
        raise LLMFailureError(
            model_id=model_id,
            underlying_exception=f"response truncated at max_output_tokens ({max_tokens}); the tool call is incomplete",
        )

    output_items = [item.model_dump(mode="json", exclude_none=True) for item in response.output]
    text_parts: list[str] = []
    tool_uses: list[ToolUse] = []
    for item in output_items:
        if item.get("type") == "function_call":
            tool_uses.append(
                ToolUse(
                    toolUseId=item["call_id"],
                    name=item["name"],
                    input=_parse_arguments(model_id, item.get("arguments")),
                )
            )
        elif item.get("type") == "message":
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    text_parts.append(part.get("text", ""))
                elif part.get("type") == "refusal":
                    raise LLMFailureError(
                        model_id=model_id,
                        underlying_exception=f"the model declined the request: {part.get('refusal', '')}",
                    )

    usage = response.usage
    tokens_in = int(getattr(usage, "input_tokens", 0) or 0)
    tokens_out = int(getattr(usage, "output_tokens", 0) or 0)
    tokens_total = int(getattr(usage, "total_tokens", 0) or 0) or (tokens_in + tokens_out)

    return LLMResponse(
        text="".join(text_parts),
        model_id=model_id,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        tokens_total=tokens_total,
        latency_seconds=latency_seconds,
        raw={"status": response.status},
        tool_uses=tuple(tool_uses),
        # Stored in this API's own item shape and echoed back verbatim next turn: reasoning
        # items must accompany the function_call items they preceded, or the model rejects the
        # continuation. The translator passes any block that already has a "type" straight
        # through, exactly like the Anthropic transport does with its blocks.
        assistant_content=tuple(output_items),
    )


def _client() -> openai.OpenAI:
    global _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is None:
            _CLIENT = openai.OpenAI(timeout=600.0, max_retries=3)
        return _CLIENT


def _parse_arguments(model_id: str, arguments: str | None) -> dict[str, Any]:
    if not arguments:
        return {}
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError as exc:
        raise LLMFailureError(
            model_id=model_id,
            underlying_exception=f"tool call arguments are not valid JSON: {exc}",
        ) from exc
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def _to_input(messages: list["ChatMessage"]) -> tuple[str, list[dict[str, Any]]]:
    """Translate the agent-built conversation into Responses API input items.

    System turns lift into `instructions`. Echoed assistant turns are this API's own output
    items (reasoning / message / function_call) and pass through verbatim. Converse-shaped
    toolResult blocks become function_call_output items; any surrounding text follows as a
    user message so nothing the agents wrote is dropped.
    """
    instruction_parts: list[str] = []
    items: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "system":
            instruction_parts.append(
                message.content if isinstance(message.content, str) else str(message.content)
            )
            continue
        if isinstance(message.content, str):
            items.append({"role": message.role, "content": message.content})
            continue
        text_parts: list[str] = []
        for block in message.content:
            if not isinstance(block, dict):
                text_parts.append(str(block))
            elif "type" in block:
                # Already in this API's shape: an output item we are echoing back.
                items.append(block)
            elif "toolResult" in block:
                result = block["toolResult"]
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": result["toolUseId"],
                        "output": _result_text(result.get("content", [])),
                    }
                )
            elif "toolUse" in block:
                use = block["toolUse"]
                items.append(
                    {
                        "type": "function_call",
                        "call_id": use["toolUseId"],
                        "name": use["name"],
                        "arguments": json.dumps(use.get("input") or {}, sort_keys=True),
                    }
                )
            elif "text" in block:
                text_parts.append(block["text"])
        if text_parts:
            items.append({"role": message.role, "content": "\n\n".join(text_parts)})
    return "\n\n".join(instruction_parts), items


def _result_text(items: list[Any]) -> str:
    parts = []
    for item in items:
        if isinstance(item, dict) and "json" in item:
            parts.append(json.dumps(item["json"], sort_keys=True))
        elif isinstance(item, dict) and "text" in item:
            parts.append(item["text"])
        else:
            parts.append(str(item))
    return "\n".join(parts)


def _tools(tools: dict[str, Any]) -> list[dict[str, Any]]:
    """Translate the Converse toolConfig the agents build into this API's flat function specs."""
    specs = []
    for entry in tools.get("tools", []):
        spec = entry.get("toolSpec", entry)
        specs.append(
            {
                "type": "function",
                "name": spec["name"],
                "description": spec.get("description", ""),
                "parameters": spec.get("inputSchema", {}).get("json", spec.get("parameters", {})),
            }
        )
    return specs
