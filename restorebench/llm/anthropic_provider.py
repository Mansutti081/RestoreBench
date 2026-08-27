# ABOUTME: Serves Claude models over the first-party Anthropic Messages API behind the LLM facade.
# ABOUTME: Translates the Converse-shaped blocks the agents build, and echoes assistant turns back.
from __future__ import annotations

import time
from threading import Lock
from typing import TYPE_CHECKING, Any

import anthropic

from restorebench.llm import models
from restorebench.schemas.errors import LLMFailureError

if TYPE_CHECKING:  # pragma: no cover - import cycle at runtime, types only
    from restorebench.llm.providers import ChatMessage, LLMResponse


# Haiku 4.5 predates adaptive thinking and needs an explicit budget below max_tokens. Opus 5 and
# Sonnet 5 reject budget_tokens with a 400 and decide their own depth.
THINKING_BUDGET_TOKENS = 1024

_CLIENT: anthropic.Anthropic | None = None
_CLIENT_LOCK = Lock()


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
    """Run one Messages API turn and return the shared LLMResponse."""
    from restorebench.llm.providers import LLMResponse, ToolUse

    system, turns = _split_system(messages)
    request: dict[str, Any] = {
        "model": model_id,
        "max_tokens": max_tokens,
        "messages": turns,
    }
    if system:
        request["system"] = system
    if model_id in models.ANTHROPIC_ACCEPTS_SAMPLING:
        # Opus 5 and Sonnet 5 reject every sampling parameter; sending one is a 400, not a nudge.
        request["temperature"] = temperature
    if thinking:
        request["thinking"] = _thinking_config(model_id, max_tokens)
    if tools is not None:
        request["tools"] = _tools(tools)

    start = time.monotonic()
    message = _client().messages.create(**request)
    latency_seconds = time.monotonic() - start

    if message.stop_reason == "max_tokens" and raise_on_truncation:
        # The partial turn still comes back with a half-written tool call, which downstream reads
        # as an empty tool input and scores our own cap as a model failure.
        raise LLMFailureError(
            model_id=model_id,
            underlying_exception=f"response truncated at max_tokens ({max_tokens}); the tool call is incomplete",
        )
    if message.stop_reason == "refusal":
        raise LLMFailureError(
            model_id=model_id,
            underlying_exception="the model declined the request (stop_reason=refusal)",
        )

    blocks = [block.model_dump(mode="json", exclude_none=True) for block in message.content]
    tool_uses = tuple(
        ToolUse(toolUseId=block["id"], name=block["name"], input=block.get("input") or {})
        for block in blocks
        if block.get("type") == "tool_use"
    )
    text = "".join(block.get("text", "") for block in blocks if block.get("type") == "text")
    reasoning = "".join(block.get("thinking", "") for block in blocks if block.get("type") == "thinking")

    usage = message.usage
    tokens_in = int(usage.input_tokens or 0)
    tokens_out = int(usage.output_tokens or 0)
    # There is no provider-side total on this API: cache reads and writes are separate counters,
    # so the billed total is their sum and not tokens_in + tokens_out.
    tokens_total = (
        tokens_in
        + tokens_out
        + int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
        + int(getattr(usage, "cache_read_input_tokens", 0) or 0)
    )

    return LLMResponse(
        text=text,
        model_id=model_id,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        tokens_total=tokens_total,
        latency_seconds=latency_seconds,
        raw={"stop_reason": message.stop_reason, "reasoning": reasoning or None},
        tool_uses=tool_uses,
        # Stored in the provider's own shape: it is only ever echoed back to this provider, and
        # thinking blocks must return unchanged or the next turn is rejected.
        assistant_content=tuple(blocks),
    )


def _client() -> anthropic.Anthropic:
    global _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is None:
            _CLIENT = anthropic.Anthropic(timeout=600.0, max_retries=3)
        return _CLIENT


def _thinking_config(model_id: str, max_tokens: int) -> dict[str, Any]:
    if model_id == models.ANTHROPIC_HAIKU_4_5:
        return {"type": "enabled", "budget_tokens": min(THINKING_BUDGET_TOKENS, max_tokens - 1)}
    return {"type": "adaptive"}


def _split_system(messages: list["ChatMessage"]) -> tuple[str, list[dict[str, Any]]]:
    """Lift system turns into the top-level field this API expects."""
    system_parts: list[str] = []
    turns: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "system":
            system_parts.append(message.content if isinstance(message.content, str) else str(message.content))
            continue
        turns.append({"role": message.role, "content": _content(message.content)})
    return "\n\n".join(system_parts), turns


def _content(content: Any) -> Any:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return [_block(item) for item in content]


def _block(block: Any) -> dict[str, Any]:
    """Translate one agent-built Converse block, or pass through a block we produced."""
    if not isinstance(block, dict):
        return block
    if "type" in block:
        # Already in this API's shape: an assistant turn we are echoing back.
        return block
    if "text" in block:
        return {"type": "text", "text": block["text"]}
    if "toolResult" in block:
        result = block["toolResult"]
        return {
            "type": "tool_result",
            "tool_use_id": result["toolUseId"],
            "content": [_result_item(item) for item in result.get("content", [])],
            "is_error": result.get("status") == "error",
        }
    if "toolUse" in block:
        use = block["toolUse"]
        return {"type": "tool_use", "id": use["toolUseId"], "name": use["name"], "input": use.get("input") or {}}
    return block


def _result_item(item: Any) -> dict[str, Any]:
    if isinstance(item, dict) and "json" in item:
        import json

        return {"type": "text", "text": json.dumps(item["json"], sort_keys=True)}
    if isinstance(item, dict) and "text" in item:
        return {"type": "text", "text": item["text"]}
    return {"type": "text", "text": str(item)}


def _tools(tools: dict[str, Any]) -> list[dict[str, Any]]:
    """Translate the Converse toolConfig the agents build into this API's tool list."""
    specs = []
    for entry in tools.get("tools", []):
        spec = entry.get("toolSpec", entry)
        specs.append(
            {
                "name": spec["name"],
                "description": spec.get("description", ""),
                "input_schema": spec.get("inputSchema", {}).get("json", spec.get("input_schema", {})),
            }
        )
    return specs
