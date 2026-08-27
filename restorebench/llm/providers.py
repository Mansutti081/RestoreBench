# ABOUTME: Provides a uniform chat-completion facade over Amazon Bedrock's Converse API.
# ABOUTME: Keeps provider SDK details out of agent and evaluation code.
from __future__ import annotations

import os
import time
from copy import deepcopy
from threading import Lock
from typing import Any, Literal

from botocore.exceptions import ClientError
from pydantic import BaseModel, ConfigDict, Field, model_validator

from restorebench.llm import models
from restorebench.schemas.errors import LLMFailureError

MAX_RETRIES = 2
BASE_RETRY_DELAY_SECONDS = 0.5
DEFAULT_REGION = "us-east-1"
# Per-call read budget: the pre-Bedrock benchmark client pinned 180s; botocore defaults to 60s.
LLM_READ_TIMEOUT_SECONDS = 180
_BEDROCK_CLIENT: Any | None = None
_BEDROCK_CLIENT_LOCK = Lock()

# Haiku 4.5 predates adaptive thinking and needs an explicit budget. Bedrock
# requires the budget to stay below max_tokens.
THINKING_BUDGET_TOKENS = 1024

# Reasoning tokens are spent inside maxTokens, so the cap has to cover the thinking budget plus
# the tool call the model still has to write. At 2048 the guillotine fell mid-toolUse and Bedrock
# returned the partial message: an empty tool input that reads as a model failure. Opus as analyst
# then measured ~7000-8000 output tokens per call (one hit 7916), so the cap sits well clear of
# that too. Output tokens are billed as generated, not as capped, so headroom is free.
DEFAULT_MAX_TOKENS = 16384

# Bedrock exposes reasoning through a different field per model provider, and
# silently ignores fields it does not recognise. Every entry here was verified
# against a live Converse call. Only "high" enables reasoning on Qwen; "medium"
# is accepted and does nothing.
_REASONING_FIELDS: dict[str, dict[str, Any]] = {
    models.DEEPSEEK_V3_2: {"reasoning_effort": "high"},
    models.KIMI_K2_5: {"reasoning_effort": "high"},
    models.GLM_5: {"reasoning_effort": "high"},
    # Not in the current suite, kept because the mapping is verified and harmless. The Claude
    # pair served the de-risking cells before the family moved to the Anthropic API.
    models.OPUS_4_6: {"thinking": {"type": "adaptive"}},
    models.HAIKU_4_5: {"reasoning_config": {"type": "enabled", "budget_tokens": THINKING_BUDGET_TOKENS}},
    models.QWEN3_32B: {"reasoning_effort": "high"},
    models.GPT_OSS_120B: {"reasoning_effort": "high"},
}

_TRANSIENT_ERROR_CODES = frozenset(
    {
        "ThrottlingException",
        "ServiceUnavailableException",
        "InternalServerException",
        "ModelNotReadyException",
        "ModelTimeoutException",
    }
)


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str | list[dict[str, Any]]


class ToolUse(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    tool_use_id: str = Field(alias="toolUseId")
    name: str
    input: dict[str, Any]


class ToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    tool_use_id: str = Field(alias="toolUseId")
    content: list[dict[str, Any]]
    status: Literal["success", "error"]


class LLMResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    model_id: str
    tokens_in: int
    tokens_out: int
    # Bedrock's own totalTokens. Not always inputTokens + outputTokens: cache-read and
    # cache-write tokens are billed and counted here, so record what the provider says.
    # Reasoning/thinking tokens are NOT reported separately — they sit inside outputTokens.
    tokens_total: int = 0
    latency_seconds: float
    raw: dict[str, Any] = Field(default_factory=dict)
    tool_use: ToolUse | None = None
    tool_uses: tuple[ToolUse, ...] = ()
    assistant_content: tuple[dict[str, Any], ...] = ()

    @model_validator(mode="after")
    def keep_tool_use_compatibility(self) -> "LLMResponse":
        if self.tool_uses:
            self.tool_use = self.tool_uses[0]
        elif self.tool_use is not None:
            self.tool_uses = (self.tool_use,)
        return self


def llm_call(
    model_id: str,
    messages: list[ChatMessage],
    *,
    temperature: float = 1.0,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    thinking: bool = False,
    tools: dict[str, Any] | None = None,
    raise_on_truncation: bool = True,
) -> LLMResponse:
    """Call the model on whichever transport serves it, and return the shared response shape.

    Claude ids without a provider prefix are served by the first-party Anthropic Messages API;
    everything else goes to Bedrock Converse. Callers never choose: the id decides, so a run
    cannot silently take a different transport than its model registry says it did.

    When `thinking` is set, a first-turn response with no reasoning block is rejected
    for budget/effort-style models (guaranteed emission — a missing block means the
    reasoning field was mistyped or silently ignored). Adaptive-thinking models
    (Opus) are exempt: the model itself decides when to think, so a response without
    reasoning is legitimate there. Continuation turns after a toolResult are always
    tolerant.

    `raise_on_truncation` (default True) fails a max_tokens stop, so agents never mistake a
    cut-off tool call for a bad answer. The standalone benchmark harness records truncation as
    its own outcome and opts out with False to receive the partial response.
    """
    if not messages:
        raise LLMFailureError(model_id=model_id, underlying_exception="at least one message is required")
    if models.provider_for(model_id) == "openai":
        from restorebench.llm import openai_provider

        return openai_provider.call(
            model_id,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking=thinking,
            tools=tools,
            raise_on_truncation=raise_on_truncation,
        )
    if models.provider_for(model_id) == "anthropic":
        from restorebench.llm import anthropic_provider

        return anthropic_provider.call(
            model_id,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking=thinking,
            tools=tools,
            raise_on_truncation=raise_on_truncation,
        )
    return _call_with_retries(
        model_id,
        lambda: _converse(
            model_id,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking=thinking,
            tools=tools,
            raise_on_truncation=raise_on_truncation,
        ),
    )


def _converse(
    model_id: str,
    messages: list[ChatMessage],
    *,
    temperature: float,
    max_tokens: int,
    thinking: bool,
    tools: dict[str, Any] | None,
    raise_on_truncation: bool = True,
) -> LLMResponse:
    system, turns = _split_system_messages(messages)
    request: dict[str, Any] = {
        "modelId": model_id,
        "messages": turns,
        "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature},
    }
    if system:
        request["system"] = system
    if thinking:
        request["additionalModelRequestFields"] = _reasoning_fields(model_id)
    if tools is not None:
        request["toolConfig"] = tools

    start = time.monotonic()
    response = _bedrock_client().converse(**request)
    latency_seconds = time.monotonic() - start

    blocks = response["output"]["message"]["content"]
    stop_reason = response.get("stopReason")
    if stop_reason == "max_tokens" and raise_on_truncation:
        # Bedrock still returns the partial message: the toolUse block is cut off or missing, so
        # the caller would see an empty tool input and score our own output cap as a model failure.
        raise LLMFailureError(
            model_id=model_id,
            underlying_exception=f"response truncated at max_tokens ({max_tokens}); the tool call is incomplete",
        )
    reasoning = _reasoning_text(blocks)
    tool_uses = _tool_uses(blocks)
    if thinking and reasoning is None and not _is_continuation(messages) and not _reasoning_is_optional(model_id):
        raise LLMFailureError(
            model_id=model_id,
            underlying_exception="thinking was requested but the response carried no reasoning block",
        )

    usage = response.get("usage", {})
    raw: dict[str, Any] = {
        "usage": dict(usage),
        "stop_reason": stop_reason,
        "assistant_content": deepcopy(blocks),
    }
    if reasoning is not None:
        raw["reasoning"] = reasoning
    if tool_uses:
        raw["tool_use"] = tool_uses[0].model_dump(mode="json", by_alias=True)
        raw["tool_uses"] = [tool_use.model_dump(mode="json", by_alias=True) for tool_use in tool_uses]

    tokens_in = int(usage.get("inputTokens", 0))
    tokens_out = int(usage.get("outputTokens", 0))
    return LLMResponse(
        text=_response_text(blocks),
        model_id=model_id,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        tokens_total=int(usage.get("totalTokens", tokens_in + tokens_out)),
        latency_seconds=latency_seconds,
        raw=raw,
        tool_uses=tool_uses,
        assistant_content=tuple(deepcopy(blocks)),
    )


def _reasoning_fields(model_id: str) -> dict[str, Any]:
    try:
        return _REASONING_FIELDS[model_id]
    except KeyError:
        raise LLMFailureError(
            model_id=model_id,
            underlying_exception=(
                f"unknown model {model_id!r}: no reasoning field is known for it, and Bedrock would "
                "ignore a guessed one without error"
            ),
        ) from None


def _reasoning_is_optional(model_id: str) -> bool:
    # Adaptive thinking lets the model itself decide when to think, so a response without
    # a reasoning block is legitimate — raising would turn every such run into LLM_FAILURE.
    # Budget/effort-style fields guarantee emission; for those a missing block still means
    # a mistyped or silently-ignored field and the strict check stays (the silent-ignore trap).
    fields = _REASONING_FIELDS.get(model_id, {})
    return fields.get("thinking", {}).get("type") == "adaptive"


def _split_system_messages(messages: list[ChatMessage]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    system = [_content_blocks(message)[0] for message in messages if message.role == "system"]
    turns = [
        {"role": message.role, "content": _content_blocks(message)}
        for message in messages
        if message.role != "system"
    ]
    return system, turns


def _content_blocks(message: ChatMessage) -> list[dict[str, Any]]:
    if isinstance(message.content, str):
        return [{"text": message.content}]
    return deepcopy(message.content)


def _response_text(blocks: list[dict[str, Any]]) -> str:
    return "".join(block["text"] for block in blocks if "text" in block)


def _reasoning_text(blocks: list[dict[str, Any]]) -> str | None:
    for block in blocks:
        reasoning = block.get("reasoningContent")
        if reasoning:
            return str(reasoning.get("reasoningText", {}).get("text", ""))
    return None


def _tool_uses(blocks: list[dict[str, Any]]) -> tuple[ToolUse, ...]:
    return tuple(ToolUse.model_validate(block["toolUse"]) for block in blocks if block.get("toolUse"))


def _is_continuation(messages: list[ChatMessage]) -> bool:
    for message in messages:
        if isinstance(message.content, list) and any("toolResult" in block for block in message.content):
            return True
    return False


def _call_with_retries(model_id: str, call):
    attempts_used = 0
    while True:
        try:
            return call()
        except LLMFailureError:
            raise
        except Exception as exc:
            if attempts_used >= MAX_RETRIES or not _is_transient_error(exc):
                raise LLMFailureError(model_id=model_id, underlying_exception=str(exc)) from exc
            _sleep(BASE_RETRY_DELAY_SECONDS * (2**attempts_used))
            attempts_used += 1


def _is_transient_error(exc: Exception) -> bool:
    if isinstance(exc, ClientError):
        if exc.response.get("Error", {}).get("Code") in _TRANSIENT_ERROR_CODES:
            return True
        status_code = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if status_code is not None:
            if status_code == 429:
                return True
            if 500 <= int(status_code) < 600:
                return True
            return False
    haystack = f"{exc.__class__.__name__} {exc}".lower()
    return any(marker in haystack for marker in ("timeout", "temporar", "rate limit", "rate_limit", "unavailable"))


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def _bedrock_client():
    global _BEDROCK_CLIENT
    if _BEDROCK_CLIENT is not None:
        return _BEDROCK_CLIENT

    import boto3
    from botocore.config import Config

    with _BEDROCK_CLIENT_LOCK:
        if _BEDROCK_CLIENT is None:
            # Explicit read timeout: botocore's 60s default silently regressed the 180s
            # budget the pre-Bedrock benchmark client pinned; long generations need it.
            _BEDROCK_CLIENT = boto3.client(
                "bedrock-runtime",
                region_name=os.environ.get("AWS_REGION", DEFAULT_REGION),
                config=Config(connect_timeout=10, read_timeout=LLM_READ_TIMEOUT_SECONDS),
            )
        return _BEDROCK_CLIENT
