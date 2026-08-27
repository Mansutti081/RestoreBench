# ABOUTME: Repairs tool payloads whose nested objects arrived as JSON strings instead of objects.
# ABOUTME: Fixes an encoding, never a meaning: anything that will not parse is left to the validator.
from __future__ import annotations

import json
import types
import typing
from typing import Any

from pydantic import BaseModel


def coerce_nested_json(payload: Any, model: type[BaseModel]) -> Any:
    """Decode top-level fields that a model serialised as a JSON string.

    Some providers answer a structured-output tool call with the nested object encoded as a
    string — the content is right, the encoding is not. Left alone this validates as
    MALFORMED_OUTPUT, which is recorded as the model's result: on the E3 gate DeepSeek V3.2
    hit it on ten iterations out of ten and committed no maneuver at all, while the Claude
    family hit it eight times in the whole campaign and always recovered. The defect therefore
    penalises whichever model formats least reliably, which is the comparison the benchmark
    exists to make.

    Only fields whose annotation is a model, or a list of models, may be decoded. A string
    field is never parsed: `rationale` holding text that happens to look like JSON must stay
    the model's own words. Nested dictionaries are handed to pydantic untouched — recursing
    would reintroduce exactly that risk one level down.
    """
    if not isinstance(payload, dict):
        return payload

    repaired: dict[str, Any] = dict(payload)
    for name, field in model.model_fields.items():
        value = repaired.get(name)
        if not isinstance(value, str) or not _expects_model(field.annotation):
            continue
        try:
            decoded = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            # Not JSON at all. Leave it: the validator's error names the real problem, and a
            # silent drop here would turn a malformed answer into a missing field.
            continue
        if isinstance(decoded, (dict, list)):
            repaired[name] = decoded
    return repaired


def _expects_model(annotation: Any) -> bool:
    """True when the annotation admits a BaseModel, directly or inside a list or union."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return True

    origin = typing.get_origin(annotation)
    if origin is None:
        return False
    if origin in (typing.Union, types.UnionType, list, set, tuple, frozenset):
        return any(_expects_model(arg) for arg in typing.get_args(annotation))
    return False
