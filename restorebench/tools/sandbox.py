# ABOUTME: Provides in-process sandbox handles over deepcopy pandapower networks.
# ABOUTME: Applies one bounds-checked maneuver at a time without mutating caller nets.
from __future__ import annotations

import copy
from typing import Any, TypeVar
from uuid import UUID, uuid4

from restorebench.physics.actions import apply_qv_action
from restorebench.schemas import (
    Maneuver,
    SandboxNet,
    ToolFailureError,
)


NetT = TypeVar("NetT")

_SANDBOX_REGISTRY: dict[UUID, Any] = {}
_PROMOTED_SANDBOX_ID: UUID | None = None


def create_sandbox(net: NetT, scenario_request_id: UUID | None = None) -> SandboxNet:
    sandbox = SandboxNet(
        sandbox_id=uuid4(),
        scenario_request_id=scenario_request_id or uuid4(),
    )
    _SANDBOX_REGISTRY[sandbox.sandbox_id] = copy.deepcopy(net)
    return sandbox


def resolve_net(handle_or_net: SandboxNet | NetT) -> Any | NetT:
    if isinstance(handle_or_net, SandboxNet):
        return _registered_net(handle_or_net, "SandboxServer.resolve_net")
    return handle_or_net


def apply_maneuver(
    sandbox: SandboxNet,
    maneuver: Maneuver,
    *,
    saturated_gens: frozenset[int],
) -> SandboxNet:
    net = _registered_net(sandbox, "SandboxServer.apply_maneuver")
    q_context = {gen_id: "Q_LIMITED_UPPER" for gen_id in saturated_gens}
    _SANDBOX_REGISTRY[sandbox.sandbox_id] = apply_qv_action(net, maneuver.action, q_context)

    return sandbox


def promote_sandbox(sandbox: SandboxNet) -> None:
    global _PROMOTED_SANDBOX_ID
    _registered_net(sandbox, "SandboxServer.promote_sandbox")
    _PROMOTED_SANDBOX_ID = sandbox.sandbox_id


def discard_sandbox(sandbox: SandboxNet) -> None:
    global _PROMOTED_SANDBOX_ID
    _SANDBOX_REGISTRY.pop(sandbox.sandbox_id, None)
    if _PROMOTED_SANDBOX_ID == sandbox.sandbox_id:
        _PROMOTED_SANDBOX_ID = None


def _registered_net(sandbox: SandboxNet, tool_name: str) -> Any:
    try:
        return _SANDBOX_REGISTRY[sandbox.sandbox_id]
    except KeyError as exc:
        raise ToolFailureError(tool_name, f"unknown sandbox_id {sandbox.sandbox_id}") from exc
