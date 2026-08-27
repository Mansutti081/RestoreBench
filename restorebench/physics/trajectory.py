# ABOUTME: Reconstructs private curation and public diagnostic load trajectories from immutable anchors.
# ABOUTME: Keeps the two reference contracts distinct while sharing bounded active-power scheduling.
from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from typing import Any

from restorebench.physics.active_balance import schedule_active_power
from restorebench.physics.policies import (
    CURATION_TRAJECTORY_POLICY_VERSION,
    DIAGNOSTIC_TRAJECTORY_POLICY_VERSION,
    TRAJECTORY_DECIMAL_PLACES,
)
from restorebench.schemas.physics import (
    ActiveBalancePolicy,
    CurationLoadWeight,
    CurationTrajectoryRequest,
    DiagnosticTrajectoryRequest,
    TrajectoryState,
)


def build_curation_state(
    profile_net: Any,
    *,
    stress: float,
    ordered_load_weights: Sequence[CurationLoadWeight],
    active_policy: ActiveBalancePolicy,
) -> TrajectoryState:
    """Build one profile-anchored pocket-stress state from immutable load arrays."""
    request = CurationTrajectoryRequest(
        stress=stress,
        ordered_load_weights=tuple(ordered_load_weights),
    )
    expected_ids = tuple(int(load_id) for load_id in profile_net.load.index)
    received_ids = tuple(item.load_id for item in request.ordered_load_weights)
    if received_ids != expected_ids:
        raise ValueError(
            "ordered load IDs must exactly match the immutable profile load table: "
            f"expected={expected_ids}, received={received_ids}"
        )

    target = copy.deepcopy(profile_net)
    for item in request.ordered_load_weights:
        multiplier = 1.0 + request.stress * item.weight
        _set_scaled_load(target, profile_net, item.load_id, multiplier)

    requested_delta = _load_delta(profile_net, target)
    balance = schedule_active_power(
        profile_net,
        target,
        requested_load_delta_mw=requested_delta,
        policy=active_policy,
    )
    return TrajectoryState(
        net=balance.net,
        trajectory_type="CURATION",
        coordinate=request.stress,
        requested_load_delta_mw=requested_delta,
        active_balance=balance,
        trajectory_policy_version=CURATION_TRAJECTORY_POLICY_VERSION,
    )


def build_diagnostic_state(
    snapshot_net: Any,
    *,
    lambda_value: float,
    active_policy: ActiveBalancePolicy,
) -> TrajectoryState:
    """Build one snapshot-anchored uniform-scaling state without private curation inputs."""
    request = DiagnosticTrajectoryRequest(lambda_value=lambda_value)
    target = copy.deepcopy(snapshot_net)
    for load_id in snapshot_net.load.index:
        _set_scaled_load(target, snapshot_net, int(load_id), request.lambda_value)

    requested_delta = _load_delta(snapshot_net, target)
    balance = schedule_active_power(
        snapshot_net,
        target,
        requested_load_delta_mw=requested_delta,
        policy=active_policy,
    )
    return TrajectoryState(
        net=balance.net,
        trajectory_type="DIAGNOSTIC",
        coordinate=request.lambda_value,
        requested_load_delta_mw=requested_delta,
        active_balance=balance,
        trajectory_policy_version=DIAGNOSTIC_TRAJECTORY_POLICY_VERSION,
    )


def _set_scaled_load(target: Any, reference: Any, load_id: int, multiplier: float) -> None:
    if math.isclose(multiplier, 1.0, rel_tol=0.0, abs_tol=0.0):
        target.load.at[load_id, "p_mw"] = reference.load.at[load_id, "p_mw"]
        target.load.at[load_id, "q_mvar"] = reference.load.at[load_id, "q_mvar"]
        return
    target.load.at[load_id, "p_mw"] = round(
        float(reference.load.at[load_id, "p_mw"]) * multiplier,
        TRAJECTORY_DECIMAL_PLACES,
    )
    target.load.at[load_id, "q_mvar"] = round(
        float(reference.load.at[load_id, "q_mvar"]) * multiplier,
        TRAJECTORY_DECIMAL_PLACES,
    )


def _load_delta(reference: Any, target: Any) -> float:
    delta = math.fsum(
        float(target.load.at[load_id, "p_mw"]) - float(reference.load.at[load_id, "p_mw"])
        for load_id in reference.load.index
    )
    return round(delta, TRAJECTORY_DECIMAL_PLACES)
