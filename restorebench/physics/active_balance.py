# ABOUTME: Schedules active generation with deterministic bounded participation and water-filling.
# ABOUTME: Reconstructs dispatch from an immutable reference and returns an isolated target state.
from __future__ import annotations

import copy
import math
from typing import Any

from restorebench.physics.policies import ACTIVE_BALANCE_TOLERANCE_MW
from restorebench.schemas.physics import (
    ActiveBalancePolicy,
    ActiveBalanceResult,
    GeneratorDispatch,
    GeneratorParticipation,
)


def build_active_policy(reference_net: Any) -> ActiveBalancePolicy:
    """Build frozen base-P-proportional participation over eligible generators."""
    participation: list[GeneratorParticipation] = []
    for gen_id, row in reference_net.gen.sort_index().iterrows():
        if not bool(row.get("in_service", True)):
            continue
        p_mw = float(row["p_mw"])
        min_p_mw = float(row["min_p_mw"])
        max_p_mw = float(row["max_p_mw"])
        if (
            not all(math.isfinite(value) for value in (p_mw, min_p_mw, max_p_mw))
            or max_p_mw <= 0.0
            or max_p_mw <= min_p_mw
        ):
            continue
        if p_mw <= 0.0:
            raise ValueError(f"eligible generator {int(gen_id)} needs positive base P for participation")
        participation.append(GeneratorParticipation(gen_id=int(gen_id), factor=p_mw))
    if not participation:
        raise ValueError("network has no eligible active-power participants")
    return ActiveBalancePolicy(participation=tuple(participation))


def schedule_active_power(
    reference_net: Any,
    target_net: Any,
    *,
    requested_load_delta_mw: float,
    policy: ActiveBalancePolicy,
) -> ActiveBalanceResult:
    """Schedule target generator P directly from a reference without mutating either caller."""
    _validate_matching_tables(reference_net, target_net)
    observed_delta = math.fsum(
        float(target_net.load.at[load_id, "p_mw"]) - float(reference_net.load.at[load_id, "p_mw"])
        for load_id in reference_net.load.index
    )
    if not math.isclose(
        observed_delta,
        requested_load_delta_mw,
        rel_tol=0.0,
        abs_tol=ACTIVE_BALANCE_TOLERANCE_MW,
    ):
        raise ValueError(
            "requested load delta does not match target/reference load tables: "
            f"requested={requested_load_delta_mw:.12g}, observed={observed_delta:.12g}"
        )

    eligible_ids = _eligible_generator_ids(reference_net)
    policy_ids = tuple(item.gen_id for item in policy.participation)
    if policy_ids != eligible_ids:
        raise ValueError(
            "participation IDs must exactly match eligible reference generators: "
            f"expected={eligible_ids}, received={policy_ids}"
        )

    scheduled = copy.deepcopy(target_net)
    for gen_id in reference_net.gen.index:
        scheduled.gen.at[gen_id, "p_mw"] = float(reference_net.gen.at[gen_id, "p_mw"])

    factors = {item.gen_id: float(item.factor) for item in policy.participation}
    allocations = {gen_id: 0.0 for gen_id in eligible_ids}
    remaining = float(requested_load_delta_mw)
    active_ids = list(eligible_ids)

    while active_ids and abs(remaining) > ACTIVE_BALANCE_TOLERANCE_MW:
        factor_sum = math.fsum(factors[gen_id] for gen_id in active_ids)
        proposed = {gen_id: remaining * factors[gen_id] / factor_sum for gen_id in active_ids}
        saturated_ids = [
            gen_id
            for gen_id in active_ids
            if abs(proposed[gen_id]) > _remaining_headroom(reference_net, gen_id, allocations[gen_id], remaining)
        ]
        if not saturated_ids:
            for gen_id in active_ids:
                allocations[gen_id] += proposed[gen_id]
            remaining = 0.0
            break

        allocated_this_round = 0.0
        for gen_id in saturated_ids:
            headroom = _remaining_headroom(reference_net, gen_id, allocations[gen_id], remaining)
            bounded_delta = math.copysign(headroom, remaining)
            allocations[gen_id] += bounded_delta
            allocated_this_round += bounded_delta
        remaining -= allocated_this_round
        saturated_set = set(saturated_ids)
        active_ids = [gen_id for gen_id in active_ids if gen_id not in saturated_set]

    if abs(remaining) <= ACTIVE_BALANCE_TOLERANCE_MW:
        remaining = 0.0

    dispatch: list[GeneratorDispatch] = []
    for item in policy.participation:
        gen_id = item.gen_id
        reference_p = float(reference_net.gen.at[gen_id, "p_mw"])
        target_p = reference_p + allocations[gen_id]
        min_p = float(reference_net.gen.at[gen_id, "min_p_mw"])
        max_p = float(reference_net.gen.at[gen_id, "max_p_mw"])
        if math.isclose(target_p, min_p, rel_tol=0.0, abs_tol=ACTIVE_BALANCE_TOLERANCE_MW):
            target_p = min_p
        elif math.isclose(target_p, max_p, rel_tol=0.0, abs_tol=ACTIVE_BALANCE_TOLERANCE_MW):
            target_p = max_p
        scheduled.gen.at[gen_id, "p_mw"] = target_p
        dispatch.append(
            GeneratorDispatch(
                gen_id=gen_id,
                reference_p_mw=reference_p,
                scheduled_p_mw=target_p,
                delta_p_mw=target_p - reference_p,
                participation_factor=item.factor,
                at_active_bound=math.isclose(target_p, min_p, abs_tol=ACTIVE_BALANCE_TOLERANCE_MW)
                or math.isclose(target_p, max_p, abs_tol=ACTIVE_BALANCE_TOLERANCE_MW),
            )
        )

    allocated_delta = math.fsum(item.delta_p_mw for item in dispatch)
    unallocated_delta = float(requested_load_delta_mw) - allocated_delta
    if abs(unallocated_delta) <= ACTIVE_BALANCE_TOLERANCE_MW:
        unallocated_delta = 0.0
    status = "SCHEDULED" if remaining == 0.0 else "ACTIVE_HEADROOM_EXHAUSTED"
    return ActiveBalanceResult(
        net=scheduled,
        status=status,
        requested_delta_mw=float(requested_load_delta_mw),
        allocated_delta_mw=allocated_delta,
        unallocated_delta_mw=unallocated_delta,
        generator_dispatch=tuple(dispatch),
        policy_version=policy.policy_version,
    )


def _validate_matching_tables(reference_net: Any, target_net: Any) -> None:
    reference_load_ids = tuple(int(load_id) for load_id in reference_net.load.index)
    target_load_ids = tuple(int(load_id) for load_id in target_net.load.index)
    if reference_load_ids != target_load_ids:
        raise ValueError("reference and target load IDs must match exactly")

    reference_gen_ids = tuple(int(gen_id) for gen_id in reference_net.gen.index)
    target_gen_ids = tuple(int(gen_id) for gen_id in target_net.gen.index)
    if reference_gen_ids != target_gen_ids:
        raise ValueError("reference and target generator IDs must match exactly")


def _eligible_generator_ids(reference_net: Any) -> tuple[int, ...]:
    eligible: list[int] = []
    for gen_id, row in reference_net.gen.sort_index().iterrows():
        if not bool(row.get("in_service", True)):
            continue
        current_p = float(row["p_mw"])
        min_p = float(row["min_p_mw"])
        max_p = float(row["max_p_mw"])
        if not all(math.isfinite(value) for value in (current_p, min_p, max_p)):
            continue
        if max_p <= 0.0 or max_p <= min_p:
            continue
        if current_p < min_p - ACTIVE_BALANCE_TOLERANCE_MW or current_p > max_p + ACTIVE_BALANCE_TOLERANCE_MW:
            raise ValueError(f"reference generator {int(gen_id)} is outside declared P bounds")
        eligible.append(int(gen_id))
    return tuple(eligible)


def _remaining_headroom(
    reference_net: Any,
    gen_id: int,
    allocated_delta: float,
    requested_direction: float,
) -> float:
    reference_p = float(reference_net.gen.at[gen_id, "p_mw"])
    scheduled_p = reference_p + allocated_delta
    if requested_direction > 0:
        bound = float(reference_net.gen.at[gen_id, "max_p_mw"])
        return max(0.0, bound - scheduled_p)
    bound = float(reference_net.gen.at[gen_id, "min_p_mw"])
    return max(0.0, scheduled_p - bound)
