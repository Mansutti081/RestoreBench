# ABOUTME: Builds and admits pocket-independent operating-control profiles for registered networks.
# ABOUTME: Limits profile deviations to shunt steps and transformer taps without changing availability.
from __future__ import annotations

import copy
import hashlib
import json
from itertools import combinations
from typing import Any, Literal

import pandapower as pp
from pydantic import BaseModel, ConfigDict, Field

from restorebench.physics.actions import enumerate_legal_qv_actions
from restorebench.physics.feasibility import evaluate_solved_feasibility
from restorebench.physics.solver import solve_locked_probe
from restorebench.corpus.augment import augmented_base_fingerprint


OPERATING_PROFILE_POLICY_VERSION = "pocket-independent-operating-profiles-v1"


class OperatingProfilePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_simultaneous_deviations: int = Field(default=1, ge=0, le=2)
    max_profiles: int = Field(default=8, ge=1)
    policy_version: str = OPERATING_PROFILE_POLICY_VERSION


class ProfileModification(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    component_type: Literal["SHUNT", "TRAFO"]
    component_id: int
    field: Literal["step", "tap_pos"]
    base_value: int
    target_value: int


class OperatingProfileCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    profile_id: str
    state_hash: str
    modifications: tuple[ProfileModification, ...]
    net: Any


class OperatingProfileRejection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_id: str
    reason: str


class OperatingProfileSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    profiles: tuple[OperatingProfileCandidate, ...]
    rejections: tuple[OperatingProfileRejection, ...]
    policy_version: str


def generate_operating_profile_candidates(
    augmented_base: Any,
    *,
    policy: OperatingProfilePolicy,
    network_id: str = "case118",
) -> tuple[OperatingProfileCandidate, ...]:
    """Enumerate deterministic profile settings without any pocket or stress input."""
    atomic = _atomic_profile_modifications(augmented_base)
    modification_sets: list[tuple[ProfileModification, ...]] = [()]
    for size in range(1, policy.max_simultaneous_deviations + 1):
        for candidate in combinations(atomic, size):
            if _has_component_conflict(candidate):
                continue
            modification_sets.append(candidate)

    candidates: list[OperatingProfileCandidate] = []
    seen_states: set[str] = set()
    for modifications in modification_sets:
        net = copy.deepcopy(augmented_base)
        for modification in modifications:
            table = net.shunt if modification.component_type == "SHUNT" else net.trafo
            table.at[modification.component_id, modification.field] = modification.target_value
        pp.reset_results(net)
        state_hash = augmented_base_fingerprint(
            net,
            network_id=network_id,
            profile_policy_version=policy.policy_version,
        )
        if state_hash in seen_states:
            continue
        seen_states.add(state_hash)
        candidates.append(
            OperatingProfileCandidate(
                profile_id=_profile_id(modifications, policy),
                state_hash=state_hash,
                modifications=modifications,
                net=net,
            )
        )
    return tuple(candidates)


def admit_operating_profiles(
    candidates: tuple[OperatingProfileCandidate, ...],
    *,
    policy: OperatingProfilePolicy,
) -> OperatingProfileSelection:
    """Admit a bounded deterministic library of valid, action-bearing profiles."""
    profiles: list[OperatingProfileCandidate] = []
    rejections: list[OperatingProfileRejection] = []
    inventories: set[tuple[str, ...]] = set()

    for candidate in candidates:
        if len(profiles) >= policy.max_profiles:
            rejections.append(
                OperatingProfileRejection(
                    profile_id=candidate.profile_id,
                    reason="PROFILE_LIBRARY_CAP",
                )
            )
            continue

        probe = solve_locked_probe(candidate.net)
        if probe.status != "SOLVED":
            rejections.append(
                OperatingProfileRejection(
                    profile_id=candidate.profile_id,
                    reason="NO_SOLUTION",
                )
            )
            continue
        feasibility = evaluate_solved_feasibility(probe.solved_net)
        if not feasibility.feasible:
            codes = ",".join(sorted({reason.code for reason in feasibility.failure_reasons}))
            rejections.append(
                OperatingProfileRejection(
                    profile_id=candidate.profile_id,
                    reason=f"INFEASIBLE:{codes}",
                )
            )
            continue

        actions = enumerate_legal_qv_actions(
            candidate.net,
            q_context={
                item.gen_id: item.status
                for item in feasibility.generator_q_status
            },
        )
        if not actions:
            rejections.append(
                OperatingProfileRejection(
                    profile_id=candidate.profile_id,
                    reason="NO_LEGAL_QV_ACTIONS",
                )
            )
            continue
        inventory = tuple(
            action.model_dump_json()
            for action in actions
        )
        if inventory in inventories:
            rejections.append(
                OperatingProfileRejection(
                    profile_id=candidate.profile_id,
                    reason="DUPLICATE_CONTROL_INVENTORY",
                )
            )
            continue

        inventories.add(inventory)
        profiles.append(candidate)

    if not profiles:
        raise ValueError("operating-profile admission produced an empty library")
    return OperatingProfileSelection(
        profiles=tuple(profiles),
        rejections=tuple(rejections),
        policy_version=policy.policy_version,
    )


def _atomic_profile_modifications(net: Any) -> tuple[ProfileModification, ...]:
    modifications: list[ProfileModification] = []
    for shunt_id, row in net.shunt.sort_index().iterrows():
        if not bool(row.get("in_service", True)):
            continue
        step = int(row["step"])
        max_step = int(row["max_step"])
        if max_step != 1 or step not in {0, 1}:
            continue
        modifications.append(
            ProfileModification(
                component_type="SHUNT",
                component_id=int(shunt_id),
                field="step",
                base_value=step,
                target_value=1 - step,
            )
        )

    for trafo_id, row in net.trafo.sort_index().iterrows():
        if not bool(row.get("in_service", True)) or row["tap_pos"] != row["tap_pos"]:
            continue
        current = int(row["tap_pos"])
        lower = int(row["tap_min"])
        upper = int(row["tap_max"])
        for target in (current - 1, current + 1):
            if lower <= target <= upper:
                modifications.append(
                    ProfileModification(
                        component_type="TRAFO",
                        component_id=int(trafo_id),
                        field="tap_pos",
                        base_value=current,
                        target_value=target,
                    )
                )
    return tuple(modifications)


def _has_component_conflict(
    modifications: tuple[ProfileModification, ...],
) -> bool:
    keys = [
        (modification.component_type, modification.component_id)
        for modification in modifications
    ]
    return len(keys) != len(set(keys))


def _profile_id(
    modifications: tuple[ProfileModification, ...],
    policy: OperatingProfilePolicy,
) -> str:
    payload = {
        "modifications": [
            modification.model_dump(mode="json")
            for modification in modifications
        ],
        "policy_version": policy.policy_version,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return f"OP-{hashlib.sha256(encoded).hexdigest()[:16]}"
