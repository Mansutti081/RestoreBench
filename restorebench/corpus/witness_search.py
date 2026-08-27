# ABOUTME: Searches private Q-V-only resolution witnesses without consulting component ranking.
# ABOUTME: Uses exhaustive shallow BFS, deterministic beam extension, and state fingerprints.
from __future__ import annotations

import math
from concurrent.futures import Executor, Future
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from restorebench.physics.actions import (
    GeneratorQStatus,
    apply_qv_action,
    enumerate_legal_qv_actions,
)
from restorebench.physics.feasibility import (
    evaluate_solved_feasibility,
    satisfies_non_voltage_constraints,
)
from restorebench.physics.fingerprint import state_fingerprint
from restorebench.physics.solver import solve_locked_probe
from restorebench.schemas.actions import Action
from restorebench.schemas.dataset import CurationWitness
from restorebench.schemas.power_flow import PowerFlowResult
from restorebench.physics.policies import RETREAT_COORDINATES, RETREAT_RESOLUTION
from restorebench.physics.q_saturation import q_saturation_context
from restorebench.physics.retreat import retreat_evidence


WITNESS_SEARCH_POLICY_VERSION = "deterministic-bfs-beam-qv-v4"


class WitnessSearchPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    maneuver_budget: int = Field(default=10, ge=1, le=10)
    bfs_max_depth: int = Field(default=3, ge=1, le=10)
    frontier_budget: int = Field(default=500, ge=1)
    beam_width: int = Field(default=100, ge=1)
    beam_branching_width: int = Field(default=12, ge=1)
    # The retreat contract belongs to the shared physics, not to this search: the runtime reads
    # its Q context at the shared coordinates and cannot follow a custom grid, so a divergent
    # policy would silently restore the runtime/witness split these constants exist to prevent.
    diagnostic_coordinates: tuple[float, ...] = RETREAT_COORDINATES
    diagnostic_resolution: float = Field(
        default=RETREAT_RESOLUTION,
        gt=0.0,
        allow_inf_nan=False,
    )
    policy_version: str = WITNESS_SEARCH_POLICY_VERSION

    @model_validator(mode="after")
    def diagnostic_grid_is_valid(self) -> "WitnessSearchPolicy":
        coordinates = self.diagnostic_coordinates
        if (
            len(coordinates) < 2
            or coordinates[-1] != 1.0
            or not all(math.isfinite(coordinate) and coordinate > 0.0 for coordinate in coordinates)
            or any(
                right <= left
                for left, right in zip(
                    coordinates,
                    coordinates[1:],
                    strict=False,
                )
            )
        ):
            raise ValueError("diagnostic coordinates must be finite, positive, ascending, and end at 1")
        if coordinates != RETREAT_COORDINATES or self.diagnostic_resolution != RETREAT_RESOLUTION:
            raise ValueError(
                "witness retreat must use the shared physics retreat policy; the runtime reads "
                "its Q context there and cannot follow a custom grid"
            )
        return self


class StateEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["TERMINAL", "DIVERGED", "INFEASIBLE"]
    q_context: dict[int, GeneratorQStatus]
    terminal_pf: PowerFlowResult | None
    score: float
    logical_probe_count: int = Field(ge=0)
    solver_attempt_count: int = Field(ge=0)
    diagnostics_complete: bool = True

    @model_validator(mode="after")
    def terminal_payload_matches_status(self) -> "StateEvaluation":
        if (self.status == "TERMINAL") != (self.terminal_pf is not None):
            raise ValueError("terminal_pf must be present exactly for TERMINAL states")
        return self


class WitnessSearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    witness: CurationWitness
    resolution_regime: Literal["DIRECT", "SEQUENTIAL"]
    direct_restorer_available: bool
    witness_length: int = Field(ge=1, le=10)
    witness_optimality: Literal["EXACT_MINIMUM", "UPPER_BOUND"]
    logical_probe_count: int = Field(ge=1)
    solver_attempt_count: int = Field(ge=1)
    expanded_state_count: int = Field(ge=1)
    search_policy_version: str


@dataclass(frozen=True)
class _Node:
    net: Any
    maneuvers: tuple[Action, ...]
    state_hashes: tuple[str, ...]
    evaluation: StateEvaluation


@dataclass(frozen=True)
class _PendingEvaluation:
    action: Action
    net: Any
    state_hash: str


class WitnessWorkerError(RuntimeError):
    """An exhaustive child evaluation failed in a worker process."""


def search_curation_witness(
    target_net: Any,
    *,
    scenario_id: str,
    policy: WitnessSearchPolicy,
    executor: Executor | None = None,
) -> WitnessSearchResult:
    """Find a deterministic within-budget Q-V witness without ranker assistance."""
    initial_hash = state_fingerprint(target_net).value
    initial_evaluation = _evaluate_state(target_net, policy)
    if initial_evaluation.status == "TERMINAL":
        raise ValueError("curation target already converges without a maneuver")

    logical_probes = initial_evaluation.logical_probe_count
    solver_attempts = initial_evaluation.solver_attempt_count
    expanded_states = 1
    frontier = [
        _Node(
            net=target_net,
            maneuvers=(),
            state_hashes=(initial_hash,),
            evaluation=initial_evaluation,
        )
    ]
    seen_hashes = {initial_hash}
    exhaustive_so_far = True

    for depth in range(1, policy.maneuver_budget + 1):
        next_frontier: list[_Node] = []
        terminals: list[_Node] = []
        for node in frontier:
            node, added_logical, added_attempts = _promote_for_expansion(
                node,
                policy,
            )
            logical_probes += added_logical
            solver_attempts += added_attempts
            actions = enumerate_legal_qv_actions(
                node.net,
                node.evaluation.q_context,
            )
            if not exhaustive_so_far:
                actions = sorted(
                    actions,
                    key=lambda action: (
                        _beam_bucket_order(
                            _public_action_bucket(
                                action,
                                reference_net=node.net,
                            )
                        ),
                        action.model_dump_json(),
                    ),
                )[: policy.beam_branching_width]
            # Preserve the deterministic action order, but evaluate every
            # independent child of this node as one batch. The replay below
            # still selects the first terminal in the original traversal order.
            action_batches = (tuple(actions),)
            for action_batch in action_batches:
                pending: list[_PendingEvaluation] = []
                for action in action_batch:
                    changed = apply_qv_action(
                        node.net,
                        action,
                        node.evaluation.q_context,
                    )
                    changed_hash = state_fingerprint(changed).value
                    if changed_hash in seen_hashes:
                        continue
                    seen_hashes.add(changed_hash)
                    pending.append(
                        _PendingEvaluation(
                            action=action,
                            net=changed,
                            state_hash=changed_hash,
                        )
                    )
                evaluations = _evaluate_pending(
                    pending,
                    policy=policy,
                    executor=executor,
                    scenario_id=scenario_id,
                    depth=depth,
                )
                for candidate, evaluation in zip(
                    pending,
                    evaluations,
                    strict=True,
                ):
                    logical_probes += evaluation.logical_probe_count
                    solver_attempts += evaluation.solver_attempt_count
                    expanded_states += 1
                    child = _Node(
                        net=candidate.net,
                        maneuvers=(*node.maneuvers, candidate.action),
                        state_hashes=(*node.state_hashes, candidate.state_hash),
                        evaluation=evaluation,
                    )
                    if evaluation.status == "TERMINAL":
                        if not exhaustive_so_far:
                            return _result(
                                child,
                                scenario_id=scenario_id,
                                resolution_regime="SEQUENTIAL",
                                optimality="UPPER_BOUND",
                                logical_probes=logical_probes,
                                solver_attempts=solver_attempts,
                                expanded_states=expanded_states,
                                policy=policy,
                            )
                        terminals.append(child)
                    elif evaluation.status in {"DIVERGED", "INFEASIBLE"}:
                        next_frontier.append(child)

        if terminals:
            winner = min(terminals, key=_node_stable_key)
            exact = exhaustive_so_far
            regime = "DIRECT" if depth == 1 else "SEQUENTIAL"
            return _result(
                winner,
                scenario_id=scenario_id,
                resolution_regime=regime,
                optimality="EXACT_MINIMUM" if exact else "UPPER_BOUND",
                logical_probes=logical_probes,
                solver_attempts=solver_attempts,
                expanded_states=expanded_states,
                policy=policy,
            )
        if not next_frontier:
            break
        if depth == policy.maneuver_budget:
            break

        if depth >= policy.bfs_max_depth or len(next_frontier) > policy.frontier_budget:
            exhaustive_so_far = False
            next_frontier = _select_diverse_beam(
                next_frontier,
                width=policy.beam_width,
                reference_net=target_net,
            )
        frontier = next_frontier

    raise ValueError("no valid Q-V witness found within maneuver budget")


def _evaluate_pending(
    pending: list[_PendingEvaluation],
    *,
    policy: WitnessSearchPolicy,
    executor: Executor | None,
    scenario_id: str,
    depth: int,
) -> tuple[StateEvaluation, ...]:
    """Evaluate independent children concurrently, then replay submission order."""
    if executor is None:
        return tuple(_evaluate_state(candidate.net, policy) for candidate in pending)

    futures: list[Future[StateEvaluation]] = [
        executor.submit(_evaluate_state, candidate.net, policy) for candidate in pending
    ]
    evaluations: list[StateEvaluation] = []
    for index, (candidate, future) in enumerate(zip(pending, futures, strict=True)):
        try:
            evaluations.append(future.result())
        except Exception as exc:
            for remaining in futures[index + 1 :]:
                remaining.cancel()
            raise WitnessWorkerError(
                f"witness worker failed for scenario {scenario_id}, depth {depth}, "
                f"action {candidate.action.model_dump_json()}, state {candidate.state_hash}: {exc}"
            ) from exc
    return tuple(evaluations)


def _evaluate_state(
    net: Any,
    policy: WitnessSearchPolicy,
) -> StateEvaluation:
    probe = solve_locked_probe(net)
    if probe.status == "SOLVED":
        feasibility = evaluate_solved_feasibility(probe.solved_net)
        q_context = {item.gen_id: item.status for item in feasibility.generator_q_status}
        if not satisfies_non_voltage_constraints(feasibility):
            return StateEvaluation(
                status="INFEASIBLE",
                q_context=q_context,
                terminal_pf=None,
                score=feasibility.voltage.min_vm_pu,
                logical_probe_count=1,
                solver_attempt_count=probe.solver_attempt_count,
            )
        slack = feasibility.slack_results[0] if feasibility.slack_results else None
        terminal = PowerFlowResult(
            converged=True,
            iterations=probe.attempts[-1].iterations,
            tolerance_used=probe.tolerance_used_mva,
            runtime_ms=probe.elapsed_ms,
            solver_attempt_count=probe.solver_attempt_count,
            recovery_used=probe.recovery_used,
            feasibility=feasibility,
            generator_q_status=list(feasibility.generator_q_status),
            slack=slack,
        )
        return StateEvaluation(
            status="TERMINAL",
            q_context=q_context,
            terminal_pf=terminal,
            score=feasibility.voltage.min_vm_pu,
            logical_probe_count=1,
            solver_attempt_count=probe.solver_attempt_count,
        )

    return StateEvaluation(
        status="DIVERGED",
        q_context={},
        terminal_pf=None,
        score=float("-inf"),
        logical_probe_count=1,
        solver_attempt_count=probe.solver_attempt_count,
        diagnostics_complete=False,
    )


def _promote_for_expansion(
    node: _Node,
    policy: WitnessSearchPolicy,
) -> tuple[_Node, int, int]:
    """Attach runtime-equivalent Q context only when a divergent node is expanded."""
    evaluation = node.evaluation
    if evaluation.status != "DIVERGED" or evaluation.diagnostics_complete:
        return node, 0, 0
    q_context, score, logical_count, attempt_count = _diagnose_q_context(
        node.net,
        policy,
    )
    promoted_evaluation = evaluation.model_copy(
        update={
            "q_context": q_context,
            "score": score,
            "logical_probe_count": (evaluation.logical_probe_count + logical_count),
            "solver_attempt_count": (evaluation.solver_attempt_count + attempt_count),
            "diagnostics_complete": True,
        }
    )
    return (
        _Node(
            net=node.net,
            maneuvers=node.maneuvers,
            state_hashes=node.state_hashes,
            evaluation=promoted_evaluation,
        ),
        logical_count,
        attempt_count,
    )


def _diagnose_q_context(
    snapshot_net: Any,
    policy: WitnessSearchPolicy,
) -> tuple[dict[int, GeneratorQStatus], float, int, int]:
    """Retreat on the public snapshot-anchored trajectory and count every probe."""
    evidence = retreat_evidence(
        snapshot_net,
        coordinates=policy.diagnostic_coordinates,
        resolution=policy.diagnostic_resolution,
    )
    if evidence is None:
        # No shared retreat evidence. An empty context forbids nothing, so the search would
        # accept witnesses the runtime guard refuses to apply — the corpus would certify a
        # solution no agent can reach. Fall back to the snapshot-local Q-unlimited saturation,
        # which the runtime derives the same way, so an accepted witness stays applicable.
        fallback = q_saturation_context(snapshot_net)
        return (dict(fallback or {}), float("-inf"), 0, 0)

    feasibility = evaluate_solved_feasibility(evidence.solved_net)
    q_context = {item.gen_id: item.status for item in evidence.q_status}
    score = feasibility.voltage.min_vm_pu - 0.001 * len(feasibility.q_limited_gen_ids)
    return (
        q_context,
        score,
        evidence.logical_probe_count,
        evidence.solver_attempt_count,
    )


def _result(
    node: _Node,
    *,
    scenario_id: str,
    resolution_regime: Literal["DIRECT", "SEQUENTIAL"],
    optimality: Literal["EXACT_MINIMUM", "UPPER_BOUND"],
    logical_probes: int,
    solver_attempts: int,
    expanded_states: int,
    policy: WitnessSearchPolicy,
) -> WitnessSearchResult:
    terminal_pf = node.evaluation.terminal_pf
    if terminal_pf is None:
        raise AssertionError("winning witness node has no terminal power flow")
    witness = CurationWitness(
        scenario_id=scenario_id,
        maneuvers=node.maneuvers,
        state_hashes=node.state_hashes,
        terminal_pf=terminal_pf,
        search_policy_version=policy.policy_version,
    )
    return WitnessSearchResult(
        witness=witness,
        resolution_regime=resolution_regime,
        direct_restorer_available=resolution_regime == "DIRECT",
        witness_length=len(node.maneuvers),
        witness_optimality=optimality,
        logical_probe_count=logical_probes,
        solver_attempt_count=solver_attempts,
        expanded_state_count=expanded_states,
        search_policy_version=policy.policy_version,
    )


def _node_stable_key(node: _Node) -> tuple[str, ...]:
    return tuple(action.model_dump_json() for action in node.maneuvers)


def _select_diverse_beam(
    nodes: list[_Node],
    *,
    width: int,
    reference_net: Any,
) -> list[_Node]:
    """Keep public-action families represented without private pocket ranking."""
    buckets: dict[str, list[_Node]] = {}
    for node in sorted(nodes, key=_node_stable_key):
        bucket = _public_action_bucket(
            node.maneuvers[-1],
            reference_net=reference_net,
        )
        buckets.setdefault(bucket, []).append(node)

    selected: list[_Node] = []
    ordered_buckets = sorted(buckets, key=_beam_bucket_order)
    while len(selected) < width:
        progressed = False
        for bucket in ordered_buckets:
            candidates = buckets[bucket]
            if not candidates:
                continue
            selected.append(candidates.pop(0))
            progressed = True
            if len(selected) == width:
                break
        if not progressed:
            break
    return selected


def _public_action_bucket(action: Action, *, reference_net: Any) -> str:
    """Classify only public action direction/role for deterministic beam diversity."""
    if action.type == "GEN_V_SETPOINT":
        try:
            current = float(reference_net.gen.at[action.gen_id, "vm_pu"])
        except (AttributeError, KeyError, TypeError):
            return "GEN"
        direction = "UP" if action.new_vm_pu > current else "DOWN"
        return f"GEN_{direction}"
    if action.type == "SHUNT_STEP":
        try:
            q_mvar = float(reference_net.shunt.at[action.shunt_id, "q_mvar"])
        except (AttributeError, KeyError, TypeError):
            return "SHUNT"
        role = "CAPACITOR" if q_mvar < 0.0 else "REACTOR"
        return f"SHUNT_{role}_{action.new_step}"
    try:
        current = int(reference_net.trafo.at[action.trafo_id, "tap_pos"])
    except (AttributeError, KeyError, TypeError, ValueError):
        return "TRAFO"
    direction = "UP" if action.new_tap_pos > current else "DOWN"
    return f"TRAFO_{direction}"


def _beam_bucket_order(bucket: str) -> tuple[int, str]:
    # Generic public Q-support directions lead the beam. This never consults a
    # pocket, weak bus, control-influence oracle, or learned ranker.
    preferred = {
        "SHUNT_CAPACITOR_1": 0,
        "SHUNT_REACTOR_0": 1,
        "GEN_UP": 2,
        "TRAFO_DOWN": 3,
        "TRAFO_UP": 4,
        "GEN_DOWN": 5,
        "SHUNT_CAPACITOR_0": 6,
        "SHUNT_REACTOR_1": 7,
    }
    return preferred.get(bucket, len(preferred)), bucket
