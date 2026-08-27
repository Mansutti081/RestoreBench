# ABOUTME: Ranks legal maneuvers by the overstress each one measurably removes, and estimates depth.
# ABOUTME: Evaluations are real solves run in parallel; no static prior orders these usefully.
from __future__ import annotations

import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Literal

from restorebench.agents.progress import meaningful_change, overstress_verdict
from restorebench.physics.actions import enumerate_legal_qv_actions
from restorebench.physics.q_saturation import QContext
from restorebench.schemas.actions import Action
from restorebench.tools.power_flow import run_ac_pf

# Measured against the witnesses: neither shortest-path electrical distance, Zbus transfer
# impedance, Q headroom, nor their product ranks the solving move better than chance. So the
# tool spends real solves instead of guessing, and spreads them evenly over the legal pool.
#
# It deliberately reports no estimate of how many maneuvers remain. Dividing the gap by the best
# single gain was tried and failed validation: across 18 scenarios its correlation with the
# witness length was rho = -0.07 (p = 0.79), because gains compose non-linearly. It read 28 steps
# for a case one maneuver solves, which against a budget of 10 is worse than saying nothing.
RANKING_POLICY_VERSION = "candidate-overstress-ranking-v1"
DEFAULT_CANDIDATE_BUDGET = 12


# The runner's terminal contract is convergence AND the non-voltage constraints, so a candidate
# that merely converges is not a solution. Reporting it as one sent the agent into
# SOLVED_INFEASIBLE five times out of eight maneuvers on S0200.
CandidateStatus = Literal["solves", "converges_infeasible", "diverged"]


@dataclass(frozen=True)
class RankedCandidate:
    action: Action
    status: CandidateStatus
    overstress_after: float | None
    delta: float | None

    @property
    def converged(self) -> bool:
        return self.status == "solves"


@dataclass(frozen=True)
class CandidateRanking:
    overstress_before: float | None
    candidates: tuple[RankedCandidate, ...]
    n_available: int
    any_converges: bool


def select_candidates(pool: Sequence[Action], *, budget: int) -> list[Action]:
    """Take an evenly spread subset, deterministically.

    Deterministic because a benchmark repetition must be reproducible; evenly spread because
    the pool is ordered by element id, and a prefix would only ever probe low-numbered buses.
    """
    if budget <= 0:
        return []
    if len(pool) <= budget:
        return list(pool)
    stride = len(pool) / budget
    return [pool[min(len(pool) - 1, int(index * stride))] for index in range(budget)]


def rank_candidates(
    net: Any,
    *,
    budget: int = DEFAULT_CANDIDATE_BUDGET,
    q_context: QContext | None = None,
    blocked_actions: Sequence[Action] = (),
) -> CandidateRanking:
    """Evaluate a spread of legal maneuvers and order them by measured progress."""
    pool = [
        action
        for action in enumerate_legal_qv_actions(net, q_context)
        if not any(action == blocked for blocked in blocked_actions)
    ]
    chosen = select_candidates(pool, budget=budget)
    before = _baseline_overstress(net)
    outcomes = evaluate_candidates(net, chosen) if chosen else []

    candidates = [
        RankedCandidate(
            action=action,
            status=status,
            overstress_after=after,
            delta=meaningful_change(before, after),
        )
        for action, (status, after) in zip(chosen, outcomes)
    ]
    candidates.sort(key=_rank_key, reverse=True)
    return CandidateRanking(
        overstress_before=before,
        candidates=tuple(candidates),
        n_available=len(pool),
        any_converges=any(candidate.converged for candidate in candidates),
    )


def ranking_payload(ranking: CandidateRanking) -> dict[str, Any]:
    """Render the ranking for a tool result."""
    return {
        "overstress_before": ranking.overstress_before,
        "any_candidate_converges": ranking.any_converges,
        "n_legal_actions_available": ranking.n_available,
        "n_evaluated": len(ranking.candidates),
        "candidates": [
            {
                "action": candidate.action.model_dump(mode="json"),
                "solves_the_case": candidate.status == "solves",
                "overstress_after": candidate.overstress_after,
                "overstress_removed": candidate.delta,
                "verdict": (
                    "solves the case"
                    if candidate.status == "solves"
                    else "converges but violates the feasibility limits, so it would waste a maneuver"
                    if candidate.status == "converges_infeasible"
                    else overstress_verdict(ranking.overstress_before, candidate.overstress_after)
                ),
            }
            for candidate in ranking.candidates
        ],
    }


def evaluate_candidates(net: Any, actions: Sequence[Action]) -> list[tuple[CandidateStatus, float | None]]:
    """Solve every candidate on its own copy, in parallel.

    Each solve costs seconds and they are independent, so the wall clock of a whole shortlist
    is roughly that of one solve times the ceiling of len(actions) / workers.
    """
    if not actions:
        return []
    # A harness running several cells at once multiplies this pool by the cell concurrency, and
    # oversubscribing the cores makes every evaluation slower than it needs to be. The runner that
    # knows how many cells it is running sets the cap; alone, the tool uses every core.
    cap = int(os.environ.get("RESTOREBENCH_RANKING_WORKERS", "0")) or (os.cpu_count() or 1)
    workers = min(len(actions), cap)
    if workers <= 1:
        return [_evaluate_one(net, action) for action in actions]
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(_evaluate_one, [net] * len(actions), actions))
    except Exception:
        # A worker pool that cannot start (no fork, a net that will not pickle) must degrade to
        # a slower answer, never to no answer: the agent has nothing else to rank with.
        return [_evaluate_one(net, action) for action in actions]


def _evaluate_one(net: Any, action: Action) -> tuple[CandidateStatus, float | None]:
    """Apply one action to an isolated copy and report the outcome under the runner's contract."""
    from restorebench.physics.actions import apply_qv_action
    from restorebench.physics.feasibility import satisfies_non_voltage_constraints

    try:
        result = run_ac_pf(apply_qv_action(net, action))
    except Exception:
        return ("diverged", None)
    if result.converged:
        if satisfies_non_voltage_constraints(result.feasibility):
            return ("solves", None)
        return ("converges_infeasible", None)
    return ("diverged", None if result.diagnostics is None else result.diagnostics.overstress)


def _baseline_overstress(net: Any) -> float | None:
    result = run_ac_pf(net)
    if result.converged or result.diagnostics is None:
        return None
    return result.diagnostics.overstress


def _rank_key(candidate: RankedCandidate) -> tuple[int, float]:
    if candidate.status == "solves":
        return (2, 0.0)
    if candidate.delta is None:
        # Unmeasurable, or convergent but infeasible: neither is evidence of progress, and the
        # infeasible one actively costs a maneuver slot if proposed.
        return (0, 0.0)
    return (1, candidate.delta)
