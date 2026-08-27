# ABOUTME: Runs one isolated locked NR probe with a single recovery attempt and explicit accounting.
# ABOUTME: Returns no failed result tables and never starts retreat diagnostics.
from __future__ import annotations

import copy
import time
from typing import Any

import pandapower as pp
from pandapower.auxiliary import LoadflowNotConverged

from restorebench.physics.policies import SOLVER_PROBE_POLICY_VERSION
from restorebench.schemas.physics import LockedProbeResult, SolverAttempt


ALGORITHM = "nr"
ENFORCE_Q_LIMS = True
INIT = "dc"
MAX_ITERATION = 30
CHECK_CONNECTIVITY = True
PRIMARY_TOLERANCE_MVA = 1e-8
RECOVERY_TOLERANCE_MVA = 1e-6


def solve_locked_probe(net: Any) -> LockedProbeResult:
    """Run the shared lightweight solver policy on fresh isolated copies."""
    started = time.perf_counter()
    attempts: list[SolverAttempt] = []
    last_error: str | None = None

    for attempt_number, tolerance in enumerate(
        (PRIMARY_TOLERANCE_MVA, RECOVERY_TOLERANCE_MVA),
        start=1,
    ):
        attempt_net = copy.deepcopy(net)
        attempt_started = time.perf_counter()
        try:
            _run_locked_pf(attempt_net, tolerance)
        except LoadflowNotConverged as exc:
            last_error = str(exc)
            attempts.append(
                SolverAttempt(
                    attempt_number=attempt_number,
                    tolerance_mva=tolerance,
                    status="NO_SOLUTION",
                    iterations=_iterations(attempt_net, default=MAX_ITERATION),
                    elapsed_ms=_elapsed_ms(attempt_started),
                    error_message=last_error,
                )
            )
            continue

        attempts.append(
            SolverAttempt(
                attempt_number=attempt_number,
                tolerance_mva=tolerance,
                status="SOLVED",
                iterations=_iterations(attempt_net),
                elapsed_ms=_elapsed_ms(attempt_started),
            )
        )
        return LockedProbeResult(
            status="SOLVED",
            solved_net=attempt_net,
            attempts=tuple(attempts),
            solver_attempt_count=len(attempts),
            tolerance_used_mva=tolerance,
            recovery_used=attempt_number == 2,
            elapsed_ms=_elapsed_ms(started),
            policy_version=SOLVER_PROBE_POLICY_VERSION,
        )

    return LockedProbeResult(
        status="NO_SOLUTION",
        solved_net=None,
        attempts=tuple(attempts),
        solver_attempt_count=2,
        tolerance_used_mva=RECOVERY_TOLERANCE_MVA,
        recovery_used=True,
        elapsed_ms=_elapsed_ms(started),
        error_message=last_error,
        policy_version=SOLVER_PROBE_POLICY_VERSION,
    )


def _run_locked_pf(net: Any, tolerance_mva: float) -> None:
    pp.runpp(
        net,
        algorithm=ALGORITHM,
        enforce_q_lims=ENFORCE_Q_LIMS,
        init=INIT,
        tolerance_mva=tolerance_mva,
        max_iteration=MAX_ITERATION,
        check_connectivity=CHECK_CONNECTIVITY,
    )
    if not bool(getattr(net, "converged", False)):
        raise LoadflowNotConverged(f"Power Flow {ALGORITHM} did not converge after {MAX_ITERATION} iterations!")


def _iterations(net: Any, default: int = 0) -> int:
    if isinstance(net, dict):
        ppc = net.get("_ppc", {}) or {}
    else:
        ppc = getattr(net, "_ppc", {}) or {}
    value = ppc.get("iterations", default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0
