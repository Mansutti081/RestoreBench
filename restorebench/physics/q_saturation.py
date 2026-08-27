# ABOUTME: Derives generator Q status from the Q-unlimited solution of a non-convergent snapshot.
# ABOUTME: One public, snapshot-local Q context that runtime and curation can both reproduce.
from __future__ import annotations

import copy
import math
from typing import Any

import pandapower as pp
from pandapower.auxiliary import LoadflowNotConverged

from restorebench.physics.actions import GeneratorQStatus, QContext
from restorebench.physics.solver import (
    ALGORITHM,
    CHECK_CONNECTIVITY,
    MAX_ITERATION,
    PRIMARY_TOLERANCE_MVA,
    RECOVERY_TOLERANCE_MVA,
)

# A snapshot that does not converge has no solved state to read generator Q from, and retreating
# along a trajectory makes the answer depend on which trajectory the caller walked. Releasing the
# Q limits keeps the question inside the snapshot: every generator solves to the reactive power it
# would need, and the ones that exceed a declared bound are exactly those the constrained solve
# would pin. Admitted targets converge this way by construction — it is the Q-unlimited gate.
Q_SATURATION_POLICY_VERSION = "q-unlimited-snapshot-saturation-v1"

# Below this the exceedance is solver noise rather than a generator sitting on its bound.
SATURATION_EPS_MVAR = 1e-3


def q_saturation_context(net: Any) -> QContext | None:
    """Return the generator Q status implied by the snapshot's Q-unlimited solution.

    Returns None when the snapshot does not converge even with the Q limits released, which
    means the failure is not reactive and no saturation statement is defensible.
    """
    solved = _solve_q_unlimited(net)
    if solved is None:
        return None

    context: dict[int, GeneratorQStatus] = {}
    for gen_id, row in solved.gen.sort_index().iterrows():
        if not bool(row.get("in_service", True)):
            continue
        if gen_id not in solved.res_gen.index:
            continue
        q_mvar = float(solved.res_gen.at[gen_id, "q_mvar"])
        min_q = float(row["min_q_mvar"])
        max_q = float(row["max_q_mvar"])
        if not all(math.isfinite(value) for value in (q_mvar, min_q, max_q)):
            continue
        if q_mvar >= max_q + SATURATION_EPS_MVAR:
            context[int(gen_id)] = "Q_LIMITED_UPPER"
        elif q_mvar <= min_q - SATURATION_EPS_MVAR:
            context[int(gen_id)] = "Q_LIMITED_LOWER"
        else:
            context[int(gen_id)] = "PV_CONTROLLABLE"
    return context


def _solve_q_unlimited(net: Any) -> Any | None:
    """Solve the snapshot with the locked policy except that Q limits are released."""
    for tolerance in (PRIMARY_TOLERANCE_MVA, RECOVERY_TOLERANCE_MVA):
        attempt = copy.deepcopy(net)
        try:
            pp.runpp(
                attempt,
                algorithm=ALGORITHM,
                enforce_q_lims=False,
                init="dc",
                tolerance_mva=tolerance,
                max_iteration=MAX_ITERATION,
                check_connectivity=CHECK_CONNECTIVITY,
            )
        except LoadflowNotConverged:
            continue
        if bool(getattr(attempt, "converged", False)):
            return attempt
    return None
