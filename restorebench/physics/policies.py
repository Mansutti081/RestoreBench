# ABOUTME: Defines versioned constants for deterministic shared electrical policies.
# ABOUTME: Centralizes policy identities used by trajectory, scheduling, solver, and cache contracts.

from restorebench.schemas.physics import ACTIVE_BALANCE_POLICY_VERSION as ACTIVE_BALANCE_POLICY_VERSION

CURATION_TRAJECTORY_POLICY_VERSION = "profile-anchored-pocket-stress-v1"
DIAGNOSTIC_TRAJECTORY_POLICY_VERSION = "snapshot-anchored-uniform-scaling-v1"
TRAJECTORY_DECIMAL_PLACES = 10
ACTIVE_BALANCE_TOLERANCE_MW = 1e-9
SOLVER_PROBE_POLICY_VERSION = "locked-nr-q-limited-v1"
FEASIBILITY_POLICY_VERSION = "solved-state-feasibility-v1"
Q_LIMIT_STATUS_POLICY_VERSION = "directional-q-limit-v1"
Q_LIMIT_EPS_MVAR = 1e-3
HARD_VM_MIN_PU = 0.90
HARD_VM_MAX_PU = 1.10
RUNTIME_VM_MIN_PU = 0.95
RUNTIME_VM_MAX_PU = 1.05
BOUNDARY_POLICY_VERSION = "scan-first-boundary-v1"
# One retreat contract for runtime diagnostics and corpus witness search: a Q context read
# at a different retreat point saturates generators the other side considers controllable.
RETREAT_POLICY_VERSION = "snapshot-anchored-retreat-v1"
# Identity of every policy that decides which components a run is steered toward. The shared
# retreat is the only one today, because it fixes the Q context the applicability guard reads.
# The ranker of plan 17 must fold its own policy version in here when it ships: leaving it out
# would let a ranker change produce incomparable runs that still look interchangeable.
RANKING_POLICY_VERSION = f"retreat={RETREAT_POLICY_VERSION}"
RETREAT_COORDINATES = (0.25, 0.5, 0.75, 1.0)
RETREAT_RESOLUTION = 0.02
ELECTRICAL_DISTANCE_POLICY_VERSION = "impedance-weighted-graph-distance-v1"
FINGERPRINT_POLICY_VERSION = "electrical-state-sha256-v1"
