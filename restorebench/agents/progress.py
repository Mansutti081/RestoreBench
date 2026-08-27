# ABOUTME: Names the direction of travel between two overstress readings, above measurement noise.
# ABOUTME: Shared so the history, the preview payload, the repeat guard and the ranking agree.
from __future__ import annotations

from typing import Literal

Verdict = Literal["improved", "worsened", "unchanged", "unknown"]

# Overstress is read off a 22-step bisection starting from a load scale of 0.5, so its quantum is
# 0.5 / 2**22 ~= 1.2e-7. Differences near that size are the measurement, not the grid. Calling one
# "improved" would rank noise above a real candidate, and — since an improving preview is exempt
# from the repeat guard — would let the agent propose the same inert move for the whole budget.
OVERSTRESS_QUANTUM = 0.5 / 2**22
MIN_ABSOLUTE_CHANGE = 10 * OVERSTRESS_QUANTUM
MIN_RELATIVE_CHANGE = 1e-3


def meaningful_change(before: float | None, after: float | None) -> float | None:
    """Return the signed change once it clears the noise floor, else 0.0; None if unmeasured.

    The floor is relative to the reading so it stays meaningful as the case approaches solvable:
    at overstress 0.03 it is 3e-5, at 0.0016 it is 1.6e-6, and both sit far above the quantum.
    """
    if before is None or after is None:
        return None
    change = before - after
    floor = max(MIN_ABSOLUTE_CHANGE, MIN_RELATIVE_CHANGE * abs(before))
    return change if abs(change) >= floor else 0.0


def overstress_verdict(before: float | None, after: float | None) -> Verdict:
    """Say whether a move closed or widened the gap to solvability by a measurable amount.

    'unknown' is not 'unchanged': a state carrying no nose diagnostics has no reading at all,
    and reporting that as no change would invite the agent to read a missing value as zero.
    """
    change = meaningful_change(before, after)
    if change is None:
        return "unknown"
    if change > 0:
        return "improved"
    if change < 0:
        return "worsened"
    return "unchanged"
