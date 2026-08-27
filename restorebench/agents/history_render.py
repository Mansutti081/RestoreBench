# ABOUTME: Renders the accepted-maneuver history as a trajectory of overstress readings.
# ABOUTME: Lets an agent whose conversation restarts each iteration see whether it is improving.
from __future__ import annotations

import json
from collections.abc import Sequence

from restorebench.agents.progress import overstress_verdict
from restorebench.schemas.feedback import AcceptedManeuver

EMPTY_HISTORY = "(no maneuvers accepted yet)"


def render_accepted_history(history: Sequence[AcceptedManeuver]) -> str:
    """Return one line per accepted maneuver, each carrying its effect on overstress."""
    if not history:
        return EMPTY_HISTORY
    return "\n".join(
        f"step {index}: {_action_text(accepted)} | {_progress_text(accepted)}"
        for index, accepted in enumerate(history, start=1)
    )


def _action_text(accepted: AcceptedManeuver) -> str:
    return json.dumps(accepted.maneuver.action.model_dump(mode="json"), sort_keys=True)


def _progress_text(accepted: AcceptedManeuver) -> str:
    before, after = accepted.overstress_before, accepted.overstress_after
    verdict = overstress_verdict(before, after)
    if verdict == "unknown":
        return "overstress unknown for this step"
    assert before is not None and after is not None
    return f"overstress {before:.5f} -> {after:.5f} ({verdict})"
