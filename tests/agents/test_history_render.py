# ABOUTME: Verifies the accepted-maneuver history carries each maneuver's effect on overstress.
# ABOUTME: Without the trajectory the agent sees what it did but never whether it helped.
from __future__ import annotations

from restorebench.agents.history_render import render_accepted_history
from restorebench.schemas.actions import GenVoltageSetpointAction, Maneuver
from restorebench.schemas.feedback import AcceptedManeuver


def _maneuver(gen_id: int = 3, vm_pu: float = 1.02) -> Maneuver:
    return Maneuver(
        action=GenVoltageSetpointAction(type="GEN_V_SETPOINT", gen_id=gen_id, new_vm_pu=vm_pu),
        diagnosed_cause="REACTIVE_DEFICIT",
        rationale="Raise reactive support near the weak bus.",
    )


def test_empty_history_renders_a_placeholder() -> None:
    assert render_accepted_history(()) == "(no maneuvers accepted yet)"


def test_each_step_shows_the_overstress_before_and_after() -> None:
    history = (
        AcceptedManeuver(maneuver=_maneuver(), overstress_before=0.0333, overstress_after=0.0281),
    )

    rendered = render_accepted_history(history)

    assert "step 1" in rendered
    assert "0.0333" in rendered
    assert "0.0281" in rendered


def test_an_improving_step_is_labelled_as_progress() -> None:
    history = (
        AcceptedManeuver(maneuver=_maneuver(), overstress_before=0.0333, overstress_after=0.0281),
    )

    assert "improved" in render_accepted_history(history)


def test_a_worsening_step_is_labelled_so_the_agent_can_change_direction() -> None:
    history = (
        AcceptedManeuver(maneuver=_maneuver(), overstress_before=0.0281, overstress_after=0.0340),
    )

    assert "worsened" in render_accepted_history(history)


def test_missing_overstress_is_reported_rather_than_guessed() -> None:
    """A converged-but-infeasible step carries no nose diagnostics; silence would read as zero."""
    history = (AcceptedManeuver(maneuver=_maneuver(), overstress_before=0.0281, overstress_after=None),)

    rendered = render_accepted_history(history)

    assert "unknown" in rendered
    assert "improved" not in rendered
    assert "worsened" not in rendered


def test_steps_are_numbered_in_the_order_they_were_accepted() -> None:
    history = (
        AcceptedManeuver(maneuver=_maneuver(gen_id=3), overstress_before=0.0333, overstress_after=0.0281),
        AcceptedManeuver(maneuver=_maneuver(gen_id=7), overstress_before=0.0281, overstress_after=0.0244),
    )

    lines = render_accepted_history(history).splitlines()

    assert len(lines) == 2
    assert lines[0].startswith("step 1")
    assert lines[1].startswith("step 2")
    assert '"gen_id": 3' in lines[0]
    assert '"gen_id": 7' in lines[1]
