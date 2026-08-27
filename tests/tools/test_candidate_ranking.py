# ABOUTME: Covers the candidate ranking tool: selection, ordering, depth estimate and blocking.
# ABOUTME: Every power flow is stubbed, so the suite stays free of real solves.
from __future__ import annotations

from typing import Any

import pytest

from restorebench.schemas.actions import GenVoltageSetpointAction, TapAdjustmentAction
from restorebench.tools import candidate_ranking as ranking


def _gen(gen_id: int, vm_pu: float = 1.02) -> GenVoltageSetpointAction:
    return GenVoltageSetpointAction(type="GEN_V_SETPOINT", gen_id=gen_id, new_vm_pu=vm_pu)


def _tap(trafo_id: int, tap: int = -1) -> TapAdjustmentAction:
    return TapAdjustmentAction(type="TAP_ADJUSTMENT", trafo_id=trafo_id, new_tap_pos=tap)


def test_selection_is_deterministic_for_the_same_pool() -> None:
    """A benchmark run must be reproducible; a random subset would add variance across repetitions."""
    pool = [_gen(index) for index in range(40)]

    first = ranking.select_candidates(pool, budget=8)
    second = ranking.select_candidates(pool, budget=8)

    assert first == second
    assert len(first) == 8


def test_selection_spreads_across_the_pool_rather_than_taking_a_prefix() -> None:
    """No cheap prior orders these usefully, so an even spread beats the first k by index."""
    pool = [_gen(index) for index in range(40)]

    chosen = ranking.select_candidates(pool, budget=4)

    assert [action.gen_id for action in chosen] != [0, 1, 2, 3]
    assert len(set(action.gen_id for action in chosen)) == 4


def test_selection_returns_the_whole_pool_when_it_fits_the_budget() -> None:
    pool = [_gen(1), _gen(2)]

    assert ranking.select_candidates(pool, budget=8) == pool


def _rank(monkeypatch: pytest.MonkeyPatch, outcomes: dict[int, Any], *, before: float = 0.0333, **kwargs: Any):
    """Rank three generator candidates against stubbed solver outcomes keyed by gen_id."""
    pool = [_gen(1), _gen(2), _gen(3)]
    monkeypatch.setattr(ranking, "enumerate_legal_qv_actions", lambda *_a, **_k: list(pool))
    monkeypatch.setattr(ranking, "_baseline_overstress", lambda *_a, **_k: before)

    def fake_evaluate(_net: Any, actions: list[Any]) -> list[tuple[str, float | None]]:
        return [outcomes[action.gen_id] for action in actions]

    monkeypatch.setattr(ranking, "evaluate_candidates", fake_evaluate)
    return ranking.rank_candidates(object(), budget=8, **kwargs)


def test_candidates_are_ordered_by_how_much_they_close_the_gap(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _rank(
        monkeypatch,
        {1: ("diverged", 0.0300), 2: ("diverged", 0.0157), 3: ("diverged", 0.0400)},
    )

    assert [candidate.action.gen_id for candidate in result.candidates] == [2, 1, 3]
    assert result.candidates[0].delta == pytest.approx(0.0333 - 0.0157)
    assert result.candidates[-1].delta == pytest.approx(0.0333 - 0.0400)


def test_a_solving_candidate_is_reported_and_ranked_first(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _rank(
        monkeypatch,
        {1: ("diverged", 0.0300), 2: ("solves", None), 3: ("diverged", 0.0157)},
    )

    assert result.any_converges is True
    assert result.candidates[0].action.gen_id == 2
    assert result.candidates[0].status == "solves"


def test_a_candidate_that_converges_but_is_infeasible_does_not_count_as_solving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runner's terminal contract is convergence AND the non-voltage constraints.

    Ranking a converged-but-infeasible candidate first sent the agent straight into
    SOLVED_INFEASIBLE: on S0200 five of its eight maneuvers ended that way, and it oscillated
    one generator up and down because each direction 'converged'.
    """
    result = _rank(
        monkeypatch,
        {1: ("diverged", 0.0157), 2: ("converges_infeasible", None), 3: ("diverged", 0.0300)},
    )

    assert result.any_converges is False
    assert result.candidates[0].action.gen_id == 1
    assert result.candidates[-1].status == "converges_infeasible"





def test_a_shortlist_where_nothing_improves_is_reported_as_such(monkeypatch: pytest.MonkeyPatch) -> None:
    """Measured on 2 of 20 scenarios: the spread can miss, and the agent must be able to tell."""
    result = _rank(monkeypatch, {1: ("diverged", 0.0340), 2: ("diverged", 0.0350), 3: ("diverged", 0.0400)})

    assert result.any_converges is False
    assert all(candidate.delta is None or candidate.delta <= 0 for candidate in result.candidates)


def test_no_depth_estimate_is_reported() -> None:
    """Gap over best single gain failed validation: rho = -0.07 against witness length, p = 0.79."""
    import dataclasses

    fields = {field.name for field in dataclasses.fields(ranking.CandidateRanking)}
    assert "estimated_remaining_steps" not in fields


def test_blocked_actions_are_never_offered(monkeypatch: pytest.MonkeyPatch) -> None:
    """The agent is told not to repeat them, so ranking them wastes a solve and invites a repeat."""
    result = _rank(
        monkeypatch,
        {1: ("diverged", 0.0300), 3: ("diverged", 0.0157)},
        blocked_actions=[_gen(2)],
    )

    assert [candidate.action.gen_id for candidate in result.candidates] == [3, 1]
    assert result.n_available == 2


def test_payload_is_json_serialisable_and_names_the_verdict(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _rank(monkeypatch, {1: ("diverged", 0.0300), 2: ("diverged", 0.0157), 3: ("diverged", 0.0400)})

    payload = ranking.ranking_payload(result)

    assert payload["overstress_before"] == pytest.approx(0.0333)
    assert payload["any_candidate_converges"] is False
    assert payload["candidates"][0]["verdict"] == "improved"
    assert payload["candidates"][-1]["verdict"] == "worsened"
    import json

    json.dumps(payload)


def test_ranking_survives_a_candidate_whose_solve_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    """One bad candidate must not lose the other results; it is reported as unmeasurable."""
    result = _rank(monkeypatch, {1: ("diverged", None), 2: ("diverged", 0.0157), 3: ("diverged", 0.0400)})

    assert len(result.candidates) == 3
    unmeasured = [candidate for candidate in result.candidates if candidate.overstress_after is None]
    assert len(unmeasured) == 1
    assert unmeasured[0].delta is None
    # ranked last: an unmeasurable candidate is not evidence of progress
    assert result.candidates[-1].action.gen_id == 1


def test_the_payload_marks_every_candidate_as_applicable_now() -> None:
    """The shortlist is drawn from the legal pool under the runtime Q context.

    On S0200 the agent re-verified ranked candidates with run_ac_pf and proposed maneuvers from
    outside the list, spending 4 of its 10 budget slots on INVALID_ACTION. It has no way to know
    the list is authoritative unless the payload says so.
    """
    from restorebench.agents.tool_loop import RANK_CANDIDATE_MANEUVERS_TOOL_NAME, default_diagnostic_tools

    tool = next(
        t for t in default_diagnostic_tools() if t.name == RANK_CANDIDATE_MANEUVERS_TOOL_NAME
    )

    assert "applicable" in tool.description
    assert "INVALID_ACTION" in tool.description
    assert "already been evaluated" in tool.description
