# ABOUTME: Pins the computations behind the paper figures: event shares, maneuver progress,
# ABOUTME: and timeout slot accounting, on synthetic responses with known expected values.
from __future__ import annotations

import numpy as np

from restorebench.schemas.feedback import FailureFeedback
from restorebench.sweeps import build_paper_figures as figs

from builders import diagnostics, maneuver, response


def _feedback(iteration: int, kind: str, overstress: float | None = None) -> FailureFeedback:
    diag = None
    if overstress is not None:
        diag = diagnostics().model_copy(update={"overstress": overstress})
    return FailureFeedback(iteration=iteration, kind=kind, diagnostics=diag, detail=None, maneuver=None)


def _episode(*, status: str, configuration: int, feedback: list[FailureFeedback],
             n_committed: int, baseline_overstress: float = 10.0, n_llm_calls: int | None = None):
    committed = [maneuver({"type": "GEN_V_SETPOINT", "gen_id": 1, "new_vm_pu": 1.0})] * n_committed
    run = response(
        scenario_id="S0001",
        maneuvers=committed,
        status=status,
        converged=status == "SUCCESS",
        configuration=configuration,
    )
    baseline_diag = diagnostics().model_copy(update={"overstress": baseline_overstress})
    trace = run.trace.model_copy(update={
        "events": [run.trace.events[0].model_copy(update={
            "phase": "baseline",
            "event_name": "baseline_diagnostics",
            "payload": {"diagnostics": baseline_diag.model_dump()},
        })],
        "n_llm_calls": n_llm_calls if n_llm_calls is not None else run.trace.n_llm_calls,
    })
    return run.model_copy(update={"failure_feedback": feedback, "trace": trace})


def test_budget_event_shares_count_only_slot_consuming_kinds() -> None:
    run = _episode(
        status="BUDGET_EXHAUSTED", configuration=1, n_committed=2,
        feedback=[
            _feedback(1, "STILL_DIVERGED", 9.0),
            _feedback(2, "INVALID_ACTION"),
            _feedback(3, "MALFORMED_OUTPUT"),
            _feedback(4, "SOLVED_INFEASIBLE"),
            _feedback(4, "PREVIEW_DIVERGED"),  # within-iteration attempt: no slot
        ],
    )

    shares = figs.budget_event_shares([run])

    assert np.allclose(shares[1], [25.0, 25.0, 25.0, 25.0])


def test_maneuver_progress_chains_before_and_after_overstress() -> None:
    # baseline 10 -> 8 (improved), 8 -> 9 (worsened), then SUCCESS terminal 9 -> 0 (improved)
    run = _episode(
        status="SUCCESS", configuration=2, n_committed=3,
        feedback=[_feedback(1, "STILL_DIVERGED", 8.0), _feedback(2, "STILL_DIVERGED", 9.0)],
    )

    shares = figs.maneuver_progress_shares([run])

    improved, unchanged, worsened = shares[(2, "SUCCESS")]
    assert np.isclose(improved, 200 / 3) and np.isclose(worsened, 100 / 3) and unchanged == 0


def test_solved_infeasible_counts_as_zero_overstress() -> None:
    # baseline 10 -> converged-infeasible (0): improved
    run = _episode(
        status="BUDGET_EXHAUSTED", configuration=3, n_committed=1,
        feedback=[_feedback(1, "SOLVED_INFEASIBLE")],
    )

    [(before, after)] = figs.committed_overstress_chain(run)

    assert before == 10.0 and after == 0.0


def test_mismatched_chain_refuses_to_classify() -> None:
    # two committed maneuvers but only one feedback entry: an aborted episode must yield
    # no classification instead of misattributed progress.
    run = _episode(
        status="BUDGET_EXHAUSTED", configuration=2, n_committed=2,
        feedback=[_feedback(1, "STILL_DIVERGED", 8.0)],
    )

    assert figs.committed_overstress_chain(run) == [None, None]


def test_timeout_slot_accounting_and_llm_ratio() -> None:
    run = _episode(
        status="TIMEOUT", configuration=3, n_committed=2, n_llm_calls=90,
        feedback=[
            _feedback(1, "STILL_DIVERGED", 9.0),
            _feedback(2, "INVALID_ACTION"),
            _feedback(3, "STILL_DIVERGED", 8.0),
        ],
    )
    zero_committed = _episode(
        status="TIMEOUT", configuration=3, n_committed=0, n_llm_calls=40,
        feedback=[_feedback(1, "MALFORMED_OUTPUT")],
    )

    consumed, ratios = figs.timeout_slots([run, zero_committed])

    assert consumed.tolist() == [3, 1]          # remaining: 7 and 9 of the 10-slot budget
    assert ratios.tolist() == [45.0]            # 90 calls / 2 maneuvers; zero-committed excluded


def test_success_rate_and_cost_tables_from_episode_records() -> None:
    priced = {"llm_assignment": {"single_agent": "deepseek.v3.2"}}
    win = _episode(status="SUCCESS", configuration=2, n_committed=1,
                   feedback=[], n_llm_calls=1).model_copy(update=priced)
    loss = _episode(status="BUDGET_EXHAUSTED", configuration=2, n_committed=1,
                    feedback=[_feedback(1, "STILL_DIVERGED", 9.0)], n_llm_calls=1).model_copy(update=priced)
    runs = [win, loss]

    rates = figs.success_rate_table(runs)
    costs = figs.cost_table(runs)

    [(key, rate)] = list(rates.items())
    assert rate == 50.0
    assert costs[key] > 0.0


def test_voltage_band_uses_the_corpus_bus_count() -> None:
    from builders import quality

    run = _episode(status="SUCCESS", configuration=3, n_committed=1, feedback=[])
    run = run.model_copy(update={
        "quality": quality(n_buses_out_of_band=59),
        "dataset_version": "ieee118-reactive-deficit-v1",
    })

    samples = figs.voltage_band_samples([run])

    [(key, values)] = list(samples.items())
    assert np.isclose(values[0], 100.0 * (118 - 59) / 118)
