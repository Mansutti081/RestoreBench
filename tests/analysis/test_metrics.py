# ABOUTME: Tests pure metric functions over canned ResolutionResponse objects.
# ABOUTME: No disk, solver, LLM, or Bedrock calls are involved.

from restorebench.analysis import metrics
from restorebench.schemas.feedback import FailureFeedback

from builders import diagnostics, maneuver, quality, response


def _response(
    scenario_id: str,
    *,
    status: str = "SUCCESS",
    n_maneuvers: int = 1,
    runtime: float = 10.0,
    clean: bool = True,
    buses_out: int = 0,
    worst_vm: float = 0.99,
):
    converged = status == "SUCCESS"
    maneuvers = [
        maneuver({"type": "GEN_V_SETPOINT", "gen_id": 10, "new_vm_pu": 1.02})
        for _ in range(n_maneuvers if converged else 0)
    ]
    result = response(
        scenario_id=scenario_id,
        status=status,
        converged=converged,
        result_quality=quality(clean=clean, n_buses_out_of_band=buses_out, worst_vm_pu=worst_vm)
        if converged
        else None,
        maneuvers=maneuvers,
    )
    return result.model_copy(update={"n_maneuvers": n_maneuvers, "total_runtime_seconds": runtime})


def test_success_rate_uses_all_attempted_denominator() -> None:
    runs = [
        _response("S0001", status="SUCCESS"),
        _response("S0002", status="BUDGET_EXHAUSTED"),
        _response("S0003", status="TIMEOUT"),
        _response("S0004", status="SUCCESS"),
    ]

    result = metrics.success_rate(runs)

    assert result.numerator == 2
    assert result.denominator == 4
    assert result.value == 0.5


def test_mean_maneuvers_ignores_non_success_runs() -> None:
    runs = [
        _response("S0001", status="SUCCESS", n_maneuvers=1),
        _response("S0002", status="BUDGET_EXHAUSTED", n_maneuvers=10),
        _response("S0003", status="SUCCESS", n_maneuvers=3),
    ]

    result = metrics.mean_maneuvers(runs)

    assert result.denominator == 2
    assert result.value == 2.0


def test_mean_time_reports_mean_and_median_over_all_attempted() -> None:
    result = metrics.time_metrics(
        [
            _response("S0001", runtime=1.0),
            _response("S0002", status="LLM_FAILURE", runtime=100.0),
            _response("S0003", runtime=4.0),
        ]
    )

    assert result.denominator == 3
    assert result.mean_seconds == 35.0
    assert result.median_seconds == 4.0


def test_quality_metrics_use_success_denominator_only() -> None:
    result = metrics.quality_metrics(
        [
            _response("S0001", clean=True, buses_out=0, worst_vm=0.99),
            _response("S0002", status="TOOL_FAILURE"),
            _response("S0003", clean=False, buses_out=2, worst_vm=0.94),
        ]
    )

    assert result.clean_rate.numerator == 1
    assert result.clean_rate.denominator == 2
    assert result.mean_buses_out_of_band.denominator == 2
    assert result.mean_buses_out_of_band.value == 1.0
    assert result.mean_worst_vm_pu.value == 0.965


def test_invalid_maneuver_rate_counts_failure_feedback_entries_per_attempt() -> None:
    invalid = FailureFeedback(
        iteration=0,
        kind="INVALID_ACTION",
        diagnostics=diagnostics(),
        detail="bad action",
        maneuver=None,
    )
    malformed = invalid.model_copy(update={"kind": "MALFORMED_OUTPUT"})
    run = _response("S0001").model_copy(update={"failure_feedback": [invalid, invalid, malformed]})

    result = metrics.invalid_maneuver_rate([run, _response("S0002")])

    assert result.numerator == 2
    assert result.denominator == 2
    assert result.value == 1.0


def test_token_totals_are_counts_not_currency() -> None:
    first = _response("S0001")
    second = _response("S0002")
    first_trace = first.trace.model_copy(update={"total_llm_tokens_in": 10, "total_llm_tokens_out": 5})
    second_trace = second.trace.model_copy(update={"total_llm_tokens_in": 30, "total_llm_tokens_out": 15})

    result = metrics.token_totals(
        [first.model_copy(update={"trace": first_trace}), second.model_copy(update={"trace": second_trace})]
    )

    assert result.n_runs == 2
    assert result.tokens_in == 40
    assert result.tokens_out == 20
    assert result.mean_tokens_in == 20.0
    assert result.mean_tokens_out == 10.0


def test_token_totals_include_the_bedrock_billed_total() -> None:
    run = _response("S0001")
    trace = run.trace.model_copy(
        update={"total_llm_tokens_in": 100, "total_llm_tokens_out": 40, "total_llm_tokens": 180}
    )

    totals = metrics.token_totals([run.model_copy(update={"trace": trace})])

    assert totals.tokens_in == 100
    assert totals.tokens_out == 40
    assert totals.tokens_total == 180  # Bedrock billed more than in+out (cache tokens)
