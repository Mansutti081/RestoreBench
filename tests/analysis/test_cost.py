# ABOUTME: Verifies the Bedrock cost layer: the frozen price table and per-run/cell cost derivation.
# ABOUTME: Costs are derived from saved token counts; no pricing is baked into the result files.
import pytest

from restorebench.analysis import cost
from restorebench.llm import models

from builders import maneuver, quality, response


def _run(scenario_id: str, *, configuration: int, model_id: str, tokens_in: int, tokens_out: int):
    run = response(
        scenario_id=scenario_id,
        configuration=configuration,
        status="SUCCESS",
        converged=True,
        result_quality=quality(),
        maneuvers=[maneuver({"type": "GEN_V_SETPOINT", "gen_id": 10, "new_vm_pu": 1.02})],
    )
    assignment = (
        {"single_agent": model_id}
        if configuration in {1, 2, 4}
        else {"analyst": model_id, "executor": model_id, "orchestrator": model_id}
    )
    trace = run.trace.model_copy(
        update={
            "total_llm_tokens_in": tokens_in,
            "total_llm_tokens_out": tokens_out,
            "total_llm_tokens": tokens_in + tokens_out,
        }
    )
    return run.model_copy(update={"llm_assignment": assignment, "trace": trace})


def test_price_table_covers_every_benchmark_model() -> None:
    # A subset, not an equality: retired models keep their price so their stored runs stay
    # repriceable. Every model still in the suite must have one.
    assert set(models.BENCHMARK_MODELS) <= set(cost.BEDROCK_PRICES)
    for price in cost.BEDROCK_PRICES.values():
        assert price.input_usd_per_mtok > 0
        assert price.output_usd_per_mtok > 0


def test_run_cost_uses_input_and_output_rates_separately() -> None:
    # Haiku 4.5: $1.00 / 1M in, $5.00 / 1M out
    run = _run("S0001", configuration=2, model_id=models.HAIKU_4_5, tokens_in=1_000_000, tokens_out=1_000_000)

    assert cost.run_cost_usd(run) == pytest.approx(6.00)


def test_run_cost_scales_with_token_counts() -> None:
    run = _run("S0001", configuration=2, model_id=models.HAIKU_4_5, tokens_in=500_000, tokens_out=100_000)

    assert cost.run_cost_usd(run) == pytest.approx(0.5 * 1.00 + 0.1 * 5.00)


def test_unknown_model_is_a_hard_error_not_a_silent_zero() -> None:
    run = _run("S0001", configuration=2, model_id="not-a-benchmark-model", tokens_in=1000, tokens_out=100)

    with pytest.raises(KeyError):
        cost.run_cost_usd(run)


def test_cost_by_cell_groups_by_model_and_configuration() -> None:
    runs = [
        _run("S0001", configuration=2, model_id=models.HAIKU_4_5, tokens_in=1_000_000, tokens_out=0),
        _run("S0002", configuration=2, model_id=models.HAIKU_4_5, tokens_in=1_000_000, tokens_out=0),
        _run("S0001", configuration=3, model_id=models.GLM_5, tokens_in=1_000_000, tokens_out=0),
    ]

    rows = cost.cost_by_cell(runs)

    by_key = {(row.model_id, row.configuration): row for row in rows}
    haiku = by_key[(models.HAIKU_4_5, 2)]
    assert haiku.n_runs == 2
    assert haiku.total_usd == pytest.approx(2.00)
    assert haiku.mean_usd_per_run == pytest.approx(1.00)
    assert haiku.tokens_in == 2_000_000
    glm = by_key[(models.GLM_5, 3)]
    assert glm.total_usd == pytest.approx(1.00)  # GLM-5: $1.00 / 1M in


def test_total_cost_sums_every_run() -> None:
    runs = [
        _run("S0001", configuration=2, model_id=models.HAIKU_4_5, tokens_in=1_000_000, tokens_out=0),
        _run("S0002", configuration=3, model_id=models.GLM_5, tokens_in=1_000_000, tokens_out=0),
    ]

    assert cost.total_cost_usd(runs) == pytest.approx(2.00)


def test_prices_carry_their_provenance() -> None:
    # A price with no source is unciteable in the paper.
    for price in cost.BEDROCK_PRICES.values():
        assert price.source
