# ABOUTME: Verifies benchmark model metadata that is shared by harness filenames and exports.
# ABOUTME: Keeps the public HOWTO slugs aligned with the Bedrock model id suite.
import pytest

from restorebench.llm import models


def test_every_benchmark_model_has_the_documented_slug() -> None:
    slugs = {models.model_slug(model_id) for model_id in models.BENCHMARK_MODELS}

    assert slugs == {"deepseek-v3.2", "kimi-k2.5", "glm-5"}


def test_the_bedrock_suite_holds_only_the_models_the_anthropic_run_did_not_cover() -> None:
    """Opus 5, Sonnet 5 and Haiku 4.5 were measured on the first-party Anthropic API, so their
    Bedrock counterparts would spend money to re-measure the same family."""
    assert models.BENCHMARK_MODELS == (models.DEEPSEEK_V3_2, models.KIMI_K2_5, models.GLM_5)
    assert models.OPUS_4_6 not in models.BENCHMARK_MODELS
    assert models.HAIKU_4_5 not in models.BENCHMARK_MODELS


def test_model_slug_rejects_unknown_model_ids() -> None:
    with pytest.raises(KeyError, match="unknown benchmark model"):
        models.model_slug("provider.unreviewed-model")


def test_price_table_covers_every_benchmark_model() -> None:
    from restorebench.llm.models import BEDROCK_PRICES, BENCHMARK_MODELS

    assert set(BENCHMARK_MODELS) <= set(BEDROCK_PRICES)
    for price in BEDROCK_PRICES.values():
        assert price.input_usd_per_mtok > 0
        assert price.output_usd_per_mtok > 0
        assert price.source  # a price with no citation is unpublishable


def test_retired_models_keep_their_price_so_their_stored_runs_stay_repriceable() -> None:
    """The de-risking run holds Opus 4.6 and Bedrock
    Haiku 4.5 cells. Cost is derived from saved token counts, so dropping their prices would
    turn those results into a KeyError instead of a number."""
    for retired in (models.OPUS_4_6, models.HAIKU_4_5):
        assert models.token_cost_usd(retired, tokens_in=1_000_000, tokens_out=0) > 0
        assert models.model_slug(retired)


def test_run_cost_uses_input_and_output_rates_separately() -> None:
    from restorebench.llm.models import HAIKU_4_5, token_cost_usd

    # Haiku 4.5: $1.00 / 1M in, $5.00 / 1M out
    assert token_cost_usd(HAIKU_4_5, tokens_in=1_000_000, tokens_out=1_000_000) == pytest.approx(6.00)
    assert token_cost_usd(HAIKU_4_5, tokens_in=500_000, tokens_out=100_000) == pytest.approx(1.0)


def test_unknown_model_cost_raises_instead_of_costing_zero() -> None:
    from restorebench.llm.models import token_cost_usd

    with pytest.raises(KeyError):
        token_cost_usd("not-a-benchmark-model", tokens_in=1000, tokens_out=100)
