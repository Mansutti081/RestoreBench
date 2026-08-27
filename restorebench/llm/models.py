# ABOUTME: The single source of truth for the Amazon Bedrock models under evaluation.
# ABOUTME: Every caller imports the ids from here so a rename cannot rot in a forgotten file.
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

# Bedrock model ids, region us-east-1. The Anthropic models are only reachable through
# an inference profile, hence the `global.` prefix; the open-weight models are on-demand
# and take their bare model id.
DEEPSEEK_V3_2: Final = "deepseek.v3.2"
KIMI_K2_5: Final = "moonshotai.kimi-k2.5"
GLM_5: Final = "zai.glm-5"

# Retired from the suite once the Claude family was measured on the first-party Anthropic API
# (Opus 5, Sonnet 5, Haiku 4.5): running these would spend Bedrock money to re-measure the same
# family. Their ids, slugs and prices stay so the de-risking cells under
# earlier de-risking runs remain repriceable and their result files remain readable.
OPUS_4_6: Final = "global.anthropic.claude-opus-4-6-v1"
HAIKU_4_5: Final = "global.anthropic.claude-haiku-4-5-20251001-v1:0"

# Evaluated on the card-comprehension gate and dropped (2026-07-11). Kept as valid Bedrock
# ids for the historical reports and the tests that reference them; not in the suite.
QWEN3_32B: Final = "qwen.qwen3-32b-v1:0"      # 0.79 overall vs 0.85 floor: does not read the card
GPT_OSS_120B: Final = "openai.gpt-oss-120b-1:0"  # F6 p_max_mw 0.85: misreads the generator table

# First-party Anthropic Messages API ids. Bare strings, no provider prefix: the `global.`
# inference-profile form above is Bedrock's. Haiku 4.5 exists on both sides and is NOT the same
# run: prices, token accounting and prompt caching differ, so it carries its own id and slug and
# the two can never collapse into one cell.
ANTHROPIC_OPUS_5: Final = "claude-opus-5"
ANTHROPIC_SONNET_5: Final = "claude-sonnet-5"
ANTHROPIC_HAIKU_4_5: Final = "claude-haiku-4-5"

ANTHROPIC_MODELS: Final[frozenset[str]] = frozenset(
    {ANTHROPIC_OPUS_5, ANTHROPIC_SONNET_5, ANTHROPIC_HAIKU_4_5}
)

# OpenAI Responses API ids, credentialed by OPENAI_API_KEY. Verified against the account's
# model list on 2026-08-20.
OPENAI_SOL: Final = "gpt-5.6-sol"

OPENAI_MODELS: Final[frozenset[str]] = frozenset({OPENAI_SOL})

# Opus 5 and Sonnet 5 reject temperature, top_p and top_k with a 400; Haiku 4.5 still accepts them.
ANTHROPIC_ACCEPTS_SAMPLING: Final[frozenset[str]] = frozenset({ANTHROPIC_HAIKU_4_5})


def provider_for(model_id: str) -> str:
    """Return which transport serves this model id."""
    if model_id in ANTHROPIC_MODELS:
        return "anthropic"
    if model_id in OPENAI_MODELS:
        return "openai"
    return "bedrock"


# The default open-weight suite: the three top-tier open-weight models, all of which clear the
# card-comprehension gate. Runners fall back to this when no model is named explicitly; pass
# --model to run any id in SUPPORTED_MODELS instead.
BENCHMARK_MODELS: Final[tuple[str, ...]] = (
    DEEPSEEK_V3_2,
    KIMI_K2_5,
    GLM_5,
)

_MODEL_SLUGS: Final[dict[str, str]] = {
    OPENAI_SOL: "sol",
    ANTHROPIC_OPUS_5: "opus-5",
    ANTHROPIC_SONNET_5: "sonnet-5",
    # Not "haiku-4-5": that slug is the Bedrock run's, and the slug names the results directory.
    ANTHROPIC_HAIKU_4_5: "haiku-4-5-anthropic",
    OPUS_4_6: "opus-4-6",
    HAIKU_4_5: "haiku-4-5",
    DEEPSEEK_V3_2: "deepseek-v3.2",
    KIMI_K2_5: "kimi-k2.5",
    GLM_5: "glm-5",
}

# Every model the runners can name a results directory for, and therefore every model the
# evaluation CLIs accept.
SUPPORTED_MODELS: Final[tuple[str, ...]] = tuple(_MODEL_SLUGS)

# Cheapest model in the suite on combined input and output rates. The dataset comprehension gate
# runs 40 scenarios across several seeds and models, so it defaults here rather than to the
# frontier model. The 2026-07-11 gate itself ran on Haiku 4.5, which was the cheapest model of the
# suite as it stood then; its recorded results name their own model and are unaffected.
CHEAPEST_MODEL: Final = DEEPSEEK_V3_2

TOKENS_PER_MILLION: Final = 1_000_000


@dataclass(frozen=True)
class ModelPrice:
    input_usd_per_mtok: float
    output_usd_per_mtok: float
    source: str


# Price snapshot: us-east-1, on-demand, standard tier, taken 2026-07-13. Prices live here with
# the ids — they are model metadata, so both the harness (dry-run spend) and the analysis layer
# read one table instead of drifting apart. Cost is always DERIVED from saved token counts and
# never written into a result file (`08 §2.3`, revised 2026-07-13): prices drift, and a frozen
# figure inside 6,250 JSONs would be a dead snapshot. Re-run the analysis to reprice.
# Not modelled (the harness uses none of them): the ~10% regional-endpoint premium, the
# batch/flex 50% discounts, and prompt-caching rebates.
# platform.claude.com/docs/en/pricing, taken 2026-07-31. Sonnet 5 is in an introductory window
# that ends 2026-08-31 ($2.00/$10.00); the standard rate is $3.00/$15.00. A run priced under one
# regime cannot be repriced under the other without saying which applied.
ANTHROPIC_PRICES: Final[dict[str, ModelPrice]] = {
    ANTHROPIC_OPUS_5: ModelPrice(5.00, 25.00, "platform.claude.com/docs/en/pricing (Claude Opus 5)"),
    ANTHROPIC_SONNET_5: ModelPrice(
        2.00, 10.00, "platform.claude.com/docs/en/pricing (Claude Sonnet 5, intro through 2026-08-31)"
    ),
    ANTHROPIC_HAIKU_4_5: ModelPrice(1.00, 5.00, "platform.claude.com/docs/en/pricing (Claude Haiku 4.5)"),
}


BEDROCK_PRICES: Final[dict[str, ModelPrice]] = {
    OPUS_4_6: ModelPrice(5.00, 25.00, "aws.amazon.com/bedrock/pricing (Claude Opus 4.6)"),
    HAIKU_4_5: ModelPrice(1.00, 5.00, "aws.amazon.com/bedrock/pricing (Claude Haiku 4.5)"),
    DEEPSEEK_V3_2: ModelPrice(0.62, 1.85, "aws.amazon.com/bedrock/pricing (DeepSeek V3.2, us-east)"),
    KIMI_K2_5: ModelPrice(0.60, 3.00, "llmreference.com/model/kimi-k2-5/aws-bedrock"),
    GLM_5: ModelPrice(1.00, 3.20, "aws.amazon.com/bedrock/pricing (GLM-5)"),
}


# Standard tier, like every other entry; batch/flex/fast tiers are not modelled.
OPENAI_PRICES: Final[dict[str, ModelPrice]] = {
    OPENAI_SOL: ModelPrice(5.00, 30.00, "developers.openai.com/api/docs/pricing (GPT-5.6 Sol, standard, short context)"),
}


def model_slug(model_id: str) -> str:
    try:
        return _MODEL_SLUGS[model_id]
    except KeyError:
        raise KeyError(f"unknown benchmark model {model_id!r}") from None


def token_cost_usd(model_id: str, *, tokens_in: int, tokens_out: int) -> float:
    """USD cost of one model's token usage. Unknown ids raise — a silent zero would hide spend."""
    if model_id in ANTHROPIC_MODELS:
        price = ANTHROPIC_PRICES[model_id]
    elif model_id in OPENAI_MODELS:
        price = OPENAI_PRICES[model_id]
        if price.input_usd_per_mtok == 0.0 and price.output_usd_per_mtok == 0.0:
            raise KeyError(
                f"{model_id!r} has a provisional zero price; set the real numbers in OPENAI_PRICES"
            )
    else:
        price = BEDROCK_PRICES[model_id]
    return (
        tokens_in / TOKENS_PER_MILLION * price.input_usd_per_mtok
        + tokens_out / TOKENS_PER_MILLION * price.output_usd_per_mtok
    )
