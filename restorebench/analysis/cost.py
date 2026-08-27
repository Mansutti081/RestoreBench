# ABOUTME: Derives Amazon Bedrock inference cost from the token counts saved in each ResolutionResponse.
# ABOUTME: Prices are a frozen, cited snapshot; nothing monetary is written into the result files.
from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from restorebench.eval.store import model_id_from_response
from restorebench.llm.models import BEDROCK_PRICES, ModelPrice, token_cost_usd
from restorebench.schemas.response import ResolutionResponse


# The price table lives with the model ids (`restorebench/llm/models.py`) so the harness's dry-run
# spend line and this layer read one table. Re-exported here for the notebooks' convenience.
__all__ = ["BEDROCK_PRICES", "CostRow", "ModelPrice", "cost_by_cell", "run_cost_usd", "total_cost_usd"]


@dataclass(frozen=True)
class CostRow:
    model_id: str
    configuration: int
    n_runs: int
    tokens_in: int
    tokens_out: int
    total_usd: float
    mean_usd_per_run: float


def run_cost_usd(response: ResolutionResponse) -> float:
    """Inference cost of one run. Unknown models raise — a silent zero would hide real spend."""
    return token_cost_usd(
        model_id_from_response(response),
        tokens_in=response.trace.total_llm_tokens_in,
        tokens_out=response.trace.total_llm_tokens_out,
    )


def total_cost_usd(responses: Sequence[ResolutionResponse]) -> float:
    return sum(run_cost_usd(response) for response in responses)


def cost_by_cell(responses: Sequence[ResolutionResponse]) -> list[CostRow]:
    """Cost per (model x configuration) cell — the grid the paper reports spend on."""
    grouped: dict[tuple[str, int], list[ResolutionResponse]] = defaultdict(list)
    for response in responses:
        grouped[(model_id_from_response(response), response.configuration)].append(response)

    rows = []
    for (model_id, configuration), runs in sorted(grouped.items()):
        total = sum(run_cost_usd(run) for run in runs)
        rows.append(
            CostRow(
                model_id=model_id,
                configuration=configuration,
                n_runs=len(runs),
                tokens_in=sum(run.trace.total_llm_tokens_in for run in runs),
                tokens_out=sum(run.trace.total_llm_tokens_out for run in runs),
                total_usd=total,
                mean_usd_per_run=total / len(runs),
            )
        )
    return rows
