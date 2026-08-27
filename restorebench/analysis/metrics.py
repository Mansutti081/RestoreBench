# ABOUTME: Pure metric functions over saved ResolutionResponse objects.
# ABOUTME: Computes deterministic headline and secondary metrics without disk, LLM, or agent imports.
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import mean, median

from restorebench.schemas.response import ResolutionResponse


@dataclass(frozen=True)
class RateMetric:
    numerator: int
    denominator: int
    value: float


@dataclass(frozen=True)
class MeanMetric:
    denominator: int
    value: float | None


@dataclass(frozen=True)
class TimeMetrics:
    denominator: int
    mean_seconds: float | None
    median_seconds: float | None


@dataclass(frozen=True)
class QualityMetrics:
    clean_rate: RateMetric
    mean_buses_out_of_band: MeanMetric
    mean_worst_vm_pu: MeanMetric


@dataclass(frozen=True)
class TokenTotals:
    n_runs: int
    tokens_in: int
    tokens_out: int
    # Bedrock's own billed total, which also counts cache tokens — so it is not always in + out.
    tokens_total: int
    mean_tokens_in: float
    mean_tokens_out: float


def success_rate(responses: Sequence[ResolutionResponse]) -> RateMetric:
    attempted = len(responses)
    successes = sum(1 for response in responses if response.status == "SUCCESS")
    return _rate(successes, attempted)


def mean_maneuvers(responses: Sequence[ResolutionResponse]) -> MeanMetric:
    successes = [response.n_maneuvers for response in responses if response.status == "SUCCESS"]
    return _mean(successes)


def time_metrics(responses: Sequence[ResolutionResponse]) -> TimeMetrics:
    values = [response.total_runtime_seconds for response in responses]
    if not values:
        return TimeMetrics(denominator=0, mean_seconds=None, median_seconds=None)
    return TimeMetrics(denominator=len(values), mean_seconds=float(mean(values)), median_seconds=float(median(values)))


def quality_metrics(responses: Sequence[ResolutionResponse]) -> QualityMetrics:
    successes = [response for response in responses if response.status == "SUCCESS"]
    qualities = []
    for response in successes:
        if response.quality is None:
            raise ValueError(f"successful response {response.scenario_id} has no quality")
        qualities.append(response.quality)

    return QualityMetrics(
        clean_rate=_rate(sum(1 for quality in qualities if quality.clean), len(successes)),
        mean_buses_out_of_band=_mean([quality.n_buses_out_of_band for quality in qualities]),
        mean_worst_vm_pu=_mean([quality.worst_vm_pu for quality in qualities]),
    )


def invalid_maneuver_rate(responses: Sequence[ResolutionResponse]) -> RateMetric:
    invalid = sum(
        1
        for response in responses
        for feedback in response.failure_feedback
        if feedback.kind == "INVALID_ACTION"
    )
    return _rate(invalid, len(responses))


def token_totals(responses: Sequence[ResolutionResponse]) -> TokenTotals:
    tokens_in = sum(response.trace.total_llm_tokens_in for response in responses)
    tokens_out = sum(response.trace.total_llm_tokens_out for response in responses)
    tokens_total = sum(response.trace.total_llm_tokens for response in responses)
    n_runs = len(responses)
    return TokenTotals(
        n_runs=n_runs,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        tokens_total=tokens_total,
        mean_tokens_in=tokens_in / n_runs if n_runs else 0.0,
        mean_tokens_out=tokens_out / n_runs if n_runs else 0.0,
    )


def _rate(numerator: int, denominator: int) -> RateMetric:
    return RateMetric(numerator=numerator, denominator=denominator, value=numerator / denominator if denominator else 0.0)


def _mean(values: Sequence[float | int]) -> MeanMetric:
    return MeanMetric(denominator=len(values), value=float(mean(values)) if values else None)

