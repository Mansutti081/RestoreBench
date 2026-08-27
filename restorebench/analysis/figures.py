# ABOUTME: Prepares data for the two headline analysis figures and saves them when plotting deps exist.
# ABOUTME: Keeps notebooks thin by moving grouping and pairing logic into tested Python functions.
from __future__ import annotations

import importlib
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from restorebench.analysis import aggregate
from restorebench.schemas.response import ResolutionResponse


@dataclass(frozen=True)
class LadderPoint:
    model_id: str
    configuration: int
    mean_success_rate: float
    stddev_success_rate: float
    n_repetitions: int


def architecture_ladder_data(responses: Sequence[ResolutionResponse]) -> list[LadderPoint]:
    by_repetition: dict[tuple[str, int, int], list[ResolutionResponse]] = defaultdict(list)
    for response in responses:
        by_repetition[(_model_id(response), response.configuration, response.repetition_index)].append(response)

    by_cell: dict[tuple[str, int], list[float]] = defaultdict(list)
    for (model_id, configuration, _repetition), runs in by_repetition.items():
        successes = sum(1 for run in runs if run.status == "SUCCESS")
        by_cell[(model_id, configuration)].append(successes / len(runs))

    return [
        LadderPoint(
            model_id=model_id,
            configuration=configuration,
            mean_success_rate=float(mean(values)),
            stddev_success_rate=float(stdev(values)) if len(values) > 1 else 0.0,
            n_repetitions=len(values),
        )
        for (model_id, configuration), values in sorted(by_cell.items())
    ]


def _maneuvers_delta(
    model_responses: Sequence[ResolutionResponse],
    *,
    control: int,
    treatment: int,
    expected_scenarios: int,
    expected_repetitions: int,
) -> tuple[float | None, int]:
    try:
        pairs = aggregate.paired_scenario_means(
            model_responses,
            control_config=control,
            treatment_config=treatment,
            metric="mean_maneuvers",
            expected_scenarios=expected_scenarios,
            expected_repetitions=expected_repetitions,
        )
    except ValueError as exc:
        # The only legitimate miss: no scenario has a defined maneuvers pair (zero
        # successes in one arm everywhere). Shape/completeness errors keep propagating —
        # the success-rate pairing above already vouched for the data's completeness.
        if "no scenario has a defined" in str(exc):
            return None, 0
        raise
    delta = float(mean(right - left for left, right in zip(pairs.control, pairs.treatment)))
    return delta, len(pairs.scenario_ids)


def save_architecture_ladder(responses: Sequence[ResolutionResponse], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    points = architecture_ladder_data(responses)
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(8, 4.8))
    for model_id in sorted({point.model_id for point in points}):
        series = [point for point in points if point.model_id == model_id]
        axis.errorbar(
            [point.configuration for point in series],
            [point.mean_success_rate for point in series],
            yerr=[point.stddev_success_rate for point in series],
            marker="o",
            label=model_id,
        )
    axis.set_xlabel("Configuration")
    axis.set_ylabel("Success rate")
    axis.set_xticks([1, 2, 3, 4, 5])
    axis.set_ylim(0, 1)
    if points:
        axis.legend()
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path


def _model_id(response: ResolutionResponse) -> str:
    for role in ("single_agent", "analyst", "executor", "orchestrator"):
        model_id = response.llm_assignment.get(role)
        if model_id:
            return model_id
    return "unknown"


def _pyplot() -> Any:
    return importlib.import_module("matplotlib.pyplot")
