# ABOUTME: Aggregates saved evaluation responses into stratified tables and paired statistics.
# ABOUTME: Reads only serialized results and dataset labels; it never imports LLM or agent code.
from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Literal

from scipy import stats

from restorebench.analysis import metrics
from restorebench.eval.store import load_response
from restorebench.physics.actions import ACTION_POLICY_VERSION
from restorebench.physics.policies import RANKING_POLICY_VERSION, SOLVER_PROBE_POLICY_VERSION
from restorebench.schemas.dataset import DatasetManifest, ScenarioLabel
from restorebench.schemas.response import RESULT_SCHEMA_VERSION, ResolutionResponse


# Statistical resampling is the one legitimate RNG use in this project. The fixed seed makes
# regenerated tables byte-identical while preserving bootstrap uncertainty estimates.
BOOTSTRAP_SEED = 20260712
BOOTSTRAP_SAMPLES = 5_000
BONFERRONI_FAMILY_SIZE = 4
SIGNIFICANCE_ALPHA = 0.05
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRIVATE_LABELS_PATH = ROOT / "dataset/ieee118/private/labels.json"
DEFAULT_DATASET_MANIFEST_PATH = ROOT / "dataset/ieee118/manifest.json"

MetricName = Literal["success_rate", "mean_maneuvers", "mean_time", "clean_rate"]
MetricKind = Literal["proportion", "continuous"]

# Direction of "better" per metric: a positive treatment-minus-control delta is an
# improvement only for higher-is-better metrics. Maneuvers and wall-clock are costs.
METRIC_ORIENTATION: dict[str, bool] = {
    "success_rate": True,
    "clean_rate": True,
    "mean_maneuvers": False,
    "mean_time": False,
}


class ResultCompatibilityError(ValueError):
    """Base error for results that cannot enter benchmark aggregation."""


class LegacyResultNotComparableError(ResultCompatibilityError):
    """Raised when a result without a complete version stamp reaches comparative analysis."""


class ResultVersionMismatchError(ResultCompatibilityError):
    """Raised when a result version differs from its comparison corpus or peers."""

    def __init__(self, field: str, *, expected: object, actual: object) -> None:
        self.field = field
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"incompatible {field}: expected {expected!r}, got {actual!r}"
        )


@dataclass(frozen=True)
class StratumInfo:
    operating_profile_id: str | None = None
    resolution_regime: str | None = None
    witness_length: int | None = None
    witness_optimality: str | None = None


@dataclass(frozen=True)
class StratumMetricRow:
    key: str
    numerator: int
    denominator: int
    value: float


@dataclass(frozen=True)
class StratumMetricsRow:
    """Every headline metric for one stratum group, computed by analysis.metrics —
    the single source of the metric definitions (plan 14: every metric stratified)."""

    key: str
    success_rate: metrics.RateMetric
    mean_maneuvers: metrics.MeanMetric
    time: metrics.TimeMetrics
    quality: metrics.QualityMetrics
    invalid_maneuver_rate: metrics.RateMetric


@dataclass(frozen=True)
class PairedSamples:
    scenario_ids: tuple[str, ...]
    control: tuple[float, ...]
    treatment: tuple[float, ...]
    # Scenarios excluded because a SUCCESS-conditioned metric (mean_maneuvers, clean_rate)
    # had zero successes in one arm: the pair is undefined, dropped WITH visibility —
    # never silently, never by crashing the whole comparison.
    dropped_scenario_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ComparisonResult:
    n_pairs: int
    delta_mean: float
    p_value: float
    adjusted_p_value: float
    effect_size: float
    effect_size_label: str
    verdict: str


def load_results(results_dir: str | Path) -> list[ResolutionResponse]:
    cells = Path(results_dir) / "cells"
    return [load_response(path) for path in sorted(cells.glob("*.json"))]


def build_strata(
    responses: Sequence[ResolutionResponse],
    *,
    private_labels_path: Path = DEFAULT_PRIVATE_LABELS_PATH,
    dataset_manifest_path: Path = DEFAULT_DATASET_MANIFEST_PATH,
) -> dict[str, StratumInfo]:
    scenario_ids = [response.scenario_id for response in responses]
    labels = _load_private_labels(private_labels_path)
    strata = {}
    selected_labels: list[ScenarioLabel] = []
    for scenario_id in sorted(set(scenario_ids)):
        try:
            label = labels[scenario_id]
        except KeyError as exc:
            raise ValueError(f"unknown scenario_id {scenario_id}") from exc
        selected_labels.append(label)
        strata[scenario_id] = StratumInfo(
            operating_profile_id=label.recipe.operating_profile_id,
            resolution_regime=label.resolution_regime,
            witness_length=label.witness_length,
            witness_optimality=label.witness_optimality,
        )
    _validate_target_versions(
        responses,
        labels=selected_labels,
        dataset_manifest_path=dataset_manifest_path,
    )
    return strata


def success_by_stratum(
    responses: Sequence[ResolutionResponse],
    strata: Mapping[str, StratumInfo],
    key: Literal[
        "operating_profile_id",
        "resolution_regime",
        "witness_length",
        "witness_optimality",
    ]
    | None,
) -> list[StratumMetricRow]:
    _require_comparable_responses(responses)
    grouped: dict[str, list[ResolutionResponse]] = defaultdict(list)
    for response in responses:
        if response.scenario_id not in strata:
            raise ValueError(f"response scenario_id {response.scenario_id} missing from strata")
        group = "ALL" if key is None else str(getattr(strata[response.scenario_id], key))
        grouped[group].append(response)
    return [_success_row(group, grouped[group]) for group in sorted(grouped)]


def metrics_by_stratum(
    responses: Sequence[ResolutionResponse],
    strata: Mapping[str, StratumInfo],
    key: Literal[
        "operating_profile_id",
        "resolution_regime",
        "witness_length",
        "witness_optimality",
    ]
    | None,
) -> list[StratumMetricsRow]:
    _require_comparable_responses(responses)
    grouped: dict[str, list[ResolutionResponse]] = defaultdict(list)
    for response in responses:
        if response.scenario_id not in strata:
            raise ValueError(f"response scenario_id {response.scenario_id} missing from strata")
        group = "ALL" if key is None else str(getattr(strata[response.scenario_id], key))
        grouped[group].append(response)
    return [
        StratumMetricsRow(
            key=group,
            success_rate=metrics.success_rate(grouped[group]),
            mean_maneuvers=metrics.mean_maneuvers(grouped[group]),
            time=metrics.time_metrics(grouped[group]),
            quality=metrics.quality_metrics(grouped[group]),
            invalid_maneuver_rate=metrics.invalid_maneuver_rate(grouped[group]),
        )
        for group in sorted(grouped)
    ]


def paired_scenario_means(
    responses: Sequence[ResolutionResponse],
    *,
    control_config: int,
    treatment_config: int,
    metric: MetricName,
    expected_scenarios: int = 50,
    expected_repetitions: int = 5,
) -> PairedSamples:
    _require_comparable_responses(responses)
    control = _group_by_scenario(responses, control_config)
    treatment = _group_by_scenario(responses, treatment_config)
    if set(control) != set(treatment):
        mismatch = sorted(set(control) ^ set(treatment))
        raise ValueError(f"control and treatment scenario sets differ: {mismatch}")
    scenario_ids = tuple(sorted(control))
    if len(scenario_ids) != expected_scenarios:
        raise ValueError(f"expected {expected_scenarios} paired scenarios, got {len(scenario_ids)}")
    _validate_repetition_counts(control, treatment, scenario_ids, expected_repetitions)

    kept: list[str] = []
    dropped: list[str] = []
    control_means: list[float] = []
    treatment_means: list[float] = []
    for scenario_id in scenario_ids:
        control_mean = _scenario_mean(control[scenario_id], metric)
        treatment_mean = _scenario_mean(treatment[scenario_id], metric)
        if control_mean is None or treatment_mean is None:
            dropped.append(scenario_id)
            continue
        kept.append(scenario_id)
        control_means.append(control_mean)
        treatment_means.append(treatment_mean)
    if not kept:
        raise ValueError(f"no scenario has a defined {metric} in both configurations")
    return PairedSamples(
        scenario_ids=tuple(kept),
        control=tuple(control_means),
        treatment=tuple(treatment_means),
        dropped_scenario_ids=tuple(dropped),
    )


def compare_paired_values(
    control: Sequence[float],
    treatment: Sequence[float],
    *,
    metric_kind: MetricKind,
    higher_is_better: bool,
    expected_pairs: int = 50,
    n_comparisons: int = BONFERRONI_FAMILY_SIZE,
) -> ComparisonResult:
    if len(control) != expected_pairs or len(treatment) != expected_pairs:
        raise ValueError(f"expected {expected_pairs} paired scenario means")
    differences = [right - left for left, right in zip(control, treatment)]
    delta = float(mean(differences)) if differences else 0.0
    p_value = _p_value(control, treatment, differences, metric_kind)
    adjusted = min(1.0, p_value * n_comparisons)
    effect = _effect_size(control, treatment, differences, metric_kind)
    return ComparisonResult(
        n_pairs=len(differences),
        delta_mean=delta,
        p_value=p_value,
        adjusted_p_value=adjusted,
        effect_size=effect,
        effect_size_label=_effect_size_label(abs(effect)),
        verdict=_verdict(delta, adjusted, higher_is_better),
    )


def render_table(rows: Sequence[Mapping[str, object]]) -> str:
    if not rows:
        return ""
    columns = sorted({key for row in rows for key in row})
    ordered = sorted(rows, key=lambda row: tuple(_format_cell(row.get(column, "")) for column in columns))
    lines = [",".join(columns)]
    for row in ordered:
        lines.append(",".join(_format_cell(row.get(column, "")) for column in columns))
    return "\n".join(lines) + "\n"


def _success_row(key: str, responses: Sequence[ResolutionResponse]) -> StratumMetricRow:
    numerator = sum(1 for response in responses if response.status == "SUCCESS")
    denominator = len(responses)
    return StratumMetricRow(
        key=key,
        numerator=numerator,
        denominator=denominator,
        value=numerator / denominator if denominator else 0.0,
    )


def _load_private_labels(path: Path) -> dict[str, ScenarioLabel]:
    labels = [
        ScenarioLabel.model_validate(row)
        for row in json.loads(path.read_text(encoding="utf-8"))
    ]
    return {label.scenario_id: label for label in labels}


def _validate_target_versions(
    responses: Sequence[ResolutionResponse],
    *,
    labels: Sequence[ScenarioLabel],
    dataset_manifest_path: Path,
) -> None:
    manifest = DatasetManifest.model_validate_json(
        dataset_manifest_path.read_text(encoding="utf-8")
    )
    shared_versions = {
        (
            label.generation_metadata.shared_policy_versions.solver_probe,
            label.generation_metadata.shared_policy_versions.action,
        )
        for label in labels
    }
    if len(shared_versions) > 1:
        raise ResultVersionMismatchError(
            "corpus_shared_policy_versions",
            expected="one version set",
            actual=sorted(shared_versions),
        )
    solver_version, action_policy_version = (
        next(iter(shared_versions))
        if shared_versions
        else (SOLVER_PROBE_POLICY_VERSION, ACTION_POLICY_VERSION)
    )
    expected = {
        "dataset_version": manifest.dataset_version,
        "solver_version": solver_version,
        "action_policy_version": action_policy_version,
        "ranking_policy_version": RANKING_POLICY_VERSION,
        "result_schema_version": RESULT_SCHEMA_VERSION,
    }
    for response in responses:
        _require_complete_version_stamp(response)
        for field, expected_value in expected.items():
            actual_value = getattr(response, field)
            if actual_value != expected_value:
                raise ResultVersionMismatchError(
                    field,
                    expected=expected_value,
                    actual=actual_value,
                )


def _require_comparable_responses(
    responses: Sequence[ResolutionResponse],
) -> None:
    expected: dict[str, str] | None = None
    for response in responses:
        stamp = _require_complete_version_stamp(response)
        if expected is None:
            expected = stamp
            continue
        for field, expected_value in expected.items():
            actual_value = stamp[field]
            if actual_value != expected_value:
                raise ResultVersionMismatchError(
                    field,
                    expected=expected_value,
                    actual=actual_value,
                )


def _require_complete_version_stamp(
    response: ResolutionResponse,
) -> dict[str, str]:
    fields = (
        "dataset_version",
        "solver_version",
        "action_policy_version",
        "ranking_policy_version",
        "result_schema_version",
    )
    stamp = {field: getattr(response, field) for field in fields}
    missing = [field for field, value in stamp.items() if value is None]
    if missing:
        raise LegacyResultNotComparableError(
            f"result {response.request_id} is LEGACY_NON_COMPARABLE: "
            f"missing version stamp fields {missing}"
        )
    return {field: value for field, value in stamp.items() if value is not None}


def _group_by_scenario(
    responses: Sequence[ResolutionResponse],
    configuration: int,
) -> dict[str, list[ResolutionResponse]]:
    grouped: dict[str, list[ResolutionResponse]] = defaultdict(list)
    for response in responses:
        if response.configuration == configuration:
            grouped[response.scenario_id].append(response)
    return grouped


def _validate_repetition_counts(
    control: Mapping[str, Sequence[ResolutionResponse]],
    treatment: Mapping[str, Sequence[ResolutionResponse]],
    scenario_ids: Sequence[str],
    expected_repetitions: int,
) -> None:
    for scenario_id in scenario_ids:
        if len(control[scenario_id]) != expected_repetitions or len(treatment[scenario_id]) != expected_repetitions:
            raise ValueError(f"{scenario_id} does not have {expected_repetitions} repetitions per configuration")


def _scenario_mean(responses: Sequence[ResolutionResponse], metric: MetricName) -> float | None:
    values = [_run_metric_value(response, metric) for response in responses]
    finite = [value for value in values if value is not None]
    if not finite:
        return None  # SUCCESS-conditioned metric with zero successes: undefined, caller drops the pair
    return float(mean(finite))


def _run_metric_value(response: ResolutionResponse, metric: MetricName) -> float | None:
    if metric == "success_rate":
        return 1.0 if response.status == "SUCCESS" else 0.0
    if metric == "mean_time":
        return response.total_runtime_seconds
    if metric == "mean_maneuvers":
        return float(response.n_maneuvers) if response.status == "SUCCESS" else None
    if metric == "clean_rate":
        if response.status != "SUCCESS":
            return None
        if response.quality is None:
            raise ValueError(f"successful response {response.scenario_id} has no quality")
        return 1.0 if response.quality.clean else 0.0
    raise ValueError(f"unsupported metric {metric}")


def _p_value(
    control: Sequence[float],
    treatment: Sequence[float],
    differences: Sequence[float],
    metric_kind: MetricKind,
) -> float:
    if all(difference == 0.0 for difference in differences):
        return 1.0
    if metric_kind == "proportion":
        return _paired_bootstrap_p_value(differences)
    try:
        return float(stats.wilcoxon(treatment, control).pvalue)
    except ValueError:
        return 1.0


def _paired_bootstrap_p_value(differences: Sequence[float]) -> float:
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(differences)
    means = [
        mean(differences[rng.randrange(n)] for _ in range(n))
        for _ in range(BOOTSTRAP_SAMPLES)
    ]
    le_zero = sum(1 for value in means if value <= 0.0) / BOOTSTRAP_SAMPLES
    ge_zero = sum(1 for value in means if value >= 0.0) / BOOTSTRAP_SAMPLES
    return min(1.0, 2.0 * min(le_zero, ge_zero))


def _effect_size(
    control: Sequence[float],
    treatment: Sequence[float],
    differences: Sequence[float],
    metric_kind: MetricKind,
) -> float:
    if metric_kind == "proportion":
        return _cohens_h(mean(control), mean(treatment))
    if len(differences) < 2:
        return 0.0
    spread = stdev(differences)
    return 0.0 if spread == 0.0 else float(mean(differences) / spread)


def _cohens_h(control_rate: float, treatment_rate: float) -> float:
    left = max(0.0, min(1.0, control_rate))
    right = max(0.0, min(1.0, treatment_rate))
    return 2.0 * math.asin(math.sqrt(right)) - 2.0 * math.asin(math.sqrt(left))


def _effect_size_label(effect: float) -> str:
    if effect < 0.2:
        return "negligible"
    if effect < 0.5:
        return "small"
    if effect < 0.8:
        return "medium"
    return "large"


def _verdict(delta: float, adjusted_p_value: float, higher_is_better: bool) -> str:
    if adjusted_p_value >= SIGNIFICANCE_ALPHA or delta == 0.0:
        return "not significantly different"
    # A positive treatment-minus-control delta is an improvement only for
    # higher-is-better metrics; for maneuvers/time it means the treatment is worse.
    treatment_improved = (delta > 0.0) == higher_is_better
    return "treatment better" if treatment_improved else "control better"


def _format_cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)
