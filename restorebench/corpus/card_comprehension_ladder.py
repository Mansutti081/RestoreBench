# ABOUTME: Prepares the Card comprehension ladder report across the evaluation model suite.
# ABOUTME: Runs through card_comprehension_review.review when explicitly invoked by an operator.
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Sequence

from restorebench.llm.models import BENCHMARK_MODELS
from restorebench.corpus import card_comprehension_review as review

LADDER_MODEL_IDS = BENCHMARK_MODELS
SEEDS = (42, 1337, 7)
N_SCENARIOS = 40
# The Qwen3.5 ladder in card_comprehension_ladder_report.md was run against models served
# outside Bedrock and is a historical artifact. It is never overwritten: this run writes its
# own report, and the two are not comparable — different models, different serving stack.
DEFAULT_REPORT = review.DATASET_DIR / "card_comprehension_bedrock_ladder_report.md"


@dataclass(frozen=True)
class LadderRun:
    seed: int
    report: review.CardComprehensionReport


def run_ladder(
    *,
    n: int = N_SCENARIOS,
    models: Sequence[str] = LADDER_MODEL_IDS,
    seeds: Sequence[int] = SEEDS,
    report_path: str | Path = DEFAULT_REPORT,
) -> Path:
    runs: list[LadderRun] = []
    for model in models:
        for seed in seeds:
            runs.append(LadderRun(seed=seed, report=review.review(n=n, model=model, seed=seed)))

    output_path = Path(report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_markdown(runs=runs, n=n, seeds=seeds), encoding="utf-8")
    return output_path


def render_markdown(
    *,
    runs: Sequence[LadderRun],
    n: int,
    seeds: Sequence[int],
) -> str:
    lines = [
        "# Card-comprehension ladder report",
        "",
        "Prepared by `restorebench/corpus/card_comprehension_ladder.py`.",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Probe: `review(n={n}, seed in {list(seeds)})` across the evaluation model suite.",
        "",
        _floor_line(),
        "",
        "## Multi-seed validation",
        "",
        "```",
        _run_header(),
        *(_run_row(run) for run in runs),
        "```",
        "",
        "## Aggregates",
        "",
        "```",
        _aggregate_header(),
        *(_aggregate_rows(runs)),
        "```",
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=N_SCENARIOS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    args = parser.parse_args(argv)

    models = tuple(args.models) if args.models else LADDER_MODEL_IDS
    seeds = tuple(args.seeds) if args.seeds else SEEDS
    run_ladder(n=args.n, models=models, seeds=seeds, report_path=args.report)
    return 0


def _floor_line() -> str:
    family_floors = " | ".join(
        f"`{family} {review.FLOOR_PER_FAMILY[family]:.2f}`" for family in review.FAMILIES
    )
    return f"Floors: `overall {review.FLOOR_OVERALL:.2f}` | {family_floors}"


def _run_header() -> str:
    families = " ".join(f"{family:>6}" for family in review.FAMILIES)
    return f"{'model':<48} {'seed':>5} {'overall':>8} {families} {'passed':>7}"


def _run_row(run: LadderRun) -> str:
    report = run.report
    families = " ".join(f"{report.per_family_accuracy.get(family, 0.0):>6.3f}" for family in review.FAMILIES)
    return (
        f"{report.model:<48} {run.seed:>5} "
        f"{report.overall_accuracy:>8.3f} {families} {str(report.passed):>7}"
    )


def _aggregate_header() -> str:
    families = " ".join(f"{family:>6}" for family in review.FAMILIES)
    return f"{'model':<48} {'metric':>6} {'overall':>8} {families}"


def _aggregate_rows(runs: Sequence[LadderRun]) -> list[str]:
    grouped: dict[str, list[review.CardComprehensionReport]] = defaultdict(list)
    for run in runs:
        grouped[run.report.model].append(run.report)

    rows: list[str] = []
    for model, model_reports in grouped.items():
        rows.append(_aggregate_row(model, "mean", model_reports))
        rows.append(_aggregate_row(model, "min", model_reports))
    return rows


def _aggregate_row(model: str, metric: str, reports: Sequence[review.CardComprehensionReport]) -> str:
    if metric == "mean":
        overall = mean(report.overall_accuracy for report in reports)
        families = [mean(report.per_family_accuracy.get(family, 0.0) for report in reports) for family in review.FAMILIES]
    elif metric == "min":
        overall = min(report.overall_accuracy for report in reports)
        families = [min(report.per_family_accuracy.get(family, 0.0) for report in reports) for family in review.FAMILIES]
    else:
        raise ValueError(f"unsupported aggregate metric: {metric}")
    family_values = " ".join(f"{value:>6.3f}" for value in families)
    return f"{model:<48} {metric:>6} {overall:>8.3f} {family_values}"


if __name__ == "__main__":
    raise SystemExit(main())
