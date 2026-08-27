# ABOUTME: Builds the primary long-format dataset and the two summary tables from stored results.
# ABOUTME: Every figure derives from the CSV this writes; nothing here calls a provider.
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from restorebench.llm import models

ROOT = Path(__file__).resolve().parents[2]
# One store per provider campaign. They are pooled only after every cell is confirmed to carry
# the same five-field version stamp; coverage is still judged inside each store, because a case
# finished on one provider does not become partial by being unfinished on the other.
STORES = (
    ROOT / "results/ieee118-anthropic/cells",
    ROOT / "results/ieee118-bedrock/cells",
    ROOT / "results/ieee118-openai/cells",
)
LABELS = ROOT / "dataset/ieee118/private/labels.json"
OUT_DIR = ROOT / "reports/figure_data"

# The whole augmented network is in service, so one manifest-declared denominator applies per dataset.
TOTAL_BUSES = 118
FULL_COVERAGE_CELLS = 9

ARCHITECTURE = {
    1: "chatbot",
    2: "single agent",
    3: "multi-agent",
}

# Combinations that exist in the design but carry no data yet. They are written into every
# output so a plot that reads the CSV sees them as missing rather than as absent.
PLANNED_MODELS = [
    (models.OPENAI_SOL, "GPT-5.6 Sol", "OpenAI", "openai"),
    (models.DEEPSEEK_V3_2, "DeepSeek V3.2", "DeepSeek", "bedrock"),
    (models.KIMI_K2_5, "Kimi K2.5", "Moonshot AI", "bedrock"),
    (models.GLM_5, "GLM-5", "Z.ai", "bedrock"),
]
DISPLAY = {
    "sol": "GPT-5.6 Sol",
    "haiku-4-5-anthropic": "Claude Haiku 4.5",
    "sonnet-5": "Claude Sonnet 5",
    "opus-5": "Claude Opus 5",
}
# DISPLAY covers the Anthropic family; the open-weight labels live here instead.
BEDROCK_DISPLAY = {
    "deepseek-v3.2": "DeepSeek V3.2",
    "kimi-k2.5": "Kimi K2.5",
    "glm-5": "GLM-5",
}
# The five fields that make two cells comparable. A disagreement is fatal, never a warning.
VERSION_STAMP_FIELDS = (
    "dataset_version",
    "solver_version",
    "action_policy_version",
    "ranking_policy_version",
    "result_schema_version",
)
BASELINE = ("Claude Haiku 4.5", "chatbot")


def _labels() -> dict[str, dict[str, Any]]:
    return {row["scenario_id"]: row for row in json.loads(LABELS.read_text(encoding="utf-8"))}


def _model_of(result: dict[str, Any]) -> str:
    assignment = result["llm_assignment"]
    return assignment.get("single_agent") or assignment.get("analyst")


def _display_name(slug: str) -> str:
    return DISPLAY.get(slug) or BEDROCK_DISPLAY[slug]


def _assert_one_version_stamp(stored: list[dict[str, Any]]) -> None:
    """Refuse to pool cells measured under different contracts.

    Two campaigns share a table only if the dataset, the solver, the action and ranking policies
    and the result schema all agree. Merging across a difference would publish one number built
    from two experiments, so this fails the build rather than annotating the output.
    """
    for field in VERSION_STAMP_FIELDS:
        seen = {result[field] for result in stored}
        if len(seen) > 1:
            raise ValueError(
                f"cells disagree on {field}: {sorted(seen)}. "
                "Incomparable campaigns cannot enter one table."
            )


def _rows() -> tuple[list[dict[str, Any]], list[str]]:
    """One row per stored cell, plus placeholder rows for everything still to run."""
    labels = _labels()
    stored: list[dict[str, Any]] = []
    full_by_store: list[set[str]] = []
    partial_cases: set[str] = set()
    for store in STORES:
        cells = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(store.glob("*.json"))
            if "manifest" not in path.name
        ]
        coverage = Counter(result["scenario_id"] for result in cells)
        store_full = {case for case, n in coverage.items() if n == FULL_COVERAGE_CELLS}
        for result in cells:
            result["_full_coverage"] = result["scenario_id"] in store_full
        partial_cases.update(case for case, n in coverage.items() if n != FULL_COVERAGE_CELLS)
        full_by_store.append(store_full)
        stored.extend(cells)

    _assert_one_version_stamp(stored)
    full = sorted(set().union(*full_by_store) if full_by_store else set())
    partial = sorted(partial_cases)

    rows: list[dict[str, Any]] = []
    for result in stored:
        case = result["scenario_id"]
        model_id = _model_of(result)
        slug = models.model_slug(model_id)
        quality = result.get("quality") or {}
        out_of_band = quality.get("n_buses_out_of_band")
        in_band = None if out_of_band is None else TOTAL_BUSES - out_of_band
        trace = result["trace"]
        tokens_in = trace.get("total_llm_tokens_in") or 0
        tokens_out = trace.get("total_llm_tokens_out") or 0
        kinds = Counter(entry["kind"] for entry in (result.get("failure_feedback") or ()))
        label = labels[case]

        rows.append(
            {
                "case_id": case,
                "model_id": model_id,
                "model_slug": slug,
                "model_display": _display_name(slug),
                "architecture": ARCHITECTURE[result["configuration"]],
                "configuration": result["configuration"],
                "repetition_index": result["repetition_index"],
                # A partial case is excluded from every aggregation, but its rows are kept so the
                # exclusion is auditable instead of invisible.
                "test_status": "Tested" if result["_full_coverage"] else "Partially tested",
                "resolution_regime": label["resolution_regime"],
                "witness_length": label["witness_length"],
                "solved": int(result["status"] == "SUCCESS"),
                "status": result["status"],
                "n_maneuvers": result["n_maneuvers"],
                "input_tokens": tokens_in,
                "output_tokens": tokens_out,
                "total_tokens": trace.get("total_llm_tokens") or 0,
                "n_llm_calls": trace.get("n_llm_calls"),
                "n_tool_calls": trace.get("n_tool_calls"),
                "n_power_flows": trace.get("n_power_flows"),
                "execution_time_s": result.get("total_runtime_seconds"),
                "input_cost_usd": models.token_cost_usd(model_id, tokens_in=tokens_in, tokens_out=0),
                "output_cost_usd": models.token_cost_usd(model_id, tokens_in=0, tokens_out=tokens_out),
                "total_cost_usd": models.token_cost_usd(
                    model_id, tokens_in=tokens_in, tokens_out=tokens_out
                ),
                "total_buses": TOTAL_BUSES,
                "buses_in_band": in_band,
                "buses_in_band_pct": None if in_band is None else in_band / TOTAL_BUSES * 100,
                "worst_vm_pu": quality.get("worst_vm_pu"),
                "clean": quality.get("clean"),
                # Verified across the campaign: a final state exists exactly when the cell solved.
                "final_state_available": bool(quality),
                "error_type": kinds.most_common(1)[0][0] if kinds else None,
                "n_invalid_action": kinds.get("INVALID_ACTION", 0),
                "n_malformed_output": kinds.get("MALFORMED_OUTPUT", 0),
                "n_preview_diverged": kinds.get("PREVIEW_DIVERGED", 0),
                "dataset_version": result["dataset_version"],
                "solver_version": result["solver_version"],
                "action_policy_version": result["action_policy_version"],
                "ranking_policy_version": result["ranking_policy_version"],
                "result_schema_version": result["result_schema_version"],
                "transport": models.provider_for(model_id),
                "notes": "",
            }
        )

    # A placeholder only stands in for a cell that does not exist. Once a model has measured
    # rows, emitting one too would put a "reserved for a future campaign" line next to its data.
    measured = {(row["model_slug"], row["configuration"]) for row in rows}
    for model_id, display, provider, transport in PLANNED_MODELS:
        slug = models.model_slug(model_id) if model_id in models.SUPPORTED_MODELS else display
        for configuration in (1, 2, 3):
            if (slug, configuration) in measured:
                continue
            rows.append(
                _empty_row(full[0] if full else "TBD", model_id, slug, display, transport, configuration)
            )
    return rows, partial


def _empty_row(case, model_id, slug, display, transport, configuration) -> dict[str, Any]:
    """A combination with no data. Every metric is blank, never zero."""
    return {
        "case_id": case,
        "model_id": model_id,
        "model_slug": slug,
        "model_display": display,
        "architecture": ARCHITECTURE[configuration],
        "configuration": configuration,
        "repetition_index": 0,
        "test_status": "Not tested",
        "resolution_regime": None,
        "witness_length": None,
        "solved": None,
        "status": None,
        "n_maneuvers": None,
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "n_llm_calls": None,
        "n_tool_calls": None,
        "n_power_flows": None,
        "execution_time_s": None,
        "input_cost_usd": None,
        "output_cost_usd": None,
        "total_cost_usd": None,
        "total_buses": TOTAL_BUSES,
        "buses_in_band": None,
        "buses_in_band_pct": None,
        "worst_vm_pu": None,
        "clean": None,
        "final_state_available": False,
        "error_type": None,
        "n_invalid_action": None,
        "n_malformed_output": None,
        "n_preview_diverged": None,
        "dataset_version": None,
        "solver_version": None,
        "action_policy_version": None,
        "ranking_policy_version": None,
        "result_schema_version": None,
        "transport": transport,
        "notes": "reserved for a future campaign",
    }


def _wilson(successes: int, total: int) -> tuple[float, float]:
    """95% Wilson interval. A point estimate on 46 cases implies a precision it does not have."""
    if total == 0:
        return (math.nan, math.nan)
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return ((centre - half) * 100, (centre + half) * 100)


def _summarise(rows: list[dict[str, Any]], regime: str | None = None) -> list[dict[str, Any]]:
    tested = [r for r in rows if r["test_status"] == "Tested"]
    if regime:
        tested = [r for r in tested if r["resolution_regime"] == regime]

    keys = sorted({(r["model_display"], r["architecture"], r["configuration"]) for r in rows})
    table: list[dict[str, Any]] = []
    for display, architecture, configuration in sorted(keys, key=lambda k: (k[2], k[0])):
        cells = [r for r in tested if r["model_display"] == display and r["architecture"] == architecture]
        if not cells:
            table.append(
                {
                    "model": display,
                    "architecture": architecture,
                    "status": "Not tested",
                    **{k: None for k in _METRIC_KEYS},
                    "n_band_observations": 0,
                }
            )
            continue

        solved = sum(r["solved"] for r in cells)
        band = [r["buses_in_band_pct"] for r in cells if r["buses_in_band_pct"] is not None]
        band_abs = [r["buses_in_band"] for r in cells if r["buses_in_band"] is not None]
        low, high = _wilson(solved, len(cells))
        times = [r["execution_time_s"] for r in cells if r["execution_time_s"] is not None]
        costs = [r["total_cost_usd"] for r in cells]

        table.append(
            {
                "model": display,
                "architecture": architecture,
                "status": "Tested",
                "solved_cases": solved,
                "evaluated_cases": len(cells),
                "success_rate_pct": solved / len(cells) * 100,
                "success_rate_ci95": f"[{low:.1f}, {high:.1f}]",
                "avg_input_tokens": statistics.mean(r["input_tokens"] for r in cells),
                "avg_output_tokens": statistics.mean(r["output_tokens"] for r in cells),
                "avg_total_tokens": statistics.mean(r["total_tokens"] for r in cells),
                "avg_time_s": statistics.mean(times) if times else None,
                "std_time_s": statistics.stdev(times) if len(times) > 1 else None,
                "total_cost_usd": sum(costs),
                "avg_cost_usd": statistics.mean(costs),
                "std_cost_usd": statistics.stdev(costs) if len(costs) > 1 else None,
                "avg_buses_in_band": statistics.mean(band_abs) if band_abs else None,
                "avg_buses_in_band_pct": statistics.mean(band) if band else None,
                "std_buses_in_band_pct": statistics.stdev(band) if len(band) > 1 else None,
                "n_band_observations": len(band),
            }
        )
    return table


_METRIC_KEYS = (
    "solved_cases",
    "evaluated_cases",
    "success_rate_pct",
    "success_rate_ci95",
    "avg_input_tokens",
    "avg_output_tokens",
    "avg_total_tokens",
    "avg_time_s",
    "std_time_s",
    "total_cost_usd",
    "avg_cost_usd",
    "std_cost_usd",
    "avg_buses_in_band",
    "avg_buses_in_band_pct",
    "std_buses_in_band_pct",
)


def _relative(method: float | None, base: float | None) -> float | None:
    if method is None or base in (None, 0):
        return None
    return (method - base) / base * 100


def _baseline_table(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    base = next(
        row for row in summary if (row["model"], row["architecture"]) == BASELINE
    )
    out: list[dict[str, Any]] = []
    for row in summary:
        is_base = (row["model"], row["architecture"]) == BASELINE
        if row["status"] != "Tested":
            out.append(
                {
                    "model": row["model"],
                    "architecture": row["architecture"],
                    "status": "Not tested",
                    **{k: None for k in _DELTA_KEYS},
                }
            )
            continue

        extra_solved = row["solved_cases"] - base["solved_cases"]
        extra_cost = row["total_cost_usd"] - base["total_cost_usd"]
        out.append(
            {
                "model": row["model"],
                "architecture": row["architecture"],
                "status": "Baseline" if is_base else "Tested",
                "success_rate_pct": row["success_rate_pct"],
                "delta_success_pp": row["success_rate_pct"] - base["success_rate_pct"],
                "delta_success_relative_pct": _relative(
                    row["success_rate_pct"], base["success_rate_pct"]
                ),
                "avg_cost_usd": row["avg_cost_usd"],
                "delta_cost_pct": _relative(row["avg_cost_usd"], base["avg_cost_usd"]),
                "avg_time_s": row["avg_time_s"],
                "delta_time_pct": _relative(row["avg_time_s"], base["avg_time_s"]),
                "avg_buses_in_band_pct": row["avg_buses_in_band_pct"],
                "n_band_obs": row["n_band_observations"],
                "delta_band_pp": (
                    None
                    if row["avg_buses_in_band_pct"] is None
                    else row["avg_buses_in_band_pct"] - base["avg_buses_in_band_pct"]
                ),
                "delta_band_relative_pct": _relative(
                    row["avg_buses_in_band_pct"], base["avg_buses_in_band_pct"]
                ),
                "additional_solved_cases": extra_solved,
                # A negative denominator would yield a number that reads as a saving and is a loss.
                "additional_cost_per_success_usd": (
                    extra_cost / extra_solved if extra_solved > 0 else "N/A"
                ),
            }
        )
    return out


_DELTA_KEYS = (
    "success_rate_pct",
    "delta_success_pp",
    "delta_success_relative_pct",
    "avg_cost_usd",
    "delta_cost_pct",
    "avg_time_s",
    "delta_time_pct",
    "avg_buses_in_band_pct",
    "n_band_obs",
    "delta_band_pp",
    "delta_band_relative_pct",
    "additional_solved_cases",
    "additional_cost_per_success_usd",
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any) -> str:
    if value is None:
        return "NaN"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.2f}" if abs(value) >= 0.01 or value == 0 else f"{value:.4f}"
    return str(value)


def _markdown(rows: list[dict[str, Any]]) -> str:
    header = list(rows[0])
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(_fmt(row[key]) for key in header) + " |" for row in rows]
    return "\n".join(lines)


def main() -> None:
    global LABELS, STORES, TOTAL_BUSES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "dataset/ieee118",
        help="Dataset root providing private labels and the network-size denominator.",
    )
    parser.add_argument(
        "--store",
        dest="stores",
        type=Path,
        action="append",
        help="Phase-B result directory; repeat to pool compatible provider campaigns.",
    )
    args = parser.parse_args()

    LABELS = args.data_dir / "private/labels.json"
    TOTAL_BUSES = _dataset_bus_count(args.data_dir)
    if args.stores:
        STORES = tuple(args.stores)

    populated = [store for store in STORES if any(Path(store).glob("*.json"))]
    if not populated:
        raise SystemExit(
            "no result cells found under "
            + ", ".join(str(store) for store in STORES)
            + " — run a campaign first (restorebench-sweep --campaign ...), or point --store at "
            "an existing cells directory"
        )

    rows, partial = _rows()
    _write_csv(args.out_dir / "primary_long.csv", rows)

    overall = _summarise(rows)
    _write_csv(args.out_dir / "table1_overall.csv", overall)
    _write_csv(args.out_dir / "table2_vs_baseline.csv", _baseline_table(overall))
    for regime in ("DIRECT", "SEQUENTIAL"):
        _write_csv(args.out_dir / f"table1_{regime.lower()}.csv", _summarise(rows, regime))

    report = args.out_dir / "TABLES.md"
    tested = [r for r in rows if r["test_status"] == "Tested"]
    report.write_text(
        "\n\n".join(
            [
                "# Result tables",
                "Generated by `restorebench/sweeps/build_figure_dataset.py` from "
                + ", ".join(f"`{store.parent.relative_to(ROOT)}/`" for store in STORES)
                + ". Do not edit by hand.",
                f"Cells in the aggregations: **{len(tested)}**. "
                "Coverage is judged inside each store, so a case can be complete for one "
                "provider and partial for the other. Partial in at least one store: "
                f"{', '.join(partial) or 'none'}.",
                "## Table 1 — overall",
                _markdown(overall),
                "## Table 1 — DIRECT",
                _markdown(_summarise(rows, "DIRECT")),
                "## Table 1 — SEQUENTIAL",
                _markdown(_summarise(rows, "SEQUENTIAL")),
                f"## Table 2 — against the baseline ({BASELINE[0]} + {BASELINE[1]})",
                _markdown(_baseline_table(overall)),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"{len(rows)} rows -> {args.out_dir}/primary_long.csv")
    print(f"{len(tested)} cells enter the aggregations; excluded: {partial}")
    print(f"tables -> {report}")


def _dataset_bus_count(data_dir: Path) -> int:
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    try:
        count = int(manifest["environment"]["bus_count"])
    except (KeyError, TypeError, ValueError):
        # Frozen IEEE-118 manifests predate the generic network metadata.
        if manifest.get("dataset_version", "").startswith("ieee118-"):
            return 118
        raise ValueError("dataset manifest does not declare environment.bus_count") from None
    if count <= 0:
        raise ValueError("dataset bus count must be positive")
    return count


if __name__ == "__main__":
    main()
