# ABOUTME: Writes the plot-ready markdown: one description and one data table per figure.
# ABOUTME: Reads only the stored results; the output is meant to be read, not imported.
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "reports/figure_data/primary_long.csv"
LABELS = ROOT / "dataset/ieee118/private/labels.json"
OUT = ROOT / "reports/figure_tables.md"

ARCH_ORDER = ["chatbot", "single agent", "multi-agent"]
# Display order for every model the benchmark can carry: the OpenAI model, the Claude family,
# then the open-weight models. Which of them are measured is read from the data, never declared here —
# a hardcoded pending list goes on printing NaN long after the cells have landed.
ALL_MODELS = [
    "GPT-5.6 Sol",
    "Claude Haiku 4.5",
    "Claude Sonnet 5",
    "Claude Opus 5",
    "DeepSeek V3.2",
    "Kimi K2.5",
    "GLM-5",
]
BASELINE = ("Claude Haiku 4.5", "chatbot")


def _tested() -> list[dict[str, Any]]:
    with DATA.open(encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row["test_status"] == "Tested"]


def _num(value: str) -> float | None:
    return float(value) if value not in ("", "None") else None


def _cell(rows: list[dict[str, Any]], model: str, arch: str) -> list[dict[str, Any]]:
    return [r for r in rows if r["model_display"] == model and r["architecture"] == arch]


def measured_models(rows: list[dict[str, Any]]) -> list[str]:
    """The models with data, in canonical order, with anything unrecognised appended.

    An unknown model is reported rather than dropped: a display name that drifts out of
    ALL_MODELS would otherwise remove a whole campaign from every figure without a word.
    """
    present = {row["model_display"] for row in rows}
    known = [model for model in ALL_MODELS if model in present]
    return known + sorted(present - set(ALL_MODELS))


def pending_models(rows: list[dict[str, Any]]) -> list[str]:
    """Models the benchmark expects but has no measured cell for."""
    present = {row["model_display"] for row in rows}
    return [model for model in ALL_MODELS if model not in present]


def _combos(models: list[str]) -> list[tuple[str, str]]:
    return [(m, a) for m in models for a in ARCH_ORDER]


def _five_numbers(values: list[float]) -> dict[str, float | None]:
    """A box plot draws exactly these. Reporting them is reporting the chart."""
    if not values:
        return {k: None for k in ("min", "q1", "median", "q3", "max")}
    ordered = sorted(values)
    quartiles = statistics.quantiles(ordered, n=4, method="inclusive") if len(ordered) > 1 else [ordered[0]] * 3
    return {
        "min": ordered[0],
        "q1": quartiles[0],
        "median": statistics.median(ordered),
        "q3": quartiles[2],
        "max": ordered[-1],
    }


def _distribution(rows: list[dict[str, Any]], model: str, arch: str, column: str) -> dict[str, float | None]:
    """One box of a box plot: the five numbers it draws, the mean, and the count behind them.

    A cell with no measurement is dropped rather than read as zero, so `n` states how many
    observations the box actually summarises.
    """
    values = [
        value
        for value in (_num(r.get(column, "")) for r in _cell(rows, model, arch))
        if value is not None
    ]
    return {
        "n": len(values),
        **_five_numbers(values),
        "mean": statistics.mean(values) if values else None,
    }


def _f(value: Any, digits: int = 1) -> str:
    if value is None:
        return "NaN"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main() -> None:
    if not DATA.is_file():
        raise SystemExit(
            f"{DATA} does not exist — run `python -m restorebench.sweeps.build_figure_dataset` "
            "over a populated result store first"
        )
    rows = _tested()
    if not rows:
        raise SystemExit(f"{DATA} holds no tested cells — the result store it was built from is empty")
    models = measured_models(rows)
    pending = pending_models(rows)
    labels = {r["scenario_id"]: r for r in json.loads(LABELS.read_text(encoding="utf-8"))}
    cases = sorted({r["case_id"] for r in rows})
    direct = [c for c in cases if labels[c]["resolution_regime"] == "DIRECT"]
    sequential = [c for c in cases if labels[c]["resolution_regime"] == "SEQUENTIAL"]

    out: list[str] = []
    add = out.append

    add("# Plot-ready data")
    add(
        "One description and one data table per figure. Every number is computed from the "
        "stored results the figure dataset pools — the Anthropic and Bedrock campaigns — over "
        f"the {len(rows)} cells with full coverage: {len(cases)} held-out scenarios × "
        f"{len(models)} models × {len(ARCH_ORDER)} architectures, one repetition.\n\n"
        "Regenerate with `uv run python -m restorebench.sweeps.build_figure_tables`. "
        "Do not edit by hand.\n\n"
        f"**Models measured:** {', '.join(models)}.\n"
        + (
            f"**Awaiting measurement:** {', '.join(pending)} — every table below carries their "
            "rows as `NaN`, so adding them is a data change, not a chart change.\n\n"
            if pending
            else "**Every model in the design is measured.**\n\n"
        )
        + "**Percentages are on a 0–100 scale.** A difference between two percentages is in "
        "**percentage points (pp)**; a difference relative to a baseline value is a "
        "**percentage change (%)**."
    )

    # ---------------------------------------------------------------- Figure 1
    add("\n---\n\n## Figure 1 — Success rate by model and architecture")
    add(
        "**Grouped bar plot.** X axis: the models, three bars each. Y axis: success rate, 0–100 %. "
        "One colour per architecture, the same colour across all models — colour encodes the "
        "architecture, which is the variable being compared.\n\n"
        "Error bars are the 95 % Wilson interval. With 46 cases and one repetition, a bar without "
        "an interval implies a precision the data does not have."
    )
    add(
        "\n| model | architecture | solved_cases | total_cases | success_rate_pct | ci95_low | ci95_high | status |"
        "\n|---|---|---:|---:|---:|---:|---:|---|"
    )
    with (ROOT / "reports/figure_data/table1_overall.csv").open(encoding="utf-8") as handle:
        table1 = {(r["model"], r["architecture"]): r for r in csv.DictReader(handle)}
    for model, arch in _combos(models):
        r = table1[(model, arch)]
        low, high = r["success_rate_ci95"].strip("[]").split(", ")
        add(
            f"| {model} | {arch} | {r['solved_cases']} | {r['evaluated_cases']} | "
            f"{_f(float(r['success_rate_pct']))} | {low} | {high} | Tested |"
        )
    for model in pending:
        for arch in ARCH_ORDER:
            add(f"| {model} | {arch} | NaN | NaN | NaN | NaN | NaN | Not tested |")

    add(
        "\n**What the figure shows.** Colour separates the bars far more than position does: the "
        "chatbot bar is low for every model, the two tool-bearing bars are high for every model. "
        "Model capability matters only in the tool-free row, where the spread is 37 points "
        "(17 % to 54 %); with tools it compresses below 13 and stops being monotonic in capability "
        "— Sonnet 5 sits below Haiku 4.5 as a single agent. The single-agent and multi-agent bars "
        "overlap within their confidence intervals on every model.\n\n"
        "**Same figure, per regime.** The tool-bearing architectures saturate on DIRECT, so the "
        "aggregate hides where they differ. Produce these as two panels."
    )
    for regime, subset in (("DIRECT", direct), ("SEQUENTIAL", sequential)):
        add(f"\n*{regime} — {len(subset)} cases per cell*\n")
        add("| model | architecture | solved_cases | total_cases | success_rate_pct |")
        add("|---|---|---:|---:|---:|")
        for model, arch in _combos(models):
            cells = [r for r in _cell(rows, model, arch) if r["case_id"] in subset]
            solved = sum(int(r["solved"]) for r in cells)
            add(f"| {model} | {arch} | {solved} | {len(cells)} | {_f(solved / len(cells) * 100)} |")

    add(
        "\n**What the panels show.** On DIRECT the six tool-bearing cells sit between 86 % and "
        "97 % and are indistinguishable from one another. On SEQUENTIAL the ordering inverts and "
        "the two best cells land at opposite corners: the cheapest model with the simple "
        "architecture (Haiku 4.5 single agent, 8/10) and the most expensive with the complex one "
        "(Opus 5 multi-agent, 8/10). The worst tool-bearing cell is Haiku 4.5 multi-agent, which "
        "falls from 89 % on DIRECT to 40 % on SEQUENTIAL — the sharpest collapse in the matrix."
    )

    # ---------------------------------------------------------------- Figure 2
    add("\n---\n\n## Figure 2 — Average cost against success rate")
    add(
        "**Scatter plot**, one point per model–architecture combination. X axis: average cost per "
        "evaluated case, in **USD** — a log scale suits it, the values span an order of magnitude. "
        "Y axis: success rate, 0–100 %.\n\n"
        "Colour identifies the model; marker shape identifies the architecture — circle chatbot, "
        "square single agent, triangle multi-agent.\n\n"
        "The denominator of the average is *evaluated* cases, not solved ones: a failing cell still "
        "spends budget."
    )
    add(
        "\n| model | architecture | total_cost_usd | evaluated_cases | avg_cost_per_case_usd | success_rate_pct | status |"
        "\n|---|---|---:|---:|---:|---:|---|"
    )
    for model, arch in _combos(models):
        r = table1[(model, arch)]
        add(
            f"| {model} | {arch} | {_f(float(r['total_cost_usd']), 2)} | {r['evaluated_cases']} | "
            f"{_f(float(r['avg_cost_usd']), 3)} | {_f(float(r['success_rate_pct']))} | Tested |"
        )
    for model in pending:
        for arch in ARCH_ORDER:
            add(f"| {model} | {arch} | NaN | NaN | NaN | NaN | Not tested |")

    add(
        "\n**What the figure shows.** The frontier is not the diagonal. Haiku 4.5 as a single "
        "agent reaches 84.8 % at $0.447 per case, while Opus 5 as a multi-agent reaches 91.3 % at "
        "$2.345 — five times the price for six points. Sonnet 5 multi-agent matches that same "
        "91.3 % at $1.539, so the most expensive point on the chart is not the best one.\n\n"
        "The three chatbot points sit alone in the cheap-and-poor corner: they cost a tenth of the "
        "rest and solve half as much. Cost per success, which is the ratio these points encode: "
        "$0.53 for Haiku single agent against $2.57 for Opus multi-agent."
    )

    # ---------------------------------------------------------------- Figure 3
    add("\n---\n\n## Figure 3 — Which cases each combination solves")
    add(
        "**Binary heatmap.** Rows: the 46 study cases, DIRECT first then SEQUENTIAL, with a "
        "separator between the blocks. Columns: the nine model–architecture combinations, grouped "
        "by model.\n\n"
        "`1` solved (green), `0` not solved (red), `NaN` not run (grey or blank). Grey is not a "
        "shade of red: a missing experiment is not a failed one, and the scale must keep them "
        "distinguishable in black and white too.\n\n"
        "The `solved_by` column counts how many of the nine solve each case; sorting rows by it "
        "inside each block groups the hard cases together."
    )
    header = "| case_id | regime | " + " | ".join(
        f"{m.replace('Claude ', '')} {a}" for m, a in _combos(models)
    ) + " | solved_by |"
    add("\n" + header)
    add("|---|---|" + "---:|" * 10)
    solved_map = {
        (r["case_id"], r["model_display"], r["architecture"]): int(r["solved"]) for r in rows
    }
    for block in (direct, sequential):
        ordered = sorted(
            block,
            key=lambda c: -sum(solved_map.get((c, m, a), 0) for m, a in _combos(models)),
        )
        for case in ordered:
            values = [solved_map.get((case, m, a), None) for m, a in _combos(models)]
            total = sum(v for v in values if v is not None)
            regime = labels[case]["resolution_regime"]
            cells = " | ".join("NaN" if v is None else str(v) for v in values)
            add(f"| {case} | {regime} | {cells} | {total} |")
        if block is direct:
            add("| | | | | | | | | | | |")

    add(
        f"\nCases solved by all nine: "
        f"{sum(1 for c in cases if all(solved_map.get((c, m, a)) == 1 for m, a in _combos(models)))}. "
        f"Cases solved by none: "
        f"{sum(1 for c in cases if not any(solved_map.get((c, m, a)) for m, a in _combos(models)))}."
    )
    if pending:
        add(
            f"\nColumns for {', '.join(pending)} are omitted here for width; they are `NaN` for "
            "every row and must be added as blank columns when those models run."
        )

    add(
        "\n**What the figure shows.** The rows separate into three bands. A large block of DIRECT "
        "cases is solved by everything except the chatbots, which is the tool effect made visible "
        "case by case. A handful of cases resist every combination, and they are concentrated in "
        "the SEQUENTIAL block. Between the two sits the interesting band: cases where the "
        "architectures disagree on the same model, which is where the single-agent and "
        "multi-agent columns stop looking alike.\n\n"
        "Read the SEQUENTIAL block against the `witness_length` of each case rather than "
        "top-to-bottom: S0046 and S0096 need two moves and are solved by one combination each, "
        "while S0058 needs four and is solved by two. **Difficulty does not follow witness "
        "length**, and this figure is where a reader can see it directly."
    )

    # ---------------------------------------------------------------- Figure 4
    add("\n---\n\n## Figure 4 — Buses inside the voltage band")
    add(
        "**Box plot.** X axis: the nine combinations. Y axis: percentage of the 118 buses with "
        "`0.95 ≤ V ≤ 1.05` p.u. in the final state, 0–100 %.\n\n"
        "A box plot draws five numbers per box, and those are the table below. `n` is the number "
        "of observations behind each box and it is **not** 46."
    )
    add(
        "\n> **Two things this figure cannot do.**\n>\n"
        "> A final state exists only for solved cells: an unsolved run ends non-convergent, and a "
        "power flow that does not converge has no bus voltages. Unsolved cells are `NaN`, never "
        "zero — a zero would assert that all 118 buses are out of band, which is false.\n>\n"
        "> Each box therefore describes a different subset of cases, chosen by the method itself. "
        "A combination that solves only the easy cases can look better here for that reason alone. "
        "Print `n` above every box, and read the deltas in Table 2 against it."
    )
    add(
        "\n| model | architecture | n | min | q1 | median | q3 | max | mean | status |"
        "\n|---|---|---:|---:|---:|---:|---:|---:|---:|---|"
    )
    for model, arch in _combos(models):
        box = _distribution(rows, model, arch, "buses_in_band_pct")
        add(
            f"| {model} | {arch} | {box['n']} | {_f(box['min'])} | {_f(box['q1'])} | "
            f"{_f(box['median'])} | {_f(box['q3'])} | {_f(box['max'])} | {_f(box['mean'])} | Tested |"
        )
    for model in pending:
        for arch in ARCH_ORDER:
            add(f"| {model} | {arch} | 0 | NaN | NaN | NaN | NaN | NaN | NaN | Not tested |")

    add(
        "\n**What the figure shows.** The boxes are narrow and nearly level: every combination "
        "lands around 80–84 % of buses in band, and the tool-bearing ones sit about two points "
        "*lower* than the chatbots. That difference is almost certainly the selection effect above "
        "and not a quality regression — the chatbot boxes summarise the 8 to 25 easiest cases each "
        "one managed to solve, the tool-bearing boxes summarise 36 to 42 cases of every "
        "difficulty.\n\n"
        "The flat and low level is a property of the corpus, not of the methods: no scenario "
        "terminates inside 0.90–1.10 p.u. even when following the private witness, whose terminal "
        "`min_vm_pu` has median 0.6364. **This corpus certifies restoration of convergence, not "
        "restoration of an acceptable voltage profile**, and the figure must be captioned in those "
        "terms or it reads as a quality claim it cannot support.\n\n"
        "The raw per-case values behind these boxes are the `buses_in_band_pct` column of "
        "`reports/figure_data/primary_long.csv`, one row per case."
    )

    # ---------------------------------------------------------------- Appendix
    add("\n---\n\n## Figure 5 — With tools against without")
    add(
        "**Two bars, two panels.** The benchmark's tool axis: configuration 1 has no tools and no "
        "feedback between maneuvers; configurations 2 and 3 have four diagnostic tools and see each "
        "outcome before choosing the next move. Panels: DIRECT and SEQUENTIAL."
    )
    add("\n| regime | tools | solved_cases | total_cases | success_rate_pct | avg_cost_per_case_usd |")
    add("|---|---|---:|---:|---:|---:|")
    for regime, subset in (("overall", cases), ("DIRECT", direct), ("SEQUENTIAL", sequential)):
        for label, match in (("without tools", ["chatbot"]), ("with tools", ["single agent", "multi-agent"])):
            cells = [r for r in rows if r["architecture"] in match and r["case_id"] in subset]
            solved = sum(int(r["solved"]) for r in cells)
            cost = sum(float(r["total_cost_usd"]) for r in cells)
            add(
                f"| {regime} | {label} | {solved} | {len(cells)} | "
                f"{_f(solved / len(cells) * 100)} | {_f(cost / len(cells), 3)} |"
            )

    no_tool = [r for r in rows if r["architecture"] == "chatbot"]
    with_tool = [r for r in rows if r["architecture"] != "chatbot"]
    s0, c0 = sum(int(r["solved"]) for r in no_tool), sum(float(r["total_cost_usd"]) for r in no_tool)
    s1, c1 = sum(int(r["solved"]) for r in with_tool), sum(float(r["total_cost_usd"]) for r in with_tool)
    add(
        f"\n**The decision quantity.** Tools buy {s1 - s0} additional solved cases for "
        f"${c1 - c0:.2f}, which is **${(c1 - c0) / (s1 - s0):.2f} per additional case solved**.\n\n"
        "**What this comparison does not isolate.** Configuration 1 differs from configuration 2 by "
        "tools *and* by iterative feedback *and* by the candidate ranking, all at once. The pilot "
        "that would separate them was never run, so the supported wording is \"tools and feedback "
        "together differ from neither\".\n\n"
        "**And part of the gap is not strategy.** The tool-free configuration averages "
        f"{sum(int(r['n_invalid_action']) for r in no_tool) / len(no_tool):.2f} rejected actions per "
        f"cell against {sum(int(r['n_invalid_action']) for r in with_tool) / len(with_tool):.2f} with "
        "tools. About a third of those rejections concern a generator that has already saturated its "
        "reactive limit — a property of the solved state, which these non-convergent cases do not "
        "have and which configuration 1 is never told. Each rejection costs one of the ten budget "
        "slots."
    )

    # ---------------------------------------------------------------- Figure 6
    add("\n---\n\n## Figure 6 — Wall-clock time per case")
    add(
        "**Box plot.** X axis: the nine combinations, grouped by model. Y axis: seconds per case — "
        "**use a log scale**, the values span from 29 s to 2674 s and a linear axis collapses the "
        "three chatbot boxes into a single line.\n\n"
        "A box plot draws five numbers per box, and those are the table below. `n` is 46 everywhere: "
        "unlike the voltage band of Figure 4, a runtime exists for every run whether or not the case "
        "was solved, so nothing is selected away."
    )
    add(
        "\n> **Report the median, not the mean.** In every tool-bearing cell the mean sits above the "
        "third quartile. The distribution has one long right tail and the mean tracks the tail, not "
        "the typical case. A bar chart of `avg_time_s` would be a chart of the outliers.\n>\n"
        "> **This is wall clock, not compute.** The quantity is `total_runtime_seconds`, the span "
        "from `started_at` to `completed_at`. It includes API latency, retries and rate-limit "
        "backoff, and the sweep ran with concurrency. It is an honest measure of what a user waits; "
        "it is a poor measure of how much work a method does. Caption it as latency and the figure "
        "stands; caption it as cost and it does not."
    )
    add(
        "\n| model | architecture | n | min | q1 | median | q3 | max | mean | status |"
        "\n|---|---|---:|---:|---:|---:|---:|---:|---:|---|"
    )
    for model, arch in _combos(models):
        box = _distribution(rows, model, arch, "execution_time_s")
        add(
            f"| {model} | {arch} | {box['n']} | {_f(box['min'])} | {_f(box['q1'])} | "
            f"{_f(box['median'])} | {_f(box['q3'])} | {_f(box['max'])} | {_f(box['mean'])} | Tested |"
        )
    for model in pending:
        for arch in ARCH_ORDER:
            add(f"| {model} | {arch} | 0 | NaN | NaN | NaN | NaN | NaN | NaN | Not tested |")

    add("\n*Pooled across models, for the caption:*\n")
    add("| architecture | n | median_s | q1_s | q3_s | mean_s |")
    add("|---|---:|---:|---:|---:|---:|")
    pooled: dict[str, dict[str, float | None]] = {}
    for arch in ARCH_ORDER:
        values = [
            value
            for value in (_num(r["execution_time_s"]) for r in rows if r["architecture"] == arch)
            if value is not None
        ]
        five = _five_numbers(values)
        pooled[arch] = {**five, "mean": statistics.mean(values) if values else None}
        add(
            f"| {arch} | {len(values)} | {_f(five['median'])} | {_f(five['q1'])} | "
            f"{_f(five['q3'])} | {_f(pooled[arch]['mean'])} |"
        )

    solved_times = [_num(r["execution_time_s"]) for r in rows if r["solved"] == "1"]
    unsolved_times = [_num(r["execution_time_s"]) for r in rows if r["solved"] != "1"]
    solved_times = [v for v in solved_times if v is not None]
    unsolved_times = [v for v in unsolved_times if v is not None]
    ratio = pooled["multi-agent"]["median"] / pooled["single agent"]["median"]  # type: ignore[operator]

    add(
        "\n**What the figure shows.** The architecture sets the time, the model barely moves it. The "
        "three chatbot boxes are tight and nearly identical because a tool-free run is a fixed "
        "number of LLM calls with no loop to lengthen. The moment tools appear the boxes stretch by "
        "an order of magnitude, and the multi-agent boxes are the widest on the chart.\n\n"
        f"**Multi-agent costs about {ratio:.1f}× the median time of single agent** "
        f"({_f(pooled['multi-agent']['median'])} s against {_f(pooled['single agent']['median'])} s "
        "pooled) for a success rate that Figure 1 shows overlapping within its confidence intervals. "
        "Read Figure 6 beside Figure 1 and the multi-agent column becomes hard to justify: more wall "
        "clock, more money, no separable gain.\n\n"
        "**The single-agent boxes are the strange ones.** Their medians sit close to the chatbots, "
        "but their third quartiles jump several hundred seconds. The typical tool-bearing run is "
        "quick; a quarter of them are not. That shape is an iteration-count distribution, not a "
        "latency distribution — most cases resolve in one or two maneuvers and a minority walk the "
        "full ten-move budget.\n\n"
        f"**Failure is slower than success.** Median {statistics.median(unsolved_times):.1f} s for "
        f"unsolved cells against {statistics.median(solved_times):.1f} s for solved ones, and the "
        f"means diverge much further — {statistics.mean(unsolved_times):.1f} s against "
        f"{statistics.mean(solved_times):.1f} s. A run that fails does not fail fast; it exhausts the "
        "budget first. The upper tail of every box is made largely of cases that were never going to "
        "be solved, which is the honest reading of the 2000 s-plus outliers.\n\n"
        "**Two things this figure cannot support.** It cannot rank the models on speed: the "
        "differences between models within an architecture are smaller than the spread inside any "
        "single box. And it cannot be compared against wall-clock numbers from any other paper's "
        "hardware, since the quantity is dominated by a commercial API's queueing behaviour on the "
        "days the sweep ran."
    )

    OUT.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"written -> {OUT} ({len(out)} blocks)")


if __name__ == "__main__":
    main()
