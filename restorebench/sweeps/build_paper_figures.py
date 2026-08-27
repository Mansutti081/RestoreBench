# ABOUTME: Builds the seven paper figures directly from a result store, computing every number.
# ABOUTME: Reads stored ResolutionResponse cells only; no LLM calls, no hand-entered data.
"""Build the seven paper figures from stored results.

    uv run restorebench-figures --store results/ieee118-bedrock --store results/ieee118-anthropic

Every number in every figure is computed here from the stored cells. The script prints the
computed tables alongside the rendered PDFs, so the figures are auditable against the store.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from restorebench.eval.store import load_response, model_id_from_response
from restorebench.llm.models import token_cost_usd
from restorebench.schemas.response import ResolutionResponse

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STORES = (
    ROOT / "results/ieee118-anthropic",
    ROOT / "results/ieee118-bedrock",
    ROOT / "results/ieee118-openai",
    ROOT / "results/pegase89-anthropic",
    ROOT / "results/pegase89-bedrock",
    ROOT / "results/pegase89-openai",
)
DEFAULT_OUT_DIR = ROOT / "reports/figures"

MANEUVER_BUDGET = 10
ARCHITECTURES = {1: "Chatbot", 2: "Single agent", 3: "Multi-agent"}
ARCH_LABELS = {1: "chatbot", 2: "single agent", 3: "multi-agent"}
ARCH_COLORS = {1: "#0072B2", 2: "#E69F00", 3: "#009E73"}
TIMEOUT_COLOR = "#009E73"

MODEL_DISPLAY = {
    "gpt-5.6-sol": "GPT-5.6 Sol",
    "claude-haiku-4-5": "Claude Haiku 4.5",
    "claude-sonnet-5": "Claude Sonnet 5",
    "claude-opus-5": "Claude Opus 5",
    "global.anthropic.claude-haiku-4-5-20251001-v1:0": "Claude Haiku 4.5",
    "deepseek.v3.2": "DeepSeek V3.2",
    "moonshotai.kimi-k2.5": "Kimi K2.5",
    "zai.glm-5": "GLM-5",
}
MODEL_ORDER = [
    "GPT-5.6 Sol",
    "Claude Haiku 4.5",
    "Claude Sonnet 5",
    "Claude Opus 5",
    "DeepSeek V3.2",
    "Kimi K2.5",
    "GLM-5",
]
MODEL_MARKERS = {
    "GPT-5.6 Sol": "X",
    "Claude Haiku 4.5": "o",
    "Claude Sonnet 5": "s",
    "Claude Opus 5": "^",
    "DeepSeek V3.2": "D",
    "Kimi K2.5": "v",
    "GLM-5": "P",
}
CLAUDE_DISPLAYS = ("Claude Haiku 4.5", "Claude Sonnet 5", "Claude Opus 5")

# The % of buses inside the voltage band needs the network size, which follows from the corpus
# every pooled cell was measured on (the version-stamp guard ensures there is exactly one).
BUS_COUNTS = {"ieee118": 118, "pegase89": 89}

# The four ways a budget slot can be consumed without ending the episode. PREVIEW_* entries are
# within-iteration attempts and LLM_FAILURE aborts the episode; neither consumes a slot.
BUDGET_EVENT_LABELS = {
    "STILL_DIVERGED": "Still diverged",
    "MALFORMED_OUTPUT": "Structured-output failure",
    "INVALID_ACTION": "Invalid action",
    "SOLVED_INFEASIBLE": "Solved infeasible",
}
BUDGET_EVENT_ORDER = ("STILL_DIVERGED", "MALFORMED_OUTPUT", "INVALID_ACTION", "SOLVED_INFEASIBLE")

RCPARAMS = {
    "font.size": 16,
    "axes.labelsize": 18,
    "axes.titlesize": 17,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
}

# Overstress is a solver-derived float; equality within this tolerance is "unchanged".
OVERSTRESS_EPS = 1e-9


def load_cells(stores: Sequence[Path]) -> list[ResolutionResponse]:
    responses = []
    for store in stores:
        cells = Path(store) / "cells"
        if not cells.exists():
            continue
        for path in sorted(cells.glob("*.json")):
            if "manifest" in path.name:
                continue
            responses.append(load_response(path))
    if not responses:
        raise SystemExit(
            "no result cells found under "
            + ", ".join(str(store) for store in stores)
            + " — run a campaign first (restorebench-sweep --campaign ...)"
        )
    _require_one_version_stamp(responses)
    return responses


def _require_one_version_stamp(responses: Sequence[ResolutionResponse]) -> None:
    """Refuse to pool results produced under different corpus or policy versions."""
    stamps = {
        (
            r.dataset_version,
            r.solver_version,
            r.action_policy_version,
            r.ranking_policy_version,
            r.result_schema_version,
        )
        for r in responses
    }
    if len(stamps) > 1:
        raise SystemExit(f"stores mix {len(stamps)} distinct version stamps; they are not comparable")
    if any(field is None for field in next(iter(stamps))):
        raise SystemExit("results carry an incomplete version stamp and cannot enter figures")


# ============================================================================
# Figure 1 — composition of budget-consuming events in BUDGET_EXHAUSTED episodes
# ============================================================================

def budget_event_shares(responses: Sequence[ResolutionResponse]) -> dict[int, np.ndarray]:
    """Per architecture: the % share of each budget-consuming event kind, over all such events."""
    shares: dict[int, np.ndarray] = {}
    for configuration in ARCHITECTURES:
        counts: Counter[str] = Counter()
        for response in responses:
            if response.configuration != configuration or response.status != "BUDGET_EXHAUSTED":
                continue
            for feedback in response.failure_feedback:
                if feedback.kind in BUDGET_EVENT_LABELS:
                    counts[feedback.kind] += 1
        total = sum(counts.values())
        if total == 0:
            continue
        shares[configuration] = np.array(
            [100.0 * counts[kind] / total for kind in BUDGET_EVENT_ORDER]
        )
    return shares


def figure_failure_composition(responses, out_dir: Path, fmt: str) -> Path | None:
    import matplotlib.pyplot as plt

    shares = budget_event_shares(responses)
    if not shares:
        print("skipped figure_failure_composition: no BUDGET_EXHAUSTED episodes")
        return None
    for configuration, values in sorted(shares.items()):
        row = ", ".join(
            f"{BUDGET_EVENT_LABELS[k]}={v:.2f}%" for k, v in zip(BUDGET_EVENT_ORDER, values)
        )
        print(f"  fig-failure {ARCHITECTURES[configuration]}: {row}")

    categories = [BUDGET_EVENT_LABELS[k] for k in BUDGET_EVENT_ORDER]
    fig, ax = plt.subplots(figsize=(12, 6.5))
    present = sorted(shares)
    bar_height = 0.72 / max(len(present), 1)
    centers = np.arange(len(categories))[::-1]
    offsets = {c: (len(present) - 1) / 2 - i for i, c in enumerate(present)}
    for configuration in present:
        ax.barh(
            centers + offsets[configuration] * bar_height,
            shares[configuration],
            height=bar_height,
            color=ARCH_COLORS[configuration],
            edgecolor="white",
            linewidth=0.6,
            label=ARCHITECTURES[configuration],
            zorder=3,
        )
    ax.set_yticks(centers)
    ax.set_yticklabels(categories)
    ax.set_xlabel("Share of Budget-Consuming Events [%]")
    _percent_axis(ax)
    ax.legend(frameon=False, loc="lower right")
    fig.suptitle(
        "Composition of budget-consuming events in BUDGET_EXHAUSTED episodes",
        x=0.5, y=0.98, ha="center", fontsize=17,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return _save(fig, out_dir / f"figure_failure_composition.{fmt}")


# ============================================================================
# Figure 2 — progress of committed maneuvers by architecture and terminal outcome
# ============================================================================

def committed_overstress_chain(response: ResolutionResponse) -> list[tuple[float, float] | None]:
    """(before, after) overstress for each committed maneuver, None when unclassifiable.

    The response stores the bare maneuvers, but each committed non-terminal maneuver produced a
    STILL_DIVERGED (diverged again: after = its diagnostics' overstress) or SOLVED_INFEASIBLE
    (converged: distance from solvability is zero) feedback entry, in iteration order. The
    terminal maneuver of a SUCCESS episode converged, so its after is zero as well. The chain
    starts from the baseline diagnostics recorded in the trace.
    """
    before = _baseline_overstress(response)
    pairs: list[tuple[float, float] | None] = []
    for feedback in response.failure_feedback:
        if feedback.kind == "STILL_DIVERGED":
            after = feedback.diagnostics.overstress if feedback.diagnostics else None
            pairs.append((before, after) if before is not None and after is not None else None)
            before = after
        elif feedback.kind == "SOLVED_INFEASIBLE":
            after = 0.0
            pairs.append((before, after) if before is not None else None)
            before = after
    if response.status == "SUCCESS":
        pairs.append((before, 0.0) if before is not None else None)
    if len(pairs) != response.n_maneuvers:
        # An aborted episode (TOOL_FAILURE mid-application) can break the invariant; refuse to
        # classify it rather than misattribute progress.
        return [None] * response.n_maneuvers
    return pairs


def _baseline_overstress(response: ResolutionResponse) -> float | None:
    for event in response.trace.events:
        if event.phase == "baseline" and event.event_name == "baseline_diagnostics":
            diagnostics = event.payload.get("diagnostics") or {}
            return diagnostics.get("overstress")
    return None


def maneuver_progress_shares(
    responses: Sequence[ResolutionResponse],
) -> dict[tuple[int, str], np.ndarray]:
    """Per (architecture, outcome): % of classified maneuvers Improved / Unchanged / Worsened."""
    shares: dict[tuple[int, str], np.ndarray] = {}
    for configuration in ARCHITECTURES:
        for status in ("SUCCESS", "BUDGET_EXHAUSTED"):
            improved = unchanged = worsened = 0
            for response in responses:
                if response.configuration != configuration or response.status != status:
                    continue
                for pair in committed_overstress_chain(response):
                    if pair is None:
                        continue
                    before, after = pair
                    if after < before - OVERSTRESS_EPS:
                        improved += 1
                    elif after > before + OVERSTRESS_EPS:
                        worsened += 1
                    else:
                        unchanged += 1
            total = improved + unchanged + worsened
            if total == 0:
                continue
            shares[(configuration, status)] = np.array(
                [100.0 * improved / total, 100.0 * unchanged / total, 100.0 * worsened / total]
            )
    return shares


def figure_maneuver_progress(responses, out_dir: Path, fmt: str) -> Path | None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    shares = maneuver_progress_shares(responses)
    if not shares:
        print("skipped figure_maneuver_progress: no classifiable committed maneuvers")
        return None
    for (configuration, status), values in sorted(shares.items()):
        print(
            f"  fig-progress {ARCHITECTURES[configuration]} / {status}: "
            f"Improved={values[0]:.2f}% Unchanged={values[1]:.2f}% Worsened={values[2]:.2f}%"
        )

    verdicts = ["Improved", "Unchanged", "Worsened"]
    series = [
        (configuration, status)
        for configuration in sorted(ARCHITECTURES)
        for status in ("SUCCESS", "BUDGET_EXHAUSTED")
        if (configuration, status) in shares
    ]
    fig, ax = plt.subplots(figsize=(13, 7.5))
    bar_height = 0.78 / max(len(series), 1)
    centers = np.arange(len(verdicts))[::-1]
    for index, key in enumerate(series):
        configuration, status = key
        offset = ((len(series) - 1) / 2 - index) * bar_height
        ax.barh(
            centers + offset,
            shares[key],
            height=bar_height,
            color=ARCH_COLORS[configuration],
            hatch="//" if status == "BUDGET_EXHAUSTED" else None,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
    for line_y in (0.5, 1.5):
        ax.axhline(line_y, color="0.6", linestyle="--", linewidth=0.8, zorder=1)
    ax.set_yticks(centers)
    ax.set_yticklabels(verdicts)
    ax.set_xlabel("Share of Classified Maneuvers [%]")
    _percent_axis(ax)
    ax.set_title("Progress of committed maneuvers by architecture and terminal outcome", pad=16)
    handles = [
        Patch(
            facecolor=ARCH_COLORS[configuration],
            edgecolor="white",
            hatch="//" if status == "BUDGET_EXHAUSTED" else None,
            label=f"{ARCHITECTURES[configuration]} — "
            + ("Budget exhausted" if status == "BUDGET_EXHAUSTED" else "Success"),
        )
        for configuration, status in series
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right", ncol=2)
    fig.tight_layout()
    return _save(fig, out_dir / f"figure_maneuver_progress.{fmt}")


# ============================================================================
# Figures 3 and 4 — TIMEOUT episodes
# ============================================================================

def timeout_slots(responses: Sequence[ResolutionResponse]) -> tuple[np.ndarray, np.ndarray]:
    """(consumed slots, llm calls per committed maneuver ratios) over TIMEOUT episodes."""
    consumed_list, ratios = [], []
    excluded_zero_maneuvers = 0
    for response in responses:
        if response.status != "TIMEOUT":
            continue
        consumed = sum(
            1 for feedback in response.failure_feedback if feedback.kind in BUDGET_EVENT_LABELS
        )
        if consumed > MANEUVER_BUDGET:
            raise SystemExit(f"{response.request_id}: {consumed} consumed slots exceed the budget")
        consumed_list.append(consumed)
        if response.n_maneuvers > 0:
            ratios.append(response.trace.n_llm_calls / response.n_maneuvers)
        else:
            excluded_zero_maneuvers += 1
    if excluded_zero_maneuvers:
        print(f"  fig-llm excluded zero-maneuver TIMEOUT episodes: {excluded_zero_maneuvers}")
    return np.array(consumed_list), np.array(ratios)


def figure_timeout_maneuvers_remaining(responses, out_dir: Path, fmt: str) -> Path | None:
    import matplotlib.pyplot as plt

    consumed, _ = timeout_slots(responses)
    if len(consumed) == 0:
        print("skipped figure_timeout_maneuvers_remaining: no TIMEOUT episodes")
        return None
    remaining = MANEUVER_BUDGET - consumed
    levels, counts = np.unique(remaining, return_counts=True)
    level_shares = 100.0 * counts / len(remaining)
    assert np.isclose(level_shares.sum(), 100)
    print("  fig-timeout remaining->share%: " + ", ".join(f"{int(k)}:{s:.1f}" for k, s in zip(levels, level_shares)))

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    positions = np.arange(len(levels))
    ax.barh(positions, level_shares, height=0.65, color=TIMEOUT_COLOR,
            edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_yticks(positions)
    ax.set_yticklabels([str(int(level)) for level in levels])
    ax.invert_yaxis()
    ax.set_xlabel("Share of TIMEOUT Episodes [%]")
    ax.set_ylabel("Maneuvers Remaining at Timeout")
    _percent_axis(ax)
    ax.set_title("Maneuvers remaining when timeout occurs", pad=16)
    fig.tight_layout()
    return _save(fig, out_dir / f"figure_timeout_maneuvers_remaining.{fmt}")


def figure_timeout_llm_calls(responses, out_dir: Path, fmt: str) -> Path | None:
    import matplotlib.pyplot as plt

    _, ratios = timeout_slots(responses)
    if len(ratios) == 0:
        print("skipped figure_timeout_llm_calls_per_maneuver: no TIMEOUT episodes with maneuvers")
        return None
    bin_labels = ["0–10", "10–20", "20–40", "40–60", ">60"]
    membership = np.vstack([
        (ratios >= 0) & (ratios < 10),
        (ratios >= 10) & (ratios < 20),
        (ratios >= 20) & (ratios < 40),
        (ratios >= 40) & (ratios <= 60),
        ratios > 60,
    ])
    assert np.all(membership.sum(axis=0) == 1)
    bin_shares = 100.0 * membership.sum(axis=1) / len(ratios)
    assert np.isclose(bin_shares.sum(), 100)
    print("  fig-llm bins->share%: " + ", ".join(f"{k}:{s:.1f}" for k, s in zip(bin_labels, bin_shares)))

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    positions = np.arange(len(bin_labels))
    ax.barh(positions, bin_shares, height=0.65, color=TIMEOUT_COLOR,
            edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_yticks(positions)
    ax.set_yticklabels(bin_labels)
    ax.invert_yaxis()
    ax.set_xlabel("Percentage of TIMEOUT Episodes [%]")
    ax.set_ylabel("LLM Calls per Committed Maneuver")
    _percent_axis(ax)
    ax.set_title("LLM interaction efficiency before timeout", pad=16)
    fig.tight_layout()
    return _save(fig, out_dir / f"figure_timeout_llm_calls_per_maneuver.{fmt}")


# ============================================================================
# Figure 1 — success rate by model and architecture
# ============================================================================

def _display(response: ResolutionResponse) -> str:
    model_id = model_id_from_response(response)
    return MODEL_DISPLAY.get(model_id, model_id)


def _present_models(responses: Sequence[ResolutionResponse]) -> list[str]:
    seen = {_display(r) for r in responses}
    ordered = [m for m in MODEL_ORDER if m in seen]
    return ordered + sorted(seen - set(ordered))


def success_rate_table(responses: Sequence[ResolutionResponse]) -> dict[tuple[str, int], float]:
    """(model display, configuration) -> success rate %, over every stored episode."""
    table: dict[tuple[str, int], float] = {}
    for model in _present_models(responses):
        for configuration in ARCHITECTURES:
            episodes = [
                r for r in responses
                if _display(r) == model and r.configuration == configuration
            ]
            if episodes:
                wins = sum(1 for r in episodes if r.status == "SUCCESS")
                table[(model, configuration)] = 100.0 * wins / len(episodes)
    return table


def figure_success_rate(responses, out_dir: Path, fmt: str) -> Path | None:
    import matplotlib.pyplot as plt

    table = success_rate_table(responses)
    if not table:
        print("skipped figure_success_rate: no episodes")
        return None
    models = _present_models(responses)
    for (model, configuration), value in sorted(table.items()):
        print(f"  fig-success {model} / {ARCH_LABELS[configuration]}: {value:.1f}%")

    fig, ax = plt.subplots(figsize=(10, 6.5))
    n_arch = len(ARCHITECTURES)
    bar_width = 0.8 / n_arch
    x = np.arange(len(models))
    for index, configuration in enumerate(sorted(ARCHITECTURES)):
        offsets = x - 0.4 + bar_width * (index + 0.5)
        values = [table.get((model, configuration), np.nan) for model in models]
        ax.bar(
            offsets, values, width=bar_width * 0.92,
            color=ARCH_COLORS[configuration], label=ARCH_LABELS[configuration],
            edgecolor="white", linewidth=0.6, zorder=3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Success Rate [%]")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=n_arch)
    ax.grid(axis="y", color="0.85", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return _save(fig, out_dir / f"figure_success_rate.{fmt}")


# ============================================================================
# Figure 2 — average cost per case against success rate
# ============================================================================

def cost_table(responses: Sequence[ResolutionResponse]) -> dict[tuple[str, int], float]:
    """(model display, configuration) -> mean USD per case, from stored token counts."""
    table: dict[tuple[str, int], float] = {}
    for model in _present_models(responses):
        for configuration in ARCHITECTURES:
            episodes = [
                r for r in responses
                if _display(r) == model and r.configuration == configuration
            ]
            if episodes:
                costs = [
                    token_cost_usd(
                        model_id_from_response(r),
                        tokens_in=r.trace.total_llm_tokens_in,
                        tokens_out=r.trace.total_llm_tokens_out,
                    )
                    for r in episodes
                ]
                table[(model, configuration)] = float(np.mean(costs))
    return table


def figure_cost_vs_success(responses, out_dir: Path, fmt: str) -> Path | None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    success = success_rate_table(responses)
    costs = cost_table(responses)
    if not success:
        print("skipped figure_cost_vs_success: no episodes")
        return None
    for key in sorted(costs):
        model, configuration = key
        print(
            f"  fig-cost {model} / {ARCH_LABELS[configuration]}: "
            f"${costs[key]:.3f}/case at {success[key]:.1f}%"
        )

    fig, ax = plt.subplots(figsize=(10, 6.5))
    markers = dict(MODEL_MARKERS)
    spare_markers = iter("Xshd*8")
    for key, rate in success.items():
        model, configuration = key
        if model not in markers:
            markers[model] = next(spare_markers, "o")
        marker = markers[model]
        ax.scatter(
            costs[key], rate, s=160, color=ARCH_COLORS[configuration], marker=marker,
            edgecolor="white", linewidth=0.8, zorder=3,
        )
    ax.set_xlabel("Average Cost per Case [$]")
    ax.set_ylabel("Success Rate [%]")
    ax.set_ylim(0, 100)
    ax.grid(True, color="0.85", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    present = _present_models(responses)
    claude = [m for m in present if m in CLAUDE_DISPLAYS]
    others = [m for m in present if m not in CLAUDE_DISPLAYS]

    def model_handles(names):
        return [
            Line2D([0], [0], marker=markers[name], linestyle="none", markersize=10,
                   markerfacecolor="0.4", markeredgecolor="white", label=name)
            for name in names
        ]

    arch_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=10,
               markerfacecolor=ARCH_COLORS[c], markeredgecolor="white", label=ARCH_LABELS[c])
        for c in sorted(ARCHITECTURES)
    ]
    row_y = 0.99
    for handles in (model_handles(claude), model_handles(others), arch_handles):
        if handles:
            fig.legend(handles=handles, loc="upper center",
                       bbox_to_anchor=(0.5, row_y), ncol=len(handles), frameon=False)
            row_y -= 0.05
    fig.subplots_adjust(top=0.80)
    return _save(fig, out_dir / f"figure_cost_vs_success.{fmt}")


# ============================================================================
# Figure 4 — buses inside the voltage band (box plot over SUCCESS episodes)
# ============================================================================

def _bus_count(responses: Sequence[ResolutionResponse]) -> int:
    version = responses[0].dataset_version or ""
    for prefix, count in BUS_COUNTS.items():
        if version.startswith(prefix):
            return count
    raise SystemExit(f"cannot infer the network size from dataset_version {version!r}")


def voltage_band_samples(responses: Sequence[ResolutionResponse]) -> dict[tuple[str, int], np.ndarray]:
    """(model display, configuration) -> % of buses inside the band, one value per SUCCESS."""
    buses = _bus_count(responses)
    samples: dict[tuple[str, int], np.ndarray] = {}
    for model in _present_models(responses):
        for configuration in ARCHITECTURES:
            values = [
                100.0 * (buses - r.quality.n_buses_out_of_band) / buses
                for r in responses
                if _display(r) == model and r.configuration == configuration
                and r.status == "SUCCESS" and r.quality is not None
            ]
            if values:
                samples[(model, configuration)] = np.array(values)
    return samples


def figure_voltage_band(responses, out_dir: Path, fmt: str) -> Path | None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    samples = voltage_band_samples(responses)
    if not samples:
        print("skipped figure_voltage_band: no SUCCESS episodes with quality")
        return None

    models = _present_models(responses)
    stats, archs, group_of = [], [], []
    for model in models:
        for configuration in sorted(ARCHITECTURES):
            key = (model, configuration)
            if key not in samples:
                continue
            values = samples[key]
            quartiles = np.percentile(values, [25, 50, 75])
            stats.append({
                "label": "", "whislo": float(values.min()), "q1": float(quartiles[0]),
                "med": float(quartiles[1]), "q3": float(quartiles[2]),
                "whishi": float(values.max()), "mean": float(values.mean()), "fliers": [],
            })
            archs.append(configuration)
            group_of.append(model)
            print(
                f"  fig-band {model} / {ARCH_LABELS[configuration]}: n={len(values)} "
                f"median={quartiles[1]:.1f}%"
            )

    fig, ax = plt.subplots(figsize=(16, 6))
    positions = np.arange(1, len(stats) + 1)
    boxes = ax.bxp(stats, positions=positions, widths=0.6, patch_artist=True,
                   showmeans=True, meanline=True)
    for patch, configuration in zip(boxes["boxes"], archs):
        patch.set_facecolor(ARCH_COLORS[configuration])
        patch.set_edgecolor("black")
        patch.set_alpha(0.75)
    for element in ("whiskers", "caps", "medians"):
        for line in boxes[element]:
            line.set_color("black")
    for line in boxes["means"]:
        line.set_color("black")
        line.set_linestyle(":")

    ax.set_xticks(positions)
    ax.set_xticklabels([])
    start = 0
    for model in models:
        size = sum(1 for g in group_of if g == model)
        if size == 0:
            continue
        center = positions[start:start + size].mean()
        ax.text(center, -0.1, model, ha="center", va="top", transform=ax.get_xaxis_transform())
        if start + size < len(positions):
            ax.axvline(positions[start + size - 1] + 0.5, color="0.5",
                       linestyle="--", linewidth=1, zorder=1)
        start += size

    ax.set_xlim(0.3, len(stats) + 0.7)
    ax.set_ylabel("Buses in Voltage Band [%]")
    ax.set_ylim(60, 100)
    ax.grid(axis="y", color="0.85", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    handles = [
        Patch(facecolor=ARCH_COLORS[c], edgecolor="black", alpha=0.75, label=ARCH_LABELS[c])
        for c in sorted(ARCHITECTURES)
    ]
    ax.legend(handles=handles, frameon=False, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), ncol=len(handles))
    fig.tight_layout()
    return _save(fig, out_dir / f"figure_voltage_band.{fmt}")


# ============================================================================
# Shared rendering helpers
# ============================================================================

def _percent_axis(ax) -> None:
    from matplotlib.ticker import PercentFormatter

    ax.set_xlim(0, 100)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.grid(axis="x", color="0.85", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save(fig, path: Path) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--store", dest="stores", type=Path, action="append",
        help="result directory holding a cells/ subfolder; repeat to pool compatible stores",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--format", choices=("pdf", "png"), default="pdf")
    args = parser.parse_args(argv)

    try:
        import matplotlib
    except ImportError:
        raise SystemExit("matplotlib is not installed — run: uv sync --group plots")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(RCPARAMS)

    stores = args.stores or [store for store in DEFAULT_STORES if (store / "cells").exists()]
    responses = load_cells(stores)
    by_status = Counter(response.status for response in responses)
    print(f"{len(responses)} cells from {len(stores)} store(s); status counts: {dict(by_status)}")

    built = [
        figure_success_rate(responses, args.out_dir, args.format),
        figure_cost_vs_success(responses, args.out_dir, args.format),
        figure_voltage_band(responses, args.out_dir, args.format),
        figure_failure_composition(responses, args.out_dir, args.format),
        figure_maneuver_progress(responses, args.out_dir, args.format),
        figure_timeout_maneuvers_remaining(responses, args.out_dir, args.format),
        figure_timeout_llm_calls(responses, args.out_dir, args.format),
    ]
    print(f"{sum(1 for path in built if path)} of 7 figures built -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
