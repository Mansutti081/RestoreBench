# ABOUTME: Exports saved ResolutionResponse.trace.reasoning entries to readable Markdown transcripts.
# ABOUTME: The JSON result remains source of truth; this module formats named runs on demand.
from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from restorebench.eval.store import load_response, model_id_from_response
from restorebench.llm.models import model_slug
from restorebench.schemas.response import ReasoningEntry, ResolutionResponse


DEFAULT_RESULTS_DIR = Path("results")
CONFIG_NAMES = {
    1: "chatbot",
    2: "single-agent",
    3: "multi-agent",
}


def export_reasoning(run_ids: Sequence[str], *, results_dir: str | Path = DEFAULT_RESULTS_DIR) -> list[Path]:
    root = Path(results_dir)
    outputs = []
    for run_id in run_ids:
        response = load_response(_find_result(run_id, root))
        target = root / "reasoning" / f"{run_id}.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(_render_markdown(response), encoding="utf-8")
        outputs.append(target)
    return outputs


def _find_result(run_id: str, results_dir: Path) -> Path:
    candidates = [
        results_dir / "cells" / f"{run_id}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    looked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"run id {run_id!r} not found; looked in {looked}")


def _render_markdown(response: ResolutionResponse) -> str:
    lines = [_header(response)]
    if not response.trace.reasoning:
        return "\n".join(lines) + "\n"

    for iteration, entries in _entries_by_iteration(response.trace.reasoning):
        lines.extend(["", f"## Iteration {iteration + 1}"])
        lines.extend(f"**{entry.role}** — {entry.text}" for entry in entries)
    return "\n".join(lines) + "\n"


def _header(response: ResolutionResponse) -> str:
    model = model_slug(model_id_from_response(response))
    config_name = CONFIG_NAMES[response.configuration]
    return (
        f"# {response.scenario_id} · {config_name} · {model} · rep {response.repetition_index} · "
        f"{response.status} in {response.n_maneuvers} maneuvers"
    )


def _entries_by_iteration(entries: Sequence[ReasoningEntry]) -> list[tuple[int, list[ReasoningEntry]]]:
    grouped: dict[int, list[ReasoningEntry]] = defaultdict(list)
    for entry in entries:
        grouped[entry.iteration].append(entry)
    return sorted(grouped.items(), key=lambda item: item[0])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export saved run reasoning to Markdown.")
    parser.add_argument("run_id", nargs="+")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    args = parser.parse_args(argv)
    for path in export_reasoning(args.run_id, results_dir=args.results_dir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
