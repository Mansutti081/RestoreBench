# ABOUTME: Runs evaluation sweeps over one manifest-bound dataset.
# ABOUTME: Fans out synchronous resolve() calls, checkpoints every run, and resumes from files.
from __future__ import annotations

import argparse
import hashlib
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeVar, cast

from restorebench.agents.baseline_chatbot import make_baseline_chatbot
from restorebench.agents.multi_agent import make_multi_agent
from restorebench.agents.single_agent import make_single_agent
from restorebench.environment.orchestrator import AgentStep, ProposeSequence, resolve
from restorebench.environment.scenarios import DEFAULT_DATA_DIR, held_out_ids, load_scenario
from restorebench.eval.manifest import DatasetManifestEntry, update_manifest
from restorebench.eval.store import RunKey, is_done, model_id_from_response, save_response
from restorebench.llm.models import BENCHMARK_MODELS, SUPPORTED_MODELS, token_cost_usd
from restorebench.schemas.config import LLMAssignment, OrchestratorConfig
from restorebench.schemas.dataset import DatasetManifest, Scenario
from restorebench.schemas.response import ResolutionResponse


DEFAULT_CONCURRENCY = 10
DEFAULT_RESULTS_DIR = Path("results")
RETRY_STATUSES = frozenset({"LLM_FAILURE", "TOOL_FAILURE"})

T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True)
class HarnessSummary:
    n_planned: int
    n_run: int
    n_skipped: int
    result_paths: tuple[Path, ...]


def run_evaluation(
    models: Sequence[str],
    *,
    configurations: Sequence[int],
    results_dir: str | Path,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    repetitions: int = 5,
    concurrency: int = DEFAULT_CONCURRENCY,
    force: bool = False,
    limit: int | None = None,
) -> HarnessSummary:
    result_root = Path(results_dir)
    cells_dir = result_root / "cells"
    dataset_root = Path(data_dir)
    dataset = _dataset_manifest_entry(dataset_root)
    update_manifest(result_root, model_ids=models, dataset=dataset)
    scenario_ids = held_out_ids() if dataset_root == DEFAULT_DATA_DIR else held_out_ids(data_dir=dataset_root)
    # --limit bounds (scenario x model x config) CELLS, per the plan's dry run; each
    # selected cell still runs all its repetitions. Limiting expanded keys instead would
    # make --limit 5 run five repetitions of the first cell.
    cells = [
        (scenario_id, model_id, configuration)
        for scenario_id in scenario_ids
        for model_id in models
        for configuration in configurations
    ]
    if limit is not None:
        cells = cells[:limit]
    all_keys = [
        RunKey(scenario_id, model_id, configuration, repetition)
        for scenario_id, model_id, configuration in cells
        for repetition in range(repetitions)
    ]

    keys = [
        key
        for key in all_keys
        if force or not is_done(
            results_dir=cells_dir,
            scenario_id=key.scenario_id,
            model_id=key.model_id,
            configuration=key.configuration,
            repetition=key.repetition,
        )
    ]

    def worker(key: RunKey) -> Path:
        started = time.monotonic()
        path = _run_key(key, cells_dir, dataset_root)
        if limit is not None:
            _print_dry_run_line(path, started)
        return path

    paths = _run_bounded(keys, concurrency, worker)
    update_manifest(
        result_root,
        model_ids=models,
        dataset=dataset,
        run_counts={"cells": len(list(cells_dir.glob("*.json")))},
    )
    return HarnessSummary(
        n_planned=len(all_keys),
        n_run=len(paths),
        n_skipped=len(all_keys) - len(keys),
        result_paths=tuple(paths),
    )


def _run_key(
    key: RunKey,
    cells_dir: Path,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> Path:
    scenario = load_scenario(key.scenario_id) if data_dir == DEFAULT_DATA_DIR else load_scenario(
        key.scenario_id,
        data_dir=data_dir,
    )
    config = _config_for_key(key)
    agent = _agent_for_key(key)
    response = _resolve_with_retry(scenario, config, agent)
    return save_response(response, cells_dir)


def _resolve_with_retry(scenario: Scenario, config: OrchestratorConfig, agent: AgentStep | ProposeSequence) -> ResolutionResponse:
    response = resolve(scenario, config, agent)
    if response.status in RETRY_STATUSES:
        response = resolve(scenario, config, agent)
    return response


def _config_for_key(key: RunKey) -> OrchestratorConfig:
    if key.configuration in {1, 2}:
        assignment = LLMAssignment(single_agent=key.model_id, analyst=None, executor=None, orchestrator=None)
    else:
        assignment = LLMAssignment(
            single_agent=None,
            analyst=key.model_id,
            executor=key.model_id,
            orchestrator=key.model_id,
        )
    return OrchestratorConfig(
        CONFIGURATION=cast(Literal[1, 2, 3], key.configuration),
        LLM_ASSIGNMENT=assignment,
        repetition_index=key.repetition,
    )


def _agent_for_key(key: RunKey) -> AgentStep | ProposeSequence:
    if key.configuration == 1:
        return make_baseline_chatbot(key.model_id)
    if key.configuration == 2:
        return make_single_agent(key.model_id)
    if key.configuration == 3:
        return make_multi_agent()
    raise ValueError(f"unsupported configuration {key.configuration}")


def _run_bounded(items: Sequence[T], concurrency: int, worker: Callable[[T], R]) -> list[R]:
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    if concurrency == 1:
        return [worker(item) for item in items]

    results: list[R] = []
    iterator = iter(items)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        pending: dict[Future[R], T] = {}
        for _ in range(min(concurrency, len(items))):
            item = next(iterator)
            pending[pool.submit(worker, item)] = item
        while pending:
            for future in as_completed(tuple(pending)):
                pending.pop(future)
                results.append(future.result())
                try:
                    item = next(iterator)
                except StopIteration:
                    pass
                else:
                    pending[pool.submit(worker, item)] = item
                break
    return results


def _print_dry_run_line(path: Path, started: float) -> None:
    # The dry run's whole job is to replace the cost ESTIMATE with a measured number before the
    # sweep, so it reports the USD spend of this run, not just the raw counts.
    response = ResolutionResponse.model_validate_json(path.read_text(encoding="utf-8"))
    trace = response.trace
    elapsed = time.monotonic() - started
    usd = token_cost_usd(
        model_id_from_response(response),
        tokens_in=trace.total_llm_tokens_in,
        tokens_out=trace.total_llm_tokens_out,
    )
    print(
        f"{path.stem}: status={response.status} "
        f"llm_calls={trace.n_llm_calls} "
        f"tokens_in={trace.total_llm_tokens_in} "
        f"tokens_out={trace.total_llm_tokens_out} "
        f"tokens_total={trace.total_llm_tokens} "
        f"usd={usd:.4f} "
        f"wall_seconds={elapsed:.2f}"
    )


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _dataset_manifest_entry(data_dir: Path) -> DatasetManifestEntry | None:
    path = data_dir / "manifest.json"
    if not path.exists():
        return None
    dataset = DatasetManifest.model_validate_json(path.read_text(encoding="utf-8"))
    return DatasetManifestEntry(
        dataset_version=dataset.dataset_version,
        base_network_hash=dataset.base_network_hash,
        split_manifest_hash=dataset.split_manifest_hash,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    selected_models = tuple(args.model or BENCHMARK_MODELS)
    run_evaluation(
        selected_models,
        configurations=tuple(args.configuration or [1, 2, 3]),
        results_dir=args.results_dir,
        data_dir=args.data_dir,
        repetitions=args.repeats,
        concurrency=args.concurrency,
        force=args.force,
        limit=args.limit,
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run RestoreBench evaluation sweeps.")
    # choices: an unknown id would otherwise run (and pay for) the LLM calls, then crash in
    # save_response when model_slug raises KeyError, aborting the sweep.
    parser.add_argument("--model", action="append", choices=list(SUPPORTED_MODELS))
    parser.add_argument("--configuration", type=int, action="append", choices=[1, 2, 3])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--force", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
