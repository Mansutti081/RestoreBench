# ABOUTME: Persists ResolutionResponse objects for evaluation runs.
# ABOUTME: Defines run ids, atomic writes, loading, and filesystem-based resume helpers.
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from restorebench.llm.models import model_slug
from restorebench.schemas.response import ResolutionResponse


@dataclass(frozen=True)
class RunKey:
    scenario_id: str
    model_id: str
    configuration: int
    repetition: int

    @property
    def run_id(self) -> str:
        return run_id(self.scenario_id, self.configuration, self.model_id, self.repetition)


def run_id(scenario_id: str, configuration: int, model_id: str, repetition: int) -> str:
    return f"{scenario_id}__config{configuration}__{model_slug(model_id)}__rep{repetition}"


def save_response(response: ResolutionResponse, results_dir: str | Path) -> Path:
    target = response_path(response, results_dir)
    atomic_write_text(target, _response_json(response))
    return target


def load_response(path: str | Path) -> ResolutionResponse:
    return ResolutionResponse.model_validate_json(Path(path).read_text(encoding="utf-8"))


def response_path(response: ResolutionResponse, results_dir: str | Path) -> Path:
    directory = Path(results_dir)
    model_id = model_id_from_response(response)
    return directory / f"{run_id(response.scenario_id, response.configuration, model_id, response.repetition_index)}.json"


def model_id_from_response(response: ResolutionResponse) -> str:
    assignment = response.llm_assignment
    for role in ("single_agent", "analyst", "executor", "orchestrator"):
        model_id = assignment.get(role)
        if model_id:
            return model_id
    raise ValueError("ResolutionResponse.llm_assignment does not contain a model id")


def is_done(
    *,
    results_dir: str | Path,
    scenario_id: str,
    model_id: str,
    configuration: int,
    repetition: int = 0,
) -> bool:
    return cell_path(results_dir, scenario_id, configuration, model_id, repetition).exists()


def cell_path(
    results_dir: str | Path,
    scenario_id: str,
    configuration: int,
    model_id: str,
    repetition: int,
) -> Path:
    return Path(results_dir) / f"{run_id(scenario_id, configuration, model_id, repetition)}.json"


def missing_runs(
    *,
    results_dir: str | Path,
    scenario_ids: Iterable[str],
    model_ids: Iterable[str],
    configurations: Iterable[int],
    repetitions: Iterable[int],
) -> list[RunKey]:
    missing: list[RunKey] = []
    for scenario_id in scenario_ids:
        for model_id in model_ids:
            for configuration in configurations:
                for repetition in repetitions:
                    if not is_done(
                        results_dir=results_dir,
                        scenario_id=scenario_id,
                        model_id=model_id,
                        configuration=configuration,
                        repetition=repetition,
                    ):
                        missing.append(RunKey(scenario_id, model_id, configuration, repetition))
    return missing


def atomic_write_text(target: Path, text: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        _replace_temp(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()


def _replace_temp(tmp: Path, target: Path) -> None:
    tmp.replace(target)


def _response_json(response: ResolutionResponse) -> str:
    return json.dumps(response.model_dump(mode="json"), indent=2) + "\n"
