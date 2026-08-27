# ABOUTME: Runs one-shot Bedrock benchmark calls and checkpoints scorer attempt files.
# ABOUTME: Persists prompts, raw LLM responses, token counts, and validation failures per call.
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import pandapower as pp
from pydantic import ValidationError

from restorebench.llm.providers import ChatMessage, LLMResponse, llm_call
from restorebench.llm import providers
from restorebench.llm.models import GPT_OSS_120B
from restorebench.environment.card_render import render_scenario_card
from restorebench.schemas.actions import ManeuverSequence
from restorebench.scoring.score_maneuvers import MANEUVER_BUDGET


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT / "dataset/ieee118/full"
DEFAULT_DATASET_ROOT = ROOT / "dataset/ieee118"
DEFAULT_EVALUATION_MANIFEST_PATH = DEFAULT_DATASET_ROOT / "evaluation_manifest.json"
DEFAULT_SPLIT_PATH = DEFAULT_EVALUATION_MANIFEST_PATH
DEFAULT_OUT_DIR = ROOT / "results/bedrock_benchmark"
PROPOSE_MANEUVERS_TOOL_NAME = "propose_maneuvers"
DEFAULT_MAX_TOKENS = 8192
GPT_OSS_MAX_TOKENS = 16384


@dataclass(frozen=True)
class RunResult:
    scenario_id: str
    model_id: str
    repeat_index: int
    attempt_path: Path
    record_path: Path
    skipped: bool
    validation_error: str | None = None
    truncated: bool = False


def maneuver_sequence_tool_config() -> dict[str, Any]:
    return {
        "tools": [
            {
                "toolSpec": {
                    "name": PROPOSE_MANEUVERS_TOOL_NAME,
                    "description": "Return an ordered ManeuverSequence that restores AC power-flow convergence.",
                    "inputSchema": {"json": ManeuverSequence.model_json_schema()},
                }
            }
        ]
    }


def run_one(
    scenario_id: str,
    model_id: str,
    *,
    repeat_index: int,
    out_dir: Path = DEFAULT_OUT_DIR,
    data_dir: Path = DEFAULT_DATA_DIR,
    force: bool = False,
) -> RunResult:
    target_attempt = attempt_path(out_dir, model_id, scenario_id, repeat_index)
    target_record = record_path(out_dir, model_id, scenario_id, repeat_index)
    if target_attempt.exists() and not force:
        return RunResult(scenario_id, model_id, repeat_index, target_attempt, target_record, skipped=True)

    prompt = build_prompt(load_scenario_card(scenario_id, data_dir=data_dir))
    response = call_model(model_id, prompt)
    maneuvers, validation_error = extract_attempt_maneuvers(response)
    attempt = {"scenario_id": scenario_id, "maneuvers": maneuvers, "source": model_id}
    record = build_run_record(scenario_id, model_id, repeat_index, prompt, response, validation_error)
    write_json(target_attempt, attempt)
    write_json(target_record, record)
    return RunResult(
        scenario_id,
        model_id,
        repeat_index,
        target_attempt,
        target_record,
        False,
        validation_error,
        truncated=record["truncated"],
    )


def run_many(
    scenario_ids: Sequence[str],
    model_id: str,
    *,
    repeats: int,
    out_dir: Path = DEFAULT_OUT_DIR,
    data_dir: Path = DEFAULT_DATA_DIR,
    force: bool = False,
    limit: int | None = None,
) -> list[RunResult]:
    jobs = [(scenario_id, repeat) for scenario_id in scenario_ids for repeat in range(repeats)]
    if limit is not None:
        jobs = jobs[:limit]
    return [
        run_one(scenario_id, model_id, repeat_index=repeat, out_dir=out_dir, data_dir=data_dir, force=force)
        for scenario_id, repeat in jobs
    ]


def attempt_path(out_dir: Path, model_id: str, scenario_id: str, repeat_index: int) -> Path:
    return out_dir / "attempts" / model_slug(model_id) / f"{scenario_id}__r{repeat_index:03d}.json"


def record_path(out_dir: Path, model_id: str, scenario_id: str, repeat_index: int) -> Path:
    return out_dir / "records" / model_slug(model_id) / f"{scenario_id}__r{repeat_index:03d}.json"


def model_slug(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model_id).strip("_")


def load_scenario_card(scenario_id: str, *, data_dir: Path = DEFAULT_DATA_DIR) -> str:
    net = pp.from_json(str(data_dir / f"{scenario_id}.json"))
    return render_scenario_card(net)


def build_prompt(scenario_card: str) -> list[ChatMessage]:
    return [
        ChatMessage(role="system", content=_system_prompt()),
        ChatMessage(role="user", content=f"# Scenario Card\n\n{scenario_card}"),
    ]


def call_model(model_id: str, prompt: list[ChatMessage]) -> LLMResponse:
    # This harness records a max_tokens stop as its own `truncated` outcome (see is_truncated),
    # so it opts out of the default raise to receive the partial response.
    return llm_call(
        model_id,
        prompt,
        temperature=1.0,
        max_tokens=max_tokens_for_model(model_id),
        thinking=True,
        tools=maneuver_sequence_tool_config(),
        raise_on_truncation=False,
    )


def extract_attempt_maneuvers(response: LLMResponse) -> tuple[list[dict[str, Any]], str | None]:
    if is_truncated(response):
        return [], None
    if response.tool_use is None:
        return [], "response did not contain a toolUse block"
    try:
        sequence = ManeuverSequence.model_validate(response.tool_use.input)
    except ValidationError as exc:
        return [], str(exc)
    maneuvers = [maneuver.action.model_dump(mode="json") for maneuver in sequence.maneuvers[:MANEUVER_BUDGET]]
    return maneuvers, None


def build_run_record(
    scenario_id: str,
    model_id: str,
    repeat_index: int,
    prompt: list[ChatMessage],
    response: LLMResponse,
    validation_error: str | None,
) -> dict[str, Any]:
    return {
        "scenario_id": scenario_id,
        "model_id": model_id,
        "repeat_index": repeat_index,
        "region": os.environ.get("AWS_REGION", providers.DEFAULT_REGION),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "prompt": [message.model_dump(mode="json") for message in prompt],
        "raw_response": response.raw,
        "response_text": response.text,
        "tokens_in": response.tokens_in,
        "tokens_out": response.tokens_out,
        "latency_seconds": response.latency_seconds,
        "reasoning_present": "reasoning" in response.raw,
        "truncated": is_truncated(response),
        "validation_error": validation_error,
    }


def is_truncated(response: LLMResponse) -> bool:
    return response.raw.get("stop_reason") == "max_tokens"


def max_tokens_for_model(model_id: str) -> int:
    """GPT-OSS reasons by default and carries a much heavier chat template, so it needs
    more headroom than the others before Bedrock truncates the tool call."""
    return GPT_OSS_MAX_TOKENS if model_id == GPT_OSS_120B else DEFAULT_MAX_TOKENS


def scenario_ids_from_index(index_path: Path) -> list[str]:
    loaded = json.loads(index_path.read_text(encoding="utf-8"))
    return [str(label["scenario_id"]) for label in loaded["scenarios"]]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    scenario_ids = select_scenario_ids(args.scenarios, args.scenario_index)
    results = run_many(
        scenario_ids,
        args.model,
        repeats=args.repeats,
        out_dir=args.out,
        force=args.force,
        limit=args.limit,
    )
    print(json.dumps([result_summary(result) for result in results], indent=2, sort_keys=True))


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one-shot Bedrock benchmark attempts.")
    parser.add_argument("scenarios", nargs="*", help="scenario IDs such as S0001")
    parser.add_argument("--scenario-index", type=Path, help="JSON scenario index to run")
    parser.add_argument("--model", required=True, help="Bedrock modelId")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser.parse_args(argv)


def select_scenario_ids(scenarios: Sequence[str], scenario_index: Path | None) -> list[str]:
    if scenarios:
        return list(scenarios)
    index_path = scenario_index or DEFAULT_SPLIT_PATH
    return scenario_ids_from_index(index_path)


def result_summary(result: RunResult) -> dict[str, Any]:
    return {
        "scenario_id": result.scenario_id,
        "model_id": result.model_id,
        "repeat_index": result.repeat_index,
        "attempt_path": str(result.attempt_path),
        "record_path": str(result.record_path),
        "skipped": result.skipped,
        "validation_error": result.validation_error,
        "truncated": result.truncated,
    }


def _system_prompt() -> str:
    return (
        "You are resolving one non-converging IEEE-118 AC power-flow scenario. "
        "Return exactly one tool call with a ManeuverSequence of at most ten ordered maneuvers. "
        "The tool input must be an object with exactly `maneuvers` and `reconstruction_summary`. "
        "Each maneuver must contain `action`, `diagnosed_cause`, and `rationale`. "
        "`diagnosed_cause` must be null or exactly one of `REACTIVE_DEFICIT`, `CORRIDOR_OVERSTRESS`, "
        "`BAD_SETPOINTS`, `QLIM_INSTABILITY`, or `WEAK_SLACK`; put explanatory prose only in `rationale`. "
        "The `action` field must be an object, never a string. Valid action object shapes are: "
        "`{\"type\":\"GEN_V_SETPOINT\",\"gen_id\":int,\"new_vm_pu\":float}`, "
        "`{\"type\":\"SHUNT_STEP\",\"shunt_id\":int,\"new_step\":int}`, "
        "`{\"type\":\"TAP_ADJUSTMENT\",\"trafo_id\":int,\"new_tap_pos\":int}`. "
        "Do not use alternate keys such as `index`, `generator_index`, `tap_pos`, `vm_pu`, "
        "`p_mw`, `in_service`, `target_state`, or `shunt_index`."
    )


if __name__ == "__main__":
    main()
