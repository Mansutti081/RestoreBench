# ABOUTME: Probes whether an LLM can read Scenario Cards using deterministic ground truth.
# ABOUTME: Scores topology, lever, bounds, and load-context comprehension without solving grids.
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence, TextIO

import pandapower as pp
from pydantic import BaseModel, ConfigDict

from restorebench.llm.models import CHEAPEST_MODEL
from restorebench.llm.providers import ChatMessage, llm_call
from restorebench.schemas.errors import LLMFailureError
from restorebench.environment import card_render as render

ROOT = Path(__file__).resolve().parents[2]
SEED = 42
N_DEFAULT = 20
# This is a dataset quality gate, not the benchmark: it runs many scenarios across several
# seeds and models, and its pass floors below were calibrated without reasoning. Reasoning
# stays off here, and the default is the cheapest model in the suite.
MODEL_DEFAULT = CHEAPEST_MODEL
# The probe scores by exact match on sets, indices and numbers. It is a measurement of the
# Scenario Card, not of a model's creativity, so it samples as deterministically as the
# provider allows. Passed explicitly: llm_call defaults to 1.0 for the benchmark, and the
# floors below were calibrated at 0.
PROBE_TEMPERATURE = 0.0
FLOOR_OVERALL = 0.85
FLOOR_PER_FAMILY = {"F1": 0.80, "F2": 0.85, "F3": 0.85, "F4": 0.90, "F5": 0.85, "F6": 0.90}

DATASET_DIR = ROOT / "dataset/ieee118"
FULL_DIR = DATASET_DIR / "full"
LLM_DIR = DATASET_DIR / "llm"
EVALUATION_MANIFEST_PATH = DATASET_DIR / "evaluation_manifest.json"
DEFAULT_REPORT = DATASET_DIR / "card_comprehension_report.json"
FAMILIES = tuple(FLOOR_PER_FAMILY)

Family = Literal["F1", "F2", "F3", "F4", "F5", "F6"]
QuestionType = Literal[
    "neighbors",
    "degree",
    "gen_indices_at_bus",
    "shunt_indices_at_bus",
    "incident_tappable_trafo_indices_at_bus",
    "shunt_type",
    "tap_position",
    "gen_p_max_mw",
    "gen_dispatchable",
]


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")

    qid: str
    family: Family
    qtype: QuestionType
    target_id: int
    text: str


class ReviewRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    qid: str
    family: Family
    question: str
    expected: Any
    model_answer: Any = None
    correct: bool


class CardComprehensionReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    timestamp: str
    n_scenarios: int
    n_questions: int
    per_family_accuracy: dict[str, float]
    overall_accuracy: float
    floor: dict[str, float]
    passed: bool
    rows: list[ReviewRow]


def sample_scenarios(n: int, seed: int = SEED) -> list[str]:
    """Return a seeded sample of scenario ids that have both FULL JSON and Card files.

    The probe measures whether the Scenario Card is readable, not whether a model can solve the
    scenario, so it may look at any card: no answer key is involved.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    full_ids = {path.stem for path in FULL_DIR.glob("S*.json")}
    llm_ids = {path.stem for path in LLM_DIR.glob("S*.md")}
    scenario_ids = sorted(full_ids & llm_ids)
    if n > len(scenario_ids):
        raise ValueError(f"requested {n} scenarios, but only {len(scenario_ids)} complete pairs exist")
    return random.Random(seed).sample(scenario_ids, n)


def build_questions(full_net, seed: int) -> list[Question]:
    """Build one deterministic battery of Card-answerable questions for a scenario."""
    rng = random.Random(seed)
    questions: list[Question] = []
    neighbors = render._neighbors(full_net)
    bus_ids = sorted(bus for bus, values in neighbors.items() if values)

    f1_bus = _pick(rng, bus_ids)
    questions.append(
        Question(
            qid=f"F1_neighbors_bus_{f1_bus}",
            family="F1",
            qtype="neighbors",
            target_id=f1_bus,
            text=f"Which buses are directly connected to bus {f1_bus}?",
        )
    )

    f2_bus = _pick(rng, bus_ids)
    questions.append(
        Question(
            qid=f"F2_degree_bus_{f2_bus}",
            family="F2",
            qtype="degree",
            target_id=f2_bus,
            text=f"How many buses is bus {f2_bus} directly connected to?",
        )
    )

    active_gens = render._active(full_net.gen).sort_index()
    gen_bus = _pick(rng, sorted(int(bus) for bus in active_gens["bus"].unique()))
    questions.append(
        Question(
            qid=f"F3_generators_bus_{gen_bus}",
            family="F3",
            qtype="gen_indices_at_bus",
            target_id=gen_bus,
            text=f"Which generator index(es) sit at bus {gen_bus}?",
        )
    )

    shunt_bus = _pick(rng, sorted(int(bus) for bus in full_net.shunt["bus"].unique()))
    questions.append(
        Question(
            qid=f"F3_shunts_bus_{shunt_bus}",
            family="F3",
            qtype="shunt_indices_at_bus",
            target_id=shunt_bus,
            text=f"Which shunt index(es) sit at bus {shunt_bus}?",
        )
    )

    active_tappable = _active_tappable_trafos(full_net)
    trafo_buses = sorted(
        {int(row["hv_bus"]) for _, row in active_tappable.iterrows()}
        | {int(row["lv_bus"]) for _, row in active_tappable.iterrows()}
    )
    if trafo_buses:
        trafo_bus = _pick(rng, trafo_buses)
        questions.append(
            Question(
                qid=f"F3_tappable_transformers_bus_{trafo_bus}",
                family="F3",
                qtype="incident_tappable_trafo_indices_at_bus",
                target_id=trafo_bus,
                text=f"Which tappable transformer index(es) are incident to bus {trafo_bus}?",
            )
        )

    shunt_ids = _representative_shunts(full_net)
    for shunt_id in shunt_ids:
        questions.append(
            Question(
                qid=f"F4_shunt_type_{shunt_id}",
                family="F4",
                qtype="shunt_type",
                target_id=shunt_id,
                text=f"Is shunt {shunt_id} a capacitor or a reactor?",
            )
        )

    if not active_tappable.empty:
        trafo_id = _pick(rng, [int(idx) for idx in active_tappable.index])
        questions.append(
            Question(
                qid=f"F5_tap_position_{trafo_id}",
                family="F5",
                qtype="tap_position",
                target_id=trafo_id,
                text=f"Is transformer {trafo_id}'s tap at its minimum, maximum, or interior?",
            )
        )

    dispatchable = active_gens[active_gens["min_p_mw"] < active_gens["max_p_mw"]]
    nondispatchable = active_gens[active_gens["min_p_mw"] >= active_gens["max_p_mw"]]
    pmax_gen = _pick(rng, [int(idx) for idx in dispatchable.index] or [int(idx) for idx in active_gens.index])
    questions.append(
        Question(
            qid=f"F6_gen_p_max_mw_{pmax_gen}",
            family="F6",
            qtype="gen_p_max_mw",
            target_id=pmax_gen,
            text=f"What is the maximum active power p_max_mw of generator {pmax_gen}?",
        )
    )
    dispatch_question_gen = _pick(
        rng,
        [int(idx) for idx in nondispatchable.index] or [int(idx) for idx in active_gens.index],
    )
    questions.append(
        Question(
            qid=f"F6_gen_dispatchable_{dispatch_question_gen}",
            family="F6",
            qtype="gen_dispatchable",
            target_id=dispatch_question_gen,
            text=f"Is generator {dispatch_question_gen} dispatchable?",
        )
    )

    return questions


def ground_truth(full_net, question: Question) -> Any:
    """Compute the expected answer using only quantities displayed in the Card."""
    if question.qtype == "neighbors":
        return render._neighbors(full_net)[question.target_id]
    if question.qtype == "degree":
        return len(render._neighbors(full_net)[question.target_id])
    if question.qtype == "gen_indices_at_bus":
        rows = render._active(full_net.gen)
        return _sorted_indices(rows[rows["bus"] == question.target_id])
    if question.qtype == "shunt_indices_at_bus":
        return _sorted_indices(full_net.shunt[full_net.shunt["bus"] == question.target_id])
    if question.qtype == "incident_tappable_trafo_indices_at_bus":
        rows = _active_tappable_trafos(full_net)
        mask = (rows["hv_bus"] == question.target_id) | (rows["lv_bus"] == question.target_id)
        return _sorted_indices(rows[mask])
    if question.qtype == "shunt_type":
        q_mvar = float(full_net.shunt.at[question.target_id, "q_mvar"])
        if q_mvar < 0:
            return "capacitor"
        if q_mvar > 0:
            return "reactor"
        return "neutral"
    if question.qtype == "tap_position":
        row = full_net.trafo.loc[question.target_id]
        tap_pos = int(row["tap_pos"])
        tap_min = int(row["tap_min"])
        tap_max = int(row["tap_max"])
        if tap_pos == tap_min:
            return "minimum"
        if tap_pos == tap_max:
            return "maximum"
        return "interior"
    if question.qtype == "gen_p_max_mw":
        return float(full_net.gen.at[question.target_id, "max_p_mw"])
    if question.qtype == "gen_dispatchable":
        return bool(full_net.gen.at[question.target_id, "min_p_mw"] < full_net.gen.at[question.target_id, "max_p_mw"])
    raise ValueError(f"unsupported question type: {question.qtype}")


def render_prompt(card_md: str, questions: Sequence[Question]) -> str:
    """Render one Card plus the deterministic question battery into a structured-answer prompt."""
    question_lines = "\n".join(f"- {question.qid}: {question.text}" for question in questions)
    return (
        "You are reading one benchmark Scenario Card. Answer only from the Card text.\n"
        "Return only JSON in this exact shape: {\"answers\": {\"QUESTION_ID\": ANSWER}}.\n"
        "Use arrays of integers for index/set answers, numbers for numeric answers, booleans for yes/no, "
        "and lowercase strings for categories.\n\n"
        "SCENARIO CARD:\n"
        f"{card_md}\n"
        "QUESTIONS:\n"
        f"{question_lines}\n"
    )


def parse_answers(raw: str) -> dict[str, Any]:
    """Parse model answers; malformed output returns no answers and never raises."""
    for candidate in _json_candidates(raw):
        try:
            parsed = json.loads(candidate)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(parsed, dict) and isinstance(parsed.get("answers"), dict):
            return {str(key): value for key, value in parsed["answers"].items()}
        if isinstance(parsed, dict):
            return {str(key): value for key, value in parsed.items()}
    return {}


def score(expected: Any, got: Any, family: str) -> bool:
    """Return exact correctness under each family scoring rule."""
    if got is None:
        return False
    if family in {"F1", "F3"}:
        expected_set = _as_int_set(expected)
        got_set = _as_int_set(got)
        return expected_set is not None and got_set is not None and expected_set == got_set
    if family in {"F2", "F6"}:
        return _scalar_equal(expected, got)
    if family in {"F4", "F5"}:
        return str(expected).strip().lower() == str(got).strip().lower()
    return False


def dry_run(n: int = N_DEFAULT, seed: int = SEED, stream: TextIO = sys.stdout) -> list[ReviewRow]:
    """Build and print the deterministic question/ground-truth battery without any LLM call."""
    rows: list[ReviewRow] = []
    print(f"DRY RUN: building Card comprehension battery for n={n}", file=stream)
    for scenario_index, scenario_id in enumerate(sample_scenarios(n, seed=seed)):
        net = _load_full_net(scenario_id)
        print(f"\n[{scenario_id}]", file=stream)
        for question in build_questions(net, seed + scenario_index):
            expected = ground_truth(net, question)
            print(f"{question.qid} ({question.family}): {question.text}", file=stream)
            print(f"  expected: {expected}", file=stream)
            rows.append(
                ReviewRow(
                    scenario_id=scenario_id,
                    qid=question.qid,
                    family=question.family,
                    question=question.text,
                    expected=expected,
                    model_answer=None,
                    correct=False,
                )
            )
    return rows


def review(
    n: int,
    model: str = MODEL_DEFAULT,
    *,
    seed: int = SEED,
) -> CardComprehensionReport:
    """Run the live Card-comprehension review over a seeded scenario sample."""
    rows: list[ReviewRow] = []
    for scenario_index, scenario_id in enumerate(sample_scenarios(n, seed=seed)):
        net = _load_full_net(scenario_id)
        card_md = _load_card(scenario_id)
        questions = build_questions(net, seed + scenario_index)
        prompt = render_prompt(card_md, questions)
        response = llm_call(model, [ChatMessage(role="user", content=prompt)], temperature=PROBE_TEMPERATURE)
        parsed_answers = parse_answers(response.text)
        for question in questions:
            expected = ground_truth(net, question)
            model_answer = parsed_answers.get(question.qid)
            rows.append(
                ReviewRow(
                    scenario_id=scenario_id,
                    qid=question.qid,
                    family=question.family,
                    question=question.text,
                    expected=expected,
                    model_answer=model_answer,
                    correct=score(expected, model_answer, question.family),
                )
            )
    return _build_report(rows=rows, model=model, n_scenarios=n)


def write_report(report: CardComprehensionReport, path: str | Path = DEFAULT_REPORT) -> None:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=N_DEFAULT)
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.dry_run:
        dry_run(n=args.n, seed=args.seed, stream=sys.stdout)
        return 0

    try:
        report = review(n=args.n, model=args.model, seed=args.seed)
    except LLMFailureError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    write_report(report, args.report)
    print(f"overall_accuracy={report.overall_accuracy:.3f} passed={report.passed}")
    for family in FAMILIES:
        print(f"{family}={report.per_family_accuracy.get(family, 0.0):.3f}")
    return 0 if report.passed else 1


def _build_report(rows: list[ReviewRow], model: str, n_scenarios: int) -> CardComprehensionReport:
    per_family = {}
    for family in FAMILIES:
        family_rows = [row for row in rows if row.family == family]
        per_family[family] = _accuracy(family_rows)
    overall = _accuracy(rows)
    floor = {"overall": FLOOR_OVERALL, **FLOOR_PER_FAMILY}
    passed = overall >= FLOOR_OVERALL and all(per_family[family] >= FLOOR_PER_FAMILY[family] for family in FAMILIES)
    return CardComprehensionReport(
        model=model,
        timestamp=datetime.now(timezone.utc).isoformat(),
        n_scenarios=n_scenarios,
        n_questions=len(rows),
        per_family_accuracy=per_family,
        overall_accuracy=overall,
        floor=floor,
        passed=passed,
        rows=rows,
    )


def _accuracy(rows: Sequence[ReviewRow]) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if row.correct) / len(rows)


def _load_full_net(scenario_id: str):
    path = FULL_DIR / f"{scenario_id}.json"
    if not path.exists():
        raise RuntimeError(f"FULL scenario missing: {path}")
    return pp.from_json(str(path))


def _load_card(scenario_id: str) -> str:
    path = LLM_DIR / f"{scenario_id}.md"
    if not path.exists():
        raise RuntimeError(f"Scenario Card missing: {path}")
    return path.read_text(encoding="utf-8")


def _pick(rng: random.Random, values: Sequence[int]) -> int:
    if not values:
        raise ValueError("cannot pick from an empty sequence")
    return int(rng.choice(list(values)))


def _active_tappable_trafos(net):
    rows = render._active(net.trafo)
    return rows[rows["tap_pos"].notna()].sort_index()


def _representative_shunts(net) -> list[int]:
    capacitors = [int(idx) for idx in net.shunt.index[net.shunt["q_mvar"] < 0]]
    reactors = [int(idx) for idx in net.shunt.index[net.shunt["q_mvar"] > 0]]
    selected = []
    if capacitors:
        selected.append(min(capacitors))
    selected.extend(sorted(reactors))
    return selected


def _sorted_indices(rows) -> list[int]:
    return [int(idx) for idx in sorted(rows.index)]


def _json_candidates(raw: str) -> list[str]:
    text = (raw or "").strip()
    if not text:
        return []
    candidates = [text]
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            cleaned = part.strip()
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
            if cleaned:
                candidates.append(cleaned)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and start < end:
        candidates.append(text[start : end + 1])
    return candidates


def _as_int_set(value: Any) -> set[int] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, (list, tuple, set)):
        return None
    try:
        return {int(item) for item in value}
    except (TypeError, ValueError):
        return None


def _scalar_equal(expected: Any, got: Any) -> bool:
    if isinstance(expected, bool):
        got_bool = _as_bool(got)
        return got_bool is not None and got_bool is expected
    if isinstance(expected, int) and not isinstance(expected, bool):
        try:
            return int(got) == expected
        except (TypeError, ValueError):
            return False
    if isinstance(expected, float):
        try:
            return float(got) == expected
        except (TypeError, ValueError):
            return False
    return expected == got


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return None


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
