# ABOUTME: Verifies Gate 5 Card comprehension probe deterministic core.
# ABOUTME: Covers sampling, question ground truth, parsing, scoring, dry-run, and leakage guards.
import io
import json
import os
import sys

import pytest

from restorebench.llm.models import QWEN3_32B
from restorebench.llm.providers import ChatMessage, LLMResponse
from restorebench.schemas.errors import LLMFailureError
from restorebench.corpus import card_comprehension_review as review
from restorebench.corpus.augment import build_augmented_base
from restorebench.corpus import render_scenario_card as render


FORBIDDEN_LEAKAGE = [
    "res_bus",
    "res_line",
    "res_trafo",
    "loading",
    "loading_percent",
    "mismatch",
    "diagnostic",
    "nr",
    "newton",
    "pre_weakening",
    "outage",
    "removed",
    "impedance",
    "thermal",
]


def test_sample_scenarios_is_seeded_and_existing():
    first = review.sample_scenarios(20, seed=42)
    second = review.sample_scenarios(20, seed=42)

    assert first == second
    assert len(first) == 20
    assert len(set(first)) == 20
    for scenario_id in first:
        assert (review.FULL_DIR / f"{scenario_id}.json").exists()
        assert (review.LLM_DIR / f"{scenario_id}.md").exists()


def test_sample_scenarios_stays_inside_the_corpus():
    corpus_ids = {path.stem for path in review.FULL_DIR.glob("S*.json")}

    for seed in (42, 1337, 7):
        sampled = review.sample_scenarios(40, seed=seed)
        assert set(sampled) <= corpus_ids


def test_build_questions_and_ground_truth_match_card_view_on_augmented_base():
    net = build_augmented_base()
    questions = review.build_questions(net, seed=42)
    families = {question.family for question in questions}
    neighbor_map = render._neighbors(net)

    assert families == {"F1", "F2", "F3", "F4", "F5", "F6"}

    f1 = next(question for question in questions if question.family == "F1")
    assert review.ground_truth(net, f1) == neighbor_map[f1.target_id]
    for neighbor in review.ground_truth(net, f1):
        assert f1.target_id in neighbor_map[neighbor]

    f2 = next(question for question in questions if question.family == "F2")
    assert review.ground_truth(net, f2) == len(neighbor_map[f2.target_id])

    f4_answers = {
        question.target_id: review.ground_truth(net, question)
        for question in questions
        if question.family == "F4"
    }
    capacitor_id = int(net.shunt.index[net.shunt.q_mvar < 0][0])
    reactor_ids = [int(idx) for idx in net.shunt.index[net.shunt.q_mvar > 0]]
    assert f4_answers[capacitor_id] == "capacitor"
    for reactor_id in reactor_ids:
        assert f4_answers[reactor_id] == "reactor"

    dispatchable = []
    condensers = []
    for gen_id in net.gen.index:
        question = review.Question(
            qid=f"dispatchable_{int(gen_id)}",
            family="F6",
            qtype="gen_dispatchable",
            target_id=int(gen_id),
            text=f"Is generator {int(gen_id)} dispatchable?",
        )
        if review.ground_truth(net, question):
            dispatchable.append(int(gen_id))
        else:
            condensers.append(int(gen_id))
    assert len(dispatchable) == 18
    assert len(condensers) == 35


def test_score_and_parse_answers_are_exact_and_malformed_input_is_wrong():
    assert review.score([1, 2], [2, 1], "F1")
    assert not review.score([1, 2], [1], "F1")
    assert review.score([4, 5], [5, 4], "F3")
    assert review.score("capacitor", "Capacitor", "F4")
    assert review.score("interior", "interior", "F5")
    assert review.score(100.0, 100.0, "F6")
    assert not review.score(100.0, 99.999, "F6")

    parsed = review.parse_answers("not valid json")

    assert parsed == {}
    assert not review.score([1, 2], parsed.get("missing"), "F1")


def test_render_prompt_and_parse_answers_use_question_ids():
    net = build_augmented_base()
    questions = review.build_questions(net, seed=7)[:2]
    prompt = review.render_prompt(render.render_scenario_card(net), questions)

    assert "Return only JSON" in prompt
    for question in questions:
        assert question.qid in prompt
        assert question.text in prompt

    raw = json.dumps({"answers": {questions[0].qid: [1, 2], questions[1].qid: 3}})
    assert review.parse_answers(raw) == {questions[0].qid: [1, 2], questions[1].qid: 3}


def test_dry_run_builds_ground_truth_without_openai_client_import():
    before = sys.modules.get("openai")
    output = io.StringIO()

    rows = review.dry_run(n=1, seed=42, stream=output)

    assert rows
    assert "DRY RUN" in output.getvalue()
    assert "expected" in output.getvalue()
    assert sys.modules.get("openai") is before


def test_review_uses_provider_layer_with_explicit_seed(monkeypatch):
    question = review.Question(
        qid="F2_degree_bus_1",
        family="F2",
        qtype="degree",
        target_id=1,
        text="How many buses is bus 1 directly connected to?",
    )
    seen = {}

    def fake_sample_scenarios(n: int, seed: int) -> list[str]:
        seen["sample"] = (n, seed)
        return ["S9999"]

    def fake_build_questions(net, seed: int) -> list[review.Question]:
        seen["question_seed"] = seed
        return [question]

    def fake_llm_call(model_id: str, messages: list[ChatMessage], **kwargs) -> LLMResponse:
        seen["llm"] = (model_id, messages)
        seen["temperature"] = kwargs.get("temperature")
        return LLMResponse(
            text=json.dumps({"answers": {question.qid: 3}}),
            model_id=model_id,
            tokens_in=12,
            tokens_out=5,
            latency_seconds=0.01,
        )

    monkeypatch.setattr(review, "sample_scenarios", fake_sample_scenarios)
    monkeypatch.setattr(review, "_load_full_net", lambda scenario_id: object())
    monkeypatch.setattr(review, "_load_card", lambda scenario_id: "card markdown")
    monkeypatch.setattr(review, "build_questions", fake_build_questions)
    monkeypatch.setattr(review, "render_prompt", lambda card_md, questions: "rendered prompt")
    monkeypatch.setattr(review, "ground_truth", lambda net, candidate: 3)
    monkeypatch.setattr(review, "llm_call", fake_llm_call)

    report = review.review(n=1, model=QWEN3_32B, seed=1337)

    assert seen["sample"] == (1, 1337)
    assert seen["question_seed"] == 1337
    assert seen["llm"] == (QWEN3_32B, [ChatMessage(role="user", content="rendered prompt")])
    assert seen["temperature"] == review.PROBE_TEMPERATURE == 0.0
    assert report.n_scenarios == 1
    assert report.rows[0].correct
    assert report.rows[0].model_answer == 3


def test_questions_and_ground_truth_do_not_leak_hidden_or_pf_quantities():
    net = build_augmented_base()
    questions = review.build_questions(net, seed=42)

    for question in questions:
        expected = review.ground_truth(net, question)
        haystack = f"{question.text} {expected}".lower()
        for forbidden in FORBIDDEN_LEAKAGE:
            assert forbidden not in haystack


@pytest.mark.skipif(
    os.environ.get("RESTOREBENCH_LLM_INTEGRATION") != "1",
    reason="real LLM integration is opt-in and not part of the deterministic unit gate",
)
def test_real_llm_review_n5_when_operator_opts_in(tmp_path):
    report_path = tmp_path / "card_comprehension_report.json"

    try:
        report = review.review(n=5, model=review.MODEL_DEFAULT)
    except LLMFailureError as exc:
        pytest.skip(f"LLM endpoint unreachable or unavailable: {exc}")

    review.write_report(report, report_path)
    parsed = json.loads(report_path.read_text(encoding="utf-8"))
    assert parsed["n_scenarios"] == 5
    assert parsed["n_questions"] == len(parsed["rows"])
