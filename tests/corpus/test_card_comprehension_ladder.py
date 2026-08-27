# ABOUTME: Verifies the Card comprehension ladder report generator.
# ABOUTME: Uses a fake review function so no provider or dataset live path is exercised.
from restorebench.llm.models import OPUS_4_6, QWEN3_32B
from restorebench.corpus import card_comprehension_ladder as ladder
from restorebench.corpus import card_comprehension_review as review


def _fake_report(model: str, n: int, accuracy: float) -> review.CardComprehensionReport:
    return review.CardComprehensionReport(
        model=model,
        timestamp="2026-07-08T00:00:00+00:00",
        n_scenarios=n,
        n_questions=10,
        per_family_accuracy={family: accuracy for family in review.FAMILIES},
        overall_accuracy=accuracy,
        floor={"overall": review.FLOOR_OVERALL, **review.FLOOR_PER_FAMILY},
        passed=accuracy >= review.FLOOR_OVERALL,
        rows=[],
    )


def test_run_ladder_calls_review_for_each_model_seed_and_writes_report(monkeypatch, tmp_path):
    calls = []

    def fake_review(n: int, model: str, seed: int) -> review.CardComprehensionReport:
        calls.append((n, model, seed))
        accuracy = 0.95 if seed == 42 else 0.90
        return _fake_report(model=model, n=n, accuracy=accuracy)

    monkeypatch.setattr(ladder.review, "review", fake_review)
    report_path = tmp_path / "card_comprehension_ladder_report.md"

    output_path = ladder.run_ladder(
        n=40,
        models=(OPUS_4_6, QWEN3_32B),
        seeds=(42, 7),
        report_path=report_path,
    )

    assert output_path == report_path
    assert calls == [
        (40, OPUS_4_6, 42),
        (40, OPUS_4_6, 7),
        (40, QWEN3_32B, 42),
        (40, QWEN3_32B, 7),
    ]
    rendered = report_path.read_text(encoding="utf-8")
    assert "# Card-comprehension ladder report" in rendered
    assert OPUS_4_6 in rendered
    assert QWEN3_32B in rendered
    assert "   42" in rendered
    assert "    7" in rendered
    assert "  mean" in rendered
    assert "   min" in rendered
