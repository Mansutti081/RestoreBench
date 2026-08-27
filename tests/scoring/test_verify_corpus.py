# ABOUTME: Tests the corpus verifier that replays witnesses, including how it reports failures.
# ABOUTME: Uses the real corpus for the passing path and injected witnesses for the failing ones.
from __future__ import annotations

import json
from pathlib import Path

import pytest

from restorebench.scoring import verify_corpus as verifier


DATASET_DIR = Path(__file__).resolve().parents[2] / "dataset/ieee118"


def test_real_witnesses_verify_against_the_frozen_corpus() -> None:
    report = verifier.verify_corpus(DATASET_DIR, limit=2)

    assert report["scenarios_checked"] == 2
    assert report["scenarios_verified"] == 2
    assert report["failures"] == []
    # Every witness resolves within budget, so no length may exceed it.
    lengths = [int(key) for key in report["witness_length_distribution"]]
    assert lengths and max(lengths) <= report["maneuver_budget"]


def _corpus_with_witnesses(tmp_path: Path, rows: list[dict]) -> Path:
    """A corpus whose public artifacts are the real ones and whose witnesses are ours."""
    staged = tmp_path / "corpus"
    staged.mkdir()
    for name in ("full", "lean", "llm"):
        (staged / name).symlink_to(DATASET_DIR / name)
    for name in ("manifest.json", "evaluation_manifest.json"):
        (staged / name).symlink_to(DATASET_DIR / name)
    (staged / "private").mkdir()
    (staged / "private" / "witnesses.json").write_text(json.dumps(rows), encoding="utf-8")
    return staged


def test_a_witness_that_does_not_resolve_is_reported_not_swallowed(tmp_path: Path) -> None:
    # An empty maneuver list is a legal attempt that deterministically cannot resolve anything:
    # the claim under test is failure reporting, so the corpus must be told it failed rather
    # than quietly counted as verified.
    staged = _corpus_with_witnesses(tmp_path, [
        {"scenario_id": "S0008", "maneuvers": []},
    ])

    report = verifier.verify_corpus(staged)

    assert report["scenarios_verified"] == 0
    [failure] = report["failures"]
    assert failure["scenario_id"] == "S0008"
    assert failure["claim"] == "witness_resolves"


def test_missing_witness_file_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(verifier.VerificationError, match="no witness file"):
        verifier.verify_corpus(tmp_path)


def test_empty_witness_file_fails_loudly(tmp_path: Path) -> None:
    staged = _corpus_with_witnesses(tmp_path, [])

    with pytest.raises(verifier.VerificationError, match="no witnesses found"):
        verifier.verify_corpus(staged)


def test_main_returns_nonzero_when_a_claim_fails(tmp_path: Path, capsys) -> None:
    staged = _corpus_with_witnesses(tmp_path, [
        {"scenario_id": "S0008", "maneuvers": []},
    ])

    exit_code = verifier.main(["--dataset-dir", str(staged)])

    assert exit_code == 1
    assert "FAILED" in capsys.readouterr().err
