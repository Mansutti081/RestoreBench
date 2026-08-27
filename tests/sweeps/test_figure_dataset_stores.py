# ABOUTME: Tests that the figure dataset pools the Anthropic and Bedrock stores safely.
# ABOUTME: Coverage is judged per store, and mismatched version stamps stop the build.
from __future__ import annotations

import json
from pathlib import Path

import pytest

from restorebench.sweeps import build_figure_dataset as builder

STAMP = {
    "dataset_version": "ieee118-reactive-deficit-v1",
    "solver_version": "locked-nr-q-limited-v1",
    "action_policy_version": "qv-atomic-v1",
    "ranking_policy_version": "retreat=snapshot-anchored-retreat-v1",
    "result_schema_version": "resolution-response-v2",
}


def _write_cell(store: Path, case: str, configuration: int, model_id: str, **overrides) -> None:
    store.mkdir(parents=True, exist_ok=True)
    payload = {
        "scenario_id": case,
        "configuration": configuration,
        "repetition_index": 0,
        "llm_assignment": {"single_agent": model_id, "analyst": None},
        "status": "SUCCESS",
        "n_maneuvers": 1,
        "converged": True,
        "quality": {"n_buses_out_of_band": 0},
        "trace": {"total_llm_tokens_in": 100, "total_llm_tokens_out": 10, "n_llm_calls": 1},
        "failure_feedback": [],
        "total_runtime_seconds": 1.0,
        **STAMP,
        **overrides,
    }
    (store / f"{case}__config{configuration}__{model_id}__rep0.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


@pytest.fixture
def stores(tmp_path, monkeypatch):
    anthropic = tmp_path / "anthropic" / "phase_b"
    bedrock = tmp_path / "bedrock" / "phase_b"
    monkeypatch.setattr(builder, "STORES", (anthropic, bedrock))
    labels = tmp_path / "labels.json"
    labels.write_text(
        json.dumps([{"scenario_id": "S0001", "resolution_regime": "DIRECT", "witness_length": 1}]),
        encoding="utf-8",
    )
    monkeypatch.setattr(builder, "LABELS", labels)
    return anthropic, bedrock


def _complete_case(store: Path, case: str, model_ids: list[str]) -> None:
    for model_id in model_ids:
        for configuration in (1, 2, 3):
            _write_cell(store, case, configuration, model_id)


def test_rows_come_from_both_stores(stores):
    # Arrange
    anthropic, bedrock = stores
    _complete_case(anthropic, "S0001", ["claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5"])
    _complete_case(bedrock, "S0001", ["deepseek.v3.2", "moonshotai.kimi-k2.5", "zai.glm-5"])

    # Act
    rows, _ = builder._rows()

    # Assert
    measured = [r for r in rows if r["status"] is not None]
    assert {r["model_slug"] for r in measured} == {
        "opus-5", "sonnet-5", "haiku-4-5-anthropic", "deepseek-v3.2", "kimi-k2.5", "glm-5",
    }
    assert len(measured) == 18


def test_coverage_is_judged_per_store(stores):
    # A case complete on Anthropic and half-run on Bedrock keeps its Anthropic cells testable:
    # judging coverage on the pooled count would discard six finished cells.
    anthropic, bedrock = stores
    _complete_case(anthropic, "S0001", ["claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5"])
    _write_cell(bedrock, "S0001", 1, "deepseek.v3.2")

    rows, partial = builder._rows()

    anthropic_rows = [r for r in rows if r["transport"] == "anthropic" and r["status"] is not None]
    bedrock_rows = [r for r in rows if r["transport"] == "bedrock" and r["status"] is not None]
    assert {r["test_status"] for r in anthropic_rows} == {"Tested"}
    assert {r["test_status"] for r in bedrock_rows} == {"Partially tested"}
    assert "S0001" in partial


def test_a_mismatched_version_stamp_stops_the_build(stores):
    # Pooling a cell measured under a different contract is the one thing that must never
    # happen silently: it would merge two incomparable campaigns into one table.
    anthropic, bedrock = stores
    _complete_case(anthropic, "S0001", ["claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5"])
    _write_cell(bedrock, "S0001", 1, "deepseek.v3.2", solver_version="some-other-solver-v9")

    with pytest.raises(ValueError, match="solver_version"):
        builder._rows()


def test_a_model_with_measured_cells_gets_no_placeholder_row(stores):
    anthropic, bedrock = stores
    _complete_case(anthropic, "S0001", ["claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5"])
    _complete_case(bedrock, "S0001", ["deepseek.v3.2", "moonshotai.kimi-k2.5", "zai.glm-5"])

    rows, _ = builder._rows()

    reserved = [r for r in rows if r["notes"] == "reserved for a future campaign"]
    assert not any(r["model_slug"] in {"deepseek-v3.2", "kimi-k2.5", "glm-5"} for r in reserved)


def test_bus_count_comes_from_selected_dataset_manifest(tmp_path) -> None:
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": "pegase89-reactive-deficit-v1",
                "environment": {"bus_count": "89"},
            }
        ),
        encoding="utf-8",
    )

    assert builder._dataset_bus_count(tmp_path) == 89
