# ABOUTME: Verifies the durable command plan and stage detection for long-running corpus runs.
# ABOUTME: Ensures every restarted child receives checkpoint resume flags and conservative workers.
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from restorebench.corpus.resumable_run import (
    ResumableRunConfig,
    _generation_complete,
    _validation_complete,
    build_generation_command,
    build_validation_command,
)
from restorebench.corpus.versions import (
    DATASET_VERSION,
    GENERATOR_VERSION,
    VALIDATOR_VERSION,
)


def test_resumable_generation_command_is_checkpointed_and_resumable(
    tmp_path: Path,
) -> None:
    config = ResumableRunConfig(
        run_dir=tmp_path,
        target_count=200,
        workers=3,
        minimum_family_evaluations_per_scenario=1,
        maximum_family_evaluations=4000,
    )

    initial = build_generation_command(
        config,
        workers=3,
        resume=False,
    )
    resumed = build_generation_command(
        config,
        workers=2,
        resume=True,
    )

    assert initial == [
        sys.executable,
        "-m",
        "restorebench.corpus.generate_scenarios",
        "--n",
        "200",
        "--output-dir",
        str(tmp_path / "dataset"),
        "--workers",
        "3",
        "--checkpoint-dir",
        str(tmp_path / "checkpoints/generation"),
        "--minimum-family-evaluations-per-scenario",
        "1",
        "--maximum-family-evaluations",
        "4000",
    ]
    assert resumed[-1] == "--resume"
    assert resumed[resumed.index("--workers") + 1] == "2"


def test_resumable_validation_command_resumes_per_scenario(
    tmp_path: Path,
) -> None:
    config = ResumableRunConfig(run_dir=tmp_path)

    initial = build_validation_command(config, resume=False)
    resumed = build_validation_command(config, resume=True)

    assert initial == [
        sys.executable,
        "-m",
        "restorebench.corpus.validate_dataset",
        "--dataset-dir",
        str(tmp_path / "dataset"),
        "--checkpoint-dir",
        str(tmp_path / "checkpoints/validation"),
    ]
    assert resumed[-1] == "--resume"


def test_generation_completion_requires_current_matching_identities(
    tmp_path: Path,
) -> None:
    config = ResumableRunConfig(run_dir=tmp_path, target_count=1)
    _write_json(
        config.dataset_dir / "manifest.json",
        {
            "dataset_version": DATASET_VERSION,
            "scenario_count": 1,
        },
    )
    _write_json(
        config.dataset_dir / ".generation_identity.json",
        {
            "format_version": "resumable-staging-v1",
            "identity": {
                "dataset_version": DATASET_VERSION,
                "generator_version": GENERATOR_VERSION,
                "policy_hash": "policy",
                "target_count": 1,
            },
        },
    )
    _write_json(
        config.generation_checkpoint_dir / "identity.json",
        {
            "format_version": "atomic-checkpoint-v1",
            "identity": {
                "dataset_version": DATASET_VERSION,
                "generator_version": GENERATOR_VERSION,
                "policy_hash": "policy",
                "target_count": 1,
            },
        },
    )

    assert _generation_complete(config)

    staging_path = config.dataset_dir / ".generation_identity.json"
    staging = json.loads(staging_path.read_text(encoding="utf-8"))
    staging["identity"]["generator_version"] = "stale-generator"
    _write_json(staging_path, staging)

    assert not _generation_complete(config)


def test_validation_completion_rechecks_versions_and_public_hashes(
    tmp_path: Path,
) -> None:
    config = ResumableRunConfig(run_dir=tmp_path, target_count=1)
    full = config.dataset_dir / "full/S0001.json"
    lean = config.dataset_dir / "lean/S0001.json"
    card = config.dataset_dir / "llm/S0001.md"
    _write_text(full, '{"kind":"full"}\n')
    _write_text(lean, '{"kind":"lean"}\n')
    _write_text(card, "# card\n")
    manifest = {
        "dataset_version": DATASET_VERSION,
        "scenario_count": 1,
        "scenarios": [
            {
                "scenario_id": "S0001",
                "full_artifact_hash": _sha256(full),
                "lean_artifact_hash": _sha256(lean),
                "card_artifact_hash": _sha256(card),
            }
        ],
    }
    top_level = {
        "manifest.json": manifest,
        "evaluation_manifest.json": {"scenarios": [{"scenario_id": "S0001"}]},
        "generation_report.json": {"requested_count": 1},
        "private/labels.json": [{"scenario_id": "S0001"}],
        "private/witnesses.json": [{"scenario_id": "S0001"}],
    }
    for relative, payload in top_level.items():
        _write_json(config.dataset_dir / relative, payload)
    _write_json(
        config.dataset_dir / "validation_report.json",
        {
            "valid": True,
            "validator_version": VALIDATOR_VERSION,
            "total": 1,
            "valid_count": 1,
            "invalid_count": 0,
        },
    )
    _write_json(
        config.validation_checkpoint_dir / "identity.json",
        {
            "format_version": "atomic-checkpoint-v1",
            "identity": {
                "validator_version": VALIDATOR_VERSION,
                "artifact_hashes": {
                    relative: _sha256(config.dataset_dir / relative)
                    for relative in top_level
                },
            },
        },
    )
    _write_json(
        config.validation_checkpoint_dir / "json/S0001.json",
        {"scenario_id": "S0001", "valid": True},
    )

    assert _validation_complete(config)

    _write_text(card, "# modified card\n")
    assert not _validation_complete(config)

    _write_text(card, "# card\n")
    report_path = config.dataset_dir / "validation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["validator_version"] = "stale-validator"
    _write_json(report_path, report)
    assert not _validation_complete(config)


def _write_json(path: Path, payload: object) -> None:
    _write_text(path, json.dumps(payload, sort_keys=True) + "\n")


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
