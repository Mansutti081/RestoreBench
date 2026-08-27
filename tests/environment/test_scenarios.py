# ABOUTME: Verifies public-manifest scenario loading without private curation labels.
# ABOUTME: Covers artifact hashes and access-controlled memory splits.
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandapower as pp
import pytest

from restorebench.environment.scenarios import (
    scenario_ids_for_split,
    held_out_ids,
    load_card,
    load_full_net,
    load_scenario,
    )
from restorebench.corpus.augment import build_augmented_base


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _staged_dataset(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "staging"
    full_dir = root / "full"
    card_dir = root / "llm"
    full_dir.mkdir(parents=True)
    card_dir.mkdir()
    full_path = full_dir / "S0001.json"
    card_path = card_dir / "S0001.md"
    lean_path = root / "lean" / "S0001.json"
    lean_path.parent.mkdir()
    pp.to_json(build_augmented_base(), str(full_path))
    pp.to_json(build_augmented_base(), str(lean_path))
    card_path.write_text("public current-state card\n", encoding="utf-8")

    evaluation_path = root / "evaluation_manifest.json"
    evaluation_path.write_text(
        json.dumps(
            {
                "dataset_version": "reactive-deficit-v0",
                "scenarios": [
                    {
                        "scenario_id": "S0001",
                        "memory_split": "memory_population",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_version": "reactive-deficit-v0",
                "base_network_hash": "a" * 64,
                "scenario_count": 1,
                "scenarios": [
                    {
                        "scenario_id": "S0001",
                        "full_artifact_hash": _sha(full_path),
                        "lean_artifact_hash": _sha(lean_path),
                        "card_artifact_hash": _sha(card_path),
                    }
                ],
                "split_manifest_hash": _sha(evaluation_path),
                "policy_hashes": {"curation": "c" * 64},
                "environment": {"python": "3.11", "pandapower": pp.__version__},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return root, manifest_path, evaluation_path


def test_load_scenario_uses_public_and_evaluation_manifests_only(tmp_path: Path) -> None:
    root, manifest_path, evaluation_path = _staged_dataset(tmp_path)

    scenario = load_scenario(
        "S0001",
        data_dir=root,
        manifest_path=manifest_path,
        evaluation_manifest_path=evaluation_path,
    )

    assert scenario.scenario_id == "S0001"
    assert scenario.memory_split == "memory_population"
    assert scenario.dataset_version == "reactive-deficit-v0"
    assert "label" not in type(scenario).model_fields
    assert load_card(scenario) == "public current-state card\n"
    assert isinstance(load_full_net(scenario), pp.pandapowerNet)


def test_loader_rejects_unknown_ids_and_artifact_hash_drift(tmp_path: Path) -> None:
    root, manifest_path, evaluation_path = _staged_dataset(tmp_path)

    with pytest.raises(ValueError, match="not found"):
        load_scenario(
            "S9999",
            data_dir=root,
            manifest_path=manifest_path,
            evaluation_manifest_path=evaluation_path,
        )

    (root / "llm/S0001.md").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        load_scenario(
            "S0001",
            data_dir=root,
            manifest_path=manifest_path,
            evaluation_manifest_path=evaluation_path,
        )


def test_split_enumerators_read_evaluation_manifest_deterministically(
    tmp_path: Path,
) -> None:
    _root, _manifest_path, evaluation_path = _staged_dataset(tmp_path)

    assert scenario_ids_for_split(
        "memory_population", evaluation_manifest_path=evaluation_path
    ) == ["S0001"]
    assert held_out_ids(evaluation_manifest_path=evaluation_path) == []

