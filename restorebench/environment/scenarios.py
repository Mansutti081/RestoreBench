# ABOUTME: Loads public scenario artifacts and access-controlled split assignments.
# ABOUTME: Never parses or attaches private curation labels to runtime Scenario objects.
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandapower as pp

from restorebench.schemas.dataset import (
    DatasetManifest,
    EvaluationManifest,
    MemorySplit,
    Scenario,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT / "dataset/ieee118"
DEFAULT_MANIFEST_PATH = DEFAULT_DATA_DIR / "manifest.json"
DEFAULT_EVALUATION_MANIFEST_PATH = (
    DEFAULT_DATA_DIR / "evaluation_manifest.json"
)


def load_scenario(
    scenario_id: str,
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
    manifest_path: Path | None = None,
    evaluation_manifest_path: Path | None = None,
) -> Scenario:
    """Load one runtime scenario without making private labels reachable."""
    public_manifest_path = manifest_path or data_dir / "manifest.json"
    if not public_manifest_path.exists():
        raise ValueError(f"no dataset manifest at {public_manifest_path}")
    evaluation_path = (
        evaluation_manifest_path or data_dir / "evaluation_manifest.json"
    )
    manifest = DatasetManifest.model_validate_json(
        public_manifest_path.read_text(encoding="utf-8")
    )
    evaluation = _load_evaluation_manifest(evaluation_path)
    if evaluation.dataset_version != manifest.dataset_version:
        raise ValueError(
            "public and evaluation manifest dataset versions differ"
        )
    if _sha256(evaluation_path) != manifest.split_manifest_hash:
        raise ValueError("evaluation manifest hash differs from public manifest")
    entry = next(
        (
            item
            for item in manifest.scenarios
            if item.scenario_id == scenario_id
        ),
        None,
    )
    split_entry = next(
        (
            item
            for item in evaluation.scenarios
            if item.scenario_id == scenario_id
        ),
        None,
    )
    if entry is None or split_entry is None:
        raise ValueError(
            f"scenario_id {scenario_id} not found in dataset manifests"
        )
    full_path = data_dir / "full" / f"{scenario_id}.json"
    lean_path = data_dir / "lean" / f"{scenario_id}.json"
    card_path = data_dir / "llm" / f"{scenario_id}.md"
    for path, expected_hash in (
        (full_path, entry.full_artifact_hash),
        (lean_path, entry.lean_artifact_hash),
        (card_path, entry.card_artifact_hash),
    ):
        if not path.is_file() or _sha256(path) != expected_hash:
            raise ValueError(
                f"artifact hash mismatch for {scenario_id}: {path}"
            )
    return Scenario(
        scenario_id=scenario_id,
        dataset_version=manifest.dataset_version,
        full_net_path=str(full_path.resolve()),
        card_path=str(card_path.resolve()),
        memory_split=split_entry.memory_split,
    )


def load_card(scenario: Scenario) -> str:
    return Path(scenario.card_path).read_text(encoding="utf-8")


def load_full_net(scenario: Scenario) -> Any:
    return pp.from_json(str(Path(scenario.full_net_path)))


def scenario_ids_for_split(
    split: MemorySplit,
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
    evaluation_manifest_path: Path | None = None,
) -> list[str]:
    path = evaluation_manifest_path or data_dir / "evaluation_manifest.json"
    return sorted(
        record.scenario_id
        for record in _load_evaluation_manifest(path).scenarios
        if record.memory_split == split
    )


def held_out_ids(
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
    evaluation_manifest_path: Path | None = None,
) -> list[str]:
    return scenario_ids_for_split(
        "held_out",
        data_dir=data_dir,
        evaluation_manifest_path=evaluation_manifest_path,
    )


def _load_evaluation_manifest(path: Path) -> EvaluationManifest:
    return EvaluationManifest.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
