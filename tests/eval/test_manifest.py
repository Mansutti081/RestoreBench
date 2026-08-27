# ABOUTME: Verifies results_manifest.json creation and read-modify-write updates.
# ABOUTME: Keeps model slugs, calibration hashes, memory hashes, and run counts stable.
from datetime import datetime, timezone

import pytest

from restorebench.eval import manifest
from restorebench.llm import models


NOW = datetime(2026, 7, 12, 12, 0, tzinfo=timezone.utc)


def test_manifest_refuses_to_mix_datasets(tmp_path) -> None:
    ieee = manifest.DatasetManifestEntry(
        dataset_version="ieee118-reactive-deficit-v1",
        base_network_hash="a" * 64,
        split_manifest_hash="b" * 64,
    )
    pegase = manifest.DatasetManifestEntry(
        dataset_version="pegase89-reactive-deficit-v1",
        base_network_hash="c" * 64,
        split_manifest_hash="d" * 64,
    )
    manifest.update_manifest(tmp_path, model_ids=[models.GLM_5], dataset=ieee, now=NOW)

    with pytest.raises(ValueError, match="different dataset"):
        manifest.update_manifest(tmp_path, model_ids=[models.GLM_5], dataset=pegase, now=NOW)
