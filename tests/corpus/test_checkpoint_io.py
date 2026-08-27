# ABOUTME: Verifies durable atomic checkpoint storage for long-running corpus jobs.
# ABOUTME: Prevents incompatible resumes, partial-shard reuse, and silent checkpoint gaps.
from __future__ import annotations

from pathlib import Path

import pytest

from restorebench.corpus.checkpoint_io import (
    CheckpointCompatibilityError,
    CheckpointStore,
)


def test_checkpoint_store_resumes_contiguous_atomic_pickle_shards(
    tmp_path: Path,
) -> None:
    identity = {
        "kind": "generation",
        "target_count": 200,
        "policy_hash": "abc",
    }
    store = CheckpointStore.open(
        tmp_path / "checkpoint",
        identity=identity,
        resume=False,
    )
    store.write_pickle_shard(0, key="profile-a:pocket-a", payload={"value": 1})
    store.write_pickle_shard(1, key="profile-b:pocket-b", payload={"value": 2})

    resumed = CheckpointStore.open(
        tmp_path / "checkpoint",
        identity=identity,
        resume=True,
    )

    assert resumed.load_pickle_shards(
        expected_keys=("profile-a:pocket-a", "profile-b:pocket-b")
    ) == ({"value": 1}, {"value": 2})


def test_checkpoint_store_rejects_incompatible_or_implicit_reuse(
    tmp_path: Path,
) -> None:
    root = tmp_path / "checkpoint"
    CheckpointStore.open(
        root,
        identity={"kind": "generation", "policy_hash": "abc"},
        resume=False,
    )

    with pytest.raises(FileExistsError, match="resume"):
        CheckpointStore.open(
            root,
            identity={"kind": "generation", "policy_hash": "abc"},
            resume=False,
        )
    with pytest.raises(
        CheckpointCompatibilityError,
        match="identity",
    ):
        CheckpointStore.open(
            root,
            identity={"kind": "generation", "policy_hash": "different"},
            resume=True,
        )


def test_checkpoint_store_rejects_gaps_and_ignores_uncommitted_temporary_files(
    tmp_path: Path,
) -> None:
    store = CheckpointStore.open(
        tmp_path / "checkpoint",
        identity={"kind": "generation", "policy_hash": "abc"},
        resume=False,
    )
    store.write_pickle_shard(0, key="family-0", payload={"value": 0})
    (store.shard_dir / "000001.pkl.tmp").write_bytes(b"partial")
    store.write_pickle_shard(2, key="family-2", payload={"value": 2})

    with pytest.raises(CheckpointCompatibilityError, match="contiguous"):
        store.load_pickle_shards(
            expected_keys=("family-0", "family-1", "family-2"),
        )


def test_checkpoint_store_round_trips_typed_json_shards(
    tmp_path: Path,
) -> None:
    identity = {"kind": "validation", "manifest_hash": "abc"}
    store = CheckpointStore.open(
        tmp_path / "checkpoint",
        identity=identity,
        resume=False,
    )
    store.write_json_shard("S0001", {"valid": True})

    resumed = CheckpointStore.open(
        tmp_path / "checkpoint",
        identity=identity,
        resume=True,
    )

    assert resumed.read_json_shard("S0001") == {"valid": True}
    assert resumed.read_json_shard("S0002") is None
