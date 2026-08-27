# ABOUTME: Provides atomic, identity-checked checkpoint shards for long dataset jobs.
# ABOUTME: Makes interrupted generation and validation resumable without trusting partial writes.
from __future__ import annotations

import json
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


CHECKPOINT_FORMAT_VERSION = "atomic-checkpoint-v1"
_JSON_KEY = re.compile(r"^[A-Za-z0-9_.-]+$")


class CheckpointCompatibilityError(ValueError):
    """Raised when saved work cannot safely resume under the requested identity."""


@dataclass(frozen=True)
class CheckpointStore:
    root: Path
    identity: dict[str, Any]

    @property
    def shard_dir(self) -> Path:
        return self.root / "shards"

    @property
    def json_dir(self) -> Path:
        return self.root / "json"

    @classmethod
    def open(
        cls,
        root: Path,
        *,
        identity: Mapping[str, Any],
        resume: bool,
    ) -> "CheckpointStore":
        resolved = root.resolve()
        normalized = json.loads(_canonical_json(dict(identity)))
        identity_path = resolved / "identity.json"
        if resolved.exists() and any(resolved.iterdir()):
            if not resume:
                raise FileExistsError(
                    f"checkpoint directory already contains work; pass resume=True: {resolved}"
                )
            if not identity_path.is_file():
                raise CheckpointCompatibilityError(
                    f"checkpoint identity is missing: {identity_path}"
                )
            stored = json.loads(identity_path.read_text(encoding="utf-8"))
            expected = {
                "format_version": CHECKPOINT_FORMAT_VERSION,
                "identity": normalized,
            }
            if stored != expected:
                raise CheckpointCompatibilityError(
                    "checkpoint identity does not match the requested run"
                )
        else:
            if resume:
                raise FileNotFoundError(
                    f"checkpoint directory has no resumable work: {resolved}"
                )
            resolved.mkdir(parents=True, exist_ok=True)
            _atomic_write_json(
                identity_path,
                {
                    "format_version": CHECKPOINT_FORMAT_VERSION,
                    "identity": normalized,
                },
            )

        store = cls(root=resolved, identity=normalized)
        store.shard_dir.mkdir(parents=True, exist_ok=True)
        store.json_dir.mkdir(parents=True, exist_ok=True)
        return store

    def write_pickle_shard(
        self,
        index: int,
        *,
        key: str,
        payload: Any,
    ) -> Path:
        if index < 0:
            raise ValueError("checkpoint shard index must be non-negative")
        path = self.shard_dir / f"{index:06d}.pkl"
        if path.exists():
            raise FileExistsError(f"checkpoint shard already exists: {path}")
        _atomic_write_pickle(
            path,
            {
                "format_version": CHECKPOINT_FORMAT_VERSION,
                "index": index,
                "key": key,
                "payload": payload,
            },
        )
        return path

    def load_pickle_shards(
        self,
        *,
        expected_keys: Sequence[str],
    ) -> tuple[Any, ...]:
        paths = sorted(self.shard_dir.glob("*.pkl"))
        observed_indices = tuple(
            int(path.stem)
            for path in paths
            if path.stem.isdigit()
        )
        if observed_indices != tuple(range(len(paths))):
            raise CheckpointCompatibilityError(
                "checkpoint pickle shards are not contiguous from index zero"
            )

        payloads: list[Any] = []
        for expected_index, path in enumerate(paths):
            if expected_index >= len(expected_keys):
                raise CheckpointCompatibilityError(
                    "checkpoint contains more family shards than the current schedule"
                )
            with path.open("rb") as handle:
                record = pickle.load(handle)  # noqa: S301 - local identity-checked checkpoint
            expected = (
                CHECKPOINT_FORMAT_VERSION,
                expected_index,
                expected_keys[expected_index],
            )
            observed = (
                record.get("format_version"),
                record.get("index"),
                record.get("key"),
            )
            if observed != expected:
                raise CheckpointCompatibilityError(
                    f"checkpoint shard {path.name} does not match the current family schedule"
                )
            payloads.append(record["payload"])
        return tuple(payloads)

    def write_json_shard(
        self,
        key: str,
        payload: Mapping[str, Any],
    ) -> Path:
        path = self._json_path(key)
        _atomic_write_json(path, dict(payload))
        return path

    def read_json_shard(self, key: str) -> dict[str, Any] | None:
        path = self._json_path(key)
        if not path.is_file():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise CheckpointCompatibilityError(
                f"checkpoint JSON shard must contain an object: {path}"
            )
        return value

    def _json_path(self, key: str) -> Path:
        if not _JSON_KEY.fullmatch(key):
            raise ValueError(f"unsafe checkpoint JSON shard key: {key!r}")
        return self.json_dir / f"{key}.json"


def atomic_write_json(path: Path, payload: Any) -> None:
    """Expose the same durable JSON primitive to pipeline status writers."""
    _atomic_write_json(path, payload)


def _atomic_write_json(path: Path, payload: Any) -> None:
    encoded = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        )
        + "\n"
    ).encode()
    _atomic_write_bytes(path, encoded)


def _atomic_write_pickle(path: Path, payload: Any) -> None:
    _atomic_write_bytes(
        path,
        pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL),
    )


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
