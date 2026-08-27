# ABOUTME: Maintains results/results_manifest.json for eval sweeps.
# ABOUTME: Records model slugs, the dataset identity, run counts, and UTC update times.
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from restorebench.eval.store import atomic_write_text
from restorebench.llm.models import BENCHMARK_MODELS, model_slug


@dataclass(frozen=True)
class DatasetManifestEntry:
    dataset_version: str
    base_network_hash: str
    split_manifest_hash: str


def update_manifest(
    results_dir: str | Path,
    *,
    model_ids: Sequence[str] = BENCHMARK_MODELS,
    dataset: DatasetManifestEntry | None = None,
    run_counts: Mapping[str, int] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    path = manifest_path(results_dir)
    current = load_manifest(results_dir) if path.exists() else {}
    timestamp = _format_utc(now or _utc_now())

    current.setdefault("created_at", timestamp)
    current["updated_at"] = timestamp
    current["models"] = [{"model_id": model_id, "slug": model_slug(model_id)} for model_id in model_ids]
    if dataset is not None:
        stored_dataset = current.get("dataset")
        requested_dataset = asdict(dataset)
        if stored_dataset is not None and stored_dataset != requested_dataset:
            raise ValueError("results directory belongs to a different dataset")
        current["dataset"] = requested_dataset
    if run_counts:
        counts = dict(current.get("run_counts", {}))
        counts.update(dict(run_counts))
        current["run_counts"] = counts

    atomic_write_text(path, json.dumps(current, indent=2, sort_keys=True) + "\n")
    return current


def load_manifest(results_dir: str | Path) -> dict[str, Any]:
    return json.loads(manifest_path(results_dir).read_text(encoding="utf-8"))


def manifest_path(results_dir: str | Path) -> Path:
    return Path(results_dir) / "results_manifest.json"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _format_utc(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("manifest timestamp must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat()
