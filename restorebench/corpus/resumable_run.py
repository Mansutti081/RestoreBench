# ABOUTME: Supervises crash-resumable corpus generation and validation with durable logs.
# ABOUTME: Restarts failed children from atomic checkpoints and lowers CPU concurrency after crashes.
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from restorebench.corpus.checkpoint_io import (
    CHECKPOINT_FORMAT_VERSION,
    atomic_write_json,
)
from restorebench.corpus.run_pipeline import check_final_summary
from restorebench.corpus.versions import (
    DATASET_VERSION,
    GENERATOR_VERSION,
    VALIDATOR_VERSION,
)


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ResumableRunConfig:
    run_dir: Path
    target_count: int = 200
    workers: int = 3
    minimum_family_evaluations_per_scenario: int = 1
    maximum_family_evaluations: int = 4000
    max_restarts: int = 20
    heartbeat_seconds: int = 60

    @property
    def dataset_dir(self) -> Path:
        return self.run_dir / "dataset"

    @property
    def generation_checkpoint_dir(self) -> Path:
        return self.run_dir / "checkpoints/generation"

    @property
    def validation_checkpoint_dir(self) -> Path:
        return self.run_dir / "checkpoints/validation"

    @property
    def log_dir(self) -> Path:
        return self.run_dir / "logs"

    @property
    def status_path(self) -> Path:
        return self.run_dir / "status.json"


def build_generation_command(
    config: ResumableRunConfig,
    *,
    workers: int,
    resume: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "restorebench.corpus.generate_scenarios",
        "--n",
        str(config.target_count),
        "--output-dir",
        str(config.dataset_dir),
        "--workers",
        str(workers),
        "--checkpoint-dir",
        str(config.generation_checkpoint_dir),
        "--minimum-family-evaluations-per-scenario",
        str(config.minimum_family_evaluations_per_scenario),
        "--maximum-family-evaluations",
        str(config.maximum_family_evaluations),
    ]
    if resume:
        command.append("--resume")
    return command


def build_validation_command(
    config: ResumableRunConfig,
    *,
    resume: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "restorebench.corpus.validate_dataset",
        "--dataset-dir",
        str(config.dataset_dir),
        "--checkpoint-dir",
        str(config.validation_checkpoint_dir),
    ]
    if resume:
        command.append("--resume")
    return command


def run_resumable(config: ResumableRunConfig) -> None:
    _validate_config(config)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    config.log_dir.mkdir(parents=True, exist_ok=True)
    (config.run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    _record_status(config, stage="starting", attempt=0)

    if not _generation_complete(config):
        _run_supervised_stage(
            config,
            stage="generation",
            log_path=config.log_dir / "generation.log",
            command_builder=lambda attempt: build_generation_command(
                config,
                workers=max(1, config.workers - attempt),
                resume=config.generation_checkpoint_dir.joinpath(
                    "identity.json"
                ).is_file(),
            ),
        )
    else:
        _append_supervisor_log(
            config,
            "generation already complete; skipping to validation",
        )

    if not _validation_complete(config):
        _run_supervised_stage(
            config,
            stage="validation",
            log_path=config.log_dir / "validation.log",
            command_builder=lambda _attempt: build_validation_command(
                config,
                resume=config.validation_checkpoint_dir.joinpath(
                    "identity.json"
                ).is_file(),
            ),
        )
    else:
        _append_supervisor_log(
            config,
            "validation already complete; running final summary only",
        )

    ok, counts = check_final_summary(
        config.target_count,
        config.dataset_dir,
    )
    ok = ok and _validation_complete(config)
    _record_status(
        config,
        stage="complete" if ok else "final_check_failed",
        attempt=0,
        extra={"counts": counts},
    )
    _append_supervisor_log(
        config,
        f"final summary ok={ok} counts={json.dumps(counts, sort_keys=True)}",
    )
    if not ok:
        raise SystemExit(1)


def _run_supervised_stage(
    config: ResumableRunConfig,
    *,
    stage: str,
    log_path: Path,
    command_builder: Callable[[int], list[str]],
) -> None:
    for attempt in range(config.max_restarts + 1):
        command = command_builder(attempt)
        _append_supervisor_log(
            config,
            f"{stage} attempt={attempt + 1} command={' '.join(command)}",
        )
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHONUNBUFFERED": "1",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "MPLCONFIGDIR": str(config.run_dir / "matplotlib"),
            }
        )
        (config.run_dir / "matplotlib").mkdir(
            parents=True,
            exist_ok=True,
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8", buffering=1) as log:
            log.write(
                f"\n[{_timestamp()}] START attempt={attempt + 1} "
                f"command={' '.join(command)}\n"
            )
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            _record_status(
                config,
                stage=stage,
                attempt=attempt + 1,
                child_pid=process.pid,
                extra={"command": command},
            )
            while process.poll() is None:
                time.sleep(config.heartbeat_seconds)
                health = _health_snapshot(config)
                _append_supervisor_log(
                    config,
                    f"{stage} heartbeat child_pid={process.pid} "
                    f"health={json.dumps(health, sort_keys=True)}",
                )
                _record_status(
                    config,
                    stage=stage,
                    attempt=attempt + 1,
                    child_pid=process.pid,
                    extra={"health": health},
                )
            return_code = int(process.returncode or 0)
            log.write(
                f"[{_timestamp()}] EXIT attempt={attempt + 1} "
                f"return_code={return_code}\n"
            )
            log.flush()
            os.fsync(log.fileno())

        if return_code == 0:
            _append_supervisor_log(
                config,
                f"{stage} completed on attempt={attempt + 1}",
            )
            return
        _append_supervisor_log(
            config,
            f"{stage} failed return_code={return_code}; "
            "the next attempt will resume committed checkpoints",
        )
        if attempt == config.max_restarts:
            _record_status(
                config,
                stage=f"{stage}_failed",
                attempt=attempt + 1,
                extra={"return_code": return_code},
            )
            raise SystemExit(return_code or 1)
        time.sleep(min(300, 30 * (attempt + 1)))


def _generation_complete(config: ResumableRunConfig) -> bool:
    manifest = _read_json_object(config.dataset_dir / "manifest.json")
    staging = _read_json_object(
        config.dataset_dir / ".generation_identity.json"
    )
    checkpoint = _read_json_object(
        config.generation_checkpoint_dir / "identity.json"
    )
    if manifest is None or staging is None or checkpoint is None:
        return False
    staging_identity = staging.get("identity")
    checkpoint_identity = checkpoint.get("identity")
    if not isinstance(staging_identity, dict) or not isinstance(
        checkpoint_identity,
        dict,
    ):
        return False
    return (
        int(manifest.get("scenario_count", 0)) == config.target_count
        and manifest.get("dataset_version") == DATASET_VERSION
        and staging.get("format_version") == "resumable-staging-v1"
        and staging_identity.get("target_count") == config.target_count
        and staging_identity.get("dataset_version") == DATASET_VERSION
        and staging_identity.get("generator_version") == GENERATOR_VERSION
        and checkpoint.get("format_version") == CHECKPOINT_FORMAT_VERSION
        and checkpoint_identity.get("target_count") == config.target_count
        and checkpoint_identity.get("dataset_version") == DATASET_VERSION
        and checkpoint_identity.get("generator_version")
        == GENERATOR_VERSION
        and checkpoint_identity.get("policy_hash")
        == staging_identity.get("policy_hash")
    )


def _validation_complete(config: ResumableRunConfig) -> bool:
    report = _read_json_object(
        config.dataset_dir / "validation_report.json"
    )
    checkpoint = _read_json_object(
        config.validation_checkpoint_dir / "identity.json"
    )
    manifest = _read_json_object(config.dataset_dir / "manifest.json")
    if report is None or checkpoint is None or manifest is None:
        return False
    checkpoint_identity = checkpoint.get("identity")
    if not isinstance(checkpoint_identity, dict):
        return False
    artifact_hashes = checkpoint_identity.get("artifact_hashes")
    return (
        bool(report.get("valid", False))
        and report.get("validator_version") == VALIDATOR_VERSION
        and int(report.get("total", 0)) == config.target_count
        and int(report.get("valid_count", 0)) == config.target_count
        and int(report.get("invalid_count", -1)) == 0
        and checkpoint.get("format_version") == CHECKPOINT_FORMAT_VERSION
        and checkpoint_identity.get("validator_version")
        == VALIDATOR_VERSION
        and _count_files(
            config.validation_checkpoint_dir / "json",
            "S*.json",
        )
        == config.target_count
        and _stored_artifact_hashes_match(
            config.dataset_dir,
            artifact_hashes,
        )
        and _public_artifact_hashes_match(
            config.dataset_dir,
            manifest,
            target_count=config.target_count,
        )
    )


def _stored_artifact_hashes_match(
    dataset_dir: Path,
    artifact_hashes: Any,
) -> bool:
    if not isinstance(artifact_hashes, dict) or not artifact_hashes:
        return False
    for relative, expected in artifact_hashes.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            return False
        path = dataset_dir / relative
        if not path.is_file() or _sha256_file(path) != expected:
            return False
    return True


def _public_artifact_hashes_match(
    dataset_dir: Path,
    manifest: dict[str, Any],
    *,
    target_count: int,
) -> bool:
    scenarios = manifest.get("scenarios")
    if not isinstance(scenarios, list) or len(scenarios) != target_count:
        return False
    artifacts = (
        ("full", "full_artifact_hash", ".json"),
        ("lean", "lean_artifact_hash", ".json"),
        ("llm", "card_artifact_hash", ".md"),
    )
    for entry in scenarios:
        if not isinstance(entry, dict):
            return False
        scenario_id = entry.get("scenario_id")
        if (
            not isinstance(scenario_id, str)
            or len(scenario_id) != 5
            or not scenario_id.startswith("S")
            or not scenario_id[1:].isdigit()
        ):
            return False
        for directory, hash_field, suffix in artifacts:
            expected = entry.get(hash_field)
            path = dataset_dir / directory / f"{scenario_id}{suffix}"
            if (
                not isinstance(expected, str)
                or not path.is_file()
                or _sha256_file(path) != expected
            ):
                return False
    return True


def _read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record_status(
    config: ResumableRunConfig,
    *,
    stage: str,
    attempt: int,
    child_pid: int | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "updated_at": _timestamp(),
        "stage": stage,
        "attempt": attempt,
        "supervisor_pid": os.getpid(),
        "child_pid": child_pid,
        "target_count": config.target_count,
        "run_dir": str(config.run_dir.resolve()),
        "generation_checkpoint_count": _count_files(
            config.generation_checkpoint_dir / "shards",
            "*.pkl",
        ),
        "validation_checkpoint_count": _count_files(
            config.validation_checkpoint_dir / "json",
            "S*.json",
        ),
    }
    if extra:
        payload.update(extra)
    atomic_write_json(config.status_path, payload)


def _append_supervisor_log(
    config: ResumableRunConfig,
    message: str,
) -> None:
    path = config.log_dir / "supervisor.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{_timestamp()}] {message}\n")
        handle.flush()
        os.fsync(handle.fileno())


def _health_snapshot(config: ResumableRunConfig) -> dict[str, Any]:
    memory = _memory_available_kib()
    disk = shutil.disk_usage(config.run_dir)
    load = os.getloadavg()
    return {
        "load_1m": round(load[0], 2),
        "load_5m": round(load[1], 2),
        "load_15m": round(load[2], 2),
        "memory_available_mib": (
            round(memory / 1024, 1) if memory is not None else None
        ),
        "disk_free_gib": round(disk.free / (1024**3), 2),
        "generation_checkpoint_count": _count_files(
            config.generation_checkpoint_dir / "shards",
            "*.pkl",
        ),
        "validation_checkpoint_count": _count_files(
            config.validation_checkpoint_dir / "json",
            "S*.json",
        ),
    }


def _memory_available_kib() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text(
            encoding="utf-8"
        ).splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


def _count_files(directory: Path, pattern: str) -> int:
    return len(tuple(directory.glob(pattern))) if directory.is_dir() else 0


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_config(config: ResumableRunConfig) -> None:
    if config.target_count <= 0:
        raise ValueError("target count must be positive")
    if config.workers <= 0:
        raise ValueError("workers must be positive")
    if config.max_restarts < 0:
        raise ValueError("max restarts must be non-negative")
    if config.heartbeat_seconds <= 0:
        raise ValueError("heartbeat seconds must be positive")
    frozen = (ROOT / "dataset/ieee118").resolve()
    resolved = config.run_dir.resolve()
    if resolved == frozen or frozen in resolved.parents:
        raise ValueError("resumable run directory cannot be inside the frozen corpus")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--minimum-family-evaluations-per-scenario",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--maximum-family-evaluations",
        type=int,
        default=4000,
    )
    parser.add_argument("--max-restarts", type=int, default=20)
    parser.add_argument("--heartbeat-seconds", type=int, default=60)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_resumable(
        ResumableRunConfig(
            run_dir=args.run_dir,
            target_count=args.n,
            workers=args.workers,
            minimum_family_evaluations_per_scenario=(
                args.minimum_family_evaluations_per_scenario
            ),
            maximum_family_evaluations=(
                args.maximum_family_evaluations
            ),
            max_restarts=args.max_restarts,
            heartbeat_seconds=args.heartbeat_seconds,
        )
    )


if __name__ == "__main__":
    main(sys.argv[1:])
