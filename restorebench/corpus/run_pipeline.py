# ABOUTME: Runs staged reactive-deficit generation and independent validation fail-fast.
# ABOUTME: Requires an explicit new output directory and never cleans the frozen corpus.
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[2]


def build_stages(
    target_n: int,
    output_dir: Path,
    *,
    workers: int = 1,
    network: str = "case118",
    memory_population_count: int | None = None,
    held_out_count: int | None = None,
    checkpoint_root: Path | None = None,
    resume: bool = False,
) -> list[list[str]]:
    generation = [
        sys.executable,
        "-m",
        "restorebench.corpus.generate_scenarios",
        "--n",
        str(target_n),
        "--output-dir",
        str(output_dir),
    ]
    if workers != 1:
        generation.extend(["--workers", str(workers)])
    if network != "case118":
        generation.extend(["--network", network])
    if memory_population_count is not None and held_out_count is not None:
        generation.extend(
            [
                "--memory-population-count",
                str(memory_population_count),
                "--held-out-count",
                str(held_out_count),
            ]
        )
    validation = [
        sys.executable,
        "-m",
        "restorebench.corpus.validate_dataset",
        "--dataset-dir",
        str(output_dir),
    ]
    if checkpoint_root is not None:
        generation.extend(["--checkpoint-dir", str(checkpoint_root / "generation")])
        validation.extend(["--checkpoint-dir", str(checkpoint_root / "validation")])
        if resume:
            generation.append("--resume")
            validation.append("--resume")
    return [generation, validation]


def run_stage(
    index: int,
    total: int,
    command: Sequence[str],
) -> float:
    print(
        f"STAGE {index}/{total}: {' '.join(command)}",
        flush=True,
    )
    started = time.perf_counter()
    completed = subprocess.run(
        list(command),
        cwd=ROOT,
        check=False,
    )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    print(f"STAGE {index}/{total} completed in {elapsed:.1f}s")
    return elapsed


def check_final_summary(
    target_n: int,
    output_dir: Path,
) -> tuple[bool, dict[str, int]]:
    manifest = _read_json(output_dir / "manifest.json")
    validation = _read_json(output_dir / "validation_report.json")
    labels = _read_json(output_dir / "private/labels.json")
    witnesses = _read_json(output_dir / "private/witnesses.json")
    counts = {
        "manifest": int(manifest.get("scenario_count", 0)),
        "full": len(list((output_dir / "full").glob("S*.json"))),
        "lean": len(list((output_dir / "lean").glob("S*.json"))),
        "cards": len(list((output_dir / "llm").glob("S*.md"))),
        "labels": len(labels) if isinstance(labels, list) else 0,
        "witnesses": len(witnesses) if isinstance(witnesses, list) else 0,
        "valid": int(validation.get("valid_count", 0)),
        "invalid": int(validation.get("invalid_count", -1)),
    }
    ok = (
        all(
            counts[name] == target_n
            for name in (
                "manifest",
                "full",
                "lean",
                "cards",
                "labels",
                "witnesses",
                "valid",
            )
        )
        and counts["invalid"] == 0
        and bool(validation.get("valid", False))
    )
    return ok, counts


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--network", choices=("case118", "case89pegase"), default="case118")
    parser.add_argument("--memory-population-count", type=int)
    parser.add_argument("--held-out-count", type=int)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Explicit new staging directory.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Independent witness-evaluation processes for the generation stage.",
    )
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.n <= 0:
        parser.error("--n must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.resume and args.checkpoint_root is None:
        parser.error("--resume requires --checkpoint-root")
    explicit_counts = (args.memory_population_count, args.held_out_count)
    if (explicit_counts[0] is None) != (explicit_counts[1] is None):
        parser.error("split counts must be specified together")
    if explicit_counts[0] is not None:
        if explicit_counts[0] < 0 or explicit_counts[1] < 0:
            parser.error("split counts must be non-negative")
        if sum(explicit_counts) != args.n:
            parser.error("split counts must sum to --n")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.output_dir.resolve() == (ROOT / "dataset/ieee118").resolve():
        raise SystemExit("refusing to use the frozen dataset/ieee118 directory")
    stages = build_stages(
        args.n,
        args.output_dir,
        workers=args.workers,
        network=args.network,
        memory_population_count=args.memory_population_count,
        held_out_count=args.held_out_count,
        checkpoint_root=args.checkpoint_root,
        resume=args.resume,
    )
    durations = [
        run_stage(index, len(stages), command)
        for index, command in enumerate(stages, start=1)
    ]
    ok, counts = check_final_summary(args.n, args.output_dir)
    print(json.dumps(counts, indent=2, sort_keys=True))
    print(
        "Stage durations: "
        + ", ".join(f"{value:.1f}s" for value in durations)
    )
    raise SystemExit(0 if ok else 1)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main(sys.argv[1:])
