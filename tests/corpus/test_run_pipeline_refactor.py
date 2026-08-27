# ABOUTME: Verifies the staged pipeline passes one explicit directory to both phases.
# ABOUTME: Prevents cleanup flags and implicit frozen-corpus paths from returning.
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from restorebench.corpus import run_pipeline


def test_pipeline_builds_generation_then_independent_validation(
    tmp_path: Path,
) -> None:
    stages = run_pipeline.build_stages(10, tmp_path)

    assert stages == [
        [
            sys.executable,
            "-m",
            "restorebench.corpus.generate_scenarios",
            "--n",
            "10",
            "--output-dir",
            str(tmp_path),
        ],
        [
            sys.executable,
            "-m",
            "restorebench.corpus.validate_dataset",
            "--dataset-dir",
            str(tmp_path),
        ],
    ]


def test_pipeline_requires_explicit_output_and_has_no_cleanup_flag() -> None:
    with pytest.raises(SystemExit):
        run_pipeline.parse_args(["--n", "10"])
    with pytest.raises(SystemExit):
        run_pipeline.parse_args(
            [
                "--n",
                "10",
                "--output-dir",
                "/tmp/stage",
                "--clean",
            ]
        )


def test_pipeline_forwards_the_explicit_worker_count_to_generation(
    tmp_path: Path,
) -> None:
    stages = run_pipeline.build_stages(10, tmp_path, workers=8)

    assert stages[0][-2:] == ["--workers", "8"]
    assert "--workers" not in stages[1]

    args = run_pipeline.parse_args(
        [
            "--n",
            "10",
            "--output-dir",
            str(tmp_path),
            "--workers",
            "8",
        ]
    )
    assert args.workers == 8


def test_pipeline_forwards_network_and_explicit_split(tmp_path: Path) -> None:
    stages = run_pipeline.build_stages(
        46,
        tmp_path,
        network="case89pegase",
        memory_population_count=0,
        held_out_count=46,
    )

    assert stages[0][-6:] == [
        "--network",
        "case89pegase",
        "--memory-population-count",
        "0",
        "--held-out-count",
        "46",
    ]


def test_pipeline_uses_separate_resumable_checkpoints(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    stages = run_pipeline.build_stages(
        46,
        tmp_path / "dataset",
        checkpoint_root=checkpoint_root,
        resume=True,
    )

    assert stages[0][-3:] == ["--checkpoint-dir", str(checkpoint_root / "generation"), "--resume"]
    assert stages[1][-3:] == ["--checkpoint-dir", str(checkpoint_root / "validation"), "--resume"]
