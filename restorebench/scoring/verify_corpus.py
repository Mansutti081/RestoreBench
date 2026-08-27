# ABOUTME: Replays every scenario's own witness through the scorer and reports the tally.
# ABOUTME: The corpus claim is that each scenario is solvable within budget; this re-derives it.
"""Verify a frozen corpus without calling any LLM.

Three independent claims are checked per scenario, all deterministic and free:

1. the public artifacts hash to what `manifest.json` says they do (the loader enforces this);
2. the scenario does *not* converge on load, so it is genuinely a restoration problem;
3. the witness in `private/witnesses.json` restores convergence within the maneuver budget.

Reviewers can run this on a clean clone and reproduce the benchmark's central claim end to end.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from restorebench.environment.scenarios import load_scenario
from restorebench.schemas.errors import CorpusIntegrityError
from restorebench.scoring.score_maneuvers import MANEUVER_BUDGET, score_attempt


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = ROOT / "dataset/ieee118"


class VerificationError(RuntimeError):
    """Raised when a corpus fails a claim it makes about itself."""


def verify_corpus(dataset_dir: Path, *, limit: int | None = None) -> dict[str, Any]:
    witnesses = _load_witnesses(dataset_dir)
    scenario_ids = sorted(witnesses)
    if limit is not None:
        scenario_ids = scenario_ids[:limit]
    if not scenario_ids:
        raise VerificationError(f"no witnesses found under {dataset_dir}")

    started = time.perf_counter()
    failures: list[dict[str, Any]] = []
    maneuver_counts: Counter[int] = Counter()

    for scenario_id in scenario_ids:
        # Loading is itself claim 1: the loader verifies every artifact hash against the manifest.
        try:
            load_scenario(scenario_id, data_dir=dataset_dir)
        except ValueError as exc:
            failures.append({"scenario_id": scenario_id, "claim": "artifact_hash", "detail": str(exc)})
            continue

        maneuvers = witnesses[scenario_id]
        try:
            # score_attempt raises CorpusIntegrityError when a scenario converges on load, which
            # is claim 2; the returned status carries claim 3.
            report = score_attempt(
                {"scenario_id": scenario_id, "maneuvers": maneuvers},
                data_dir=dataset_dir / "full",
            )
        except CorpusIntegrityError as exc:
            failures.append({"scenario_id": scenario_id, "claim": "non_convergent_on_load", "detail": str(exc)})
            continue

        if report["status"] != "SUCCESS":
            failures.append({
                "scenario_id": scenario_id,
                "claim": "witness_resolves",
                "detail": f"witness scored {report['status']}, not SUCCESS",
            })
            continue
        if report["n_invalid"]:
            failures.append({
                "scenario_id": scenario_id,
                "claim": "witness_is_applicable",
                "detail": f"{report['n_invalid']} witness maneuvers were rejected as invalid",
            })
            continue
        maneuver_counts[report["n_maneuvers"]] += 1

    return {
        "dataset_dir": str(dataset_dir),
        "scenarios_checked": len(scenario_ids),
        "scenarios_verified": len(scenario_ids) - len(failures),
        "failures": failures,
        "maneuver_budget": MANEUVER_BUDGET,
        "witness_length_distribution": {str(k): v for k, v in sorted(maneuver_counts.items())},
        "runtime_seconds": round(time.perf_counter() - started, 1),
    }


def _load_witnesses(dataset_dir: Path) -> dict[str, list[dict[str, Any]]]:
    path = dataset_dir / "private" / "witnesses.json"
    if not path.is_file():
        raise VerificationError(f"no witness file at {path}")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["scenario_id"]): list(row["maneuvers"]) for row in loaded}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = verify_corpus(args.dataset_dir, limit=args.limit)

    checked, verified = report["scenarios_checked"], report["scenarios_verified"]
    print(f"corpus:    {report['dataset_dir']}")
    print(f"verified:  {verified}/{checked} scenarios in {report['runtime_seconds']}s")
    print(f"witness length distribution: {report['witness_length_distribution']}")

    if report["failures"]:
        print(f"\nFAILED — {len(report['failures'])} scenario(s) did not hold:", file=sys.stderr)
        for failure in report["failures"][:20]:
            print(f"  {failure['scenario_id']}  {failure['claim']}: {failure['detail']}", file=sys.stderr)
        if len(report["failures"]) > 20:
            print(f"  ... and {len(report['failures']) - 20} more", file=sys.stderr)
        return 1

    print("\nOK — every scenario is hash-verified, non-convergent on load, and resolved by its witness.")
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay every scenario's witness to verify a frozen corpus. No LLM calls."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--limit", type=int, help="check only the first N scenarios, for a quick smoke test")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
