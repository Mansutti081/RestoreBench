# ABOUTME: Runs a published campaign: a frozen scenario list, model suite and configuration set.
# ABOUTME: One scenario at a time into its own store, so an interrupted sweep leaves whole cases.
"""Run one of the campaigns defined in campaigns.json.

The campaign file is the experiment definition. It names the scenarios, the models, the
configurations and the wall-clock limit, so re-running a published sweep needs nothing but a
provider credential — no prior result store, no operator knowledge about which cases were in
scope."""
from __future__ import annotations

import argparse
import json
import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from restorebench.environment.scenarios import held_out_ids
from restorebench.eval import harness
from restorebench.llm import models

# pandapower emits deprecation warnings on every solve, which would bury the per-scenario lines
# this script exists to print. They come from the runtime path, not from import, so filtering
# here is enough.
warnings.filterwarnings("ignore")

@dataclass(frozen=True)
class Campaign:
    """A complete experiment definition, loaded from campaigns.json."""

    name: str
    description: str
    data_dir: Path
    results_dir: str
    models: tuple[str, ...]
    configurations: tuple[int, ...]
    repetitions: int
    max_runtime_seconds: int
    # None means "every held-out scenario of the corpus". A list freezes the exact subset a
    # published sweep ran, so it stays reproducible without shipping that sweep's results.
    scenarios: tuple[str, ...] | None

    @property
    def witnesses(self) -> Path:
        return self.data_dir / "private" / "witnesses.json"

    @property
    def cells_per_scenario(self) -> int:
        return len(self.models) * len(self.configurations) * self.repetitions


CAMPAIGNS_PATH = Path(__file__).resolve().parent / "campaigns.json"


def load_campaigns(path: Path = CAMPAIGNS_PATH) -> dict[str, Campaign]:
    """Read the experiment definitions. This file, not a result store, is what makes a sweep
    reproducible: everything a re-run needs is in it."""
    spec = json.loads(path.read_text(encoding="utf-8"))
    campaigns = {}
    for name, entry in spec["campaigns"].items():
        scenarios = entry.get("scenarios")
        campaigns[name] = Campaign(
            name=name,
            description=entry["description"],
            data_dir=Path(entry["corpus"]),
            results_dir=entry["results_dir"],
            models=tuple(entry["models"]),
            configurations=tuple(entry["configurations"]),
            repetitions=entry["repetitions"],
            max_runtime_seconds=entry["max_runtime_seconds"],
            scenarios=None if scenarios is None else tuple(scenarios),
        )
    return campaigns


CAMPAIGNS = load_campaigns()
DEFAULT_CONCURRENCY = 3

# A provider outage is waited out, not recorded. Five minutes between attempts, and after two
# hours on the same scenario the sweep moves on and leaves it incomplete: an absent cell is
# honest, a dead one is not.
PROVIDER_RETRY_SECONDS = 300
MAX_PROVIDER_RETRIES = 24

# Reasons no amount of waiting will fix: the wallet, the credential, the account. Retrying one
# of these all night spends the queue on nothing and hides the real cause until morning, so the
# campaign stops instead. A ServiceUnavailable is deliberately not here — that is what the
# retry loop is for.
PROVIDER_FATAL_MARKERS = (
    "credit balance",
    "billing",
    "quota",
    "accessdenied",
    "unrecognizedclient",
    "expiredtoken",
    "token expired",
    "not authorized",
    "invalid api key",
    "unauthorized",
)

_original_config_for_key = harness._config_for_key


def install_runtime_limit(seconds: int) -> None:
    """Hold MAX_RUNTIME_SECONDS at the campaign's value.

    Wall clock, not model time: a busier machine turns into TIMEOUTs, and a TIMEOUT is recorded
    as the model's result. Re-running a published campaign therefore has to use its number.
    """
    harness._config_for_key = lambda key: _original_config_for_key(key).model_copy(
        update={"MAX_RUNTIME_SECONDS": seconds}
    )


def _cell_counts(store: Path) -> dict[str, int]:
    """How many result files each scenario has in a store. Missing store means no cells."""
    cells = Path(store) / "cells"
    if not cells.exists():
        return {}
    counts: dict[str, int] = {}
    for path in cells.glob("*.json"):
        if "manifest" in path.name:
            continue
        scenario_id = path.name.split("__")[0]
        counts[scenario_id] = counts.get(scenario_id, 0) + 1
    return counts


def failed_cell_paths(store: Path | str, *, scenario_id: str | None = None) -> list[Path]:
    """Cells that failed for a provider reason, not a model one.

    LLM_FAILURE means the transport never delivered an answer: a Bedrock outage, a throttle that
    outlived its retries. Keeping one is worse than having no cell at all, because the aggregate
    would score an outage as the model's failure. TIMEOUT and BUDGET_EXHAUSTED are the model's
    own results and are never touched.
    """
    cells = Path(store) / "cells"
    if not cells.exists():
        return []
    pattern = f"{scenario_id}__*.json" if scenario_id else "*.json"
    failed = []
    for path in sorted(cells.glob(pattern)):
        if "manifest" in path.name:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if payload.get("status") == "LLM_FAILURE":
            failed.append(path)
    return failed


def fatal_provider_reason(paths: Sequence[Path]) -> str | None:
    """The failure detail behind a cell that waiting cannot fix, or None.

    An empty wallet and a dead credential both surface as LLM_FAILURE, exactly like a passing
    outage. The difference is that one recovers on its own and the other burns the night.
    """
    for path in paths:
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            continue
        detail = " ".join(
            str(entry.get("detail") or "") for entry in (payload.get("failure_feedback") or [])
        )
        lowered = detail.lower()
        if any(marker in lowered for marker in PROVIDER_FATAL_MARKERS):
            return detail[:200]
    return None


def purge_failed_cells(store: Path | str, *, scenario_id: str | None = None) -> list[Path]:
    """Delete the provider-failure cells so the harness will run them again.

    The harness decides a cell is done by the existence of its result file
    (`restorebench/eval/store.py:72`), so a dead cell left in place is never retried.
    """
    purged = failed_cell_paths(store, scenario_id=scenario_id)
    for path in purged:
        path.unlink()
    return purged


def campaign_scenarios(campaign: Campaign) -> list[str]:
    """The scenarios this campaign runs, in scenario-id order.

    A frozen list is used as-is; it is what makes a published sweep re-runnable from a clean
    clone. Otherwise every held-out scenario of the campaign's corpus is in scope.
    """
    if campaign.scenarios is not None:
        held_out = set(held_out_ids(data_dir=campaign.data_dir))
        outside = sorted(set(campaign.scenarios) - held_out)
        if outside:
            raise ValueError(
                f"campaign {campaign.name} names scenarios outside the held-out split: {outside}"
            )
        return sorted(campaign.scenarios)
    return sorted(held_out_ids(data_dir=campaign.data_dir))


def load_witness_lengths(path: Path | str) -> dict[str, int]:
    """Maneuvers per scenario in the private witness file. Used only to order the queue."""
    witnesses = json.loads(Path(path).read_text(encoding="utf-8"))
    return {w["scenario_id"]: len(w["maneuvers"]) for w in witnesses}


def pending_queue(
    scenario_ids: list[str],
    store: Path | str,
    *,
    witness_lengths: dict[str, int],
    cells_per_scenario: int,
) -> list[str]:
    """Part-run scenarios first, then the sequential ones, then the direct ones.

    Part-run first so an interrupted store converges on whole cases. Sequential ahead of direct
    because those cases carry the regime contrast: if the sweep stops early, they are the ones
    worth having.
    """
    counts = _cell_counts(store)
    pending = [s for s in scenario_ids if counts.get(s, 0) < cells_per_scenario]
    started = [s for s in pending if counts.get(s, 0) > 0]
    fresh = [s for s in pending if counts.get(s, 0) == 0]
    return (
        started
        + [s for s in fresh if witness_lengths.get(s, 1) > 1]
        + [s for s in fresh if witness_lengths.get(s, 1) <= 1]
    )


def spend_so_far(store: Path | str) -> tuple[float, int]:
    """USD spent in a store, derived from the saved token counts. Never read from a result file."""
    total, n = 0.0, 0
    cells = Path(store) / "cells"
    for path in cells.glob("*.json"):
        if "manifest" in path.name:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        assignment = payload.get("llm_assignment") or {}
        model_id = assignment.get("single_agent") or assignment.get("analyst")
        if not model_id:
            continue
        trace = payload.get("trace") or {}
        try:
            total += models.token_cost_usd(
                model_id,
                tokens_in=trace.get("total_llm_tokens_in", 0),
                tokens_out=trace.get("total_llm_tokens_out", 0),
            )
        except KeyError:
            continue
        n += 1
    return total, n


def select_batch(queue: list[str], *, only: str | None = None, first: int | None = None) -> list[str]:
    """Narrow a pending queue to the scenarios this invocation should run.

    `only` wins over `first`: it names a scenario, and truncating afterwards would run a
    different one than the one asked for.
    """
    if only is not None:
        if only not in queue:
            raise ValueError(f"{only} is not pending: already complete, or outside the campaign")
        return [only]
    return queue if first is None else queue[:first]


class ProviderStopped(Exception):
    """The provider is down for a reason the sweep must not spend the night retrying."""


def _run_scenario_until_healthy(
    scenario_id: str, *, concurrency: int, campaign: Campaign
) -> tuple[int, int, str]:
    """Run one scenario, retrying while the provider keeps failing on it.

    A Bedrock outage does not crash the run: every cell comes back LLM_FAILURE and the sweep
    marches on, spending the queue on nothing. So each attempt ends by purging this scenario's
    provider failures and, if there were any, waiting for the provider to recover before trying
    the same scenario again. Cells that already succeeded are kept — the harness skips them.
    """
    ran = skipped = 0
    for attempt in range(1, MAX_PROVIDER_RETRIES + 2):
        try:
            summary = harness.run_evaluation(
                list(campaign.models),
                configurations=list(campaign.configurations),
                results_dir=campaign.results_dir,
                repetitions=campaign.repetitions,
                concurrency=concurrency,
                data_dir=campaign.data_dir,
            )
            ran, skipped = summary.n_run, summary.n_skipped
        except Exception as exc:
            # One scenario that cannot run must not end the sweep; the rest are still worth having.
            return 0, 0, f"  ERRORE {type(exc).__name__}: {str(exc)[:90]}"

        failed = failed_cell_paths(campaign.results_dir, scenario_id=scenario_id)
        fatal = fatal_provider_reason(failed)
        purged = purge_failed_cells(campaign.results_dir, scenario_id=scenario_id)
        if fatal is not None:
            raise ProviderStopped(fatal)
        if not purged:
            return ran, skipped, ""
        if attempt > MAX_PROVIDER_RETRIES:
            raise ProviderStopped(
                f"{scenario_id}: {MAX_PROVIDER_RETRIES} attempts over "
                f"{MAX_PROVIDER_RETRIES * PROVIDER_RETRY_SECONDS // 3600}h, the provider is not coming back"
            )
        print(
            f"    {scenario_id}: {len(purged)} LLM_FAILURE cells discarded, retrying in "
            f"{PROVIDER_RETRY_SECONDS // 60} min (attempt {attempt}/{MAX_PROVIDER_RETRIES})",
            flush=True,
        )
        time.sleep(PROVIDER_RETRY_SECONDS)
    return ran, skipped, ""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", metavar="SCENARIO", help="run a single scenario, for a trial run")
    parser.add_argument("--first", type=int, metavar="N", help="run only the first N scenarios of the queue")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--dry-run", action="store_true", help="print the queue and exit without calling a model")
    parser.add_argument(
        "--campaign",
        choices=sorted(CAMPAIGNS),
        default="ieee118-anthropic",
        help="which published experiment to run; every parameter comes from campaigns.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    campaign = CAMPAIGNS[args.campaign]
    install_runtime_limit(campaign.max_runtime_seconds)
    covered = campaign_scenarios(campaign)
    # Dead cells left by an earlier outage would otherwise make their scenarios look complete.
    stale = purge_failed_cells(campaign.results_dir)
    if stale:
        print(f"discarded {len(stale)} LLM_FAILURE cells left over from an earlier run", flush=True)
    pending = pending_queue(
        covered,
        campaign.results_dir,
        witness_lengths=load_witness_lengths(campaign.witnesses),
        cells_per_scenario=campaign.cells_per_scenario,
    )
    try:
        queue = select_batch(pending, only=args.only, first=args.first)
    except ValueError as exc:
        print(exc)
        return

    spend, cells = spend_so_far(campaign.results_dir)
    print(f"campaign {campaign.name}: {campaign.description}")
    print(f"corpus {campaign.data_dir} | {len(covered)} scenarios in scope "
          f"| {campaign.cells_per_scenario} cells each")
    print(f"models {[models.model_slug(m) for m in campaign.models]} "
          f"| configurations {list(campaign.configurations)} "
          f"| repetitions {campaign.repetitions} "
          f"| max runtime {campaign.max_runtime_seconds}s "
          f"| concurrency {args.concurrency}")
    print(f"to run: {len(queue)} scenarios | already spent ${spend:.2f} on {cells} cells "
          f"in {campaign.results_dir}")
    print(f"queue: {queue}", flush=True)
    if args.dry_run:
        return

    started = time.time()
    for index, scenario_id in enumerate(queue, start=1):
        harness.held_out_ids = lambda sid=scenario_id, **kwargs: [sid]
        scenario_started = time.time()
        try:
            ran, skipped, note = _run_scenario_until_healthy(
                scenario_id, concurrency=args.concurrency, campaign=campaign
            )
        except ProviderStopped as stop:
            # Exit cleanly, not by crashing: the supervisor restarts a crash, and restarting into
            # a dead credential would spend the night doing exactly what this stop prevents.
            spend, cells = spend_so_far(campaign.results_dir)
            print(f"\n=== SWEEP STOPPED === {stop}", flush=True)
            print(f"no corrupted cell was saved. ${spend:.2f} on {cells} cells so far.", flush=True)
            return
        spend, cells = spend_so_far(campaign.results_dir)
        print(
            f"[{index:>2}/{len(queue)}] {scenario_id}  ran {ran} skipped {skipped}  "
            f"{(time.time() - scenario_started) / 60:.1f} min  | cumulative ${spend:.2f}{note}",
            flush=True,
        )

    spend, cells = spend_so_far(campaign.results_dir)
    counts = _cell_counts(campaign.results_dir)
    per_scenario = campaign.cells_per_scenario
    complete = sum(1 for s in covered if counts.get(s, 0) == per_scenario)
    print(f"\n=== DONE in {(time.time() - started) / 60:.0f} min ===", flush=True)
    print(f"scenarios complete at {per_scenario}/{per_scenario} cells: {complete}/{len(covered)} "
          f"| ${spend:.2f} on {cells} cells", flush=True)


if __name__ == "__main__":
    main()
