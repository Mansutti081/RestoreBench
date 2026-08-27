# ABOUTME: Verifies staged generator CLI boundaries and deterministic pilot split counts.
# ABOUTME: Prevents implicit writes to the frozen corpus and reuse of non-empty staging paths.
from __future__ import annotations

import os
import threading
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from restorebench.corpus import generate_scenarios
from restorebench.corpus.generate_scenarios import (
    DatasetGenerationPolicy,
    _require_new_staging_directory,
    _state_evaluation_executor,
    parse_args,
    resolve_split_counts,
    split_counts,
)
from restorebench.corpus.select_corpus import CorpusCompositionPolicy
from restorebench.corpus.augment import build_augmented_base
from restorebench.corpus.witness_search import WitnessSearchPolicy, _evaluate_state


def test_frozen_pilot_policy_keeps_all_single_control_profiles_and_useful_scales() -> None:
    policy = DatasetGenerationPolicy()

    assert policy.operating_profiles.max_simultaneous_deviations == 1
    assert policy.operating_profiles.max_profiles == 33
    assert policy.pockets.distance_scales_pu == (
        0.02,
        0.05,
        0.08,
        0.1,
        0.12,
        0.14,
    )
    assert policy.target_offsets == (0.02, 0.2, 0.5)
    assert policy.composition.min_sequential_share == 0.05
    assert policy.composition.min_distinct_witness_lengths == 2


def test_generator_requires_an_explicit_output_directory() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--n", "10"])

    args = parse_args(["--n", "10", "--output-dir", "/tmp/restorebench-pilot"])

    assert args.n == 10
    assert args.output_dir == Path("/tmp/restorebench-pilot")
    assert args.workers == 1


def test_generator_accepts_an_explicit_positive_worker_count() -> None:
    args = parse_args(
        [
            "--n",
            "10",
            "--output-dir",
            "/tmp/restorebench-pilot",
            "--workers",
            "8",
        ]
    )

    assert args.workers == 8

    with pytest.raises(SystemExit):
        parse_args(
            [
                "--n",
                "10",
                "--output-dir",
                "/tmp/restorebench-pilot",
                "--workers",
                "0",
            ]
        )


def test_generator_accepts_explicit_checkpoint_resume_and_yield_policy(
    tmp_path: Path,
) -> None:
    args = parse_args(
        [
            "--n",
            "200",
            "--output-dir",
            str(tmp_path / "dataset"),
            "--checkpoint-dir",
            str(tmp_path / "checkpoint"),
            "--resume",
            "--minimum-family-evaluations-per-scenario",
            "1",
            "--maximum-family-evaluations",
            "4000",
        ]
    )

    assert args.checkpoint_dir == tmp_path / "checkpoint"
    assert args.resume is True
    assert args.minimum_family_evaluations_per_scenario == 1
    assert args.maximum_family_evaluations == 4000


def test_pool_resume_skips_every_atomically_completed_family(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    augmented = SimpleNamespace(
        load=pd.DataFrame(
            {
                "bus": [1],
                "in_service": [True],
            }
        )
    )
    profiles = (
        SimpleNamespace(profile_id="P0"),
        SimpleNamespace(profile_id="P1"),
    )
    pocket = SimpleNamespace(vector_hash="pocket")
    monkeypatch.setattr(
        generate_scenarios,
        "build_augmented_base",
        lambda: augmented,
    )
    monkeypatch.setattr(
        generate_scenarios,
        "generate_operating_profile_candidates",
        lambda *_args, **_kwargs: (),
    )
    monkeypatch.setattr(
        generate_scenarios,
        "admit_operating_profiles",
        lambda *_args, **_kwargs: SimpleNamespace(
            profiles=profiles,
            rejections=(),
        ),
    )
    monkeypatch.setattr(
        generate_scenarios,
        "generate_pocket_recipes",
        lambda *_args, **_kwargs: (pocket,),
    )
    monkeypatch.setattr(
        generate_scenarios,
        "_pool_can_select_corpus",
        lambda *_args, **_kwargs: True,
    )

    def generated_candidate(candidate_id: str) -> SimpleNamespace:
        return SimpleNamespace(
            public=SimpleNamespace(
                candidate_id=candidate_id,
                resolution_regime="DIRECT",
                witness_length=1,
                witness_component_keys=("GEN:1",),
                witness_action_families=("GEN_V_SETPOINT",),
            )
        )

    first_attempt: list[str] = []

    def crash_on_second(
        profile: SimpleNamespace,
        _pocket: SimpleNamespace,
        **_kwargs: object,
    ) -> object:
        first_attempt.append(profile.profile_id)
        if profile.profile_id == "P1":
            raise RuntimeError("simulated process crash")
        return generate_scenarios.FamilyGenerationResult(
            candidates=(generated_candidate("C0"),),
            rejection_counts={},
        )

    monkeypatch.setattr(
        generate_scenarios,
        "_evaluate_family",
        crash_on_second,
    )
    policy = DatasetGenerationPolicy(
        composition=CorpusCompositionPolicy(),
        minimum_family_evaluations_per_scenario=1,
        maximum_family_evaluations=2,
    )
    checkpoint_dir = tmp_path / "checkpoint"

    with pytest.raises(RuntimeError, match="simulated process crash"):
        generate_scenarios.generate_valid_pool(
            1,
            policy=policy,
            checkpoint_dir=checkpoint_dir,
        )

    resumed_attempt: list[str] = []

    def finish_second(
        profile: SimpleNamespace,
        _pocket: SimpleNamespace,
        **_kwargs: object,
    ) -> object:
        resumed_attempt.append(profile.profile_id)
        return generate_scenarios.FamilyGenerationResult(
            candidates=(generated_candidate("C1"),),
            rejection_counts={},
        )

    monkeypatch.setattr(
        generate_scenarios,
        "_evaluate_family",
        finish_second,
    )
    result = generate_scenarios.generate_valid_pool(
        1,
        policy=policy,
        checkpoint_dir=checkpoint_dir,
        resume=True,
    )

    assert first_attempt == ["P0", "P1"]
    assert resumed_attempt == ["P1"]
    assert result.families_evaluated == 2
    assert [item.public.candidate_id for item in result.candidates] == [
        "C0",
        "C1",
    ]


def test_pool_resume_does_not_evaluate_past_a_completed_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    augmented = SimpleNamespace(
        load=pd.DataFrame({"bus": [1], "in_service": [True]})
    )
    profiles = (
        SimpleNamespace(profile_id="P0"),
        SimpleNamespace(profile_id="P1"),
    )
    pocket = SimpleNamespace(vector_hash="pocket")
    monkeypatch.setattr(
        generate_scenarios,
        "build_augmented_base",
        lambda: augmented,
    )
    monkeypatch.setattr(
        generate_scenarios,
        "generate_operating_profile_candidates",
        lambda *_args, **_kwargs: (),
    )
    monkeypatch.setattr(
        generate_scenarios,
        "admit_operating_profiles",
        lambda *_args, **_kwargs: SimpleNamespace(
            profiles=profiles,
            rejections=(),
        ),
    )
    monkeypatch.setattr(
        generate_scenarios,
        "generate_pocket_recipes",
        lambda *_args, **_kwargs: (pocket,),
    )
    monkeypatch.setattr(
        generate_scenarios,
        "_family_schedule",
        lambda *_args, **_kwargs: tuple(
            (profile, pocket) for profile in profiles
        ),
    )
    monkeypatch.setattr(
        generate_scenarios,
        "_pool_has_composition_capacity",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        generate_scenarios,
        "_pool_can_select_corpus",
        lambda *_args, **_kwargs: True,
    )

    def generated_candidate(candidate_id: str) -> SimpleNamespace:
        return SimpleNamespace(
            public=SimpleNamespace(candidate_id=candidate_id)
        )

    monkeypatch.setattr(
        generate_scenarios,
        "_evaluate_family",
        lambda *_args, **_kwargs: (
            generate_scenarios.FamilyGenerationResult(
                candidates=(
                    generated_candidate("C0"),
                    generated_candidate("C1"),
                ),
                rejection_counts={},
            )
        ),
    )
    policy = DatasetGenerationPolicy(
        composition=CorpusCompositionPolicy(),
        minimum_family_evaluations_per_scenario=1,
        maximum_family_evaluations=2,
    )
    checkpoint_dir = tmp_path / "checkpoint"
    initial = generate_scenarios.generate_valid_pool(
        1,
        policy=policy,
        checkpoint_dir=checkpoint_dir,
    )
    assert initial.families_evaluated == 1

    monkeypatch.setattr(
        generate_scenarios,
        "_evaluate_family",
        lambda *_args, **_kwargs: pytest.fail(
            "completed checkpoint must not evaluate another family"
        ),
    )
    resumed = generate_scenarios.generate_valid_pool(
        1,
        policy=policy,
        checkpoint_dir=checkpoint_dir,
        resume=True,
    )

    assert resumed.families_evaluated == 1
    assert [item.public.candidate_id for item in resumed.candidates] == [
        "C0",
        "C1",
    ]


def test_generator_entry_limits_every_supported_blas_runtime_to_one_thread() -> None:
    assert {
        name: os.environ.get(name)
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
        )
    } == {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }


def test_pool_progress_line_exposes_work_completed_and_candidate_yield() -> None:
    assert (
        generate_scenarios._pool_progress_line(
            families_evaluated=17,
            schedule_size=120,
            candidate_count=4,
            desired_pool=18,
        )
        == "POOL_PROGRESS families=17/120 candidates=4/18"
    )


def test_family_overlaps_target_offsets_when_state_workers_are_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    family = SimpleNamespace(
        boundary=SimpleNamespace(upper=1.0),
        scenario_family_id="FAMILY",
    )
    arrivals = 0
    arrivals_lock = threading.Lock()
    all_arrived = threading.Event()

    monkeypatch.setattr(
        generate_scenarios,
        "measure_curation_family",
        lambda *_args, **_kwargs: family,
    )
    monkeypatch.setattr(
        generate_scenarios,
        "admit_target_candidate",
        lambda *_args, target_stress, **_kwargs: SimpleNamespace(
            target_state=SimpleNamespace(net=target_stress),
            target_fingerprint=str(target_stress),
            qv_evidence=SimpleNamespace(q_limited_gen_ids=()),
        ),
    )

    def exhaust_witness(
        _net: object,
        **_kwargs: object,
    ) -> object:
        nonlocal arrivals
        with arrivals_lock:
            arrivals += 1
            if arrivals == 3:
                all_arrived.set()
        assert all_arrived.wait(timeout=1.0)
        raise ValueError("no witness")

    monkeypatch.setattr(
        generate_scenarios,
        "search_curation_witness",
        exhaust_witness,
    )

    result = generate_scenarios._evaluate_family(
        SimpleNamespace(),
        SimpleNamespace(),
        policy=DatasetGenerationPolicy(),
        executor=SimpleNamespace(),
    )

    assert arrivals == 3
    assert result.candidates == ()
    assert result.rejection_counts == {"WITNESS:no witness": 3}


def test_generated_composition_unwraps_public_candidate_descriptors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = object()
    observed: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        generate_scenarios,
        "describe_corpus_composition",
        lambda candidates: observed.append(candidates) or {"total": len(candidates)},
    )

    composition = generate_scenarios._describe_generated_composition(
        (SimpleNamespace(public=descriptor),),
    )

    assert composition == {"total": 1}
    assert observed == [(descriptor,)]


def test_generated_locality_summary_keeps_remote_regions_visible() -> None:
    candidates = tuple(
        SimpleNamespace(
            admission=SimpleNamespace(
                qv_evidence=SimpleNamespace(
                    weak_region_min_distance_pu=distance,
                    weak_region_local=local,
                )
            )
        )
        for distance, local in ((0.05, True), (0.15, True), (0.4, False))
    )

    summary = generate_scenarios._describe_generated_qv_locality(candidates)

    assert summary == {
        "total": 3,
        "local_count": 2,
        "local_share": pytest.approx(2 / 3),
        "min_distance_pu": 0.05,
        "median_distance_pu": 0.15,
        "max_distance_pu": 0.4,
    }


def test_choose_corpus_enforces_the_generation_composition_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    composition = CorpusCompositionPolicy(min_sequential_share=0.25)
    policy = DatasetGenerationPolicy(composition=composition)
    observed: list[CorpusCompositionPolicy | None] = []
    monkeypatch.setattr(
        generate_scenarios,
        "select_and_assign_corpus",
        lambda *_args, composition_policy=None, **_kwargs: observed.append(composition_policy) or (),
    )
    pool = SimpleNamespace(candidates=())

    selected = generate_scenarios.choose_corpus(
        pool,
        target_count=4,
        policy=policy,
    )

    assert selected == ()
    assert observed == [composition]


def test_pool_stop_waits_for_required_composition_capacity() -> None:
    composition = CorpusCompositionPolicy(
        min_sequential_share=0.5,
        min_distinct_witness_lengths=2,
        required_action_families=("SHUNT_STEP",),
        min_action_family_share=0.5,
    )
    direct = SimpleNamespace(
        resolution_regime="DIRECT",
        witness_length=1,
        witness_component_keys=("GEN:1",),
        witness_action_families=("GEN_V_SETPOINT",),
    )
    sequential = SimpleNamespace(
        resolution_regime="SEQUENTIAL",
        witness_length=2,
        witness_component_keys=("SHUNT:1",),
        witness_action_families=("SHUNT_STEP",),
    )

    assert (
        generate_scenarios._pool_has_composition_capacity(
            (direct, direct),
            target_count=2,
            policy=composition,
        )
        is False
    )
    assert (
        generate_scenarios._pool_has_composition_capacity(
            (direct, sequential),
            target_count=2,
            policy=composition,
        )
        is True
    )
    assert (
        generate_scenarios._pool_has_composition_capacity(
            (direct, direct),
            target_count=2,
            policy=CorpusCompositionPolicy(max_component_share=0.5),
        )
        is False
    )


def test_pool_stop_requires_a_split_feasible_final_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = DatasetGenerationPolicy(
        composition=CorpusCompositionPolicy(),
    )
    candidates = (
        SimpleNamespace(candidate_id="C0"),
        SimpleNamespace(candidate_id="C1"),
    )
    monkeypatch.setattr(
        generate_scenarios,
        "select_and_assign_corpus",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("leakage groups cannot hit the exact split")),
    )

    assert (
        generate_scenarios._pool_can_select_corpus(
            candidates,
            target_count=2,
            policy=policy,
        )
        is False
    )

    monkeypatch.setattr(
        generate_scenarios,
        "select_and_assign_corpus",
        lambda *_args, **_kwargs: candidates,
    )
    assert (
        generate_scenarios._pool_can_select_corpus(
            candidates,
            target_count=2,
            policy=policy,
        )
        is True
    )


def test_process_workers_accept_real_pandapower_states() -> None:
    net = build_augmented_base()
    policy = WitnessSearchPolicy()
    serial = _evaluate_state(net, policy)

    with _state_evaluation_executor(2) as executor:
        assert executor is not None
        futures = [executor.submit(_evaluate_state, net, policy) for _ in range(2)]
        evaluations = [future.result(timeout=60) for future in futures]
        worker_thread_limits = {
            name: executor.submit(os.getenv, name).result(timeout=60)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        }

    assert [evaluation.status for evaluation in evaluations] == [
        "TERMINAL",
        "TERMINAL",
    ]
    assert [evaluation.solver_attempt_count for evaluation in evaluations] == [
        1,
        1,
    ]
    assert worker_thread_limits == {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    expected = serial.model_copy(
        update={
            "terminal_pf": serial.terminal_pf.model_copy(update={"runtime_ms": 0.0})
            if serial.terminal_pf is not None
            else None
        }
    )
    for evaluation in evaluations:
        normalized = evaluation.model_copy(
            update={
                "terminal_pf": evaluation.terminal_pf.model_copy(update={"runtime_ms": 0.0})
                if evaluation.terminal_pf is not None
                else None
            }
        )
        assert normalized == expected


def test_split_counts_match_full_and_pilot_targets() -> None:
    assert split_counts(200) == (150, 50)
    assert split_counts(10) == (8, 2)


def test_explicit_split_can_place_all_46_scenarios_in_held_out() -> None:
    assert resolve_split_counts(
        46,
        memory_population_count=0,
        held_out_count=46,
    ) == (0, 46)

    args = parse_args(
        [
            "--n",
            "46",
            "--network",
            "case89pegase",
            "--memory-population-count",
            "0",
            "--held-out-count",
            "46",
            "--output-dir",
            "/tmp/pegase-stage",
        ]
    )
    assert args.network == "case89pegase"
    assert (args.memory_population_count, args.held_out_count) == (0, 46)


def test_explicit_split_must_sum_to_requested_count() -> None:
    with pytest.raises(ValueError, match="sum to target_count"):
        resolve_split_counts(
            46,
            memory_population_count=1,
            held_out_count=46,
        )


def test_staging_directory_must_be_empty_and_not_frozen(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "new-stage"
    _require_new_staging_directory(staging)
    assert staging.is_dir()

    (staging / "existing.txt").write_text("owned", encoding="utf-8")
    with pytest.raises(FileExistsError, match="new or empty"):
        _require_new_staging_directory(staging)

    frozen = Path(__file__).resolve().parents[2] / "dataset/ieee118"
    with pytest.raises(ValueError, match="frozen"):
        _require_new_staging_directory(frozen)
    with pytest.raises(ValueError, match="frozen"):
        _require_new_staging_directory(frozen / "full")
    with pytest.raises(ValueError, match="frozen"):
        generate_scenarios._prepare_resumable_staging_directory(
            frozen / "full",
            identity={"target_count": 1},
            resume=True,
        )


def test_resumable_staging_accepts_only_the_exact_owned_run(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "dataset"
    identity = {
        "generator_version": "generator-v1",
        "target_count": 200,
        "policy_hash": "abc",
    }

    generate_scenarios._prepare_resumable_staging_directory(
        staging,
        identity=identity,
        resume=False,
    )
    (staging / "partial-artifact.json").write_text(
        "partial",
        encoding="utf-8",
    )
    generate_scenarios._prepare_resumable_staging_directory(
        staging,
        identity=identity,
        resume=True,
    )

    with pytest.raises(FileExistsError, match="new or empty"):
        generate_scenarios._prepare_resumable_staging_directory(
            staging,
            identity=identity,
            resume=False,
        )
    with pytest.raises(
        generate_scenarios.CheckpointCompatibilityError,
        match="identity",
    ):
        generate_scenarios._prepare_resumable_staging_directory(
            staging,
            identity={**identity, "policy_hash": "different"},
            resume=True,
        )


def test_resume_creates_staging_when_only_pool_checkpoint_exists(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "not-created-during-pool-generation"

    generate_scenarios._prepare_resumable_staging_directory(
        staging,
        identity={"target_count": 200, "policy_hash": "abc"},
        resume=True,
    )

    assert (staging / ".generation_identity.json").is_file()
