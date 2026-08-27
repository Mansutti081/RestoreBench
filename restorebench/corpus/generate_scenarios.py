# ABOUTME: Orchestrates deterministic reactive-deficit candidate generation and staged artifacts.
# ABOUTME: Uses shared plan-15 physics and never writes implicitly to the frozen dataset directory.
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import sys
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from statistics import median
from typing import Any, Sequence

from restorebench.corpus.numeric_runtime import configure_single_threaded_blas

import pandapower as pp
from pydantic import BaseModel, ConfigDict, Field, model_validator

from restorebench.environment.card_render import render_scenario_card
from restorebench.physics.actions import ACTION_POLICY_VERSION
from restorebench.physics.policies import (
    ACTIVE_BALANCE_POLICY_VERSION,
    CURATION_TRAJECTORY_POLICY_VERSION,
    ELECTRICAL_DISTANCE_POLICY_VERSION,
    FEASIBILITY_POLICY_VERSION,
    FINGERPRINT_POLICY_VERSION,
    SOLVER_PROBE_POLICY_VERSION,
)
from restorebench.schemas.dataset import (
    CurationPolicyVersions,
    DatasetManifest,
    EvaluationManifest,
    EvaluationScenarioEntry,
    GenerationMetadata,
    PublicScenarioEntry,
    ScenarioLabel,
    SharedPhysicsPolicyVersions,
    SolverSettings,
    TargetDepth,
)
from restorebench.corpus.augment import (
    NETWORK_SPECS,
    augmented_base_fingerprint,
    build_augmented_base,
    get_network_spec,
    write_augmented_base,
)
from restorebench.corpus.curation import (
    CURATION_SCAN_POLICY_VERSION,
    TARGET_DEPTH_POLICY_VERSION,
    CurationFamilyMeasurement,
    CurationScanPolicy,
    measure_curation_family,
)
from restorebench.corpus.checkpoint_io import (
    CheckpointCompatibilityError,
    CheckpointStore,
    atomic_write_json,
)
from restorebench.corpus.electrical_pockets import (
    POCKET_WEIGHTING_POLICY_VERSION,
    PocketWeightingPolicy,
    generate_pocket_recipes,
)
from restorebench.corpus.operating_profiles import (
    OPERATING_PROFILE_POLICY_VERSION,
    OperatingProfileCandidate,
    OperatingProfilePolicy,
    admit_operating_profiles,
    generate_operating_profile_candidates,
)
from restorebench.corpus.reactive_admission import (
    ALTERNATIVE_INIT_POLICY_VERSION,
    QV_THRESHOLDS_POLICY_VERSION,
    ReactiveDeficitThresholds,
    TargetAdmissionResult,
    admit_target_candidate,
)
from restorebench.corpus.reduce_to_lean import reduce_to_lean
from restorebench.corpus.select_corpus import (
    COMPOSITION_POLICY_VERSION,
    SPLIT_POLICY_VERSION,
    CorpusCandidate,
    CorpusCompositionPolicy,
    describe_corpus_composition,
    select_and_assign_corpus,
    witness_component_key,
)
from restorebench.corpus.witness_search import (
    WITNESS_SEARCH_POLICY_VERSION,
    WitnessSearchPolicy,
    WitnessSearchResult,
    search_curation_witness,
)
from restorebench.corpus.versions import (
    DATASET_VERSION,
    GENERATOR_VERSION,
    VALIDATOR_VERSION,
)


SEED = 42
STAGING_IDENTITY_VERSION = "resumable-staging-v1"


class DatasetGenerationPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    operating_profiles: OperatingProfilePolicy = OperatingProfilePolicy(
        max_simultaneous_deviations=1,
        max_profiles=33,
    )
    pockets: PocketWeightingPolicy = PocketWeightingPolicy(
        distance_scales_pu=(0.02, 0.05, 0.08, 0.1, 0.12, 0.14),
        weight_cutoff=0.05,
        minimum_load_count=4,
        minimum_base_p_mw=50.0,
        minimum_base_abs_q_mvar=10.0,
        near_duplicate_cosine=0.995,
    )
    scan: CurationScanPolicy = CurationScanPolicy(
        coarse_coordinates=(
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            2.5,
            3.0,
            3.5,
            4.0,
            5.0,
            6.0,
            7.0,
            7.25,
            7.5,
            8.0,
            9.0,
            10.0,
        ),
        refinement_resolution=0.02,
    )
    target_offsets: tuple[float, ...] = (0.02, 0.2, 0.5)
    reactive_thresholds: ReactiveDeficitThresholds = ReactiveDeficitThresholds(
        minimum_voltage_deterioration_pu=0.02,
        weak_bus_band_pu=0.02,
        minimum_q_headroom_reduction_mvar=20.0,
        material_q_violation_floor_mvar=5.0,
        material_q_violation_fraction=0.05,
        maximum_weak_region_distance_pu=0.2,
    )
    witness: WitnessSearchPolicy = WitnessSearchPolicy(
        maneuver_budget=10,
        bfs_max_depth=1,
        frontier_budget=16,
        beam_width=3,
        beam_branching_width=8,
        diagnostic_coordinates=(0.25, 0.5, 0.75, 1.0),
        diagnostic_resolution=0.02,
    )
    leakage_similarity_threshold: float = Field(
        default=0.995,
        gt=0.0,
        le=1.0,
    )
    # The floor was 0.1 while the SEQUENTIAL supply looked like a selection problem. It is a
    # generation problem: the pool offered 21 and eight of them carry a witness the runtime
    # applicability guard refuses, so the achievable share is 0.065 at 200 scenarios. Holding
    # 0.1 would force the corpus down to 130 without gaining a single multi-step case. Keep a
    # floor that still rejects an all-DIRECT corpus, and raise it only by generating deeper
    # targets — see 05_dataset.md §17.2.
    composition: CorpusCompositionPolicy = CorpusCompositionPolicy(
        min_sequential_share=0.05,
        min_distinct_witness_lengths=2,
    )
    pool_multiplier: float = Field(default=1.5, gt=1.0)
    minimum_family_evaluations_per_scenario: int = Field(
        default=10,
        ge=1,
    )
    maximum_family_evaluations: int = Field(default=4000, ge=1)
    network_id: str = "case118"
    dataset_version: str = DATASET_VERSION
    seed: int = SEED

    @model_validator(mode="after")
    def target_schedule_is_stable(self) -> "DatasetGenerationPolicy":
        if (
            not self.target_offsets
            or tuple(sorted(set(self.target_offsets))) != self.target_offsets
            or any(offset <= 0.0 for offset in self.target_offsets)
        ):
            raise ValueError("target offsets must be positive, unique, and ascending")
        if self.seed != SEED:
            raise ValueError("dataset generation seed is frozen at 42")
        spec = get_network_spec(self.network_id)
        if self.dataset_version != spec.dataset_version:
            raise ValueError(
                f"dataset_version for {self.network_id} must be {spec.dataset_version!r}"
            )
        return self


@dataclass(frozen=True)
class GeneratedCandidate:
    public: CorpusCandidate
    profile: OperatingProfileCandidate
    family: CurationFamilyMeasurement
    admission: TargetAdmissionResult
    witness: WitnessSearchResult


@dataclass(frozen=True)
class PoolGenerationResult:
    augmented_base: Any
    candidates: tuple[GeneratedCandidate, ...]
    rejection_counts: dict[str, int]
    profiles_considered: int
    pockets_considered: int
    families_evaluated: int


@dataclass(frozen=True)
class FamilyGenerationResult:
    candidates: tuple[GeneratedCandidate, ...]
    rejection_counts: dict[str, int]


@dataclass(frozen=True)
class _TargetGenerationResult:
    candidate: GeneratedCandidate | None
    rejection_reason: str | None


def _describe_generated_composition(
    candidates: Sequence[GeneratedCandidate],
) -> dict[str, Any]:
    """Summarize generated candidates through their selection descriptors."""
    return describe_corpus_composition(tuple(candidate.public for candidate in candidates))


def _describe_generated_qv_locality(
    candidates: Sequence[GeneratedCandidate],
) -> dict[str, Any]:
    """Summarize weak-region locality without using it as an admission gate."""
    distances = sorted(float(candidate.admission.qv_evidence.weak_region_min_distance_pu) for candidate in candidates)
    local_count = sum(candidate.admission.qv_evidence.weak_region_local for candidate in candidates)
    total = len(distances)
    return {
        "total": total,
        "local_count": local_count,
        "local_share": local_count / total if total else 0.0,
        "min_distance_pu": distances[0] if distances else None,
        "median_distance_pu": median(distances) if distances else None,
        "max_distance_pu": distances[-1] if distances else None,
    }


def _pool_has_composition_capacity(
    candidates: Sequence[CorpusCandidate],
    *,
    target_count: int,
    policy: CorpusCompositionPolicy,
) -> bool:
    """Check necessary composition counts before stopping pool generation."""
    if len(candidates) < target_count:
        return False
    sequential_required = math.ceil(policy.min_sequential_share * target_count - 1e-12)
    if sum(candidate.resolution_regime == "SEQUENTIAL" for candidate in candidates) < sequential_required:
        return False
    if len({candidate.witness_length for candidate in candidates}) < policy.min_distinct_witness_lengths:
        return False
    distinct_components = {component for candidate in candidates for component in candidate.witness_component_keys}
    if len(distinct_components) < policy.min_distinct_components:
        return False
    component_cap = math.floor(policy.max_component_share * target_count + 1e-12)
    for component in distinct_components:
        without_component = sum(component not in set(candidate.witness_component_keys) for candidate in candidates)
        unavoidable_count = max(0, target_count - without_component)
        if unavoidable_count > component_cap:
            return False
    family_required = math.ceil(policy.min_action_family_share * target_count - 1e-12)
    for family in policy.required_action_families:
        represented = sum(family in set(candidate.witness_action_families) for candidate in candidates)
        if represented < family_required:
            return False
    return True


def _pool_can_select_corpus(
    candidates: Sequence[CorpusCandidate],
    *,
    target_count: int,
    memory_population_count: int | None = None,
    held_out_count: int | None = None,
    policy: DatasetGenerationPolicy,
) -> bool:
    """Require the same grouped split and composition selection used at freeze."""
    memory_count, held_count = resolve_split_counts(
        target_count,
        memory_population_count=memory_population_count,
        held_out_count=held_out_count,
    )
    try:
        selected = select_and_assign_corpus(
            tuple(candidates),
            similarity_threshold=policy.leakage_similarity_threshold,
            memory_population_count=memory_count,
            held_out_count=held_count,
            composition_policy=policy.composition,
        )
    except (AssertionError, ValueError):
        return False
    return len(selected) == target_count


@contextmanager
def _state_evaluation_executor(
    workers: int,
) -> Iterator[Executor | None]:
    """Reuse one isolated worker pool across every witness in this generation."""
    if workers <= 0:
        raise ValueError("workers must be positive")
    if workers == 1:
        yield None
        return
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=get_context("spawn"),
        initializer=configure_single_threaded_blas,
    ) as executor:
        yield executor


def generate_valid_pool(
    target_count: int,
    *,
    policy: DatasetGenerationPolicy,
    memory_population_count: int | None = None,
    held_out_count: int | None = None,
    workers: int = 1,
    checkpoint_dir: Path | None = None,
    resume: bool = False,
) -> PoolGenerationResult:
    """Generate a hash-ordered pool that is broader than the requested corpus."""
    if target_count <= 0:
        raise ValueError("target_count must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")
    if resume and checkpoint_dir is None:
        raise ValueError("resume requires an explicit checkpoint directory")
    memory_count, held_count = resolve_split_counts(
        target_count,
        memory_population_count=memory_population_count,
        held_out_count=held_out_count,
    )
    augmented = (
        build_augmented_base()
        if policy.network_id == "case118"
        else build_augmented_base(policy.network_id)
    )
    profile_candidates = generate_operating_profile_candidates(
        augmented,
        policy=policy.operating_profiles,
        network_id=policy.network_id,
    )
    profile_selection = admit_operating_profiles(
        profile_candidates,
        policy=policy.operating_profiles,
    )
    anchors = tuple(
        sorted(
            set(
                int(bus)
                for bus in augmented.load.loc[
                    augmented.load.get("in_service", True).astype(bool),
                    "bus",
                ]
            )
        )
    )
    pockets = generate_pocket_recipes(
        augmented,
        anchor_buses=anchors,
        policy=policy.pockets,
    )
    desired_pool = max(
        target_count + 1,
        int(target_count * policy.pool_multiplier + 0.999999),
    )
    minimum_families = min(
        len(profile_selection.profiles) * len(pockets),
        target_count * policy.minimum_family_evaluations_per_scenario,
    )
    schedule = _family_schedule(profile_selection.profiles, pockets)
    rejections: Counter[str] = Counter(rejection.reason for rejection in profile_selection.rejections)
    generated: list[GeneratedCandidate] = []
    families_evaluated = 0
    schedule_size = min(
        len(schedule),
        policy.maximum_family_evaluations,
    )
    scheduled = schedule[:schedule_size]
    family_keys = tuple(_family_checkpoint_key(profile, pocket) for profile, pocket in scheduled)
    checkpoint_store: CheckpointStore | None = None
    if checkpoint_dir is not None:
        _require_safe_checkpoint_directory(checkpoint_dir)
        checkpoint_store = CheckpointStore.open(
            checkpoint_dir,
            identity=_pool_checkpoint_identity(
                target_count=target_count,
                policy=policy,
                family_keys=family_keys,
            ),
            resume=resume,
        )
        restored = checkpoint_store.load_pickle_shards(
            expected_keys=family_keys,
        )
        for index, family_result in enumerate(restored):
            if not isinstance(family_result, FamilyGenerationResult):
                raise CheckpointCompatibilityError(f"checkpoint family shard {index} has an unexpected payload")
            generated.extend(family_result.candidates)
            rejections.update(family_result.rejection_counts)
        families_evaluated = len(restored)
        if families_evaluated:
            print(
                f"POOL_RESUME families={families_evaluated}/{schedule_size} candidates={len(generated)}/{desired_pool}",
                flush=True,
            )
        if _pool_generation_can_stop(
            generated,
            families_evaluated=families_evaluated,
            minimum_families=minimum_families,
            desired_pool=desired_pool,
            target_count=target_count,
            memory_population_count=memory_count,
            held_out_count=held_count,
            policy=policy,
        ):
            return PoolGenerationResult(
                augmented_base=augmented,
                candidates=tuple(
                    sorted(
                        generated,
                        key=lambda candidate: candidate.public.candidate_id,
                    )
                ),
                rejection_counts=dict(sorted(rejections.items())),
                profiles_considered=len(profile_selection.profiles),
                pockets_considered=len(pockets),
                families_evaluated=families_evaluated,
            )

    with _state_evaluation_executor(workers) as executor:
        for schedule_index in range(families_evaluated, schedule_size):
            profile, pocket = scheduled[schedule_index]
            family_result = _evaluate_family(
                profile,
                pocket,
                policy=policy,
                executor=executor,
            )
            if checkpoint_store is not None:
                checkpoint_store.write_pickle_shard(
                    schedule_index,
                    key=family_keys[schedule_index],
                    payload=family_result,
                )
                print(
                    f"CHECKPOINT_COMMIT family={schedule_index + 1} path={checkpoint_store.shard_dir}",
                    flush=True,
                )
            generated.extend(family_result.candidates)
            rejections.update(family_result.rejection_counts)
            families_evaluated = schedule_index + 1

            print(
                _pool_progress_line(
                    families_evaluated=families_evaluated,
                    schedule_size=schedule_size,
                    candidate_count=len(generated),
                    desired_pool=desired_pool,
                ),
                flush=True,
            )
            if _pool_generation_can_stop(
                generated,
                families_evaluated=families_evaluated,
                minimum_families=minimum_families,
                desired_pool=desired_pool,
                target_count=target_count,
                memory_population_count=memory_count,
                held_out_count=held_count,
                policy=policy,
            ):
                break

    return PoolGenerationResult(
        augmented_base=augmented,
        candidates=tuple(
            sorted(
                generated,
                key=lambda candidate: candidate.public.candidate_id,
            )
        ),
        rejection_counts=dict(sorted(rejections.items())),
        profiles_considered=len(profile_selection.profiles),
        pockets_considered=len(pockets),
        families_evaluated=families_evaluated,
    )


def _pool_generation_can_stop(
    generated: Sequence[GeneratedCandidate],
    *,
    families_evaluated: int,
    minimum_families: int,
    desired_pool: int,
    target_count: int,
    memory_population_count: int | None = None,
    held_out_count: int | None = None,
    policy: DatasetGenerationPolicy,
) -> bool:
    if (
        families_evaluated < minimum_families
        or len(generated) < desired_pool
    ):
        return False
    public = tuple(candidate.public for candidate in generated)
    return _pool_has_composition_capacity(
        public,
        target_count=target_count,
        policy=policy.composition,
    ) and _pool_can_select_corpus(
        public,
        target_count=target_count,
        memory_population_count=memory_population_count,
        held_out_count=held_out_count,
        policy=policy,
    )


def _evaluate_family(
    profile: OperatingProfileCandidate,
    pocket: Any,
    *,
    policy: DatasetGenerationPolicy,
    executor: Executor | None,
) -> FamilyGenerationResult:
    rejections: Counter[str] = Counter()
    generated: list[GeneratedCandidate] = []
    try:
        family = measure_curation_family(
            profile,
            pocket,
            scan_policy=policy.scan,
        )
    except ValueError as exc:
        return FamilyGenerationResult(
            candidates=(),
            rejection_counts={f"FAMILY:{exc}": 1},
        )

    if executor is None or len(policy.target_offsets) == 1:
        target_results = tuple(
            _evaluate_target_offset(
                profile,
                pocket,
                family,
                offset=offset,
                policy=policy,
                executor=executor,
            )
            for offset in policy.target_offsets
        )
    else:
        # Each target owns an independent network snapshot. Overlap their
        # searches so serial beam phases can progress while the shared process
        # pool evaluates exhaustive children for another target.
        with ThreadPoolExecutor(
            max_workers=len(policy.target_offsets),
            thread_name_prefix="qv-target-offset",
        ) as target_executor:
            futures = tuple(
                target_executor.submit(
                    _evaluate_target_offset,
                    profile,
                    pocket,
                    family,
                    offset=offset,
                    policy=policy,
                    executor=executor,
                )
                for offset in policy.target_offsets
            )
            target_results = tuple(future.result() for future in futures)

    for result in target_results:
        if result.candidate is not None:
            generated.append(result.candidate)
        elif result.rejection_reason is not None:
            rejections[result.rejection_reason] += 1

    return FamilyGenerationResult(
        candidates=tuple(generated),
        rejection_counts=dict(sorted(rejections.items())),
    )


def _evaluate_target_offset(
    profile: OperatingProfileCandidate,
    pocket: Any,
    family: CurationFamilyMeasurement,
    *,
    offset: float,
    policy: DatasetGenerationPolicy,
    executor: Executor | None,
) -> _TargetGenerationResult:
    target_stress = family.boundary.upper + offset
    try:
        admission = admit_target_candidate(
            profile,
            family,
            target_stress=target_stress,
            thresholds=policy.reactive_thresholds,
        )
    except ValueError as exc:
        return _TargetGenerationResult(
            candidate=None,
            rejection_reason=f"TARGET:{exc}",
        )

    provisional_id = _provisional_scenario_id(
        family.scenario_family_id,
        target_stress,
    )
    try:
        witness = search_curation_witness(
            admission.target_state.net,
            scenario_id=provisional_id,
            policy=policy.witness,
            executor=executor,
        )
    except ValueError as exc:
        return _TargetGenerationResult(
            candidate=None,
            rejection_reason=f"WITNESS:{exc}",
        )

    relative_offset = offset / family.boundary.upper
    candidate_id = _candidate_id(
        family.scenario_family_id,
        target_stress,
        admission.target_fingerprint,
    )
    return _TargetGenerationResult(
        candidate=GeneratedCandidate(
            public=CorpusCandidate(
                candidate_id=candidate_id,
                scenario_family_id=family.scenario_family_id,
                operating_profile_id=profile.profile_id,
                pocket=pocket,
                resolution_regime=witness.resolution_regime,
                witness_length=witness.witness_length,
                witness_optimality=witness.witness_optimality,
                target_relative_offset=relative_offset,
                q_limited_gen_ids=admission.qv_evidence.q_limited_gen_ids,
                witness_component_keys=tuple(witness_component_key(action) for action in witness.witness.maneuvers),
                witness_action_families=tuple(action.type for action in witness.witness.maneuvers),
            ),
            profile=profile,
            family=family,
            admission=admission,
            witness=witness,
        ),
        rejection_reason=None,
    )


def choose_corpus(
    pool: PoolGenerationResult,
    *,
    target_count: int,
    policy: DatasetGenerationPolicy,
    memory_population_count: int | None = None,
    held_out_count: int | None = None,
) -> tuple[GeneratedCandidate, ...]:
    """Group, diversify, and split candidates while preserving private payloads."""
    by_id = {candidate.public.candidate_id: candidate for candidate in pool.candidates}
    memory_count, held_count = resolve_split_counts(
        target_count,
        memory_population_count=memory_population_count,
        held_out_count=held_out_count,
    )
    assigned = select_and_assign_corpus(
        tuple(candidate.public for candidate in pool.candidates),
        similarity_threshold=policy.leakage_similarity_threshold,
        memory_population_count=memory_count,
        held_out_count=held_count,
        composition_policy=policy.composition,
    )
    return tuple(
        GeneratedCandidate(
            public=descriptor,
            profile=by_id[descriptor.candidate_id].profile,
            family=by_id[descriptor.candidate_id].family,
            admission=by_id[descriptor.candidate_id].admission,
            witness=by_id[descriptor.candidate_id].witness,
        )
        for descriptor in assigned
    )


def split_counts(target_count: int) -> tuple[int, int]:
    """Return the exact 75/25 split, with deterministic pilot rounding."""
    if target_count <= 0:
        raise ValueError("target_count must be positive")
    held_out = target_count // 4
    return target_count - held_out, held_out


def resolve_split_counts(
    target_count: int,
    *,
    memory_population_count: int | None = None,
    held_out_count: int | None = None,
) -> tuple[int, int]:
    """Return an explicit split or the benchmark's default 75/25 split."""
    if memory_population_count is None and held_out_count is None:
        return split_counts(target_count)
    if memory_population_count is None or held_out_count is None:
        raise ValueError("memory and held-out counts must be specified together")
    if memory_population_count < 0 or held_out_count < 0:
        raise ValueError("split counts must be non-negative")
    if memory_population_count + held_out_count != target_count:
        raise ValueError("split counts must sum to target_count")
    return memory_population_count, held_out_count


def write_staged_corpus(
    selected: tuple[GeneratedCandidate, ...],
    *,
    pool: PoolGenerationResult,
    output_dir: Path,
    policy: DatasetGenerationPolicy,
    resume: bool = False,
) -> DatasetManifest:
    """Persist public/private artifacts under an owned resumable staging directory."""
    _prepare_resumable_staging_directory(
        output_dir,
        identity=_staging_identity(selected=selected, policy=policy),
        resume=resume,
    )
    spec = get_network_spec(policy.network_id)
    write_augmented_base(output_dir, policy.network_id)
    full_dir = output_dir / "full"
    lean_dir = output_dir / "lean"
    card_dir = output_dir / "llm"
    private_dir = output_dir / "private"
    for directory in (full_dir, lean_dir, card_dir, private_dir):
        directory.mkdir(parents=True, exist_ok=resume)

    labels: list[ScenarioLabel] = []
    witnesses = []
    public_entries: list[PublicScenarioEntry] = []
    split_entries: list[EvaluationScenarioEntry] = []
    metadata = _generation_metadata(policy)

    for ordinal, generated in enumerate(selected, start=1):
        scenario_id = f"S{ordinal:04d}"
        descriptor = generated.public
        if descriptor.leakage_group_id is None or descriptor.memory_split is None:
            raise AssertionError("selected candidate is missing leakage group or split")
        label = ScenarioLabel(
            scenario_id=scenario_id,
            scenario_class="REACTIVE_DEFICIT",
            scenario_family_id=descriptor.scenario_family_id,
            leakage_group_id=descriptor.leakage_group_id,
            recipe=generated.admission.recipe,
            convergence_boundary=generated.family.boundary,
            monotonicity=generated.family.monotonicity,
            qv_evidence=generated.admission.qv_evidence,
            q_unlimited_counterfactual=(generated.admission.q_unlimited_counterfactual),
            alternative_init_audit=(generated.admission.alternative_init_audit),
            resolvable_within_budget=True,
            resolution_regime=generated.witness.resolution_regime,
            direct_restorer_available=(generated.witness.direct_restorer_available),
            witness_length=generated.witness.witness_length,
            witness_optimality=generated.witness.witness_optimality,
            target_depth=TargetDepth(
                stress_offset=(generated.admission.recipe.target_stress - generated.family.boundary.upper),
                relative_offset=descriptor.target_relative_offset,
                policy_version=TARGET_DEPTH_POLICY_VERSION,
            ),
            memory_split=descriptor.memory_split,
            generation_metadata=metadata,
        )
        witness = generated.witness.witness.model_copy(
            update={
                "scenario_id": scenario_id,
                "terminal_pf": (generated.witness.witness.terminal_pf.model_copy(update={"runtime_ms": 0.0})),
            }
        )

        full_path = full_dir / f"{scenario_id}.json"
        lean_path = lean_dir / f"{scenario_id}.json"
        card_path = card_dir / f"{scenario_id}.md"
        full_net = copy.deepcopy(generated.admission.target_state.net)
        pp.reset_results(full_net)
        pp.to_json(full_net, str(full_path))
        lean_net = reduce_to_lean(copy.deepcopy(full_net))
        pp.to_json(lean_net, str(lean_path))
        card_path.write_text(
            render_scenario_card(full_net),
            encoding="utf-8",
        )

        labels.append(label)
        witnesses.append(witness)
        split_entries.append(
            EvaluationScenarioEntry(
                scenario_id=scenario_id,
                memory_split=descriptor.memory_split,
            )
        )
        public_entries.append(
            PublicScenarioEntry(
                scenario_id=scenario_id,
                full_artifact_hash=_sha256_file(full_path),
                lean_artifact_hash=_sha256_file(lean_path),
                card_artifact_hash=_sha256_file(card_path),
            )
        )

    evaluation = EvaluationManifest(
        dataset_version=policy.dataset_version,
        scenarios=tuple(split_entries),
    )
    evaluation_path = output_dir / "evaluation_manifest.json"
    _write_model(evaluation_path, evaluation)
    _write_models(private_dir / "labels.json", labels)
    _write_models(private_dir / "witnesses.json", witnesses)
    _write_json(
        output_dir / "generation_report.json",
        {
            "dataset_version": policy.dataset_version,
            "requested_count": len(selected),
            "split_counts": {
                "memory_population": sum(label.memory_split == "memory_population" for label in labels),
                "held_out": sum(label.memory_split == "held_out" for label in labels),
            },
            "valid_pool_count": len(pool.candidates),
            "profiles_considered": pool.profiles_considered,
            "pockets_considered": pool.pockets_considered,
            "families_evaluated": pool.families_evaluated,
            "rejection_counts": pool.rejection_counts,
            # Surfaced so a degenerate corpus is visible here rather than after the benchmark:
            # regime mix, witness lengths, and how far witnesses concentrate on few components.
            "selected_composition": _describe_generated_composition(selected),
            "pool_composition": _describe_generated_composition(pool.candidates),
            "selected_qv_locality": _describe_generated_qv_locality(selected),
            "pool_qv_locality": _describe_generated_qv_locality(pool.candidates),
            "policy": policy.model_dump(mode="json"),
        },
    )
    manifest = DatasetManifest(
        dataset_version=policy.dataset_version,
        base_network_hash=augmented_base_fingerprint(pool.augmented_base, network_id=policy.network_id),
        scenario_count=len(selected),
        scenarios=tuple(public_entries),
        split_manifest_hash=_sha256_file(evaluation_path),
        policy_hashes=_policy_hashes(policy),
        environment={
            "python": platform.python_version(),
            "pandapower": pp.__version__,
            "network_id": policy.network_id,
            "bus_count": str(spec.expected_shape["bus"]),
            "base_artifact": spec.base_filename,
            "augmented_artifact": spec.augmented_filename,
        },
    )
    _write_model(output_dir / "manifest.json", manifest)
    return manifest


def generate_and_write(
    target_count: int,
    *,
    output_dir: Path,
    policy: DatasetGenerationPolicy | None = None,
    memory_population_count: int | None = None,
    held_out_count: int | None = None,
    workers: int = 1,
    checkpoint_dir: Path | None = None,
    resume: bool = False,
) -> DatasetManifest:
    frozen_policy = policy or DatasetGenerationPolicy()
    pool = generate_valid_pool(
        target_count,
        policy=frozen_policy,
        memory_population_count=memory_population_count,
        held_out_count=held_out_count,
        workers=workers,
        checkpoint_dir=checkpoint_dir,
        resume=resume,
    )
    selected = choose_corpus(
        pool,
        target_count=target_count,
        policy=frozen_policy,
        memory_population_count=memory_population_count,
        held_out_count=held_out_count,
    )
    return write_staged_corpus(
        selected,
        pool=pool,
        output_dir=output_dir,
        policy=frozen_policy,
        resume=resume,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--network", choices=sorted(NETWORK_SPECS), default="case118")
    parser.add_argument("--memory-population-count", type=int)
    parser.add_argument("--held-out-count", type=int)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New explicit staging directory; dataset/ieee118 is never implicit.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Independent witness-evaluation processes; 1 keeps the serial path.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Durable per-family checkpoint directory outside the frozen corpus.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an identity-compatible generation checkpoint.",
    )
    parser.add_argument(
        "--minimum-family-evaluations-per-scenario",
        type=int,
        default=10,
        help="Minimum scheduled families evaluated per requested scenario.",
    )
    parser.add_argument(
        "--maximum-family-evaluations",
        type=int,
        default=4000,
        help="Hard cap on scheduled families evaluated.",
    )
    args = parser.parse_args(argv)
    if args.n <= 0:
        parser.error("--n must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.resume and args.checkpoint_dir is None:
        parser.error("--resume requires --checkpoint-dir")
    if args.minimum_family_evaluations_per_scenario <= 0:
        parser.error("--minimum-family-evaluations-per-scenario must be positive")
    if args.maximum_family_evaluations <= 0:
        parser.error("--maximum-family-evaluations must be positive")
    try:
        resolve_split_counts(
            args.n,
            memory_population_count=args.memory_population_count,
            held_out_count=args.held_out_count,
        )
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    spec = get_network_spec(args.network)
    policy = DatasetGenerationPolicy(
        network_id=spec.network_id,
        dataset_version=spec.dataset_version,
        minimum_family_evaluations_per_scenario=(args.minimum_family_evaluations_per_scenario),
        maximum_family_evaluations=args.maximum_family_evaluations,
    )
    manifest = generate_and_write(
        args.n,
        output_dir=args.output_dir,
        policy=policy,
        memory_population_count=args.memory_population_count,
        held_out_count=args.held_out_count,
        workers=args.workers,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume,
    )
    print(f"Wrote {manifest.scenario_count} staged scenarios to {args.output_dir}")


def _family_schedule(
    profiles: tuple[OperatingProfileCandidate, ...],
    pockets: tuple[Any, ...],
) -> tuple[tuple[OperatingProfileCandidate, Any], ...]:
    pairs = [(profile, pocket) for profile in profiles for pocket in pockets]
    return tuple(
        sorted(
            pairs,
            key=lambda pair: hashlib.sha256((pair[0].profile_id + ":" + pair[1].vector_hash).encode()).hexdigest(),
        )
    )


def _family_checkpoint_key(
    profile: OperatingProfileCandidate,
    pocket: Any,
) -> str:
    return f"{profile.profile_id}:{pocket.vector_hash}"


def _pool_checkpoint_identity(
    *,
    target_count: int,
    policy: DatasetGenerationPolicy,
    family_keys: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "kind": "reactive-deficit-generation",
        "generator_version": GENERATOR_VERSION,
        "target_count": target_count,
        "dataset_version": policy.dataset_version,
        "policy_hash": _sha256_payload(policy.model_dump(mode="json")),
        "schedule_hash": _sha256_payload(family_keys),
        "schedule_size": len(family_keys),
        "python_version": platform.python_version(),
        "pandapower_version": pp.__version__,
    }


def _require_safe_checkpoint_directory(checkpoint_dir: Path) -> None:
    resolved = checkpoint_dir.resolve()
    frozen = (Path(__file__).resolve().parents[2] / "dataset/ieee118").resolve()
    if resolved == frozen or frozen in resolved.parents:
        raise ValueError("refusing to place generation checkpoints in the frozen corpus")


def _pool_progress_line(
    *,
    families_evaluated: int,
    schedule_size: int,
    candidate_count: int,
    desired_pool: int,
) -> str:
    return f"POOL_PROGRESS families={families_evaluated}/{schedule_size} candidates={candidate_count}/{desired_pool}"


def _candidate_id(
    family_id: str,
    target_stress: float,
    target_fingerprint: str,
) -> str:
    payload = (f"{family_id}:{target_stress.hex()}:{target_fingerprint}").encode()
    return f"C-{hashlib.sha256(payload).hexdigest()[:24]}"


def _provisional_scenario_id(
    family_id: str,
    target_stress: float,
) -> str:
    digest = hashlib.sha256(f"{family_id}:{target_stress.hex()}".encode()).hexdigest()
    return f"S{int(digest[:8], 16) % 10000:04d}"


def _generation_metadata(
    policy: DatasetGenerationPolicy,
) -> GenerationMetadata:
    return GenerationMetadata(
        generator_version=GENERATOR_VERSION,
        validator_version=VALIDATOR_VERSION,
        python_version=platform.python_version(),
        pandapower_version=pp.__version__,
        seed=SEED,
        solver_settings=SolverSettings(),
        shared_policy_versions=SharedPhysicsPolicyVersions(
            active_balance=ACTIVE_BALANCE_POLICY_VERSION,
            action=ACTION_POLICY_VERSION,
            solver_probe=SOLVER_PROBE_POLICY_VERSION,
            feasibility=FEASIBILITY_POLICY_VERSION,
            electrical_distance=ELECTRICAL_DISTANCE_POLICY_VERSION,
            fingerprint=FINGERPRINT_POLICY_VERSION,
        ),
        curation_policy_versions=CurationPolicyVersions(
            augmentation=get_network_spec(policy.network_id).augmentation_policy_version,
            operating_profile=OPERATING_PROFILE_POLICY_VERSION,
            pocket_weighting=POCKET_WEIGHTING_POLICY_VERSION,
            load_stress=CURATION_TRAJECTORY_POLICY_VERSION,
            alternative_init=ALTERNATIVE_INIT_POLICY_VERSION,
            monotonicity_scan=CURATION_SCAN_POLICY_VERSION,
            qv_thresholds=QV_THRESHOLDS_POLICY_VERSION,
            witness_search=WITNESS_SEARCH_POLICY_VERSION,
            composition=COMPOSITION_POLICY_VERSION,
            split=SPLIT_POLICY_VERSION,
        ),
    )


def _policy_hashes(
    policy: DatasetGenerationPolicy,
) -> dict[str, str]:
    payloads = {
        "generation": policy.model_dump(mode="json"),
        "operating_profiles": policy.operating_profiles.model_dump(mode="json"),
        "pockets": policy.pockets.model_dump(mode="json"),
        "scan": policy.scan.model_dump(mode="json"),
        "reactive_thresholds": policy.reactive_thresholds.model_dump(mode="json"),
        "witness": policy.witness.model_dump(mode="json"),
        "composition": policy.composition.model_dump(mode="json"),
    }
    return {name: _sha256_payload(payload) for name, payload in sorted(payloads.items())}


def _require_new_staging_directory(output_dir: Path) -> None:
    resolved = output_dir.resolve()
    frozen = (Path(__file__).resolve().parents[2] / "dataset/ieee118").resolve()
    if resolved == frozen or frozen in resolved.parents:
        raise ValueError("refusing to overwrite the frozen dataset/ieee118 directory")
    if resolved.exists():
        if not resolved.is_dir():
            raise FileExistsError(f"staging path is not a directory: {resolved}")
        if any(resolved.iterdir()):
            raise FileExistsError(f"staging directory must be new or empty: {resolved}")
    else:
        resolved.mkdir(parents=True)


def _prepare_resumable_staging_directory(
    output_dir: Path,
    *,
    identity: dict[str, Any],
    resume: bool,
) -> None:
    identity_path = output_dir.resolve() / ".generation_identity.json"
    expected = {
        "format_version": STAGING_IDENTITY_VERSION,
        "identity": identity,
    }
    if not resume:
        _require_new_staging_directory(output_dir)
        atomic_write_json(identity_path, expected)
        return

    resolved = output_dir.resolve()
    frozen = (Path(__file__).resolve().parents[2] / "dataset/ieee118").resolve()
    if resolved == frozen or frozen in resolved.parents:
        raise ValueError("refusing to resume inside the frozen dataset/ieee118 directory")
    if not resolved.exists() or not any(resolved.iterdir()):
        _require_new_staging_directory(resolved)
        atomic_write_json(identity_path, expected)
        return
    if not identity_path.is_file():
        raise CheckpointCompatibilityError(f"resumable staging identity is missing: {identity_path}")
    stored = json.loads(identity_path.read_text(encoding="utf-8"))
    if stored != expected:
        raise CheckpointCompatibilityError("resumable staging identity does not match the requested run")


def _staging_identity(
    *,
    selected: tuple[GeneratedCandidate, ...],
    policy: DatasetGenerationPolicy,
) -> dict[str, Any]:
    return {
        "generator_version": GENERATOR_VERSION,
        "dataset_version": policy.dataset_version,
        "target_count": len(selected),
        "policy_hash": _sha256_payload(policy.model_dump(mode="json")),
        "candidate_ids_hash": _sha256_payload(tuple(item.public.candidate_id for item in selected)),
    }


def _write_model(path: Path, model: BaseModel) -> None:
    _write_json(path, model.model_dump(mode="json"))


def _write_models(path: Path, models: list[BaseModel]) -> None:
    _write_json(
        path,
        [model.model_dump(mode="json") for model in models],
    )


def _write_json(path: Path, payload: Any) -> None:
    atomic_write_json(path, payload)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_payload(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


if __name__ == "__main__":
    main(sys.argv[1:])
