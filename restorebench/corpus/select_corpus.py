# ABOUTME: Groups near-duplicate curation families and selects a deterministic diverse corpus.
# ABOUTME: Assigns complete leakage groups to exact memory-population and held-out counts.
from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from restorebench.schemas.dataset import MemorySplit, PocketRecipe
from restorebench.corpus.electrical_pockets import pocket_vector_similarity


SPLIT_POLICY_VERSION = "grouped-deterministic-configurable-v2"
COMPOSITION_POLICY_VERSION = "corpus-composition-floors-v2"


class CorpusCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str
    scenario_family_id: str
    operating_profile_id: str
    pocket: PocketRecipe
    resolution_regime: Literal["DIRECT", "SEQUENTIAL"]
    witness_length: int = Field(ge=1, le=10)
    witness_optimality: Literal["EXACT_MINIMUM", "UPPER_BOUND"]
    target_relative_offset: float = Field(gt=0.0, allow_inf_nan=False)
    q_limited_gen_ids: tuple[int, ...]
    # Which components the witness actually moves. Without these the corpus can concentrate on a
    # handful of elements, and a model that memorizes them scores well without reasoning.
    witness_component_keys: tuple[str, ...] = Field(min_length=1)
    witness_action_families: tuple[str, ...] = Field(min_length=1)
    leakage_group_id: str | None = None
    memory_split: MemorySplit | None = None

    @model_validator(mode="after")
    def witness_matches_regime(self) -> "CorpusCandidate":
        if self.resolution_regime == "DIRECT" and self.witness_length != 1:
            raise ValueError("DIRECT corpus candidates require witness length 1")
        if self.resolution_regime == "SEQUENTIAL" and self.witness_length < 2:
            raise ValueError("SEQUENTIAL corpus candidates require witness length >= 2")
        if tuple(sorted(set(self.q_limited_gen_ids))) != self.q_limited_gen_ids:
            raise ValueError("Q-limited generator IDs must be unique and ascending")
        return self


def witness_component_key(action: Any) -> str:
    """Return the stable component identity a witness maneuver moves."""
    if action.type == "GEN_V_SETPOINT":
        return f"GEN:{int(action.gen_id)}"
    if action.type == "SHUNT_STEP":
        return f"SHUNT:{int(action.shunt_id)}"
    if action.type == "TAP_ADJUSTMENT":
        return f"TRAFO:{int(action.trafo_id)}"
    raise ValueError(f"unsupported witness action type {action.type}")


def assign_leakage_groups(
    candidates: tuple[CorpusCandidate, ...],
    *,
    similarity_threshold: float,
) -> tuple[CorpusCandidate, ...]:
    """Merge same-family and near-duplicate stress vectors with deterministic union-find."""
    if not math.isfinite(similarity_threshold) or not 0.0 < similarity_threshold <= 1.0:
        raise ValueError("similarity threshold must be finite and in (0, 1]")
    ordered = sorted(candidates, key=lambda candidate: candidate.candidate_id)
    parent = list(range(len(ordered)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        lower, upper = sorted((left_root, right_root))
        parent[upper] = lower

    for left_index, left in enumerate(ordered):
        for right_index in range(left_index + 1, len(ordered)):
            right = ordered[right_index]
            if (
                left.scenario_family_id == right.scenario_family_id
                or pocket_vector_similarity(left.pocket, right.pocket)
                >= similarity_threshold
            ):
                union(left_index, right_index)

    component_families: dict[int, set[str]] = defaultdict(set)
    for index, candidate in enumerate(ordered):
        component_families[find(index)].add(candidate.scenario_family_id)
    group_ids = {
        root: _leakage_group_id(families)
        for root, families in component_families.items()
    }
    return tuple(
        candidate.model_copy(
            update={"leakage_group_id": group_ids[find(index)]}
        )
        for index, candidate in enumerate(ordered)
    )


def select_diverse_candidates(
    candidates: tuple[CorpusCandidate, ...],
    *,
    target_count: int,
    composition_policy: CorpusCompositionPolicy | None = None,
) -> tuple[CorpusCandidate, ...]:
    """Greedily maximize frozen coverage dimensions with stable tie-breaking."""
    if target_count <= 0:
        raise ValueError("target_count must be positive")
    if len(candidates) < target_count:
        raise ValueError("valid pool is smaller than requested corpus")

    remaining = sorted(candidates, key=lambda candidate: candidate.candidate_id)
    selected: list[CorpusCandidate] = []
    seen_anchors: set[int] = set()
    seen_profiles: set[str] = set()
    seen_regimes: set[str] = set()
    seen_lengths: set[int] = set()
    seen_q_patterns: set[tuple[int, ...]] = set()
    seen_depth_buckets: set[int] = set()
    seen_leakage_groups: set[str] = set()
    component_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    regime_counts: Counter[str] = Counter()

    while len(selected) < target_count:
        scored = [
            (
                _composition_priority(
                    candidate,
                    policy=composition_policy,
                    target_count=target_count,
                    component_counts=component_counts,
                    family_counts=family_counts,
                    regime_counts=regime_counts,
                )
                + _coverage_score(
                    candidate,
                    selected=selected,
                    seen_anchors=seen_anchors,
                    seen_profiles=seen_profiles,
                    seen_regimes=seen_regimes,
                    seen_lengths=seen_lengths,
                    seen_q_patterns=seen_q_patterns,
                    seen_depth_buckets=seen_depth_buckets,
                    seen_leakage_groups=seen_leakage_groups,
                ),
                -index,
                candidate,
            )
            for index, candidate in enumerate(remaining)
        ]
        _, _, winner = max(scored, key=lambda row: (row[0], row[1]))
        selected.append(winner)
        remaining.remove(winner)
        component_counts.update(set(winner.witness_component_keys))
        family_counts.update(set(winner.witness_action_families))
        regime_counts[winner.resolution_regime] += 1
        seen_anchors.add(winner.pocket.anchor_bus)
        seen_profiles.add(winner.operating_profile_id)
        seen_regimes.add(winner.resolution_regime)
        seen_lengths.add(winner.witness_length)
        seen_q_patterns.add(winner.q_limited_gen_ids)
        seen_depth_buckets.add(_depth_bucket(winner.target_relative_offset))
        if winner.leakage_group_id is not None:
            seen_leakage_groups.add(winner.leakage_group_id)
    return tuple(selected)


def select_diverse_candidates_for_split(
    candidates: tuple[CorpusCandidate, ...],
    *,
    memory_population_count: int,
    held_out_count: int,
    composition_policy: CorpusCompositionPolicy | None = None,
) -> tuple[CorpusCandidate, ...]:
    """Select exact split sides from disjoint, capacity-balanced leakage groups."""
    if memory_population_count < 0 or held_out_count < 0:
        raise ValueError("split counts must be non-negative")
    target_count = memory_population_count + held_out_count
    if target_count <= 0:
        raise ValueError("selected corpus must contain at least one candidate")
    if len(candidates) < target_count:
        raise ValueError("valid pool is smaller than requested corpus")
    if any(candidate.leakage_group_id is None for candidate in candidates):
        raise ValueError("leakage groups must be assigned before split-aware selection")
    if held_out_count == 0:
        return select_diverse_candidates(
            candidates,
            target_count=memory_population_count,
            composition_policy=composition_policy,
        )
    if memory_population_count == 0:
        return select_diverse_candidates(
            candidates,
            target_count=held_out_count,
            composition_policy=composition_policy,
        )

    groups: dict[str, tuple[CorpusCandidate, ...]] = {}
    for group_id in sorted(
        {str(candidate.leakage_group_id) for candidate in candidates}
    ):
        groups[group_id] = tuple(
            candidate
            for candidate in candidates
            if candidate.leakage_group_id == group_id
        )

    held_group_ids, memory_group_ids = _partition_group_capacities(
        groups,
        memory_population_count=memory_population_count,
        held_out_count=held_out_count,
    )
    held_candidates = tuple(
        candidate
        for group_id in held_group_ids
        for candidate in groups[group_id]
    )
    memory_candidates = tuple(
        candidate
        for group_id in memory_group_ids
        for candidate in groups[group_id]
    )
    held_selected = select_diverse_candidates(
        held_candidates,
        target_count=held_out_count,
        composition_policy=composition_policy,
    )
    memory_selected = select_diverse_candidates(
        memory_candidates,
        target_count=memory_population_count,
        composition_policy=composition_policy,
    )
    return tuple(
        sorted(
            (*held_selected, *memory_selected),
            key=lambda candidate: candidate.candidate_id,
        )
    )


def assign_memory_split(
    candidates: tuple[CorpusCandidate, ...],
    *,
    memory_population_count: int,
    held_out_count: int,
) -> tuple[CorpusCandidate, ...]:
    """Assign whole leakage groups through an exact deterministic subset-sum."""
    if memory_population_count < 0 or held_out_count < 0:
        raise ValueError("split counts must be non-negative")
    if len(candidates) != memory_population_count + held_out_count:
        raise ValueError("split counts must equal selected corpus size")
    if any(candidate.leakage_group_id is None for candidate in candidates):
        raise ValueError("leakage groups must be assigned before memory split")

    groups: dict[str, list[CorpusCandidate]] = defaultdict(list)
    for candidate in candidates:
        groups[str(candidate.leakage_group_id)].append(candidate)
    group_sizes = [
        (group_id, len(groups[group_id]))
        for group_id in sorted(groups)
    ]
    choices: dict[int, tuple[str, ...]] = {0: ()}
    for group_id, size in group_sizes:
        updated = dict(choices)
        for subtotal, selected_groups in choices.items():
            new_total = subtotal + size
            proposal = (*selected_groups, group_id)
            if (
                new_total <= held_out_count
                and (
                    new_total not in updated
                    or proposal < updated[new_total]
                )
            ):
                updated[new_total] = proposal
        choices = updated
    if held_out_count not in choices:
        raise ValueError(
            "leakage-safe exact split is impossible; expand the valid candidate pool"
        )
    held_groups = set(choices[held_out_count])
    assigned = tuple(
        candidate.model_copy(
            update={
                "memory_split": (
                    "held_out"
                    if candidate.leakage_group_id in held_groups
                    else "memory_population"
                )
            }
        )
        for candidate in sorted(
            candidates,
            key=lambda item: item.candidate_id,
        )
    )
    if sum(
        candidate.memory_split == "memory_population"
        for candidate in assigned
    ) != memory_population_count:
        raise AssertionError("deterministic split did not hit memory-population count")
    return assigned


class CorpusCompositionPolicy(BaseModel):
    """Hard composition floors the selected corpus must satisfy to be a usable benchmark."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Coverage scoring only *prefers* variety; without a floor a pool dominated by one regime
    # yields a corpus that cannot discriminate between models.
    min_sequential_share: float = Field(default=0.0, ge=0.0, le=1.0)
    min_distinct_witness_lengths: int = Field(default=1, ge=1)
    # A corpus whose witnesses concentrate on a few components rewards memorizing those
    # components instead of reasoning about the grid.
    max_component_share: float = Field(default=1.0, gt=0.0, le=1.0)
    min_distinct_components: int = Field(default=1, ge=1)
    # Families absent from every witness are never exercised by the benchmark.
    required_action_families: tuple[str, ...] = ()
    min_action_family_share: float = Field(default=0.0, ge=0.0, le=1.0)
    policy_version: str = COMPOSITION_POLICY_VERSION


def describe_corpus_composition(
    selected: tuple[CorpusCandidate, ...],
) -> dict[str, Any]:
    """Summarize the composition dimensions a degenerate corpus would collapse."""
    regimes = Counter(candidate.resolution_regime for candidate in selected)
    total = len(selected)
    # A scenario counts once per distinct component or family it exercises, so a two-step
    # witness that moves the same generator twice does not inflate that generator's share.
    components = Counter(key for c in selected for key in set(c.witness_component_keys))
    families = Counter(family for c in selected for family in set(c.witness_action_families))
    return {
        "total": total,
        "resolution_regime": dict(sorted(regimes.items())),
        "sequential_share": (regimes["SEQUENTIAL"] / total) if total else 0.0,
        "witness_length": dict(sorted(Counter(c.witness_length for c in selected).items())),
        "witness_optimality": dict(sorted(Counter(c.witness_optimality for c in selected).items())),
        "distinct_operating_profiles": len({c.operating_profile_id for c in selected}),
        "distinct_pocket_anchors": len({c.pocket.anchor_bus for c in selected}),
        "witness_component_counts": dict(sorted(components.items())),
        "distinct_witness_components": len(components),
        "max_component_share": (max(components.values()) / total) if total and components else 0.0,
        "witness_action_family_counts": dict(sorted(families.items())),
        "witness_action_family_shares": {
            family: count / total for family, count in sorted(families.items())
        }
        if total
        else {},
    }


def verify_corpus_composition(
    selected: tuple[CorpusCandidate, ...],
    *,
    policy: CorpusCompositionPolicy,
) -> None:
    """Raise when the selected corpus misses a composition floor, instead of degrading silently."""
    composition = describe_corpus_composition(selected)
    failures: list[str] = []
    if composition["sequential_share"] < policy.min_sequential_share:
        failures.append(
            f"SEQUENTIAL share {composition['sequential_share']:.3f} is below the required "
            f"{policy.min_sequential_share:.3f}"
        )
    if len(composition["witness_length"]) < policy.min_distinct_witness_lengths:
        failures.append(
            f"only {len(composition['witness_length'])} distinct witness lengths, "
            f"{policy.min_distinct_witness_lengths} required"
        )
    if composition["max_component_share"] > policy.max_component_share:
        dominant = max(composition["witness_component_counts"].items(), key=lambda row: (row[1], row[0]))
        failures.append(
            f"component {dominant[0]} is the witness of {composition['max_component_share']:.3f} "
            f"of scenarios, above the allowed {policy.max_component_share:.3f}"
        )
    if composition["distinct_witness_components"] < policy.min_distinct_components:
        failures.append(
            f"only {composition['distinct_witness_components']} distinct witness components, "
            f"{policy.min_distinct_components} required"
        )
    for family in policy.required_action_families:
        share = composition["witness_action_family_shares"].get(family, 0.0)
        if share < policy.min_action_family_share:
            failures.append(
                f"action family {family} appears in {share:.3f} of witnesses, below the "
                f"required {policy.min_action_family_share:.3f}"
            )
    if failures:
        raise ValueError("selected corpus fails composition floors: " + "; ".join(failures))


def select_and_assign_corpus(
    candidates: tuple[CorpusCandidate, ...],
    *,
    similarity_threshold: float,
    memory_population_count: int,
    held_out_count: int,
    composition_policy: CorpusCompositionPolicy | None = None,
) -> tuple[CorpusCandidate, ...]:
    """Group the pool, select split-feasibly, then canonicalize selected-group IDs."""
    grouped_pool = assign_leakage_groups(
        candidates,
        similarity_threshold=similarity_threshold,
    )
    selected = select_diverse_candidates_for_split(
        grouped_pool,
        memory_population_count=memory_population_count,
        held_out_count=held_out_count,
        composition_policy=composition_policy,
    )
    canonical_selected = assign_leakage_groups(
        selected,
        similarity_threshold=similarity_threshold,
    )
    assigned = assign_memory_split(
        canonical_selected,
        memory_population_count=memory_population_count,
        held_out_count=held_out_count,
    )
    if composition_policy is not None:
        verify_corpus_composition(assigned, policy=composition_policy)
    return assigned


def _composition_priority(
    candidate: CorpusCandidate,
    *,
    policy: CorpusCompositionPolicy | None,
    target_count: int,
    component_counts: Counter[str],
    family_counts: Counter[str],
    regime_counts: Counter[str],
) -> tuple[int, int, int]:
    """Rank a candidate by the composition floors it still helps satisfy.

    Auditing composition only after selection lets a greedy pass fail on a pool that does
    contain a valid corpus, so the unmet quotas steer the choice while it is still open.
    """
    if policy is None:
        return (0, 0, 0)

    cap = policy.max_component_share * target_count
    keeps_cap_headroom = all(
        component_counts[key] + 1 <= cap for key in set(candidate.witness_component_keys)
    )
    family_floor = policy.min_action_family_share * target_count
    serves_missing_family = any(
        family_counts[family] < family_floor
        for family in set(candidate.witness_action_families)
        if family in policy.required_action_families
    )
    sequential_floor = policy.min_sequential_share * target_count
    serves_sequential_floor = (
        candidate.resolution_regime == "SEQUENTIAL"
        and regime_counts["SEQUENTIAL"] < sequential_floor
    )
    return (int(keeps_cap_headroom), int(serves_missing_family), int(serves_sequential_floor))


def _coverage_score(
    candidate: CorpusCandidate,
    *,
    selected: list[CorpusCandidate],
    seen_anchors: set[int],
    seen_profiles: set[str],
    seen_regimes: set[str],
    seen_lengths: set[int],
    seen_q_patterns: set[tuple[int, ...]],
    seen_depth_buckets: set[int],
    seen_leakage_groups: set[str],
) -> tuple[int, int, int, int, int, int, int, float]:
    maximum_similarity = max(
        (
            pocket_vector_similarity(candidate.pocket, item.pocket)
            for item in selected
        ),
        default=0.0,
    )
    return (
        int(
            candidate.leakage_group_id is None
            or candidate.leakage_group_id not in seen_leakage_groups
        ),
        int(candidate.pocket.anchor_bus not in seen_anchors),
        int(candidate.operating_profile_id not in seen_profiles),
        int(candidate.resolution_regime not in seen_regimes),
        int(candidate.witness_length not in seen_lengths),
        int(candidate.q_limited_gen_ids not in seen_q_patterns),
        int(_depth_bucket(candidate.target_relative_offset) not in seen_depth_buckets),
        -maximum_similarity,
    )


def _depth_bucket(relative_offset: float) -> int:
    return int(relative_offset * 100.0)


def _partition_group_capacities(
    groups: dict[str, tuple[CorpusCandidate, ...]],
    *,
    memory_population_count: int,
    held_out_count: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    held: list[str] = []
    memory: list[str] = []
    held_capacity = 0
    memory_capacity = 0
    for group_id in sorted(
        groups,
        key=lambda item: (len(groups[item]), item),
    ):
        group = groups[group_id]
        held_fill = held_capacity / held_out_count
        memory_fill = memory_capacity / memory_population_count
        if held_fill <= memory_fill:
            held.append(group_id)
            held_capacity += len(group)
        else:
            memory.append(group_id)
            memory_capacity += len(group)

    def move_until_feasible(
        source: list[str],
        target: list[str],
        *,
        source_capacity: int,
        target_capacity: int,
        source_required: int,
        target_required: int,
    ) -> tuple[int, int]:
        while target_capacity < target_required:
            movable = [
                group_id
                for group_id in source
                if source_capacity - len(groups[group_id]) >= source_required
            ]
            if not movable:
                raise ValueError(
                    "leakage-safe exact split is impossible; expand the valid candidate pool"
                )
            group_id = min(
                movable,
                key=lambda item: (len(groups[item]), item),
            )
            source.remove(group_id)
            target.append(group_id)
            size = len(groups[group_id])
            source_capacity -= size
            target_capacity += size
        return source_capacity, target_capacity

    memory_capacity, held_capacity = move_until_feasible(
        memory,
        held,
        source_capacity=memory_capacity,
        target_capacity=held_capacity,
        source_required=memory_population_count,
        target_required=held_out_count,
    )
    held_capacity, memory_capacity = move_until_feasible(
        held,
        memory,
        source_capacity=held_capacity,
        target_capacity=memory_capacity,
        source_required=held_out_count,
        target_required=memory_population_count,
    )
    return tuple(sorted(held)), tuple(sorted(memory))


def _leakage_group_id(families: set[str]) -> str:
    payload = "\n".join(sorted(families)).encode()
    return f"L-{hashlib.sha256(payload).hexdigest()[:20]}"
