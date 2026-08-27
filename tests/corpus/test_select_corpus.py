# ABOUTME: Verifies deterministic diversity selection and leakage-safe exact split assignment.
# ABOUTME: Ensures near-duplicate pockets and all depths of one family remain on one split side.
from __future__ import annotations

import random

import pytest

from restorebench.schemas.dataset import PocketRecipe
from restorebench.corpus.select_corpus import (
    CorpusCandidate,
    assign_leakage_groups,
    assign_memory_split,
    select_and_assign_corpus,
    select_diverse_candidates,
    select_diverse_candidates_for_split,
)


def _pocket(anchor: int, weights: tuple[float, float]) -> PocketRecipe:
    return PocketRecipe.model_validate(
        {
            "anchor_bus": anchor,
            "distance_method": "IMPEDANCE_WEIGHTED_GRAPH_DISTANCE",
            "loads": [
                {
                    "load_id": 0,
                    "base_p_mw": 10.0,
                    "base_q_mvar": 4.0,
                    "weight": weights[0],
                },
                {
                    "load_id": 1,
                    "base_p_mw": 10.0,
                    "base_q_mvar": 4.0,
                    "weight": weights[1],
                },
            ],
            "vector_hash": f"{anchor:064x}",
            "policy_version": "pocket-v1",
        }
    )


def _candidate(
    number: int,
    *,
    family: str,
    anchor: int,
    weights: tuple[float, float],
) -> CorpusCandidate:
    return CorpusCandidate(
        candidate_id=f"C{number:04d}",
        scenario_family_id=family,
        operating_profile_id=f"OP-{number % 3}",
        pocket=_pocket(anchor, weights),
        resolution_regime="DIRECT" if number % 2 else "SEQUENTIAL",
        witness_length=1 if number % 2 else 2 + number % 4,
        witness_optimality="EXACT_MINIMUM",
        target_relative_offset=0.01 * (1 + number % 5),
        q_limited_gen_ids=(number % 7,),
        witness_component_keys=(f"GEN:{number}",),
        witness_action_families=("GEN_V_SETPOINT",),
    )


def test_near_duplicate_vectors_and_family_depths_share_leakage_group() -> None:
    candidates = (
        _candidate(1, family="F-A", anchor=10, weights=(1.0, 0.50)),
        _candidate(2, family="F-A", anchor=10, weights=(1.0, 0.50)),
        _candidate(3, family="F-B", anchor=11, weights=(1.0, 0.50001)),
        _candidate(4, family="F-C", anchor=20, weights=(0.0, 1.0)),
    )

    grouped = assign_leakage_groups(candidates, similarity_threshold=0.999999)
    groups = {candidate.candidate_id: candidate.leakage_group_id for candidate in grouped}

    assert groups["C0001"] == groups["C0002"] == groups["C0003"]
    assert groups["C0004"] != groups["C0001"]


def test_selection_is_input_order_independent_and_not_first_n() -> None:
    candidates = tuple(
        _candidate(
            number,
            family=f"F-{number // 2}",
            anchor=number % 6,
            weights=(1.0, (number % 5) / 5),
        )
        for number in range(20)
    )
    shuffled = list(candidates)
    random.Random(42).shuffle(shuffled)

    first = select_diverse_candidates(candidates, target_count=10)
    second = select_diverse_candidates(tuple(shuffled), target_count=10)

    assert [item.candidate_id for item in first] == [
        item.candidate_id for item in second
    ]
    assert {item.candidate_id for item in first} != {
        item.candidate_id for item in candidates[:10]
    }


def test_split_assignment_keeps_groups_whole_and_hits_exact_counts() -> None:
    candidates = tuple(
        _candidate(
            number,
            family=f"F-{number // 2}",
            anchor=number,
            weights=(1.0, 0.0),
        ).model_copy(update={"leakage_group_id": f"L-{number // 2}"})
        for number in range(20)
    )

    assigned = assign_memory_split(
        candidates,
        memory_population_count=14,
        held_out_count=6,
    )

    assert sum(item.memory_split == "memory_population" for item in assigned) == 14
    assert sum(item.memory_split == "held_out" for item in assigned) == 6
    by_group: dict[str, set[str]] = {}
    for item in assigned:
        by_group.setdefault(item.leakage_group_id or "", set()).add(
            item.memory_split
        )
    assert all(len(splits) == 1 for splits in by_group.values())


def test_split_rejects_pool_without_exact_grouped_partition() -> None:
    candidates = tuple(
        _candidate(
            number,
            family="F-A" if number < 3 else "F-B",
            anchor=number,
            weights=(1.0, 0.0),
        ).model_copy(
            update={"leakage_group_id": "L-A" if number < 3 else "L-B"}
        )
        for number in range(5)
    )

    with pytest.raises(ValueError, match="expand"):
        assign_memory_split(
            candidates,
            memory_population_count=4,
            held_out_count=1,
        )


def test_split_aware_selection_reserves_one_complete_group_at_exact_size() -> None:
    candidates = tuple(
        _candidate(
            number,
            family=f"F-{number // 4}",
            anchor=number,
            weights=(1.0, (number % 4) / 4),
        ).model_copy(
            update={"leakage_group_id": f"L-{number // 4}"}
        )
        for number in range(12)
    )

    selected = select_diverse_candidates_for_split(
        candidates,
        memory_population_count=8,
        held_out_count=2,
    )
    assigned = assign_memory_split(
        selected,
        memory_population_count=8,
        held_out_count=2,
    )

    assert len(selected) == 10
    assert sum(item.memory_split == "held_out" for item in assigned) == 2
    held_groups = {
        item.leakage_group_id
        for item in assigned
        if item.memory_split == "held_out"
    }
    assert len(held_groups) == 1


def test_end_to_end_selection_rehashes_groups_from_selected_families() -> None:
    candidates = tuple(
        _candidate(
            number,
            family=f"F-{number}",
            anchor=number // 4,
            weights=(1.0, 0.5 if number < 4 else 0.0),
        )
        for number in range(12)
    )

    assigned = select_and_assign_corpus(
        candidates,
        similarity_threshold=0.999,
        memory_population_count=8,
        held_out_count=2,
    )
    independently_regrouped = assign_leakage_groups(
        tuple(
            candidate.model_copy(update={"leakage_group_id": None, "memory_split": None})
            for candidate in assigned
        ),
        similarity_threshold=0.999,
    )

    assert {
        candidate.candidate_id: candidate.leakage_group_id
        for candidate in assigned
    } == {
        candidate.candidate_id: candidate.leakage_group_id
        for candidate in independently_regrouped
    }


def test_split_aware_selection_distributes_a_large_held_out_side_across_groups() -> None:
    candidates = tuple(
        _candidate(
            number,
            family=f"F-{number // 4}",
            anchor=number // 4,
            weights=(1.0, (number % 4) / 4),
        ).model_copy(
            update={"leakage_group_id": f"L-{number // 4:02d}"}
        )
        for number in range(48)
    )

    selected = select_diverse_candidates_for_split(
        candidates,
        memory_population_count=30,
        held_out_count=10,
    )
    assigned = assign_memory_split(
        selected,
        memory_population_count=30,
        held_out_count=10,
    )

    held_groups = {
        item.leakage_group_id
        for item in assigned
        if item.memory_split == "held_out"
    }
    assert len(held_groups) >= 3


def test_composition_floor_rejects_a_corpus_without_sequential_scenarios() -> None:
    # Coverage scoring only prefers variety; without a floor a DIRECT-only pool yields a corpus
    # that cannot discriminate between models, and the benchmark measures nothing.
    from restorebench.corpus.select_corpus import (
        CorpusCompositionPolicy,
        describe_corpus_composition,
        verify_corpus_composition,
    )

    direct_only = tuple(
        _candidate(number, family=f"F-{number}", anchor=10 + number, weights=(1.0, 0.5))
        for number in (1, 3, 5, 7)
    )
    composition = describe_corpus_composition(direct_only)

    assert composition["sequential_share"] == 0.0

    with pytest.raises(ValueError, match="SEQUENTIAL share"):
        verify_corpus_composition(
            direct_only,
            policy=CorpusCompositionPolicy(min_sequential_share=0.4),
        )


def test_composition_floor_accepts_a_corpus_that_meets_the_declared_share() -> None:
    from restorebench.corpus.select_corpus import (
        CorpusCompositionPolicy,
        verify_corpus_composition,
    )

    mixed = (
        _candidate(1, family="F-A", anchor=10, weights=(1.0, 0.5)),
        _candidate(2, family="F-B", anchor=11, weights=(1.0, 0.4)),
    )

    verify_corpus_composition(mixed, policy=CorpusCompositionPolicy(min_sequential_share=0.5))


def _witness_candidate(number: int, *, component: str, family: str) -> CorpusCandidate:
    return CorpusCandidate(
        candidate_id=f"W{number:04d}",
        scenario_family_id=f"F-{number}",
        operating_profile_id=f"OP-{number}",
        pocket=_pocket(10 + number, (1.0, 0.5)),
        resolution_regime="DIRECT",
        witness_length=1,
        witness_optimality="EXACT_MINIMUM",
        target_relative_offset=0.02,
        q_limited_gen_ids=(number,),
        witness_component_keys=(component,),
        witness_action_families=(family,),
    )


def test_composition_floor_rejects_a_corpus_concentrated_on_one_component() -> None:
    # A corpus whose witnesses nearly all move one generator rewards memorizing that generator
    # instead of reasoning about the grid.
    from restorebench.corpus.select_corpus import (
        CorpusCompositionPolicy,
        describe_corpus_composition,
        verify_corpus_composition,
    )

    concentrated = tuple(
        _witness_candidate(
            number,
            component="GEN:42" if number < 7 else f"GEN:{number}",
            family="GEN_V_SETPOINT",
        )
        for number in range(10)
    )
    composition = describe_corpus_composition(concentrated)

    assert composition["max_component_share"] == pytest.approx(0.7)

    with pytest.raises(ValueError, match="GEN:42"):
        verify_corpus_composition(
            concentrated,
            policy=CorpusCompositionPolicy(max_component_share=0.1),
        )


def test_composition_floor_requires_every_declared_action_family() -> None:
    from restorebench.corpus.select_corpus import (
        CorpusCompositionPolicy,
        verify_corpus_composition,
    )

    without_shunts = tuple(
        _witness_candidate(number, component=f"GEN:{number}", family="GEN_V_SETPOINT")
        for number in range(4)
    )

    with pytest.raises(ValueError, match="SHUNT_STEP"):
        verify_corpus_composition(
            without_shunts,
            policy=CorpusCompositionPolicy(
                required_action_families=("GEN_V_SETPOINT", "SHUNT_STEP"),
                min_action_family_share=0.1,
            ),
        )


def test_multi_step_witness_counts_a_repeated_component_once() -> None:
    # A two-step witness moving the same generator twice must not inflate its share.
    from restorebench.corpus.select_corpus import describe_corpus_composition

    repeated = CorpusCandidate(
        candidate_id="W9999",
        scenario_family_id="F-R",
        operating_profile_id="OP-R",
        pocket=_pocket(99, (1.0, 0.5)),
        resolution_regime="SEQUENTIAL",
        witness_length=2,
        witness_optimality="EXACT_MINIMUM",
        target_relative_offset=0.02,
        q_limited_gen_ids=(1,),
        witness_component_keys=("GEN:7", "GEN:7"),
        witness_action_families=("GEN_V_SETPOINT", "GEN_V_SETPOINT"),
    )

    composition = describe_corpus_composition((repeated,))

    assert composition["witness_component_counts"] == {"GEN:7": 1}
    assert composition["max_component_share"] == pytest.approx(1.0)


def test_constraint_aware_selection_reaches_a_valid_corpus_the_audit_alone_would_miss() -> None:
    # Auditing composition only after selection lets the greedy pass fail on a pool that does
    # contain a valid corpus; the unmet quotas must steer the choice while it is still open.
    from restorebench.corpus.select_corpus import (
        CorpusCompositionPolicy,
        select_diverse_candidates,
        verify_corpus_composition,
    )

    def candidate(number: int, *, regime: str, component: str) -> CorpusCandidate:
        return CorpusCandidate(
            candidate_id=f"S{number:04d}",
            scenario_family_id=f"F-{number}",
            operating_profile_id="OP-0",
            pocket=_pocket(10 + number, (1.0, 0.5)),
            resolution_regime=regime,
            witness_length=1 if regime == "DIRECT" else 2,
            witness_optimality="EXACT_MINIMUM",
            target_relative_offset=0.02,
            q_limited_gen_ids=(number,),
            witness_component_keys=(component,),
            witness_action_families=("GEN_V_SETPOINT",),
        )

    # Low IDs are all DIRECT on one component: a quota-blind greedy pass takes them first.
    pool = tuple(
        candidate(number, regime="DIRECT", component="GEN:42") for number in range(6)
    ) + tuple(
        candidate(number, regime="SEQUENTIAL", component=f"GEN:{number}")
        for number in range(6, 12)
    )
    policy = CorpusCompositionPolicy(min_sequential_share=0.5, max_component_share=0.5)

    selected = select_diverse_candidates(pool, target_count=4, composition_policy=policy)

    verify_corpus_composition(selected, policy=policy)

