# ABOUTME: Verifies deterministic impedance-distance load pockets and recipe hashing.
# ABOUTME: Covers localization, ordered load vectors, support gates, and near-duplicate removal.
from __future__ import annotations

from typing import Any

import pandapower as pp
import pytest

from restorebench.corpus.electrical_pockets import (
    PocketWeightingPolicy,
    build_pocket_recipe,
    generate_pocket_recipes,
    pocket_vector_similarity,
)


def _pocket_net() -> Any:
    net = pp.create_empty_network(sn_mva=100.0)
    buses = [pp.create_bus(net, vn_kv=110.0) for _ in range(5)]
    pp.create_ext_grid(
        net,
        bus=buses[0],
        vm_pu=1.0,
        min_p_mw=-100.0,
        max_p_mw=100.0,
        min_q_mvar=-100.0,
        max_q_mvar=100.0,
    )
    for left, right, x_ohm in (
        (0, 1, 1.0),
        (1, 2, 2.0),
        (2, 3, 4.0),
    ):
        pp.create_line_from_parameters(
            net,
            from_bus=buses[left],
            to_bus=buses[right],
            length_km=1.0,
            r_ohm_per_km=0.0,
            x_ohm_per_km=x_ohm,
            c_nf_per_km=0.0,
            max_i_ka=1.0,
        )
    pp.create_load(net, bus=buses[1], p_mw=10.0, q_mvar=4.0, index=3)
    pp.create_load(net, bus=buses[1], p_mw=5.0, q_mvar=2.0, index=8)
    pp.create_load(net, bus=buses[2], p_mw=20.0, q_mvar=8.0, index=11)
    pp.create_load(net, bus=buses[3], p_mw=30.0, q_mvar=12.0, index=15)
    return net


def _policy(**overrides: Any) -> PocketWeightingPolicy:
    payload = {
        "distance_scales_pu": (0.02, 0.04),
        "weight_cutoff": 0.05,
        "minimum_load_count": 2,
        "minimum_base_p_mw": 10.0,
        "minimum_base_abs_q_mvar": 1.0,
        "near_duplicate_cosine": 0.999,
    }
    payload.update(overrides)
    return PocketWeightingPolicy(**payload)


def test_pocket_recipe_is_localized_ordered_normalized_and_deterministic() -> None:
    net = _pocket_net()

    first = build_pocket_recipe(net, anchor_bus=1, distance_scale_pu=0.02, policy=_policy())
    second = build_pocket_recipe(net, anchor_bus=1, distance_scale_pu=0.02, policy=_policy())

    assert first == second
    assert first.distance_method == "IMPEDANCE_WEIGHTED_GRAPH_DISTANCE"
    assert [point.load_id for point in first.loads] == [3, 8, 11, 15]
    weights = {point.load_id: point.weight for point in first.loads}
    assert weights[3] == weights[8] == 1.0
    assert 0.0 < weights[11] < 1.0
    assert weights[15] <= weights[11]
    assert len(first.vector_hash) == 64
    assert [point.base_p_mw for point in first.loads] == [10.0, 5.0, 20.0, 30.0]


def test_pocket_rejects_disconnected_anchor_and_insufficient_support() -> None:
    net = _pocket_net()

    with pytest.raises(ValueError, match="anchor"):
        build_pocket_recipe(net, anchor_bus=4, distance_scale_pu=0.02, policy=_policy())

    with pytest.raises(ValueError, match="load count"):
        build_pocket_recipe(
            net,
            anchor_bus=3,
            distance_scale_pu=1e-6,
            policy=_policy(minimum_load_count=2),
        )


def test_generate_pockets_removes_exact_and_near_duplicate_vectors() -> None:
    net = _pocket_net()
    recipes = generate_pocket_recipes(
        net,
        anchor_buses=(1, 2, 3),
        policy=_policy(),
    )

    assert recipes
    assert [recipe.vector_hash for recipe in recipes] == list(
        dict.fromkeys(recipe.vector_hash for recipe in recipes)
    )
    for left_index, left in enumerate(recipes):
        for right in recipes[left_index + 1 :]:
            assert pocket_vector_similarity(left, right) < _policy().near_duplicate_cosine
