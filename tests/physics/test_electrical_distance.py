# ABOUTME: Verifies transparent common-base impedance-weighted bus-graph distances.
# ABOUTME: Covers lines, transformers, parallel edges, numerical floors, and unreachable buses.
from __future__ import annotations

import math
from typing import Any

import pandapower as pp
import pytest
from pydantic import ValidationError

from restorebench.physics.electrical_distance import impedance_weighted_graph_distances
from restorebench.schemas.physics import ElectricalDistancePolicy


def _distance_net() -> Any:
    net = pp.create_empty_network(sn_mva=100.0)
    buses = [pp.create_bus(net, vn_kv=110.0) for _ in range(5)]
    pp.create_line_from_parameters(
        net,
        from_bus=buses[0],
        to_bus=buses[1],
        length_km=1.0,
        r_ohm_per_km=3.0,
        x_ohm_per_km=4.0,
        c_nf_per_km=0.0,
        max_i_ka=1.0,
    )
    pp.create_line_from_parameters(
        net,
        from_bus=buses[0],
        to_bus=buses[1],
        length_km=1.0,
        r_ohm_per_km=6.0,
        x_ohm_per_km=8.0,
        c_nf_per_km=0.0,
        max_i_ka=1.0,
    )
    pp.create_transformer_from_parameters(
        net,
        hv_bus=buses[1],
        lv_bus=buses[2],
        sn_mva=50.0,
        vn_hv_kv=110.0,
        vn_lv_kv=110.0,
        vk_percent=10.0,
        vkr_percent=1.0,
        pfe_kw=0.0,
        i0_percent=0.0,
    )
    pp.create_line_from_parameters(
        net,
        from_bus=buses[2],
        to_bus=buses[3],
        length_km=1.0,
        r_ohm_per_km=0.0,
        x_ohm_per_km=0.0,
        c_nf_per_km=0.0,
        max_i_ka=1.0,
    )
    return net


def test_impedance_distance_uses_common_base_and_minimum_parallel_edge() -> None:
    net = _distance_net()
    policy = ElectricalDistancePolicy(common_mva_base=100.0, minimum_edge_weight_pu=1e-6)

    result = impedance_weighted_graph_distances(net, source_buses=(0,), policy=policy)

    line_weight = 5.0 / ((110.0**2) / 100.0)
    transformer_weight = 0.10 * (100.0 / 50.0)
    assert result.method == "IMPEDANCE_WEIGHTED_GRAPH_DISTANCE"
    assert result.distances_pu[0] == 0.0
    assert result.distances_pu[1] == pytest.approx(line_weight)
    assert result.distances_pu[2] == pytest.approx(line_weight + transformer_weight)
    assert result.distances_pu[3] == pytest.approx(line_weight + transformer_weight + 1e-6)
    assert result.distances_pu[4] is None
    assert result.unreachable_bus_ids == (4,)


def test_impedance_distance_rejects_invalid_sources_and_nonfinite_weights() -> None:
    net = _distance_net()
    policy = ElectricalDistancePolicy(common_mva_base=100.0, minimum_edge_weight_pu=1e-6)

    with pytest.raises(ValueError, match="source bus"):
        impedance_weighted_graph_distances(net, source_buses=(99,), policy=policy)

    net.line.at[0, "x_ohm_per_km"] = math.nan
    with pytest.raises(ValueError, match="finite"):
        impedance_weighted_graph_distances(net, source_buses=(0,), policy=policy)

    with pytest.raises(ValidationError):
        ElectricalDistancePolicy(
            common_mva_base=float("inf"),
            minimum_edge_weight_pu=1e-6,
        )
