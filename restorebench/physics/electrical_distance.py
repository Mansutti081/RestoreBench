# ABOUTME: Computes common-base impedance-weighted shortest-path distances on the in-service bus graph.
# ABOUTME: Exposes a transparent proximity proxy with explicit unreachable buses and numerical flooring.
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import networkx as nx

from restorebench.schemas.physics import ElectricalDistancePolicy, ElectricalDistanceResult


def impedance_weighted_graph_distances(
    net: Any,
    *,
    source_buses: Sequence[int],
    policy: ElectricalDistancePolicy,
) -> ElectricalDistanceResult:
    """Return minimum common-base series-impedance path lengths from one or more source buses."""
    sources = tuple(sorted({int(bus_id) for bus_id in source_buses}))
    if not sources:
        raise ValueError("at least one source bus is required")

    graph = nx.Graph()
    in_service_buses = {
        int(bus_id) for bus_id, row in net.bus.sort_index().iterrows() if bool(row.get("in_service", True))
    }
    graph.add_nodes_from(sorted(in_service_buses))
    for source in sources:
        if source not in in_service_buses:
            raise ValueError(f"source bus {source} does not exist or is out of service")

    open_lines, open_trafos = _open_branch_ids(net)
    for line_id, row in net.line.sort_index().iterrows():
        if not bool(row.get("in_service", True)) or int(line_id) in open_lines:
            continue
        from_bus = int(row["from_bus"])
        to_bus = int(row["to_bus"])
        if from_bus not in in_service_buses or to_bus not in in_service_buses:
            continue
        weight = _line_weight(net, row, policy)
        _add_minimum_edge(graph, from_bus, to_bus, weight)

    for trafo_id, row in net.trafo.sort_index().iterrows():
        if not bool(row.get("in_service", True)) or int(trafo_id) in open_trafos:
            continue
        hv_bus = int(row["hv_bus"])
        lv_bus = int(row["lv_bus"])
        if hv_bus not in in_service_buses or lv_bus not in in_service_buses:
            continue
        weight = _trafo_weight(row, policy)
        _add_minimum_edge(graph, hv_bus, lv_bus, weight)

    _add_closed_bus_switches(net, graph, policy.minimum_edge_weight_pu)
    lengths = nx.multi_source_dijkstra_path_length(graph, sources=sources, weight="weight")
    distances = {bus_id: float(lengths[bus_id]) if bus_id in lengths else None for bus_id in sorted(in_service_buses)}
    unreachable = tuple(bus_id for bus_id, distance in distances.items() if distance is None)
    return ElectricalDistanceResult(
        method="IMPEDANCE_WEIGHTED_GRAPH_DISTANCE",
        source_bus_ids=sources,
        distances_pu=distances,
        unreachable_bus_ids=unreachable,
        policy_version=policy.policy_version,
    )


def _line_weight(net: Any, row: Any, policy: ElectricalDistancePolicy) -> float:
    values = (
        float(row["length_km"]),
        float(row["r_ohm_per_km"]),
        float(row["x_ohm_per_km"]),
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("line length and series r/x must be finite")
    length, r_per_km, x_per_km = values
    if length < 0.0:
        raise ValueError("line length must be non-negative")
    voltage_kv = float(net.bus.at[int(row["from_bus"]), "vn_kv"])
    if not math.isfinite(voltage_kv) or voltage_kv <= 0.0:
        raise ValueError("connected line voltage base must be finite and positive")
    z_ohm = math.hypot(r_per_km * length, x_per_km * length)
    z_base_ohm = voltage_kv**2 / policy.common_mva_base
    return max(z_ohm / z_base_ohm, policy.minimum_edge_weight_pu)


def _trafo_weight(row: Any, policy: ElectricalDistancePolicy) -> float:
    vk_percent = float(row["vk_percent"])
    sn_mva = float(row["sn_mva"])
    if not math.isfinite(vk_percent) or not math.isfinite(sn_mva):
        raise ValueError("transformer short-circuit impedance and rating must be finite")
    if vk_percent < 0.0 or sn_mva <= 0.0:
        raise ValueError("transformer short-circuit impedance must be non-negative and rating positive")
    common_base_pu = (vk_percent / 100.0) * (policy.common_mva_base / sn_mva)
    return max(common_base_pu, policy.minimum_edge_weight_pu)


def _add_minimum_edge(graph: nx.Graph, left: int, right: int, weight: float) -> None:
    if graph.has_edge(left, right):
        graph[left][right]["weight"] = min(float(graph[left][right]["weight"]), weight)
    else:
        graph.add_edge(left, right, weight=weight)


def _open_branch_ids(net: Any) -> tuple[set[int], set[int]]:
    open_lines: set[int] = set()
    open_trafos: set[int] = set()
    if not hasattr(net, "switch"):
        return open_lines, open_trafos
    for _, row in net.switch.iterrows():
        if bool(row.get("closed", True)):
            continue
        element_type = str(row["et"])
        if element_type == "l":
            open_lines.add(int(row["element"]))
        elif element_type == "t":
            open_trafos.add(int(row["element"]))
    return open_lines, open_trafos


def _add_closed_bus_switches(net: Any, graph: nx.Graph, floor: float) -> None:
    if not hasattr(net, "switch"):
        return
    for _, row in net.switch.iterrows():
        if str(row["et"]) != "b" or not bool(row.get("closed", True)):
            continue
        left = int(row["bus"])
        right = int(row["element"])
        if left in graph and right in graph:
            _add_minimum_edge(graph, left, right, floor)
