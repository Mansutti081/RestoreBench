# ABOUTME: Extracts read-only grid topology summaries from pandapower nets.
# ABOUTME: Performs soft action applicability checks before sandbox mutation.
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from restorebench.physics.actions import GeneratorQStatus, get_qv_action_applicability
from restorebench.schemas import (
    Action,
    ApplicabilityResult,
    BusInfo,
    ExtGridInfo,
    GenInfo,
    LineInfo,
    LoadInfo,
    ShuntInfo,
    TopologySummary,
    TrafoInfo,
)
from restorebench.tools import sandbox as SandboxServer


def get_grid_topology(
    net_or_sandbox: Any,
    *,
    saturated_gens: frozenset[int] | None = None,
    q_context: Mapping[int, GeneratorQStatus] | None = None,
) -> TopologySummary:
    net = SandboxServer.resolve_net(net_or_sandbox)
    if q_context is not None and saturated_gens:
        raise ValueError("pass q_context or saturated_gens, not both")
    resolved_context = q_context
    if resolved_context is None and saturated_gens is not None:
        resolved_context = {
            int(gen_id): "Q_LIMITED_UPPER" if gen_id in saturated_gens else "PV_CONTROLLABLE"
            for gen_id in net.gen.index
        }

    buses = [
        BusInfo(
            bus_id=int(bus_id),
            name=_optional_str(row["name"]),
            vn_kv=float(row["vn_kv"]),
            in_service=_in_service(row),
        )
        for bus_id, row in net.bus.sort_index().iterrows()
    ]
    lines = [
        LineInfo(
            line_id=int(line_id),
            from_bus=int(row["from_bus"]),
            to_bus=int(row["to_bus"]),
            in_service=_in_service(row),
        )
        for line_id, row in net.line.sort_index().iterrows()
    ]
    trafos = [
        TrafoInfo(
            trafo_id=int(trafo_id),
            hv_bus=int(row["hv_bus"]),
            lv_bus=int(row["lv_bus"]),
            tap_pos=_optional_int(row["tap_pos"]),
            tap_min=_optional_int(row["tap_min"]),
            tap_max=_optional_int(row["tap_max"]),
            in_service=_in_service(row),
        )
        for trafo_id, row in net.trafo.sort_index().iterrows()
    ]
    gens = [
        GenInfo(
            gen_id=int(gen_id),
            bus=int(row["bus"]),
            p_mw=float(row["p_mw"]),
            vm_pu=float(row["vm_pu"]),
            min_p_mw=float(row["min_p_mw"]),
            max_p_mw=float(row["max_p_mw"]),
            min_q_mvar=float(row["min_q_mvar"]),
            max_q_mvar=float(row["max_q_mvar"]),
            dispatchable=float(row["min_p_mw"]) < float(row["max_p_mw"]),
            voltage_control_status=_voltage_control_status(
                int(gen_id),
                row,
                resolved_context,
            ),
            in_service=_in_service(row),
        )
        for gen_id, row in net.gen.sort_index().iterrows()
    ]
    ext_grids = [
        ExtGridInfo(
            ext_grid_id=int(ext_grid_id),
            bus=int(row["bus"]),
            min_p_mw=float(row["min_p_mw"]),
            max_p_mw=float(row["max_p_mw"]),
            min_q_mvar=float(row["min_q_mvar"]),
            max_q_mvar=float(row["max_q_mvar"]),
            in_service=_in_service(row),
        )
        for ext_grid_id, row in net.ext_grid.sort_index().iterrows()
    ]
    loads = [
        LoadInfo(
            load_id=int(load_id),
            bus=int(row["bus"]),
            p_mw=float(row["p_mw"]),
            q_mvar=float(row["q_mvar"]),
            in_service=_in_service(row),
        )
        for load_id, row in net.load.sort_index().iterrows()
    ]
    shunts = [
        ShuntInfo(
            shunt_id=int(shunt_id),
            bus=int(row["bus"]),
            q_mvar=float(row["q_mvar"]),
            type="capacitor" if float(row["q_mvar"]) < 0 else "reactor",
            step=int(row["step"]),
            max_step=int(row["max_step"]),
            in_service=_in_service(row),
        )
        for shunt_id, row in net.shunt.sort_index().iterrows()
    ]

    return TopologySummary(
        n_buses=len(net.bus),
        n_lines=len(net.line),
        n_trafos=len(net.trafo),
        n_gens=len(net.gen),
        n_ext_grids=len(net.ext_grid),
        n_loads=len(net.load),
        n_shunts=len(net.shunt),
        buses=buses,
        lines=lines,
        trafos=trafos,
        gens=gens,
        ext_grids=ext_grids,
        loads=loads,
        shunts=shunts,
        slack_bus=int(net.ext_grid.bus.iloc[0]),
    )


def get_action_applicability(
    net_or_sandbox: Any,
    action: Action,
    *,
    saturated_gens: frozenset[int] = frozenset(),
    q_context: Mapping[int, GeneratorQStatus] | None = None,
) -> ApplicabilityResult:
    net = SandboxServer.resolve_net(net_or_sandbox)
    if q_context is not None and saturated_gens:
        raise ValueError("pass q_context or saturated_gens, not both")
    resolved_context = q_context or {gen_id: "Q_LIMITED_UPPER" for gen_id in saturated_gens}
    return get_qv_action_applicability(net, action, resolved_context)


def _optional_str(value: Any) -> str | None:
    if pd.isna(value):
        return None
    return str(value)


def _optional_int(value: Any) -> int | None:
    if pd.isna(value):
        return None
    return int(value)


def _in_service(row: Any) -> bool:
    if "in_service" not in row.index:
        return True
    return bool(row["in_service"])


def _voltage_control_status(
    gen_id: int,
    row: Any,
    q_context: Mapping[int, GeneratorQStatus] | None,
) -> GeneratorQStatus | None:
    if q_context is None or not _in_service(row):
        return None
    return q_context.get(gen_id, "UNKNOWN")
