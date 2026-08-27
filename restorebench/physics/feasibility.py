# ABOUTME: Evaluates only valid convergent states for electrical limits, connectivity, and voltage envelopes.
# ABOUTME: Produces directional generator-Q evidence and explicit external-grid feasibility.
from __future__ import annotations

import math
from typing import Any

import networkx as nx
from pandapower.topology import create_nxgraph

from restorebench.physics.policies import (
    FEASIBILITY_POLICY_VERSION,
    HARD_VM_MAX_PU,
    HARD_VM_MIN_PU,
    Q_LIMIT_EPS_MVAR,
    Q_LIMIT_STATUS_POLICY_VERSION,
    RUNTIME_VM_MAX_PU,
    RUNTIME_VM_MIN_PU,
)
from restorebench.schemas.physics import (
    FeasibilityFailureReason,
    GeneratorQState,
    QLimitEvidence,
    SlackResult,
    SolvedFeasibility,
    VoltageEnvelope,
)


def evaluate_solved_feasibility(net: Any) -> SolvedFeasibility:
    """Evaluate hard electrical feasibility from a valid constrained AC solution."""
    _require_valid_convergent_state(net)
    failures: list[FeasibilityFailureReason] = []

    gen_p_ok = True
    for gen_id, row in net.gen.sort_index().iterrows():
        if not bool(row.get("in_service", True)):
            continue
        p_mw = float(row["p_mw"])
        min_p = float(row["min_p_mw"])
        max_p = float(row["max_p_mw"])
        if not _within_declared(p_mw, min_p, max_p):
            gen_p_ok = False
            failures.append(
                FeasibilityFailureReason(
                    code="GEN_P_LIMIT",
                    element_id=int(gen_id),
                    detail=f"generator P={p_mw:.12g} outside [{min_p:.12g}, {max_p:.12g}] MW",
                )
            )

    q_status = _generator_q_states(net)
    gen_q_ok = True
    for item in q_status:
        if item.q_mvar is None or not _within_declared(item.q_mvar, item.min_q_mvar, item.max_q_mvar):
            gen_q_ok = False
            failures.append(
                FeasibilityFailureReason(
                    code="GEN_Q_LIMIT",
                    element_id=item.gen_id,
                    detail=(f"generator Q={item.q_mvar} outside [{item.min_q_mvar:.12g}, {item.max_q_mvar:.12g}] MVAr"),
                )
            )

    slack_results: list[SlackResult] = []
    slack_ok = True
    for ext_grid_id, row in net.ext_grid.sort_index().iterrows():
        if not bool(row.get("in_service", True)):
            continue
        if ext_grid_id not in net.res_ext_grid.index:
            raise ValueError("valid convergent state is missing external-grid results")
        p_mw = float(net.res_ext_grid.at[ext_grid_id, "p_mw"])
        q_mvar = float(net.res_ext_grid.at[ext_grid_id, "q_mvar"])
        p_ok = _within_declared(p_mw, row.get("min_p_mw"), row.get("max_p_mw"))
        q_ok = _within_declared(q_mvar, row.get("min_q_mvar"), row.get("max_q_mvar"))
        slack_results.append(
            SlackResult(
                ext_grid_id=int(ext_grid_id),
                p_mw=p_mw,
                q_mvar=q_mvar,
                p_within_limits=p_ok,
                q_within_limits=q_ok,
            )
        )
        if not p_ok:
            slack_ok = False
            failures.append(
                FeasibilityFailureReason(
                    code="EXT_GRID_P_LIMIT",
                    element_id=int(ext_grid_id),
                    detail=f"external-grid P={p_mw:.12g} violates declared bounds",
                )
            )
        if not q_ok:
            slack_ok = False
            failures.append(
                FeasibilityFailureReason(
                    code="EXT_GRID_Q_LIMIT",
                    element_id=int(ext_grid_id),
                    detail=f"external-grid Q={q_mvar:.12g} violates declared bounds",
                )
            )

    connected, loads_energized = _connectivity(net)
    if not connected:
        failures.append(
            FeasibilityFailureReason(
                code="DISCONNECTED",
                detail="not every in-service bus is connected to an in-service external grid",
            )
        )
    if not loads_energized:
        failures.append(
            FeasibilityFailureReason(
                code="LOAD_UNENERGIZED",
                detail="at least one in-service load is not connected to an in-service external grid",
            )
        )

    voltage = _voltage_envelope(net)
    if not voltage.hard_envelope_ok:
        failures.append(
            FeasibilityFailureReason(
                code="HARD_VOLTAGE_ENVELOPE",
                detail=(
                    f"bus voltage range [{voltage.min_vm_pu:.12g}, {voltage.max_vm_pu:.12g}] "
                    f"outside [{HARD_VM_MIN_PU}, {HARD_VM_MAX_PU}] pu"
                ),
            )
        )

    feasible = gen_p_ok and gen_q_ok and slack_ok and connected and loads_energized and voltage.hard_envelope_ok
    q_limited_ids = tuple(item.gen_id for item in q_status if item.status in {"Q_LIMITED_UPPER", "Q_LIMITED_LOWER"})
    return SolvedFeasibility(
        feasible=feasible,
        generator_p_within_limits=gen_p_ok,
        generator_q_within_limits=gen_q_ok,
        external_grid_within_limits=slack_ok,
        connected=connected,
        loads_energized=loads_energized,
        voltage=voltage,
        generator_q_status=q_status,
        slack_results=tuple(slack_results),
        q_limited_gen_ids=q_limited_ids,
        failure_reasons=tuple(failures),
        policy_version=FEASIBILITY_POLICY_VERSION,
    )


def satisfies_non_voltage_constraints(
    feasibility: SolvedFeasibility | None,
) -> bool:
    """Return whether a solved state passes every hard check except voltage."""
    return feasibility is not None and (
        feasibility.generator_p_within_limits
        and feasibility.generator_q_within_limits
        and feasibility.external_grid_within_limits
        and feasibility.connected
        and feasibility.loads_energized
    )


def compare_q_limit_evidence(
    net: Any,
    *,
    reference_net: Any,
    reference_name: str,
) -> QLimitEvidence:
    """Compare directional Q-limited IDs between two named valid constrained solutions."""
    if not reference_name.strip():
        raise ValueError("reference_name must be non-empty")
    current = evaluate_solved_feasibility(net)
    reference = evaluate_solved_feasibility(reference_net)
    reference_limited = set(reference.q_limited_gen_ids)
    newly_limited = tuple(gen_id for gen_id in current.q_limited_gen_ids if gen_id not in reference_limited)
    return QLimitEvidence(
        reference_name=reference_name,
        generator_q_status=current.generator_q_status,
        q_limited_gen_ids=current.q_limited_gen_ids,
        newly_q_limited_gen_ids=newly_limited,
        policy_version=Q_LIMIT_STATUS_POLICY_VERSION,
    )


def _require_valid_convergent_state(net: Any) -> None:
    if not bool(getattr(net, "converged", False)):
        raise ValueError("Q and feasibility evidence requires a valid convergent constrained solution")
    for result_name, value_column in (
        ("res_bus", "vm_pu"),
        ("res_gen", "q_mvar"),
        ("res_ext_grid", "q_mvar"),
    ):
        table = getattr(net, result_name, None)
        if table is None or value_column not in table:
            raise ValueError(f"Q and feasibility evidence requires a valid convergent state with {result_name}")


def _generator_q_states(net: Any) -> tuple[GeneratorQState, ...]:
    statuses: list[GeneratorQState] = []
    for gen_id, row in net.gen.sort_index().iterrows():
        min_q = float(row["min_q_mvar"])
        max_q = float(row["max_q_mvar"])
        if not bool(row.get("in_service", True)) or gen_id not in net.res_gen.index:
            statuses.append(
                GeneratorQState(
                    gen_id=int(gen_id),
                    status="UNKNOWN",
                    q_mvar=None,
                    min_q_mvar=min_q,
                    max_q_mvar=max_q,
                    lower_headroom_mvar=None,
                    upper_headroom_mvar=None,
                    near_limit=False,
                    policy_version=Q_LIMIT_STATUS_POLICY_VERSION,
                )
            )
            continue

        q_mvar = float(net.res_gen.at[gen_id, "q_mvar"])
        if not math.isfinite(q_mvar):
            status = "UNKNOWN"
            lower_headroom = None
            upper_headroom = None
            near_limit = False
        else:
            lower_headroom = q_mvar - min_q
            upper_headroom = max_q - q_mvar
            if q_mvar >= max_q - Q_LIMIT_EPS_MVAR:
                status = "Q_LIMITED_UPPER"
            elif q_mvar <= min_q + Q_LIMIT_EPS_MVAR:
                status = "Q_LIMITED_LOWER"
            else:
                status = "PV_CONTROLLABLE"
            near_threshold = max(5.0, 0.05 * abs(max_q - min_q))
            near_limit = min(lower_headroom, upper_headroom) < near_threshold
        statuses.append(
            GeneratorQState(
                gen_id=int(gen_id),
                status=status,
                q_mvar=q_mvar if math.isfinite(q_mvar) else None,
                min_q_mvar=min_q,
                max_q_mvar=max_q,
                lower_headroom_mvar=lower_headroom,
                upper_headroom_mvar=upper_headroom,
                near_limit=near_limit,
                policy_version=Q_LIMIT_STATUS_POLICY_VERSION,
            )
        )
    return tuple(statuses)


def _voltage_envelope(net: Any) -> VoltageEnvelope:
    voltages = net.res_bus["vm_pu"].dropna().astype(float)
    if voltages.empty:
        raise ValueError("valid convergent state contains no bus-voltage results")
    low_ids = tuple(int(bus_id) for bus_id in voltages.index[voltages < RUNTIME_VM_MIN_PU])
    high_ids = tuple(int(bus_id) for bus_id in voltages.index[voltages > RUNTIME_VM_MAX_PU])
    min_vm = float(voltages.min())
    max_vm = float(voltages.max())
    return VoltageEnvelope(
        min_vm_pu=min_vm,
        max_vm_pu=max_vm,
        low_bus_ids=low_ids,
        high_bus_ids=high_ids,
        hard_envelope_ok=min_vm >= HARD_VM_MIN_PU and max_vm <= HARD_VM_MAX_PU,
        runtime_quality_ok=not low_ids and not high_ids,
    )


def _connectivity(net: Any) -> tuple[bool, bool]:
    graph = create_nxgraph(net, respect_switches=True, include_out_of_service=False)
    in_service_buses = {int(bus_id) for bus_id, row in net.bus.iterrows() if bool(row.get("in_service", True))}
    slack_buses = {
        int(row["bus"])
        for _, row in net.ext_grid.iterrows()
        if bool(row.get("in_service", True)) and int(row["bus"]) in graph
    }
    reachable: set[int] = set()
    for slack_bus in sorted(slack_buses):
        reachable.update(int(bus_id) for bus_id in nx.node_connected_component(graph, slack_bus))
    connected = bool(slack_buses) and in_service_buses <= reachable
    load_buses = {int(row["bus"]) for _, row in net.load.iterrows() if bool(row.get("in_service", True))}
    loads_energized = load_buses <= reachable
    return connected, loads_energized


def _within_declared(value: float, lower: Any, upper: Any) -> bool:
    if not math.isfinite(float(value)):
        return False
    lower_value = float(lower) if lower is not None else float("nan")
    upper_value = float(upper) if upper is not None else float("nan")
    lower_ok = not math.isfinite(lower_value) or value >= lower_value - Q_LIMIT_EPS_MVAR
    upper_ok = not math.isfinite(upper_value) or value <= upper_value + Q_LIMIT_EPS_MVAR
    return lower_ok and upper_ok
