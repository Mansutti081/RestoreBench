# ABOUTME: Renders current-state-only Scenario Cards for registered benchmark networks.
# ABOUTME: Exposes atomic Q-V controls and read-only operating context without curation metadata.
from __future__ import annotations

from typing import Any

import pandas as pd


def _fmt_bool(value: Any) -> str:
    return "true" if bool(value) else "false"


def _fmt_float(value: Any, digits: int = 1) -> str:
    return f"{float(value):.{digits}f}"


def _fmt_int(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(int(value))


def _active(df: Any) -> Any:
    if "in_service" not in df.columns:
        return df
    return df[df["in_service"].astype(bool)]


def _neighbors(net: Any) -> dict[int, list[int]]:
    incident: dict[int, set[int]] = {
        int(index): set()
        for index in sorted(net.bus.index)
    }
    for _, row in _active(net.line).sort_index().iterrows():
        left = int(row["from_bus"])
        right = int(row["to_bus"])
        incident[left].add(right)
        incident[right].add(left)
    for _, row in _active(net.trafo).sort_index().iterrows():
        left = int(row["hv_bus"])
        right = int(row["lv_bus"])
        incident[left].add(right)
        incident[right].add(left)
    return {
        bus_id: sorted(neighbors)
        for bus_id, neighbors in incident.items()
    }


def _incidents(net: Any) -> dict[int, list[str]]:
    incidents: dict[int, list[str]] = {
        int(index): []
        for index in sorted(net.bus.index)
    }
    for line_id, row in net.line.sort_index().iterrows():
        left = int(row["from_bus"])
        right = int(row["to_bus"])
        state = _fmt_bool(row.get("in_service", True))
        incidents[left].append(f"line {int(line_id)} -> bus {right} (in_service={state})")
        incidents[right].append(f"line {int(line_id)} -> bus {left} (in_service={state})")
    for trafo_id, row in net.trafo.sort_index().iterrows():
        left = int(row["hv_bus"])
        right = int(row["lv_bus"])
        state = _fmt_bool(row.get("in_service", True))
        incidents[left].append(
            f"trafo {int(trafo_id)} -> bus {right} (in_service={state})"
        )
        incidents[right].append(
            f"trafo {int(trafo_id)} -> bus {left} (in_service={state})"
        )
    return incidents


def _section_topology(net: Any) -> str:
    lines = ["## A. Topology (stable incident encoding)"]
    incidents = _incidents(net)
    for bus_id, row in net.bus.sort_index().iterrows():
        lines.append(
            f"Bus {int(bus_id)} ({_fmt_float(row['vn_kv'], 1)} kV, "
            f"in_service={_fmt_bool(row.get('in_service', True))}): "
            f"{incidents[int(bus_id)]}"
        )
    return "\n".join(lines)


def _gen_rows(net: Any) -> list[str]:
    rows = [
        "| index | bus | in_service | vm_pu | vm_min | vm_max | atomic_step |",
        "|---|---|---|---|---|---|---|",
    ]
    for gen_id, row in net.gen.sort_index().iterrows():
        rows.append(
            f"| {int(gen_id)} | {int(row['bus'])} | "
            f"{_fmt_bool(row.get('in_service', True))} | "
            f"{_fmt_float(row['vm_pu'], 3)} | 0.950 | 1.050 | 0.010 |"
        )
    return rows


def _shunt_rows(net: Any) -> list[str]:
    rows = [
        "| index | bus | in_service | q_mvar_at_1pu | role | step | max_step |",
        "|---|---|---|---|---|---|---|",
    ]
    for shunt_id, row in net.shunt.sort_index().iterrows():
        role = "capacitor" if float(row["q_mvar"]) < 0.0 else "reactor"
        rows.append(
            f"| {int(shunt_id)} | {int(row['bus'])} | "
            f"{_fmt_bool(row.get('in_service', True))} | "
            f"{_fmt_float(row['q_mvar'])} | {role} | "
            f"{_fmt_int(row['step'])} | {_fmt_int(row['max_step'])} |"
        )
    return rows


def _trafo_rows(net: Any) -> list[str]:
    rows = [
        "| index | hv_bus | lv_bus | in_service | tap_side | tap_pos | tap_min | tap_max | tap_step_percent | atomic_step |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    tappable = net.trafo[net.trafo["tap_pos"].notna()]
    for trafo_id, row in tappable.sort_index().iterrows():
        rows.append(
            f"| {int(trafo_id)} | {int(row['hv_bus'])} | {int(row['lv_bus'])} | "
            f"{_fmt_bool(row.get('in_service', True))} | {row['tap_side']} | "
            f"{_fmt_int(row['tap_pos'])} | {_fmt_int(row['tap_min'])} | "
            f"{_fmt_int(row['tap_max'])} | {_fmt_float(row['tap_step_percent'])} | 1 |"
        )
    return rows


def _section_setpoints(net: Any) -> str:
    return "\n".join(
        [
            "## B. Atomic Q-V controls",
            "",
            "### Generator voltage controls",
            *_gen_rows(net),
            "",
            "### Shunt step controls",
            *_shunt_rows(net),
            "",
            "### Transformer tap controls",
            *_trafo_rows(net),
        ]
    )


def _section_loads(net: Any) -> str:
    lines = [
        "### Loads (read-only)",
        "| index | bus | in_service | p_mw | q_mvar |",
        "|---|---|---|---|---|",
    ]
    for load_id, row in net.load.sort_index().iterrows():
        lines.append(
            f"| {int(load_id)} | {int(row['bus'])} | "
            f"{_fmt_bool(row.get('in_service', True))} | "
            f"{_fmt_float(row['p_mw'])} | {_fmt_float(row['q_mvar'])} |"
        )
    return "\n".join(lines)


def _section_generation(net: Any) -> str:
    lines = [
        "### Generation P (read-only)",
        "| index | bus | in_service | p_mw | min_p_mw | max_p_mw |",
        "|---|---|---|---|---|---|",
    ]
    for gen_id, row in net.gen.sort_index().iterrows():
        lines.append(
            f"| {int(gen_id)} | {int(row['bus'])} | "
            f"{_fmt_bool(row.get('in_service', True))} | "
            f"{_fmt_float(row['p_mw'])} | {_fmt_float(row['min_p_mw'])} | "
            f"{_fmt_float(row['max_p_mw'])} |"
        )
    return "\n".join(lines)


def _section_external_grid(net: Any) -> str:
    lines = [
        "### External-grid limits (read-only)",
        "| index | bus | in_service | vm_pu | min_p_mw | max_p_mw | min_q_mvar | max_q_mvar |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for ext_grid_id, row in net.ext_grid.sort_index().iterrows():
        lines.append(
            f"| {int(ext_grid_id)} | {int(row['bus'])} | "
            f"{_fmt_bool(row.get('in_service', True))} | "
            f"{_fmt_float(row['vm_pu'], 3)} | "
            f"{_fmt_float(row['min_p_mw'])} | {_fmt_float(row['max_p_mw'])} | "
            f"{_fmt_float(row['min_q_mvar'])} | {_fmt_float(row['max_q_mvar'])} |"
        )
    return "\n".join(lines)


def _section_operating_context(net: Any) -> str:
    return "\n\n".join(
        [
            "## C. Operating context",
            _section_loads(net),
            _section_generation(net),
            _section_external_grid(net),
        ]
    )


def render_scenario_card(net: Any) -> str:
    """Render deterministic current-state data without results or private recipes."""
    return (
        "\n\n".join(
            [
                _section_topology(net),
                _section_setpoints(net),
                _section_operating_context(net),
            ]
        )
        + "\n"
    )
