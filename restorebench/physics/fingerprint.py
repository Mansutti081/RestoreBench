# ABOUTME: Hashes canonical electrical input state and policy versions while excluding solver results.
# ABOUTME: Makes state-scoped applicability, failure vetoes, ranking, and caches deterministic.
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from restorebench.physics.actions import ACTION_POLICY_VERSION
from restorebench.physics.policies import (
    ACTIVE_BALANCE_POLICY_VERSION,
    FEASIBILITY_POLICY_VERSION,
    FINGERPRINT_POLICY_VERSION,
    SOLVER_PROBE_POLICY_VERSION,
)
from restorebench.schemas.physics import StateFingerprint


_ELECTRICAL_TABLES = (
    "bus",
    "line",
    "trafo",
    "switch",
    "load",
    "gen",
    "ext_grid",
    "shunt",
)
_NON_ELECTRICAL_COLUMNS = {
    "name",
    "geo",
}


def state_fingerprint(
    net: Any,
    *,
    policy_versions: Mapping[str, str] | None = None,
) -> StateFingerprint:
    """Return a canonical SHA-256 over electrical inputs and relevant policy identities."""
    versions = {
        "action": ACTION_POLICY_VERSION,
        "active_balance": ACTIVE_BALANCE_POLICY_VERSION,
        "feasibility": FEASIBILITY_POLICY_VERSION,
        "fingerprint": FINGERPRINT_POLICY_VERSION,
        "solver_probe": SOLVER_PROBE_POLICY_VERSION,
    }
    if policy_versions:
        versions.update({str(key): str(value) for key, value in policy_versions.items()})
    payload = {
        "network": {
            "sn_mva": _canonical_scalar(getattr(net, "sn_mva", None)),
            "f_hz": _canonical_scalar(getattr(net, "f_hz", None)),
        },
        "tables": {
            table_name: _canonical_table(getattr(net, table_name))
            for table_name in _ELECTRICAL_TABLES
            if hasattr(net, table_name)
        },
        "policy_versions": dict(sorted(versions.items())),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return StateFingerprint(
        value=hashlib.sha256(encoded).hexdigest(),
        policy_version=FINGERPRINT_POLICY_VERSION,
    )


def _canonical_table(table: pd.DataFrame) -> list[dict[str, Any]]:
    columns = sorted(str(column) for column in table.columns if str(column) not in _NON_ELECTRICAL_COLUMNS)
    records: list[dict[str, Any]] = []
    for index in sorted(table.index, key=_index_sort_key):
        record = {"id": _canonical_scalar(index)}
        for column in columns:
            record[column] = _canonical_scalar(table.at[index, column])
        records.append(record)
    return records


def _canonical_scalar(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        if math.isnan(numeric):
            return {"nonfinite": "nan"}
        if math.isinf(numeric):
            return {"nonfinite": "inf" if numeric > 0 else "-inf"}
        if numeric == 0.0:
            numeric = 0.0
        return {"float_hex": numeric.hex()}
    if isinstance(value, str):
        return value
    if pd.isna(value):
        return None
    return str(value)


def _index_sort_key(value: Any) -> tuple[str, str]:
    canonical = _canonical_scalar(value)
    return type(canonical).__name__, json.dumps(canonical, sort_keys=True)
