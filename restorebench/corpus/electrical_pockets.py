# ABOUTME: Builds private load-pocket recipes from the shared impedance-distance primitive.
# ABOUTME: Applies versioned decay, support gates, normalization, hashing, and deduplication.
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from restorebench.physics.electrical_distance import impedance_weighted_graph_distances
from restorebench.schemas.dataset import CurationLoadPoint, PocketRecipe
from restorebench.schemas.physics import ElectricalDistancePolicy


POCKET_WEIGHTING_POLICY_VERSION = "impedance-exponential-pocket-v1"


class PocketWeightingPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    common_mva_base: float = Field(default=100.0, gt=0.0, allow_inf_nan=False)
    minimum_edge_weight_pu: float = Field(
        default=1e-6,
        gt=0.0,
        allow_inf_nan=False,
    )
    distance_scales_pu: tuple[float, ...]
    weight_cutoff: float = Field(ge=0.0, lt=1.0, allow_inf_nan=False)
    minimum_load_count: int = Field(ge=1)
    minimum_base_p_mw: float = Field(gt=0.0, allow_inf_nan=False)
    minimum_base_abs_q_mvar: float = Field(gt=0.0, allow_inf_nan=False)
    near_duplicate_cosine: float = Field(gt=0.0, le=1.0, allow_inf_nan=False)
    policy_version: str = POCKET_WEIGHTING_POLICY_VERSION

    @model_validator(mode="after")
    def scales_are_stable(self) -> "PocketWeightingPolicy":
        scales = self.distance_scales_pu
        if (
            not scales
            or not all(math.isfinite(scale) and scale > 0.0 for scale in scales)
            or tuple(sorted(set(scales))) != scales
        ):
            raise ValueError(
                "distance scales must be finite, positive, unique, and ascending"
            )
        return self


def build_pocket_recipe(
    net: Any,
    *,
    anchor_bus: int,
    distance_scale_pu: float,
    policy: PocketWeightingPolicy,
) -> PocketRecipe:
    """Build one ordered immutable pocket recipe from a load-bus anchor."""
    if not math.isfinite(distance_scale_pu) or distance_scale_pu <= 0.0:
        raise ValueError("distance scale must be finite and positive")
    active_loads = net.load[
        net.load.get("in_service", True).astype(bool)
        if "in_service" in net.load
        else [True] * len(net.load)
    ]
    load_buses = {int(bus) for bus in active_loads["bus"]}
    if anchor_bus not in load_buses:
        raise ValueError(
            f"anchor bus {anchor_bus} must contain at least one in-service load"
        )

    distance_policy = ElectricalDistancePolicy(
        common_mva_base=policy.common_mva_base,
        minimum_edge_weight_pu=policy.minimum_edge_weight_pu,
    )
    distance_result = impedance_weighted_graph_distances(
        net,
        source_buses=(anchor_bus,),
        policy=distance_policy,
    )
    raw_by_bus: dict[int, float] = {}
    for bus_id in sorted(load_buses):
        distance = distance_result.distances_pu.get(bus_id)
        raw_weight = (
            math.exp(-distance / distance_scale_pu)
            if distance is not None
            else 0.0
        )
        raw_by_bus[bus_id] = (
            raw_weight if raw_weight >= policy.weight_cutoff else 0.0
        )
    maximum = max(raw_by_bus.values(), default=0.0)
    if maximum <= 0.0:
        raise ValueError("pocket has no connected non-zero load support")
    normalized_by_bus = {
        bus_id: round(weight / maximum, 12)
        for bus_id, weight in raw_by_bus.items()
    }

    points = tuple(
        CurationLoadPoint(
            load_id=int(load_id),
            base_p_mw=float(row["p_mw"]),
            base_q_mvar=float(row["q_mvar"]),
            weight=normalized_by_bus[int(row["bus"])],
        )
        for load_id, row in active_loads.sort_index().iterrows()
    )
    supported = [point for point in points if point.weight > 0.0]
    if len(supported) < policy.minimum_load_count:
        raise ValueError(
            "pocket load count below minimum: "
            f"{len(supported)} < {policy.minimum_load_count}"
        )
    total_p = math.fsum(max(point.base_p_mw, 0.0) for point in supported)
    total_abs_q = math.fsum(abs(point.base_q_mvar) for point in supported)
    if total_p < policy.minimum_base_p_mw:
        raise ValueError(
            f"pocket base MW below minimum: {total_p} < {policy.minimum_base_p_mw}"
        )
    if total_abs_q < policy.minimum_base_abs_q_mvar:
        raise ValueError(
            "pocket base absolute MVAr below minimum: "
            f"{total_abs_q} < {policy.minimum_base_abs_q_mvar}"
        )

    vector_hash = _vector_hash(
        anchor_bus=anchor_bus,
        distance_scale_pu=distance_scale_pu,
        points=points,
        policy=policy,
    )
    return PocketRecipe(
        anchor_bus=anchor_bus,
        distance_method="IMPEDANCE_WEIGHTED_GRAPH_DISTANCE",
        loads=points,
        vector_hash=vector_hash,
        policy_version=policy.policy_version,
    )


def generate_pocket_recipes(
    net: Any,
    *,
    anchor_buses: Sequence[int],
    policy: PocketWeightingPolicy,
) -> tuple[PocketRecipe, ...]:
    """Generate stable pocket recipes and remove exact or near-duplicate vectors."""
    accepted: list[PocketRecipe] = []
    seen_hashes: set[str] = set()
    for anchor_bus in sorted(set(int(bus) for bus in anchor_buses)):
        for scale in policy.distance_scales_pu:
            try:
                recipe = build_pocket_recipe(
                    net,
                    anchor_bus=anchor_bus,
                    distance_scale_pu=scale,
                    policy=policy,
                )
            except ValueError:
                continue
            if recipe.vector_hash in seen_hashes:
                continue
            if any(
                pocket_vector_similarity(recipe, existing)
                >= policy.near_duplicate_cosine
                for existing in accepted
            ):
                continue
            seen_hashes.add(recipe.vector_hash)
            accepted.append(recipe)
    return tuple(accepted)


def pocket_vector_similarity(left: PocketRecipe, right: PocketRecipe) -> float:
    """Return cosine similarity over stable load-ID-aligned pocket weights."""
    left_weights = {point.load_id: point.weight for point in left.loads}
    right_weights = {point.load_id: point.weight for point in right.loads}
    ids = sorted(set(left_weights) | set(right_weights))
    dot = math.fsum(
        left_weights.get(load_id, 0.0) * right_weights.get(load_id, 0.0)
        for load_id in ids
    )
    left_norm = math.sqrt(
        math.fsum(left_weights.get(load_id, 0.0) ** 2 for load_id in ids)
    )
    right_norm = math.sqrt(
        math.fsum(right_weights.get(load_id, 0.0) ** 2 for load_id in ids)
    )
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _vector_hash(
    *,
    anchor_bus: int,
    distance_scale_pu: float,
    points: tuple[CurationLoadPoint, ...],
    policy: PocketWeightingPolicy,
) -> str:
    payload = {
        "anchor_bus": anchor_bus,
        "distance_scale_pu": distance_scale_pu,
        "loads": [
            point.model_dump(mode="json")
            for point in points
        ],
        "policy": policy.model_dump(mode="json"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
