# ABOUTME: Builds immutable, versioned base networks used by reactive-deficit curation.
# ABOUTME: Applies each network's declared control and feasibility-bound augmentation.
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd
import pandapower as pp

from restorebench.physics.feasibility import evaluate_solved_feasibility
from restorebench.physics.fingerprint import state_fingerprint
from restorebench.physics.solver import solve_locked_probe


AUGMENTATION_POLICY_VERSION = "ieee118-declared-augmentation-v1"
VM_MIN = 0.95
VM_MAX = 1.05
TAP_MIN = -2
TAP_MAX = 2


@dataclass(frozen=True)
class NetworkSpec:
    network_id: str
    dataset_version: str
    augmentation_policy_version: str
    base_filename: str
    augmented_filename: str
    expected_shape: dict[str, int]
    expected_tappable_transformers: int
    builder: Callable[[], Any]
    freeze_nonpositive_generation: bool = False
    positive_generator_max_p_scale: float = 1.0
    ext_grid_max_q_mvar: float | None = None


NETWORK_SPECS = {
    "case118": NetworkSpec(
        network_id="case118",
        dataset_version="ieee118-reactive-deficit-v1",
        augmentation_policy_version=AUGMENTATION_POLICY_VERSION,
        base_filename="IEEE118_BASE_CASE.json",
        augmented_filename="IEEE118_AUGMENTED.json",
        expected_shape={"bus": 118, "load": 99, "gen": 53, "ext_grid": 1, "shunt": 14, "trafo": 13},
        expected_tappable_transformers=9,
        builder=pp.networks.case118,
    ),
    "case89pegase": NetworkSpec(
        network_id="case89pegase",
        dataset_version="pegase89-reactive-deficit-v1",
        augmentation_policy_version="pegase89-benchmark-augmentation-v2",
        base_filename="PEGASE89_BASE_CASE.json",
        augmented_filename="PEGASE89_AUGMENTED.json",
        expected_shape={"bus": 89, "load": 29, "gen": 11, "ext_grid": 1, "shunt": 44, "trafo": 50},
        expected_tappable_transformers=32,
        builder=pp.networks.case89pegase,
        freeze_nonpositive_generation=True,
        positive_generator_max_p_scale=1.5,
        ext_grid_max_q_mvar=2500.0,
    ),
}


def get_network_spec(network_id: str) -> NetworkSpec:
    try:
        return NETWORK_SPECS[network_id]
    except KeyError as exc:
        choices = ", ".join(sorted(NETWORK_SPECS))
        raise ValueError(f"unknown network {network_id!r}; choose one of: {choices}") from exc


def augment_network(net: Any, network_id: str) -> Any:
    """Apply the declared augmentation for one registered network in place."""
    spec = get_network_spec(network_id)
    tappable = net.trafo["tap_pos"].notna()
    net.trafo.loc[tappable, "tap_min"] = TAP_MIN
    net.trafo.loc[tappable, "tap_max"] = TAP_MAX
    net.gen.loc[:, "vm_pu"] = net.gen["vm_pu"].clip(lower=VM_MIN, upper=VM_MAX)
    frozen = net.gen["p_mw"] <= 1e-9 if spec.freeze_nonpositive_generation else net.gen["p_mw"].abs() < 1e-9
    net.gen.loc[frozen, "min_p_mw"] = net.gen.loc[frozen, "p_mw"]
    net.gen.loc[frozen, "max_p_mw"] = net.gen.loc[frozen, "p_mw"]
    positive = net.gen["p_mw"] > 1e-9
    net.gen.loc[positive, "max_p_mw"] *= spec.positive_generator_max_p_scale
    if spec.ext_grid_max_q_mvar is not None:
        net.ext_grid.loc[:, "max_q_mvar"] = spec.ext_grid_max_q_mvar
    return net


def augment_ieee118(net: Any) -> Any:
    """Apply the declared augmentation in place and return the same network."""
    return augment_network(net, "case118")


def augmented_base_fingerprint(
    net: Any,
    *,
    network_id: str = "case118",
    profile_policy_version: str | None = None,
) -> str:
    """Hash the base/profile electrical state together with its construction policies."""
    versions = {"augmentation": get_network_spec(network_id).augmentation_policy_version}
    if profile_policy_version is not None:
        versions["operating_profile"] = profile_policy_version
    return state_fingerprint(net, policy_versions=versions).value


def build_augmented_base(network_id: str = "case118") -> Any:
    """Return a fresh, valid registered base with all result tables stripped."""
    spec = get_network_spec(network_id)
    net = augment_network(spec.builder(), network_id)
    _assert_declared_shape(net, spec)
    probe = solve_locked_probe(net)
    if probe.status != "SOLVED":
        raise ValueError(f"augmented {network_id} base does not converge under the locked solver")
    feasibility = evaluate_solved_feasibility(probe.solved_net)
    if not feasibility.feasible:
        codes = sorted({reason.code for reason in feasibility.failure_reasons})
        raise ValueError(f"augmented {network_id} base is electrically infeasible: {codes}")
    pp.reset_results(net)
    return net


def write_augmented_base(output_dir: str | Path, network_id: str = "case118") -> tuple[Path, Path]:
    """Write raw and augmented bases only beneath an explicit staging directory."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    spec = get_network_spec(network_id)
    base_path = target / spec.base_filename
    augmented_path = target / spec.augmented_filename

    raw = spec.builder()
    augmented = build_augmented_base(network_id)
    pp.reset_results(raw)
    pp.to_json(raw, str(base_path))
    pp.to_json(augmented, str(augmented_path))
    return base_path, augmented_path


def _assert_declared_shape(net: Any, spec: NetworkSpec) -> None:
    expected = spec.expected_shape
    actual = {table: len(getattr(net, table)) for table in expected}
    if actual != expected:
        raise ValueError(f"unexpected {spec.network_id} shape: expected={expected}, actual={actual}")
    if int(net.trafo["tap_pos"].notna().sum()) != spec.expected_tappable_transformers:
        raise ValueError(
            f"expected exactly {spec.expected_tappable_transformers} tappable {spec.network_id} transformers"
        )
    if not net.load.get("const_z_percent", pd.Series(0.0, index=net.load.index)).fillna(0.0).eq(0.0).all():
        raise ValueError("all benchmark loads must remain constant-PQ")
    if not net.load.get("const_i_percent", pd.Series(0.0, index=net.load.index)).fillna(0.0).eq(0.0).all():
        raise ValueError("all benchmark loads must remain constant-PQ")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Explicit staging directory; the frozen dataset/ieee118 path is never implicit.",
    )
    parser.add_argument("--network", choices=sorted(NETWORK_SPECS), default="case118")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    base_path, augmented_path = write_augmented_base(args.output_dir, args.network)
    print(f"Wrote base case      -> {base_path}")
    print(f"Wrote augmented base -> {augmented_path}")


if __name__ == "__main__":
    main()
