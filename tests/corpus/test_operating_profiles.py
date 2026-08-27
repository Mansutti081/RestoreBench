# ABOUTME: Locks registered-network augmentation and pocket-independent operating-profile construction.
# ABOUTME: Ensures bases and profiles expose the controls required by the shared runtime.
from __future__ import annotations

import copy

import pandas as pd
import pandapower as pp

from restorebench.physics.actions import enumerate_legal_qv_actions
from restorebench.physics.feasibility import evaluate_solved_feasibility
from restorebench.physics.solver import solve_locked_probe
from restorebench.corpus.augment import (
    AUGMENTATION_POLICY_VERSION,
    augment_network,
    augment_ieee118,
    augmented_base_fingerprint,
    build_augmented_base,
    get_network_spec,
)
from restorebench.corpus.operating_profiles import (
    OperatingProfilePolicy,
    admit_operating_profiles,
    generate_operating_profile_candidates,
)


def test_augmentation_is_idempotent_and_changes_only_declared_fields() -> None:
    source = pp.networks.case118()
    before = copy.deepcopy(source)

    augmented = augment_ieee118(copy.deepcopy(source))
    twice = augment_ieee118(copy.deepcopy(augmented))

    pd.testing.assert_frame_equal(augmented.trafo, twice.trafo, check_exact=True)
    pd.testing.assert_frame_equal(augmented.gen, twice.gen, check_exact=True)
    pd.testing.assert_frame_equal(augmented.shunt, before.shunt, check_exact=True)
    pd.testing.assert_frame_equal(augmented.load, before.load, check_exact=True)
    pd.testing.assert_frame_equal(augmented.line, before.line, check_exact=True)
    pd.testing.assert_frame_equal(augmented.ext_grid, before.ext_grid, check_exact=True)

    tappable = augmented.trafo["tap_pos"].notna()
    assert int(tappable.sum()) == 9
    assert augmented.trafo.loc[tappable, "tap_min"].eq(-2).all()
    assert augmented.trafo.loc[tappable, "tap_max"].eq(2).all()
    assert augmented.gen["vm_pu"].between(0.95, 1.05).all()
    condensers = augmented.gen["p_mw"].abs() < 1e-9
    assert augmented.gen.loc[condensers, ["min_p_mw", "max_p_mw"]].eq(0.0).all().all()
    assert AUGMENTATION_POLICY_VERSION == "ieee118-declared-augmentation-v1"
    assert len(augmented_base_fingerprint(augmented)) == 64


def test_build_augmented_base_has_stable_shape_and_no_persisted_results() -> None:
    first = build_augmented_base()
    second = build_augmented_base()

    assert (len(first.bus), len(first.load), len(first.gen)) == (118, 99, 53)
    assert (len(first.ext_grid), len(first.shunt), len(first.trafo)) == (1, 14, 13)
    assert first.res_bus.empty
    assert augmented_base_fingerprint(first) == augmented_base_fingerprint(second)


def test_case89pegase_profile_is_runtime_compatible() -> None:
    spec = get_network_spec("case89pegase")
    net = build_augmented_base("case89pegase")

    assert spec.dataset_version == "pegase89-reactive-deficit-v1"
    assert spec.augmentation_policy_version == "pegase89-benchmark-augmentation-v2"
    assert spec.positive_generator_max_p_scale == 1.5
    assert spec.ext_grid_max_q_mvar == 2500.0
    assert (len(net.bus), len(net.load), len(net.gen)) == (89, 29, 11)
    assert (len(net.ext_grid), len(net.shunt), len(net.trafo)) == (1, 44, 50)
    assert int(net.trafo["tap_pos"].notna().sum()) == 32
    nonpositive_generators = net.gen["p_mw"] <= 1e-9
    assert net.gen.loc[nonpositive_generators, "min_p_mw"].equals(
        net.gen.loc[nonpositive_generators, "p_mw"]
    )
    assert net.gen.loc[nonpositive_generators, "max_p_mw"].equals(
        net.gen.loc[nonpositive_generators, "p_mw"]
    )
    assert net.res_bus.empty


def test_case89pegase_augmentation_changes_only_declared_electrical_bounds() -> None:
    source = pp.networks.case89pegase()
    augmented = augment_network(copy.deepcopy(source), "case89pegase")

    positive_generators = source.gen["p_mw"] > 1e-9
    pd.testing.assert_series_equal(
        augmented.gen.loc[positive_generators, "max_p_mw"],
        source.gen.loc[positive_generators, "max_p_mw"] * 1.5,
        check_names=False,
    )
    pd.testing.assert_series_equal(
        augmented.gen.loc[positive_generators, "min_p_mw"],
        source.gen.loc[positive_generators, "min_p_mw"],
    )
    assert augmented.ext_grid["max_q_mvar"].eq(2500.0).all()
    pd.testing.assert_series_equal(augmented.ext_grid["min_q_mvar"], source.ext_grid["min_q_mvar"])
    pd.testing.assert_series_equal(augmented.ext_grid["min_p_mw"], source.ext_grid["min_p_mw"])
    pd.testing.assert_series_equal(augmented.ext_grid["max_p_mw"], source.ext_grid["max_p_mw"])
    pd.testing.assert_frame_equal(augmented.line, source.line, check_exact=True)
    pd.testing.assert_frame_equal(augmented.load, source.load, check_exact=True)
    pd.testing.assert_frame_equal(augmented.shunt, source.shunt, check_exact=True)


def test_case89pegase_profile_hash_uses_its_augmentation_policy() -> None:
    base = build_augmented_base("case89pegase")
    policy = OperatingProfilePolicy(max_simultaneous_deviations=0, max_profiles=1)

    candidate = generate_operating_profile_candidates(
        base,
        policy=policy,
        network_id="case89pegase",
    )[0]

    assert candidate.state_hash == augmented_base_fingerprint(
        base,
        network_id="case89pegase",
        profile_policy_version=policy.policy_version,
    )
    assert candidate.state_hash != augmented_base_fingerprint(
        base,
        profile_policy_version=policy.policy_version,
    )


def test_profile_candidates_change_only_operating_control_positions() -> None:
    base = build_augmented_base()
    policy = OperatingProfilePolicy(max_simultaneous_deviations=1, max_profiles=8)

    candidates = generate_operating_profile_candidates(base, policy=policy)

    assert candidates
    assert candidates[0].modifications == ()
    assert len({candidate.profile_id for candidate in candidates}) == len(candidates)
    for candidate in candidates:
        assert len(candidate.modifications) <= 1
        pd.testing.assert_frame_equal(candidate.net.load, base.load, check_exact=True)
        pd.testing.assert_frame_equal(candidate.net.gen, base.gen, check_exact=True)
        pd.testing.assert_frame_equal(candidate.net.line, base.line, check_exact=True)
        pd.testing.assert_series_equal(
            candidate.net.shunt["in_service"],
            base.shunt["in_service"],
            check_exact=True,
        )
        pd.testing.assert_series_equal(
            candidate.net.trafo["in_service"],
            base.trafo["in_service"],
            check_exact=True,
        )
        changed_shunts = candidate.net.shunt["step"] != base.shunt["step"]
        changed_taps = candidate.net.trafo["tap_pos"].fillna(0) != base.trafo["tap_pos"].fillna(0)
        assert int(changed_shunts.sum() + changed_taps.sum()) == len(candidate.modifications)


def test_admitted_profiles_are_feasible_and_expose_legal_qv_actions() -> None:
    base = build_augmented_base()
    policy = OperatingProfilePolicy(max_simultaneous_deviations=1, max_profiles=6)
    candidates = generate_operating_profile_candidates(base, policy=policy)

    selection = admit_operating_profiles(candidates, policy=policy)

    assert selection.profiles
    assert len(selection.profiles) <= policy.max_profiles
    assert selection.rejections
    for profile in selection.profiles:
        probe = solve_locked_probe(profile.net)
        assert probe.status == "SOLVED"
        feasibility = evaluate_solved_feasibility(probe.solved_net)
        assert feasibility.feasible
        assert enumerate_legal_qv_actions(profile.net)
        assert profile.state_hash == augmented_base_fingerprint(
            profile.net,
            profile_policy_version=policy.policy_version,
        )
