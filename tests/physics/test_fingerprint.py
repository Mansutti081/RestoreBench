# ABOUTME: Verifies canonical electrical-state fingerprints across copies, actions, and policy versions.
# ABOUTME: Excludes stale result tables while including every field relevant to solve and applicability.
from __future__ import annotations

import copy
from typing import Any

import pandapower as pp

from restorebench.physics.actions import apply_qv_action, enumerate_legal_qv_actions
from restorebench.physics.fingerprint import state_fingerprint


def _fingerprint_net() -> Any:
    net = pp.create_empty_network()
    slack = pp.create_bus(net, vn_kv=110.0)
    controlled = pp.create_bus(net, vn_kv=110.0)
    pp.create_ext_grid(net, bus=slack, vm_pu=1.0)
    pp.create_line_from_parameters(
        net,
        from_bus=slack,
        to_bus=controlled,
        length_km=1.0,
        r_ohm_per_km=0.1,
        x_ohm_per_km=0.2,
        c_nf_per_km=0.0,
        max_i_ka=1.0,
    )
    pp.create_gen(
        net,
        bus=controlled,
        p_mw=10.0,
        vm_pu=1.0,
        min_p_mw=0.0,
        max_p_mw=20.0,
        min_q_mvar=-20.0,
        max_q_mvar=20.0,
    )
    pp.create_load(net, bus=controlled, p_mw=8.0, q_mvar=3.0)
    pp.create_shunt(net, bus=controlled, q_mvar=-4.0, p_mw=0.0, step=0, max_step=1)
    pp.create_transformer_from_parameters(
        net,
        hv_bus=slack,
        lv_bus=controlled,
        sn_mva=100.0,
        vn_hv_kv=110.0,
        vn_lv_kv=110.0,
        vk_percent=10.0,
        vkr_percent=0.5,
        pfe_kw=0.0,
        i0_percent=0.0,
        tap_side="hv",
        tap_neutral=0,
        tap_min=-1,
        tap_max=1,
        tap_step_percent=1.0,
        tap_pos=0,
    )
    return net


def test_fingerprint_is_stable_across_deep_copies_and_ignores_results() -> None:
    net = _fingerprint_net()
    copied = copy.deepcopy(net)

    before = state_fingerprint(net)
    pp.runpp(copied, algorithm="nr", enforce_q_lims=True, init="dc")
    after_solve = state_fingerprint(copied)

    assert before == after_solve


def test_every_legal_atomic_action_changes_the_fingerprint() -> None:
    net = _fingerprint_net()
    baseline = state_fingerprint(net)

    for action in enumerate_legal_qv_actions(net):
        changed = apply_qv_action(net, action)
        assert state_fingerprint(changed).value != baseline.value
        assert state_fingerprint(net) == baseline


def test_fingerprint_changes_with_electrical_inputs_and_policy_versions() -> None:
    net = _fingerprint_net()
    baseline = state_fingerprint(net)

    changed_load = copy.deepcopy(net)
    changed_load.load.at[0, "q_mvar"] += 0.001
    assert state_fingerprint(changed_load).value != baseline.value

    changed_policy = state_fingerprint(net, policy_versions={"solver_probe": "different"})
    assert changed_policy.value != baseline.value
