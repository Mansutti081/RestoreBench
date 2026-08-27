# ABOUTME: Verifies deterministic bounded active-power participation and water-filling.
# ABOUTME: Ensures trajectory scheduling starts from immutable references and never mutates callers.
from __future__ import annotations

import copy
from typing import Any

import pandapower as pp
import pandas as pd
import pytest
from pydantic import ValidationError

from restorebench.physics.active_balance import schedule_active_power
from restorebench.schemas.physics import ActiveBalancePolicy, GeneratorParticipation


def _balance_net() -> Any:
    net = pp.create_empty_network()
    bus = pp.create_bus(net, vn_kv=110.0)
    pp.create_ext_grid(net, bus=bus, vm_pu=1.0)
    pp.create_gen(
        net,
        bus=bus,
        p_mw=20.0,
        vm_pu=1.0,
        min_p_mw=10.0,
        max_p_mw=25.0,
        min_q_mvar=-20.0,
        max_q_mvar=20.0,
        index=2,
    )
    pp.create_gen(
        net,
        bus=bus,
        p_mw=40.0,
        vm_pu=1.0,
        min_p_mw=20.0,
        max_p_mw=70.0,
        min_q_mvar=-20.0,
        max_q_mvar=20.0,
        index=7,
    )
    pp.create_gen(
        net,
        bus=bus,
        p_mw=0.0,
        vm_pu=1.0,
        min_p_mw=0.0,
        max_p_mw=0.0,
        min_q_mvar=-50.0,
        max_q_mvar=50.0,
        index=9,
    )
    pp.create_load(net, bus=bus, p_mw=60.0, q_mvar=20.0)
    return net


def _policy() -> ActiveBalancePolicy:
    return ActiveBalancePolicy(
        participation=(
            GeneratorParticipation(gen_id=2, factor=1.0),
            GeneratorParticipation(gen_id=7, factor=3.0),
        )
    )


def _target_with_delta(reference: Any, delta_mw: float) -> Any:
    target = copy.deepcopy(reference)
    target.load.at[0, "p_mw"] = float(reference.load.at[0, "p_mw"]) + delta_mw
    return target


def test_active_balance_allocates_proportionally_without_saturation() -> None:
    reference = _balance_net()

    result = schedule_active_power(
        reference,
        _target_with_delta(reference, 8.0),
        requested_load_delta_mw=8.0,
        policy=_policy(),
    )

    assert result.status == "SCHEDULED"
    assert result.net.gen.at[2, "p_mw"] == pytest.approx(22.0)
    assert result.net.gen.at[7, "p_mw"] == pytest.approx(46.0)
    assert result.allocated_delta_mw == pytest.approx(8.0)
    assert result.unallocated_delta_mw == pytest.approx(0.0)


def test_active_balance_water_fills_deterministically_through_saturation() -> None:
    reference = _balance_net()

    result = schedule_active_power(
        reference,
        _target_with_delta(reference, 32.0),
        requested_load_delta_mw=32.0,
        policy=_policy(),
    )

    assert result.status == "SCHEDULED"
    assert result.net.gen.at[2, "p_mw"] == pytest.approx(25.0)
    assert result.net.gen.at[7, "p_mw"] == pytest.approx(67.0)
    assert [dispatch.gen_id for dispatch in result.generator_dispatch] == [2, 7]


def test_active_balance_uses_lower_headroom_for_negative_delta() -> None:
    reference = _balance_net()

    result = schedule_active_power(
        reference,
        _target_with_delta(reference, -25.0),
        requested_load_delta_mw=-25.0,
        policy=_policy(),
    )

    assert result.status == "SCHEDULED"
    assert result.net.gen.at[2, "p_mw"] == pytest.approx(13.75)
    assert result.net.gen.at[7, "p_mw"] == pytest.approx(21.25)


def test_active_balance_excludes_synchronous_condensers() -> None:
    reference = _balance_net()

    result = schedule_active_power(
        reference,
        _target_with_delta(reference, 5.0),
        requested_load_delta_mw=5.0,
        policy=_policy(),
    )

    assert result.net.gen.at[9, "p_mw"] == 0.0
    assert all(dispatch.gen_id != 9 for dispatch in result.generator_dispatch)


def test_active_balance_reports_typed_headroom_exhaustion() -> None:
    reference = _balance_net()

    result = schedule_active_power(
        reference,
        _target_with_delta(reference, 50.0),
        requested_load_delta_mw=50.0,
        policy=_policy(),
    )

    assert result.status == "ACTIVE_HEADROOM_EXHAUSTED"
    assert result.net.gen.at[2, "p_mw"] == 25.0
    assert result.net.gen.at[7, "p_mw"] == 70.0
    assert result.allocated_delta_mw == pytest.approx(35.0)
    assert result.unallocated_delta_mw == pytest.approx(15.0)


def test_active_balance_is_exactly_repeatable_and_does_not_mutate_inputs() -> None:
    reference = _balance_net()
    target = _target_with_delta(reference, 8.0)
    reference_before = copy.deepcopy(reference)
    target_before = copy.deepcopy(target)

    first = schedule_active_power(
        reference,
        target,
        requested_load_delta_mw=8.0,
        policy=_policy(),
    )
    second = schedule_active_power(
        reference,
        target,
        requested_load_delta_mw=8.0,
        policy=_policy(),
    )

    pd.testing.assert_frame_equal(first.net.gen, second.net.gen, check_exact=True)
    assert first.model_dump(exclude={"net"}) == second.model_dump(exclude={"net"})
    pd.testing.assert_frame_equal(reference.gen, reference_before.gen, check_exact=True)
    pd.testing.assert_frame_equal(target.gen, target_before.gen, check_exact=True)


def test_active_balance_rejects_inconsistent_delta_and_participation_ids() -> None:
    reference = _balance_net()
    target = _target_with_delta(reference, 8.0)

    with pytest.raises(ValueError, match="load delta"):
        schedule_active_power(
            reference,
            target,
            requested_load_delta_mw=7.0,
            policy=_policy(),
        )

    incomplete = ActiveBalancePolicy(participation=(GeneratorParticipation(gen_id=2, factor=1.0),))
    with pytest.raises(ValueError, match="participation IDs"):
        schedule_active_power(
            reference,
            target,
            requested_load_delta_mw=8.0,
            policy=incomplete,
        )


def test_active_balance_policy_rejects_nonfinite_participation() -> None:
    with pytest.raises(ValidationError):
        GeneratorParticipation(gen_id=2, factor=float("inf"))
