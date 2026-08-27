# ABOUTME: Locks the current-state-only LEAN and Scenario Card public boundaries.
# ABOUTME: Ensures shunt steps, atomic controls, and slack context appear without private curation data.
from __future__ import annotations

import copy

import pytest

from restorebench.environment.card_render import render_scenario_card
from restorebench.corpus.augment import build_augmented_base
from restorebench.corpus import (
    reduce_to_lean as lean_script,
    render_scenario_card as card_script,
)
from restorebench.corpus.reduce_to_lean import COLUMN_WHITELIST


PRIVATE_TOKENS = (
    "scenario_family",
    "leakage_group",
    "pocket_recipe",
    "target_stress",
    "boundary",
    "witness",
    "resolution_regime",
    "cause_hint",
    "pre_weakening",
    "lambda_nose",
)


def test_card_exposes_atomic_qv_controls_and_read_only_operating_context() -> None:
    net = build_augmented_base()

    card = render_scenario_card(net)
    lowered = card.lower()

    assert "generator voltage controls" in lowered
    assert "| vm_pu | vm_min | vm_max | atomic_step |" in lowered
    assert "shunt step controls" in lowered
    assert "| step | max_step |" in lowered
    assert "transformer tap controls" in lowered
    assert "tap_step_percent" in lowered
    assert "generation p (read-only)" in lowered
    assert "external-grid limits (read-only)" in lowered
    assert "active-power redispatch" not in lowered
    assert "in_service — toggle" not in lowered
    for token in PRIVATE_TOKENS:
        assert token not in lowered


def test_lean_keeps_current_state_and_declared_limits_but_no_results() -> None:
    net = build_augmented_base()
    lean = lean_script.reduce_to_lean(copy.deepcopy(net))

    for table, expected in COLUMN_WHITELIST.items():
        assert set(getattr(lean, table).columns) == set(expected)
    assert len(lean.ext_grid) == 1
    assert {"step", "max_step", "in_service"} <= set(lean.shunt.columns)
    assert {"tap_step_percent", "tap_side"} <= set(lean.trafo.columns)
    assert {"min_q_mvar", "max_q_mvar"} <= set(lean.gen.columns)
    assert lean.res_bus.empty
    assert lean.poly_cost.empty


@pytest.mark.parametrize(
    "parser",
    (
        lean_script.parse_args,
        card_script.parse_args,
    ),
)
def test_batch_artifact_cli_requires_explicit_dataset_dir(parser) -> None:
    with pytest.raises(SystemExit):
        parser(["--all"])

    args = parser(["--all", "--dataset-dir", "/tmp/staged-dataset"])

    assert args.all is True
