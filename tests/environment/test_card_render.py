# ABOUTME: Verifies the Scenario Card renderer used by the orchestrator and the corpus pipeline.
# ABOUTME: Guards deterministic current-state rendering and the no-diagnostics blindness contract.
from __future__ import annotations

import copy

from restorebench.environment.card_render import render_scenario_card
from restorebench.environment.scenarios import load_full_net, load_scenario


FORBIDDEN_DIAGNOSTIC_TOKENS = [
    "res_bus",
    "res_line",
    "res_trafo",
    "loading_percent",
    "mismatch",
    "iterations",
    "worst_bus",
    "lowest_vm",
    "diagnostics_source",
    "error_message",
]


def test_backend_renderer_is_deterministic_on_a_real_scenario() -> None:
    scenario = load_scenario("S0008")
    net = load_full_net(scenario)

    assert render_scenario_card(net) == render_scenario_card(copy.deepcopy(net))


def test_backend_renderer_excludes_power_flow_and_diagnostic_numbers() -> None:
    text = render_scenario_card(load_full_net(load_scenario("S0008"))).lower()

    for token in FORBIDDEN_DIAGNOSTIC_TOKENS:
        assert token not in text
