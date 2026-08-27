# ABOUTME: Locks the independent validator boundary and explicit staged-dataset CLI.
# ABOUTME: Prevents reuse of generator helpers that could reproduce the same implementation bug.
from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pandapower as pp
import pytest

from restorebench.physics.fingerprint import state_fingerprint
from restorebench.schemas.physics import SolvedFeasibility, VoltageEnvelope
from restorebench.corpus import validate_dataset
from restorebench.corpus.checkpoint_io import CheckpointStore


def test_validator_requires_explicit_dataset_directory() -> None:
    with pytest.raises(SystemExit):
        validate_dataset.parse_args([])

    args = validate_dataset.parse_args(
        ["--dataset-dir", "/tmp/restorebench-stage"]
    )

    assert args.dataset_dir == Path("/tmp/restorebench-stage")


def test_validator_accepts_explicit_checkpoint_and_resume(
    tmp_path: Path,
) -> None:
    args = validate_dataset.parse_args(
        [
            "--dataset-dir",
            str(tmp_path / "dataset"),
            "--checkpoint-dir",
            str(tmp_path / "validation-checkpoint"),
            "--resume",
        ]
    )

    assert args.checkpoint_dir == tmp_path / "validation-checkpoint"
    assert args.resume is True


def test_validator_honors_declared_all_held_out_split() -> None:
    labels = [SimpleNamespace(memory_split="held_out") for _ in range(46)]
    declared = {"memory_population": 0, "held_out": 46}

    assert validate_dataset._split_counts_match(labels, declared)


def test_validator_preserves_legacy_75_25_split_default() -> None:
    assert validate_dataset._declared_split_counts({}, scenario_count=200) == {
        "memory_population": 150,
        "held_out": 50,
    }


def test_validation_resume_skips_an_atomically_completed_scenario(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = CheckpointStore.open(
        tmp_path / "validation-checkpoint",
        identity={"kind": "validation", "manifest_hash": "abc"},
        resume=False,
    )
    stored = validate_dataset.ScenarioValidation(
        scenario_id="S0001",
        valid=True,
    )
    store.write_json_shard(
        "S0001",
        stored.model_dump(mode="json"),
    )
    monkeypatch.setattr(
        validate_dataset,
        "_validate_scenario",
        lambda **_kwargs: pytest.fail(
            "completed validation scenario must not be recomputed"
        ),
    )

    restored = validate_dataset._validate_scenario_with_checkpoint(
        root=tmp_path,
        entry=SimpleNamespace(scenario_id="S0001"),
        label=object(),
        witness={},
        raw_policy={},
        global_messages=(),
        checkpoint_store=store,
    )

    assert restored == stored


def test_validator_does_not_import_generator_or_dataset_physics_helpers() -> None:
    source = inspect.getsource(validate_dataset)

    forbidden = (
        "restorebench.corpus.generate_scenarios",
        "restorebench.corpus.curation",
        "restorebench.corpus.operating_profiles",
        "restorebench.corpus.electrical_pockets",
        "restorebench.corpus.reactive_admission",
        "restorebench.corpus.witness_search",
        "restorebench.corpus.select_corpus",
    )
    for module in forbidden:
        assert module not in source


def test_validator_independently_reconstructs_case89pegase() -> None:
    source = pp.networks.case89pegase()
    net = validate_dataset._independent_augmented_base("case89pegase")

    assert (len(net.bus), len(net.load), len(net.gen), len(net.trafo)) == (89, 29, 11, 50)
    nonpositive_generators = net.gen["p_mw"] <= 1e-9
    assert net.gen.loc[nonpositive_generators, "min_p_mw"].equals(
        net.gen.loc[nonpositive_generators, "p_mw"]
    )
    positive_generators = source.gen["p_mw"] > 1e-9
    assert net.gen.loc[positive_generators, "max_p_mw"].equals(
        source.gen.loc[positive_generators, "max_p_mw"] * 1.5
    )
    assert net.ext_grid["max_q_mvar"].eq(2500.0).all()
    assert net.ext_grid["max_p_mw"].equals(source.ext_grid["max_p_mw"])


def test_full_snapshot_comparison_normalizes_pandapower_storage_precision() -> None:
    reconstructed = pp.create_empty_network()
    bus_id = pp.create_bus(reconstructed, vn_kv=110.0)
    pp.create_ext_grid(reconstructed, bus=bus_id)
    pp.create_gen(
        reconstructed,
        bus=bus_id,
        p_mw=float.fromhex("0x1.27cd16f7da3c6p+2"),
        vm_pu=1.0,
    )
    stored = pp.from_json(pp.to_json(reconstructed))

    assert state_fingerprint(reconstructed).value != state_fingerprint(stored).value
    assert validate_dataset._storage_equivalent(reconstructed, stored)

    stored.gen.at[0, "p_mw"] += 1e-6
    assert not validate_dataset._storage_equivalent(reconstructed, stored)


def test_runtime_target_checks_use_the_stored_full_snapshot() -> None:
    source = inspect.getsource(validate_dataset._validate_scenario)

    assert "solve_locked_probe(full)" in source
    assert "_alternative_audit(full)" in source
    assert "_solve_unlimited(full)" in source
    assert "_verify_witness(\n        label,\n        witness,\n        full," in source


def _label(regime: str, length: int, optimality: str):
    """Minimal stand-in carrying only the fields the optimality audit reads."""
    from types import SimpleNamespace

    return SimpleNamespace(
        resolution_regime=regime,
        witness_length=length,
        witness_optimality=optimality,
    )


def _report():
    return validate_dataset.ScenarioValidation(scenario_id="S0001", valid=True)


def test_witness_action_context_uses_a_converged_intermediate_solution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solved_net = object()
    monkeypatch.setattr(
        validate_dataset,
        "solve_locked_probe",
        lambda _net: SimpleNamespace(
            status="SOLVED",
            solved_net=solved_net,
        ),
    )
    monkeypatch.setattr(
        validate_dataset,
        "evaluate_solved_feasibility",
        lambda net: (
            SimpleNamespace(
                generator_q_status=(
                    SimpleNamespace(
                        gen_id=7,
                        status="Q_LIMITED_UPPER",
                    ),
                )
            )
            if net is solved_net
            else pytest.fail("validator evaluated the wrong solved state")
        ),
    )
    monkeypatch.setattr(
        validate_dataset,
        "_diagnostic_q_context",
        lambda *_args, **_kwargs: pytest.fail(
            "converged intermediate state must use direct Q evidence"
        ),
    )

    context = validate_dataset._witness_action_q_context(
        object(),
        {"witness": {}},
    )

    assert context == {7: "Q_LIMITED_UPPER"}


@pytest.mark.parametrize(
    ("regime", "length", "optimality", "direct_available", "expect_failure"),
    [
        # Length 1 and 2 are decidable from the direct-restorer audit the validator already runs.
        ("DIRECT", 1, "EXACT_MINIMUM", True, False),
        ("DIRECT", 2, "EXACT_MINIMUM", True, True),
        ("DIRECT", 1, "UPPER_BOUND", True, True),
        ("SEQUENTIAL", 2, "EXACT_MINIMUM", False, False),
        ("SEQUENTIAL", 2, "EXACT_MINIMUM", True, True),
        # Beyond length 2 the claim needs exhaustive-search evidence the artifacts do not carry.
        ("SEQUENTIAL", 3, "EXACT_MINIMUM", False, True),
        ("SEQUENTIAL", 3, "UPPER_BOUND", False, False),
    ],
)
def test_witness_optimality_audit_certifies_only_decidable_claims(
    regime: str,
    length: int,
    optimality: str,
    direct_available: bool,
    expect_failure: bool,
) -> None:
    report = _report()

    validate_dataset._verify_witness_optimality(
        _label(regime, length, optimality),
        direct_restorer_available=direct_available,
        report=report,
    )

    assert bool(report.failed_checks) is expect_failure


def _terminal_feasibility() -> SolvedFeasibility:
    return SolvedFeasibility(
        feasible=False,
        generator_p_within_limits=True,
        generator_q_within_limits=True,
        external_grid_within_limits=True,
        connected=True,
        loads_energized=True,
        voltage=VoltageEnvelope(
            min_vm_pu=0.79,
            max_vm_pu=1.04,
            low_bus_ids=(44,),
            high_bus_ids=(),
            hard_envelope_ok=False,
            runtime_quality_ok=False,
        ),
        generator_q_status=(),
        slack_results=(),
        q_limited_gen_ids=(),
        failure_reasons=(),
        policy_version="test-feasibility",
    )


def test_terminal_pf_audit_rejects_a_missing_or_non_convergent_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feasibility = _terminal_feasibility()
    monkeypatch.setattr(
        validate_dataset,
        "evaluate_solved_feasibility",
        lambda _net: feasibility,
    )
    terminal = SimpleNamespace(solved_net=object())
    missing = _report()
    validate_dataset._verify_terminal_pf(terminal, {}, missing)

    diverged = _report()
    validate_dataset._verify_terminal_pf(
        terminal,
        {"terminal_pf": {"converged": False}},
        diverged,
    )

    accepted = _report()
    validate_dataset._verify_terminal_pf(
        terminal,
        {
            "terminal_pf": {
                "converged": True,
                "feasibility": feasibility.model_dump(mode="json"),
            }
        },
        accepted,
    )

    assert missing.failed_checks and diverged.failed_checks
    assert not accepted.failed_checks
    assert accepted.details["witness_terminal_voltage"] == [
        "min_vm_pu=0.79, max_vm_pu=1.04, hard_envelope_ok=False"
    ]


def test_terminal_pf_audit_rejects_voltage_metrics_that_differ_from_resolve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feasibility = _terminal_feasibility()
    monkeypatch.setattr(
        validate_dataset,
        "evaluate_solved_feasibility",
        lambda _net: feasibility,
    )
    stored = feasibility.model_dump(mode="json")
    stored["voltage"]["min_vm_pu"] = 0.80
    report = _report()

    validate_dataset._verify_terminal_pf(
        SimpleNamespace(solved_net=object()),
        {
            "terminal_pf": {
                "converged": True,
                "feasibility": stored,
            }
        },
        report,
    )

    assert report.failed_checks == ["check_12_witness_and_descriptors"]
