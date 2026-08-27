# ABOUTME: Runs deterministic pandapower AC power flows for agent-facing convergence verdicts.
# ABOUTME: Provides voltage-only quality, Q-limit warnings, and failed-solve diagnostics.
from __future__ import annotations

import copy
import math
import warnings as warnings_module
from importlib import import_module
from typing import Any, Literal

from pandapower.auxiliary import LoadflowNotConverged

from restorebench.physics import solver as locked_solver
from restorebench.physics.feasibility import evaluate_solved_feasibility
from restorebench.physics.q_saturation import q_saturation_context
from restorebench.physics.retreat import retreat_evidence
from restorebench.physics.solver import solve_locked_probe
from restorebench.schemas.errors import ToolFailureError
from restorebench.schemas.feedback import SandboxNet
from restorebench.schemas.power_flow import (
    ConvergenceResult,
    NRDiagnostics,
    OperatingWarning,
    PowerFlowResult,
    QualityResult,
    QualitySymptom,
)


INIT = locked_solver.INIT
MAX_ITERATION = locked_solver.MAX_ITERATION
DEFAULT_TOLERANCE = locked_solver.PRIMARY_TOLERANCE_MVA
RECOVERY_TOLERANCE = locked_solver.RECOVERY_TOLERANCE_MVA
VM_MIN = 0.95
VM_MAX = 1.05
Q_LIMIT_EPS = 1e-3
LOCAL_NOSE_ITERATIONS = 22
DiagnosticsSource = Literal["local_nose"]
# Only snapshots curated on the shared trajectory yield shared retreat evidence; the pre-refactor
# corpus and synthetically stressed nets do not, so the superseded retreat stays reachable and the
# default permits it. Flip this to False once the replacement corpus is frozen: from that point a
# target snapshot without shared evidence is a regression that would silently restore the
# runtime/witness Q-context split, and callers can already demand that today by passing
# allow_legacy_retreat=False.
ALLOW_LEGACY_RETREAT = True


class LegacyRetreatWarning(UserWarning):
    """Raised when diagnostics fall back to the pre-shared-trajectory retreat."""


def run_ac_pf(
    net: Any | SandboxNet,
    *,
    diagnostics_source: DiagnosticsSource = "local_nose",
) -> PowerFlowResult:
    """Run the locked AC PF and return the solver-owned convergence verdict."""
    source_net = _resolve_net(net)
    probe = solve_locked_probe(source_net)
    if probe.status == "SOLVED":
        sandbox = probe.solved_net
        feasibility = evaluate_solved_feasibility(sandbox)
        slack = feasibility.slack_results[0] if feasibility.slack_results else None
        return PowerFlowResult(
            converged=True,
            iterations=_iterations(sandbox),
            tolerance_used=probe.tolerance_used_mva,
            runtime_ms=probe.elapsed_ms,
            diagnostics=None,
            quality=assess_quality(sandbox),
            warnings=detect_warnings(sandbox),
            solver_attempt_count=probe.solver_attempt_count,
            recovery_used=probe.recovery_used,
            feasibility=feasibility,
            generator_q_status=list(feasibility.generator_q_status),
            slack=slack,
        )

    error_message = probe.error_message or "locked AC probe found no solution"
    diagnostics = extract_nr_diagnostics(
        failed_net=source_net,
        source_net=source_net,
        error_message=error_message,
        diagnostics_source=diagnostics_source,
    )
    return PowerFlowResult(
        converged=False,
        iterations=diagnostics.iterations_attempted,
        tolerance_used=probe.tolerance_used_mva,
        runtime_ms=probe.elapsed_ms,
        error_message=error_message,
        diagnostics=diagnostics,
        quality=None,
        warnings=[],
        solver_attempt_count=probe.solver_attempt_count,
        recovery_used=probe.recovery_used,
        feasibility=None,
        generator_q_status=[],
        slack=None,
    )


def check_convergence(
    net: Any | SandboxNet,
    *,
    diagnostics_source: DiagnosticsSource = "local_nose",
) -> ConvergenceResult:
    """Run the locked AC PF and return convergence plus diagnostics only."""
    source_net = _resolve_net(net)
    probe = solve_locked_probe(source_net)
    if probe.status == "SOLVED":
        return ConvergenceResult(
            converged=True,
            iterations=_iterations(probe.solved_net),
            tolerance_used=probe.tolerance_used_mva,
            diagnostics=None,
            solver_attempt_count=probe.solver_attempt_count,
            recovery_used=probe.recovery_used,
        )

    error_message = probe.error_message or "locked AC probe found no solution"
    diagnostics = extract_nr_diagnostics(
        failed_net=source_net,
        source_net=source_net,
        error_message=error_message,
        diagnostics_source=diagnostics_source,
    )
    return ConvergenceResult(
        converged=False,
        iterations=diagnostics.iterations_attempted,
        tolerance_used=probe.tolerance_used_mva,
        diagnostics=diagnostics,
        solver_attempt_count=probe.solver_attempt_count,
        recovery_used=probe.recovery_used,
    )


def assess_quality(net: Any | SandboxNet) -> QualityResult:
    """Assess voltage-only post-convergence quality on a sandbox copy."""
    sandbox = copy.deepcopy(_resolve_net(net))
    voltages = sandbox.res_bus["vm_pu"].dropna()
    symptoms: list[QualitySymptom] = []

    for bus_id, value in voltages.items():
        vm_pu = float(value)
        if vm_pu < VM_MIN:
            symptoms.append(QualitySymptom(type="V_LOW", element_id=int(bus_id), value=vm_pu, threshold=VM_MIN))
        elif vm_pu > VM_MAX:
            symptoms.append(QualitySymptom(type="V_HIGH", element_id=int(bus_id), value=vm_pu, threshold=VM_MAX))

    worst_bus, worst_vm = _worst_voltage(voltages, symptoms)
    return QualityResult(
        clean=not symptoms,
        n_buses_out_of_band=len(symptoms),
        worst_vm_pu=worst_vm,
        worst_vm_bus=worst_bus,
        symptoms=symptoms,
    )


def detect_warnings(net: Any | SandboxNet) -> list[OperatingWarning]:
    """Detect generator Q-limit operating warnings on a converged sandbox copy."""
    sandbox = copy.deepcopy(_resolve_net(net))
    warnings: list[OperatingWarning] = []
    if not hasattr(sandbox, "res_gen") or "q_mvar" not in sandbox.res_gen:
        return warnings

    for gen_id, gen in sandbox.gen.iterrows():
        if not bool(gen.get("in_service", True)) or gen_id not in sandbox.res_gen.index:
            continue

        q_mvar = float(sandbox.res_gen.at[gen_id, "q_mvar"])
        min_q = float(gen["min_q_mvar"])
        max_q = float(gen["max_q_mvar"])
        if not all(math.isfinite(value) for value in (q_mvar, min_q, max_q)):
            continue

        lower_headroom = abs(q_mvar - min_q)
        upper_headroom = abs(max_q - q_mvar)
        nearest_bound = min_q if lower_headroom <= upper_headroom else max_q
        nearest_headroom = min(lower_headroom, upper_headroom)

        if q_mvar <= min_q + Q_LIMIT_EPS or q_mvar >= max_q - Q_LIMIT_EPS:
            status = "Q_LIMITED_LOWER" if q_mvar <= min_q + Q_LIMIT_EPS else "Q_LIMITED_UPPER"
            warnings.append(
                OperatingWarning(
                    type="GEN_Q_LIMIT",
                    gen_id=int(gen_id),
                    q_mvar=q_mvar,
                    nearest_bound=float(nearest_bound),
                    voltage_control_status=status,
                )
            )
            continue

        threshold = max(5.0, 0.05 * abs(max_q - min_q))
        if nearest_headroom < threshold:
            warnings.append(
                OperatingWarning(
                    type="GEN_Q_NEAR_LIMIT",
                    gen_id=int(gen_id),
                    q_mvar=q_mvar,
                    nearest_bound=float(nearest_bound),
                    voltage_control_status="PV_CONTROLLABLE",
                )
            )

    return warnings


def extract_nr_diagnostics(
    failed_net: Any,
    source_net: Any | None = None,
    *,
    error_message: str,
    diagnostics_source: DiagnosticsSource = "local_nose",
    allow_legacy_retreat: bool = ALLOW_LEGACY_RETREAT,
) -> NRDiagnostics:
    """Extract failed-solve diagnostics using the selected explicit route."""
    iterations_attempted = _iterations(failed_net, default=MAX_ITERATION)
    if diagnostics_source == "local_nose":
        if source_net is None:
            source_net = failed_net
        return _local_nose_diagnostics(
            source_net=source_net,
            allow_legacy_retreat=allow_legacy_retreat,
            iterations_attempted=iterations_attempted,
            error_message=error_message,
        )
    raise ValueError(f"Unknown diagnostics source: {diagnostics_source}")


def _resolve_net(net: Any | SandboxNet) -> Any:
    if not isinstance(net, SandboxNet):
        return net

    try:
        sandbox_module = import_module("restorebench.tools.sandbox")
        resolver = getattr(sandbox_module, "resolve_net", None)
        if resolver is None:
            sandbox_server = getattr(sandbox_module, "SandboxServer")
            resolver = sandbox_server.resolve_net
        return resolver(net)
    except Exception as exc:
        raise ToolFailureError("PowerFlowServer", f"SandboxServer.resolve_net failed: {exc}") from exc


def _run_locked_pf(net: Any, tolerance_mva: float) -> None:
    """Compatibility seam for direct locked attempts; the public path uses solve_locked_probe."""
    locked_solver._run_locked_pf(net, tolerance_mva)


def _try_run_locked_pf(net: Any, tolerance_mva: float) -> LoadflowNotConverged | None:
    try:
        _run_locked_pf(net, tolerance_mva)
        return None
    except LoadflowNotConverged as exc:
        return exc


def _local_nose_diagnostics(
    source_net: Any,
    *,
    iterations_attempted: int,
    error_message: str,
    allow_legacy_retreat: bool = ALLOW_LEGACY_RETREAT,
) -> NRDiagnostics:
    # The corpus witness reads its Q context at the shared retreat; a runtime context taken
    # elsewhere saturates generators the witness considers controllable, which the applicability
    # guard would then forbid the agent from moving.
    evidence = retreat_evidence(source_net)
    if evidence is not None:
        lower_scale, nose_net = evidence.lambda_value, evidence.solved_net
        gens_at_q_limit = list(evidence.upper_q_limited_gen_ids)
    else:
        # Only the pre-refactor corpus legitimately fails here, because it rides another
        # trajectory. Falling back silently on a target snapshot would hide that regression and
        # quietly restore the runtime/witness split, so the fallback is opt-in and announced.
        if not allow_legacy_retreat:
            raise ToolFailureError(
                "PowerFlowServer",
                "snapshot yields no shared retreat evidence; request the legacy retreat "
                "explicitly if this snapshot predates the shared trajectory",
            )
        warnings_module.warn(
            "shared retreat produced no evidence; using the legacy retreat for voltage "
            "diagnostics and the Q-unlimited snapshot for generator Q status",
            LegacyRetreatWarning,
            stacklevel=2,
        )
        lower_scale, nose_net = _largest_converging_load_scale(source_net)
        # The voltage picture can come from the retreat, but the Q status cannot: the witness
        # search reads it from the Q-unlimited snapshot whenever the shared retreat is silent,
        # and a context taken from another trajectory would forbid the agent moves the corpus
        # certifies as the solution.
        saturation = q_saturation_context(source_net)
        gens_at_q_limit = (
            sorted(
                gen_id
                for gen_id, status in saturation.items()
                if status == "Q_LIMITED_UPPER"
            )
            if saturation is not None
            else _upper_q_limited_gen_ids(nose_net)
        )

    voltages = nose_net.res_bus["vm_pu"].dropna()
    if voltages.empty:
        raise ToolFailureError("PowerFlowServer", "local-nose diagnostic produced no bus voltages")

    lowest_bus = int(voltages.idxmin())
    lowest_vm = float(voltages.loc[lowest_bus])

    return NRDiagnostics(
        iterations_attempted=iterations_attempted,
        worst_bus=lowest_bus,
        lowest_vm_pu=lowest_vm,
        lowest_vm_bus=lowest_bus,
        gens_at_q_limit=gens_at_q_limit,
        max_mismatch_mw=None,
        max_mismatch_mvar=None,
        overstress=(1.0 / lower_scale) - 1.0 if lower_scale > 0 else None,
        error_message=error_message,
        diagnostics_source="local_nose",
    )


def _largest_converging_load_scale(source_net: Any) -> tuple[float, Any]:
    lower = 0.5
    lower_net = _solve_scaled_loads(source_net, lower)
    while lower_net is None and lower > 1e-6:
        lower *= 0.5
        lower_net = _solve_scaled_loads(source_net, lower)

    if lower_net is None:
        raise ToolFailureError("PowerFlowServer", "could not bracket a converging local-nose load scale")

    upper = 1.0
    best_scale = lower
    best_net = lower_net
    for _ in range(LOCAL_NOSE_ITERATIONS):
        mid = (lower + upper) / 2.0
        probe = _solve_scaled_loads(source_net, mid)
        if probe is None:
            upper = mid
        else:
            lower = mid
            best_scale = mid
            best_net = probe

    return best_scale, best_net


def _solve_scaled_loads(source_net: Any, scale: float) -> Any | None:
    # Deliberately the pre-shared-trajectory scaling: this feeds only the legacy fallback, whose
    # whole purpose is to reproduce the retreat a pre-refactor snapshot was built on. Snapshots
    # that ride the shared trajectory never reach here - they resolve through retreat_evidence.
    probe = copy.deepcopy(source_net)
    if len(probe.load):
        probe.load.loc[:, ["p_mw", "q_mvar"]] *= float(scale)
    result = solve_locked_probe(probe)
    if result.status == "NO_SOLUTION":
        return None
    return result.solved_net


def _upper_q_limited_gen_ids(solved_net: Any) -> list[int]:
    """Derive the upper-Q-saturated generators through the shared feasibility contract."""
    feasibility = evaluate_solved_feasibility(solved_net)
    return sorted(
        int(item.gen_id) for item in feasibility.generator_q_status if item.status == "Q_LIMITED_UPPER"
    )


def _gens_at_q_limit_from_results(net: Any) -> list[int]:
    if not hasattr(net, "res_gen") or "q_mvar" not in net.res_gen:
        return []

    saturated: list[int] = []
    for gen_id, gen in net.gen.iterrows():
        if gen_id not in net.res_gen.index:
            continue
        q_mvar = float(net.res_gen.at[gen_id, "q_mvar"])
        max_q = float(gen["max_q_mvar"])
        if not all(math.isfinite(value) for value in (q_mvar, max_q)):
            continue
        # This diagnostic drives the direction-aware voltage-setpoint guard:
        # raising vm_pu is invalid only at the upper reactive-power limit.
        if q_mvar >= max_q - Q_LIMIT_EPS:
            saturated.append(int(gen_id))
    return saturated


def _worst_voltage(voltages: Any, symptoms: list[QualitySymptom]) -> tuple[int, float]:
    if symptoms:
        worst_symptom = max(
            symptoms,
            key=lambda symptom: abs(symptom.value - symptom.threshold),
        )
        return worst_symptom.element_id, worst_symptom.value

    if len(voltages):
        deviations = (voltages.astype(float) - 1.0).abs()
        worst_bus = int(deviations.idxmax())
        return worst_bus, float(voltages.loc[worst_bus])

    return 0, float("nan")


def _iterations(net: Any, default: int = 0) -> int:
    ppc = getattr(net, "_ppc", {}) or {}
    value = ppc.get("iterations", default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)
