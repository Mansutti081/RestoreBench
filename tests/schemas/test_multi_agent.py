# ABOUTME: Verifies multi-agent message schemas for Cases 3 and 5.
# ABOUTME: Covers analyst and executor round trips plus extra-field rejection.



def round_trip(model):
    assert type(model).model_validate(model.model_dump()) == model
    assert type(model).model_validate_json(model.model_dump_json()) == model


def maneuver() -> dict:
    return {
        "action": {"type": "GEN_V_SETPOINT", "gen_id": 2, "new_vm_pu": 1.01},
        "diagnosed_cause": "REACTIVE_DEFICIT",
        "rationale": "Raise the voltage setpoint by one atomic step.",
    }


def pf_result() -> dict:
    return {
        "converged": True,
        "iterations": 7,
        "tolerance_used": 1e-8,
        "runtime_ms": 21.0,
        "error_message": None,
        "diagnostics": None,
        "quality": {
            "clean": True,
            "n_buses_out_of_band": 0,
            "worst_vm_pu": 0.99,
            "worst_vm_bus": 4,
            "symptoms": [],
        },
        "warnings": [],
    }


