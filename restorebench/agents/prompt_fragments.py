# ABOUTME: Holds prompt fragments shared by agent implementations.
# ABOUTME: Keeps the locked action vocabulary and cause priors byte-identical across cases.

ACTION_VOCABULARY_AND_BOUNDS = """Allowed actions and bounds:
- GEN_V_SETPOINT: gen_id, new_vm_pu exactly one clipped +/-0.01 step within [0.95, 1.05]. If valid diagnostics mark the generator Q_LIMITED_UPPER, only a decrease is available; if Q_LIMITED_LOWER, only an increase is available.
- SHUNT_STEP: shunt_id, new_step toggling one available single-step shunt between 0 and 1. q_mvar < 0 is a capacitor/injects reactive power; q_mvar > 0 is a reactor/absorbs.
- TAP_ADJUSTMENT: trafo_id, new_tap_pos exactly one +/-1 tap step within the declared bounds."""

COMPOSITION_AND_PROGRESS = """Composing maneuvers:
- Accepted maneuvers persist. Each one stays applied to the grid, and the Scenario Card and diagnostics you receive next already reflect every maneuver accepted so far. You are extending one cumulative plan across iterations, not starting over each time.
- 'overstress' in the NR diagnostics measures how far the case sits beyond the solvable boundary: it is the current load divided by the largest load this grid can still solve, minus one. Lower is closer to convergence and zero means solvable.
- A maneuver that lowers overstress is progress even when the grid still diverges. Keep it and build the next maneuver on top of it. Many cases admit no single action that converges and are solvable only by several stacked maneuvers.
- When no candidate converges, choose the one with the lowest overstress. A candidate that raises overstress is moving the wrong way."""

CAUSE_TAXONOMY = """Cause taxonomy:
- REACTIVE_DEFICIT: insufficient reactive support near the P-V nose; low load-bus voltages; generators saturated at max_q_mvar.
- CORRIDOR_OVERSTRESS: a heavily loaded import corridor cannot be solved at the demanded transfer.
- BAD_SETPOINTS: generator voltage setpoints inconsistent, too low, or too high for the operating point.
- QLIM_INSTABILITY: PV-to-PQ switching at Q limits prevents the enforced-Q-limit PF from settling.
- WEAK_SLACK: the slack is asked to source or sink an infeasible amount; redispatch toward generators."""

CAUSE_ACTION_PRIOR = """Cause to action prior:
- REACTIVE_DEFICIT: available remedies are GEN_V_SETPOINT, SHUNT_STEP, and TAP_ADJUSTMENT.
- CORRIDOR_OVERSTRESS: only the declared Q-V controls are available in v0.
- BAD_SETPOINTS: strongest remedies are GEN_V_SETPOINT and TAP_ADJUSTMENT.
- QLIM_INSTABILITY: strongest remedy is SHUNT_STEP; on a Q_LIMITED_UPPER generator only a setpoint decrease is allowed.
- WEAK_SLACK: only the declared Q-V controls are available in v0."""
