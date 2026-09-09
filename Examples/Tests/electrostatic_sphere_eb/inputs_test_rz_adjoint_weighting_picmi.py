#!/usr/bin/env python3
"""Validate the RZ discrete Shockley--Ramo observer and the voltage clamp.

Part 1 compares the charge induced on a grounded spherical EB in two fully
discrete ways:

1. deposit the off-node CIC particles, solve the metric-aware RZ EB Poisson
   problem, and integrate the resulting EB flux;
2. deposit the same particles and evaluate ``-sum rho_i V_i Psi_i``, with Psi
   from the exact transpose of that operator and flux functional.

Agreement exercises the RZ ``r+/-dr/2`` rows, the ``r=0`` regularity row, the
cut-edge gradient stencils, the ``2*pi*r_bnd`` surface measure, the cylindrical
nodal volumes, the sign convention, and the particle deposit.

Part 2 drives two independently biased electrodes -- the two hemispheres of the
same EB -- to different nonzero target potentials and re-measures the live field
afterwards. That exercises the capacitance matrix and its inversion, the
per-electrode unit fields, and the bias actually applied to Efield_fp.

``--actuator-gradient ordinary`` builds the unit fields from WarpX's full-grid
gradient instead of the cut-edge one. Every assertion below is unchanged: the
capacitance is measured from whichever unit fields are actually applied, so the
feedback stays self-consistent even though the individual matrix entries differ
between the two representations.
"""

import argparse

import numpy as np

from pywarpx import picmi
from pywarpx.multi_electrode_corrector import MultiElectrodeBiasCorrector
from pywarpx.multi_electrode_logger import (
    GroundedChargeCrossCheck,
    MultiElectrodeClampTelemetry,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--actuator-gradient", choices=["eb_aware", "ordinary"], default="eb_aware"
)
args, _ = parser.parse_known_args()

nr, nz = 32, 64
grid = picmi.CylindricalGrid(
    number_of_cells=[nr, nz],
    n_azimuthal_modes=1,
    lower_bound=[0.0, -0.8],
    upper_bound=[0.8, 0.8],
    lower_boundary_conditions=["none", "dirichlet"],
    upper_boundary_conditions=["dirichlet", "dirichlet"],
    lower_boundary_conditions_particles=["none", "absorbing"],
    upper_boundary_conditions_particles=["absorbing", "absorbing"],
    warpx_blocking_factor=8,
    warpx_max_grid_size=32,
)
solver = picmi.ElectromagneticSolver(grid=grid, method="Yee", cfl=0.9)
eb = picmi.EmbeddedBoundary(
    implicit_function="-(x*x+y*y+z*z-radius*radius)",
    potential=0.0,
    radius=0.18,
)

# Deliberately off-node and at several radii, including a point whose CIC
# support touches the small-volume near-axis nodes.
distribution = picmi.ParticleListDistribution(
    x=[0.0123, 0.2871, 0.4317, 0.6213],
    y=[0.0, 0.0311, -0.0247, 0.0189],
    z=[0.2637, -0.3173, 0.1179, -0.4121],
    ux=[0.0] * 4,
    uy=[0.0] * 4,
    uz=[0.0] * 4,
    weight=[1.0e9, 1.7e9, 0.8e9, 1.3e9],
)
electrons = picmi.Species(
    name="electrons",
    particle_type="electron",
    initial_distribution=distribution,
)

sim = picmi.Simulation(
    solver=solver,
    time_step_size=1.0e-12,
    max_steps=0,
    particle_shape="linear",
    warpx_embedded_boundary=eb,
    warpx_use_filter=False,
    verbose=0,
)
sim.add_species(
    electrons,
    layout=picmi.GriddedLayout(n_macroparticle_per_cell=[0, 0, 0], grid=grid),
)

# Two independently driven electrodes at different nonzero potentials: the upper
# and lower hemispheres of the same embedded sphere.
electrodes = [
    {"name": "upper", "region": "(z>0.0)", "potential": +250.0},
    {"name": "lower", "region": "(z<=0.0)", "potential": -400.0},
]

corrector = MultiElectrodeBiasCorrector(
    sim=sim,
    correction_interval=1,
    electrodes=electrodes,
    qg_mode="reciprocity",
    actuator_gradient=args.actuator_gradient,
    adjoint_tolerance=2.0e-10,
    adjoint_max_iterations=200,
    verbose=True,
)

sim.initialize_inputs()
sim.initialize_warpx()

warpx = corrector._warpx()
mfr = corrector._mfr()

# Setup drives several one-off Poisson solves, each of which overwrites
# Efield_fp, and it leaves the EB parser holding the configured bias pattern. It
# must hand the live EM field back exactly as it found it, in either actuator
# representation. Put a distinctive nonzero field there first, so this is a real
# comparison rather than a check that zero survives -- on a restart the loaded
# field is not zero either.
warpx.set_potential_on_eb("123.0*(z>0.0)")
warpx.solve_poisson_efield()
efield_before = {
    comp: mfr.get("Efield_fp", dir=corrector._Direction(comp), level=0).copy()
    for comp in (0, 1, 2)
}
assert max(field.norm0(0, 0, False, False) for field in efield_before.values()) > 0.0, (
    "the restoration check needs a nonzero live field to be meaningful"
)

corrector.setup_after_init()

for comp in (0, 1, 2):
    residual = mfr.get("Efield_fp", dir=corrector._Direction(comp), level=0).copy()
    residual.saxpy(-1.0, efield_before[comp], 0, 0, 1, 0)
    assert residual.norm0(0, 0, False, False) == 0.0, (
        f"setup_after_init did not restore Efield_fp component {comp}"
    )

# Registering setup on both afterInitEsolve and afterInitatRestart means it can
# be called twice; the second call must be a no-op rather than a re-solve.
capacitance_first = corrector.setup_state()["capacitance_matrix"].copy()
corrector.setup_after_init()
np.testing.assert_array_equal(
    corrector.setup_state()["capacitance_matrix"],
    capacitance_first,
    err_msg="a repeated setup_after_init rebuilt the calibration",
)
for comp in (0, 1, 2):
    residual = mfr.get("Efield_fp", dir=corrector._Direction(comp), level=0).copy()
    residual.saxpy(-1.0, efield_before[comp], 0, 0, 1, 0)
    assert residual.norm0(0, 0, False, False) == 0.0, (
        f"a repeated setup_after_init disturbed Efield_fp component {comp}"
    )

# ---------------------------------------------------------------------------
# Part 1: the discrete adjoint identity on the whole grounded conductor
# ---------------------------------------------------------------------------
warpx.set_potential_on_eb("0.0")
warpx.solve_poisson_efield()
q_solve = float(warpx.compute_eb_charge(weighting="1", field="Efield_fp"))
q_adjoint = float(
    sum(warpx.grounded_charge_from_adjoint(psi_fields=corrector._psi_names, lev=0))
)

abs_error = abs(q_adjoint - q_solve)
scale = max(abs(q_solve), abs(q_adjoint), 1.0e-30)
rel_error = abs_error / scale
print(
    "RZ adjoint reciprocity: "
    f"grounded_solve={q_solve:+.16e} C, "
    f"adjoint={q_adjoint:+.16e} C, rel_error={rel_error:.3e}"
)

assert np.signbit(q_adjoint) == np.signbit(q_solve), (
    "RZ adjoint and grounded solve disagree in sign: "
    f"{q_adjoint:+.16e} vs {q_solve:+.16e} C"
)
assert rel_error < 2.0e-6, (
    f"RZ discrete adjoint identity failed: relative error {rel_error:.3e} >= 2e-6"
)

# The clamp inverts the capacitance matrix every correction, so a badly
# conditioned two-electrode split would make the test meaningless.
setup = corrector.setup_state()
print(f"capacitance condition number: {setup['capacitance_condition']:.3e}")
assert setup["capacitance_condition"] < 1.0e3, (
    "two-electrode capacitance matrix is unexpectedly ill-conditioned: "
    f"{setup['capacitance_condition']:.3e}"
)

# ---------------------------------------------------------------------------
# Part 2: drive both electrodes to their targets and re-measure the live field
# ---------------------------------------------------------------------------
telemetry = MultiElectrodeClampTelemetry(
    corrector,
    out_csv="clamp_telemetry_test.csv",
    setup_json="clamp_setup_test.json",
)
cross_check = GroundedChargeCrossCheck(
    corrector,
    period=1,
    out_csv="clamp_grounded_crosscheck_test.csv",
)
telemetry.setup()
cross_check.setup()

# restore the configured bias pattern before the first correction
warpx.set_potential_on_eb(corrector.potential_expression)
corrector.correct_field()
telemetry.log()
cross_check.log()

state = corrector.last_correction_state()
assert state is not None
target = np.asarray(corrector.v_target)

# The correction must actually move the field: a no-op bias would leave
# voltage_before equal to the target and delta_voltage at zero.
assert np.max(np.abs(state["delta_voltage"])) > 1.0, (
    f"the clamp applied a negligible correction: {state['delta_voltage']}"
)

# Re-measure from the live Efield_fp. With relaxation = 1 the linear prediction
# is a fixed point, so this closes the loop through C^-1, the unit fields and
# the saxpy onto Efield_fp.
remeasured_state = corrector.measure_voltage_state()
remeasured = remeasured_state["voltage"]
print(f"target={target}, re-measured after correction={remeasured}")
np.testing.assert_allclose(remeasured, target, rtol=1.0e-9, atol=1.0e-6)

# The gain must describe the fields that were actually applied. Independently of
# C^-1 above, the measured EB flux charge must have moved by exactly C @ dV. This
# is what keeps the loop closed when the actuator representation changes: the
# individual entries of C differ between the two gradients, but each is the
# response of the observer to its own unit fields.
setup = corrector.setup_state()
assert setup["actuator_gradient"] == args.actuator_gradient
capacitance = setup["capacitance_matrix"]
measured_dq = remeasured_state["field_charge"] - state["field_charge"]
predicted_dq = capacitance @ state["delta_voltage"]
dq_scale = max(np.max(np.abs(predicted_dq)), np.finfo(float).tiny)
dq_error = np.max(np.abs(measured_dq - predicted_dq)) / dq_scale
print(f"actuator gain closure ({args.actuator_gradient}): rel. error {dq_error:.3e}")
assert dq_error < 1.0e-9, (
    "the applied unit fields do not reproduce the calibrated capacitance: "
    f"measured dQ={measured_dq}, predicted C@dV={predicted_dq}"
)

# The sparse cross-check compares the adjoint observer against an independent
# grounded solve on exactly the state the correction used.
comparison = corrector.compare_grounded_charge(state["grounded_charge"])
print(
    f"grounded cross-check max rel. difference: "
    f"{np.max(comparison['relative_difference']):.3e}"
)
assert np.max(comparison["relative_difference"]) < 1.0e-5, (
    "adjoint observer disagrees with the grounded solve per electrode: "
    f"{comparison['relative_difference']}"
)
