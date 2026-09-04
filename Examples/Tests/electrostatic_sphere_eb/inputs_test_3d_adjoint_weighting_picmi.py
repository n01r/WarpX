#!/usr/bin/env python3
"""Validate the Cartesian discrete Shockley--Ramo observer and the voltage clamp.

The 3D counterpart of ``inputs_test_rz_adjoint_weighting_picmi.py``: it checks the
same two properties on a Cartesian grid, so the 3D transpose stencil and the 3D
branch of the charge functional are covered as well.

1. The charge induced on a grounded spherical EB, measured with a Poisson solve
   plus an EB flux integral, must agree with ``-sum rho_i V_i Psi_i`` evaluated
   against the adjoint weighting potentials.
2. Two independently biased electrodes -- the two hemispheres of the same EB --
   must reach different nonzero target potentials after one correction.
"""

import numpy as np

from pywarpx import picmi
from pywarpx.multi_electrode_corrector import MultiElectrodeBiasCorrector

nx = ny = nz = 24
grid = picmi.Cartesian3DGrid(
    number_of_cells=[nx, ny, nz],
    lower_bound=[-0.8, -0.8, -0.8],
    upper_bound=[0.8, 0.8, 0.8],
    lower_boundary_conditions=["dirichlet"] * 3,
    upper_boundary_conditions=["dirichlet"] * 3,
    lower_boundary_conditions_particles=["absorbing"] * 3,
    upper_boundary_conditions_particles=["absorbing"] * 3,
    warpx_blocking_factor=8,
    warpx_max_grid_size=24,
)
solver = picmi.ElectromagneticSolver(grid=grid, method="Yee", cfl=0.9)
eb = picmi.EmbeddedBoundary(
    implicit_function="-(x*x+y*y+z*z-radius*radius)",
    potential=0.0,
    radius=0.22,
)

# deliberately off-node, spread over all octants
distribution = picmi.ParticleListDistribution(
    x=[0.3123, -0.4271, 0.1317, -0.2213],
    y=[0.2051, 0.3311, -0.4247, 0.0189],
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

corrector = MultiElectrodeBiasCorrector(
    sim=sim,
    correction_interval=1,
    electrodes=[
        {"name": "upper", "region": "(z>0.0)", "potential": +250.0},
        {"name": "lower", "region": "(z<=0.0)", "potential": -400.0},
    ],
    qg_mode="reciprocity",
    adjoint_tolerance=2.0e-10,
    adjoint_max_iterations=200,
    verbose=True,
)

sim.initialize_inputs()
sim.initialize_warpx()
corrector.setup_after_init()

warpx = corrector._warpx()

# --- the discrete adjoint identity on the whole grounded conductor ---------
warpx.set_potential_on_eb("0.0")
warpx.solve_poisson_efield()
q_solve = float(warpx.compute_eb_charge(weighting="1", field="Efield_fp"))
q_adjoint = float(
    sum(warpx.grounded_charge_from_adjoint(psi_fields=corrector._psi_names, lev=0))
)

scale = max(abs(q_solve), abs(q_adjoint), 1.0e-30)
rel_error = abs(q_adjoint - q_solve) / scale
print(
    "3D adjoint reciprocity: "
    f"grounded_solve={q_solve:+.16e} C, "
    f"adjoint={q_adjoint:+.16e} C, rel_error={rel_error:.3e}"
)
assert np.signbit(q_adjoint) == np.signbit(q_solve), (
    "3D adjoint and grounded solve disagree in sign: "
    f"{q_adjoint:+.16e} vs {q_solve:+.16e} C"
)
assert rel_error < 2.0e-6, (
    f"3D discrete adjoint identity failed: relative error {rel_error:.3e} >= 2e-6"
)

# --- drive both electrodes and re-measure the live field -------------------
warpx.set_potential_on_eb(corrector.potential_expression)
corrector.correct_field()
state = corrector.last_correction_state()
assert state is not None
assert np.max(np.abs(state["delta_voltage"])) > 1.0, (
    f"the clamp applied a negligible correction: {state['delta_voltage']}"
)

remeasured = corrector.measure_voltage_state()["voltage"]
target = np.asarray(corrector.v_target)
print(f"target={target}, re-measured after correction={remeasured}")
np.testing.assert_allclose(remeasured, target, rtol=1.0e-9, atol=1.0e-6)
