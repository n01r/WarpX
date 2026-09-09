#!/usr/bin/env python3
"""The RZ discrete Shockley--Ramo identity with a periodic axial direction.

The Cartesian counterpart is
``inputs_test_3d_adjoint_weighting_periodic_picmi.py``; see its docstring for
why a periodic direction changes which rows carry equations. RZ adds the parts
that are specific to cylindrical geometry: the ``r=0`` regularity axis is still
Neumann and is not a wall, the outer radius stays grounded, and the ``2*pi*r``
EB measure and cylindrical nodal volumes must survive the periodic wrap in z.

Charge is placed within half a cell of both z faces so its deposition support
crosses the periodic seam, and one particle sits near the axis so the small
near-axis nodal volumes are exercised at the same time.
"""

import numpy as np

from pywarpx import picmi
from pywarpx.multi_electrode_corrector import MultiElectrodeBiasCorrector

nr, nz = 24, 48
rmax, zhalf = 0.8, 0.8
radius = 0.18
dz = 2.0 * zhalf / nz

grid = picmi.CylindricalGrid(
    number_of_cells=[nr, nz],
    n_azimuthal_modes=1,
    lower_bound=[0.0, -zhalf],
    upper_bound=[rmax, zhalf],
    lower_boundary_conditions=["none", "periodic"],
    upper_boundary_conditions=["dirichlet", "periodic"],
    lower_boundary_conditions_particles=["none", "periodic"],
    upper_boundary_conditions_particles=["absorbing", "periodic"],
    warpx_blocking_factor=8,
    warpx_max_grid_size=16,
)
solver = picmi.ElectromagneticSolver(grid=grid, method="Yee", cfl=0.9)
eb = picmi.EmbeddedBoundary(
    implicit_function="-(x*x+y*y+z*z-radius*radius)",
    potential=0.0,
    radius=radius,
)

# z within half a cell of each periodic face, plus a near-axis particle.
distribution = picmi.ParticleListDistribution(
    x=[0.2871, 0.4317, 0.0123, 0.6213],
    y=[0.0311, -0.0247, 0.0, 0.0189],
    z=[zhalf - 0.4 * dz, -zhalf + 0.4 * dz, 0.2637, -0.1179],
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
    electrodes=[{"name": "sphere", "region": "1", "potential": 0.0}],
    qg_mode="reciprocity",
    adjoint_tolerance=2.0e-10,
    adjoint_max_iterations=200,
)

sim.initialize_inputs()
sim.initialize_warpx()
corrector.setup_after_init()

warpx = corrector._warpx()

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
    "RZ periodic adjoint reciprocity: "
    f"grounded_solve={q_solve:+.16e} C, "
    f"adjoint={q_adjoint:+.16e} C, rel_error={rel_error:.3e}"
)

assert abs(q_solve) > 1.0e-13, (
    f"the fixture induced too little charge to compare: {q_solve:+.6e} C"
)
assert np.signbit(q_adjoint) == np.signbit(q_solve), (
    f"adjoint and grounded solve disagree in sign: {q_adjoint:+.6e} vs {q_solve:+.6e}"
)
assert rel_error < 2.0e-6, (
    f"periodic RZ discrete adjoint identity failed: relative error {rel_error:.3e}"
)
