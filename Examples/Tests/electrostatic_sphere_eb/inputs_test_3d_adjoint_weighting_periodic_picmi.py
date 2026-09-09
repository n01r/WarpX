#!/usr/bin/env python3
"""The discrete Shockley--Ramo identity with periodic outer boundaries.

The adjoint was originally posed against a fully grounded box. A periodic
direction has no wall, and its domain-edge nodes are ordinary unknowns wrapped
onto their images rather than constrained rows. Three things therefore have to
agree: the assertion that screens boundary types, the preconditioner's domain
boundary types, and the constraint mask that decides which rows carry equations.

If the mask still marked the periodic faces as constrained -- the behaviour
before periodic support -- the solve would still converge and return a plausible
weighting potential; it would simply be the wrong operator's. So the check here
is not "does it run" but whether

    Q = -sum_a rho_a Psi[a]

still reproduces an independent grounded Poisson solve plus EB flux integral,
with charge deliberately placed so that its deposition support crosses the
periodic seam.

x and y are periodic, z is grounded PEC. The embedded sphere supplies the
Dirichlet reference.
"""

import numpy as np

from pywarpx import picmi
from pywarpx.multi_electrode_corrector import MultiElectrodeBiasCorrector

nx = ny = nz = 24
half = 0.5
radius = 0.22

grid = picmi.Cartesian3DGrid(
    number_of_cells=[nx, ny, nz],
    lower_bound=[-half, -half, -half],
    upper_bound=[half, half, half],
    lower_boundary_conditions=["periodic", "periodic", "dirichlet"],
    upper_boundary_conditions=["periodic", "periodic", "dirichlet"],
    lower_boundary_conditions_particles=["periodic", "periodic", "absorbing"],
    upper_boundary_conditions_particles=["periodic", "periodic", "absorbing"],
    warpx_blocking_factor=8,
    warpx_max_grid_size=12,
)
solver = picmi.ElectromagneticSolver(grid=grid, method="Yee", cfl=0.9)
eb = picmi.EmbeddedBoundary(
    implicit_function="-(x*x+y*y+z*z-radius*radius)",
    potential=0.0,
    radius=radius,
)

# The first two particles sit within one cell of the x and y domain edges, so
# their CIC support wraps across the periodic seam and is only deposited
# correctly if periodic ownership is handled on both the forward and the
# transposed path. dx = 1/24, so 0.4896 is about half a cell inside the face.
distribution = picmi.ParticleListDistribution(
    x=[0.4896, -0.4896, 0.3137, 0.0213],
    y=[0.0171, 0.4896, -0.2911, 0.3702],
    z=[0.1183, -0.2237, 0.3319, -0.3771],
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
    "3-D periodic adjoint reciprocity: "
    f"grounded_solve={q_solve:+.16e} C, "
    f"adjoint={q_adjoint:+.16e} C, rel_error={rel_error:.3e}"
)

# A charge this far from the conductor still induces a clearly nonzero image
# charge; a near-zero q_solve would make the relative comparison meaningless.
assert abs(q_solve) > 1.0e-13, (
    f"the fixture induced too little charge to compare: {q_solve:+.6e} C"
)
assert np.signbit(q_adjoint) == np.signbit(q_solve), (
    f"adjoint and grounded solve disagree in sign: {q_adjoint:+.6e} vs {q_solve:+.6e}"
)
assert rel_error < 2.0e-6, (
    f"periodic discrete adjoint identity failed: relative error {rel_error:.3e}"
)
