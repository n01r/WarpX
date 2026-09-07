#!/usr/bin/env python3
"""Regression test: the RZ observer under a RADIAL domain decomposition.

``inputs_test_rz_adjoint_weighting_picmi.py`` splits the domain only along z, so
every box straddles the embedded sphere. Two defects hid behind that:

1. ``WeightedChargeOnEB`` samples Efield one cell outside the cell it
   integrates, so a cut cell on a box boundary reads ghost data. Callers that
   drive an electrostatic solve at run time leave those ghosts stale, which made
   the EB flux integral -- and hence the capacitance -- depend on the
   decomposition.
2. The adjoint solve read ``MultiCutFab`` edge centroids on boxes carrying no
   cut cells. ``MultiCutFab::const_array`` only checks that with
   ``AMREX_ASSERT``, a no-op in Release, so this was undefined behaviour and
   crashed once some box missed the sphere entirely.

A radial split produces boxes that are entirely outside the conductor, which is
what exercises both. The assertions below are the same identities the
z-split test checks; they fail (or the run crashes) if either defect returns.
"""

import numpy as np

from pywarpx import picmi
from pywarpx.multi_electrode_corrector import MultiElectrodeBiasCorrector

nr, nz = 32, 64
# max_grid_size 16 splits BOTH directions: 2 in r by 4 in z. The outer-r boxes
# at large |z| contain no part of the sphere at all.
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
    warpx_max_grid_size=16,
)
solver = picmi.ElectromagneticSolver(grid=grid, method="Yee", cfl=0.9)
eb = picmi.EmbeddedBoundary(
    implicit_function="-(x*x+y*y+z*z-radius*radius)",
    potential=0.0,
    radius=0.18,
)

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

corrector = MultiElectrodeBiasCorrector(
    sim=sim,
    correction_interval=1,
    electrodes=[{"name": "sphere", "region": "1", "potential": 125.0}],
    qg_mode="reciprocity",
    adjoint_tolerance=2.0e-10,
    adjoint_max_iterations=200,
    verbose=True,
)

sim.initialize_inputs()
sim.initialize_warpx()
# Reaching this point at all is the regression guard for defect 2: the adjoint
# solve runs here and used to abort on the EB-free boxes.
corrector.setup_after_init()

warpx = corrector._warpx()
n_boxes = (
    corrector._mfr()
    .get("Efield_fp", dir=corrector._Direction(0), level=0)
    .box_array()
    .size
)
print(f"RZ decomposed adjoint test: {n_boxes} boxes")
assert n_boxes >= 8, (
    f"this test is only meaningful with a split domain, got {n_boxes} box(es)"
)

# ---------------------------------------------------------------------------
# Defect 1: the EB flux integral must not depend on the decomposition.
# The adjoint observer never reads ghost cells, so it was already correct;
# requiring the grounded solve to reproduce it is what pins the flux integral.
# ---------------------------------------------------------------------------
warpx.set_potential_on_eb("0.0")
warpx.solve_poisson_efield()
q_solve = float(warpx.compute_eb_charge(weighting="1", field="Efield_fp"))
q_adjoint = float(
    sum(warpx.grounded_charge_from_adjoint(psi_fields=corrector._psi_names, lev=0))
)
rel_error = abs(q_adjoint - q_solve) / max(abs(q_solve), abs(q_adjoint), 1.0e-30)
print(
    "RZ decomposed adjoint reciprocity: "
    f"grounded_solve={q_solve:+.16e} C, adjoint={q_adjoint:+.16e} C, "
    f"rel_error={rel_error:.3e}"
)
assert rel_error < 2.0e-6, (
    f"the EB flux integral depends on the domain decomposition: {rel_error:.3e}"
)

# The same comparison through the corrector's own path, which is what a run
# actually uses, and which also covers the capacitance built from that integral.
warpx.set_potential_on_eb(corrector.potential_expression)
state = corrector.measure_voltage_state()
comparison = corrector.compare_grounded_charge(state["grounded_charge"])
worst = float(np.max(comparison["relative_difference"]))
print(f"RZ decomposed grounded cross-check max rel. difference: {worst:.3e}")
assert worst < 1.0e-5, (
    f"adjoint observer and grounded solve disagree under decomposition: {worst:.3e}"
)

# The clamp must still drive the electrode to its target on a split domain.
corrector.correct_field()
remeasured = corrector.measure_voltage_state()["voltage"]
print(f"target={corrector.v_target}, re-measured after correction={remeasured}")
np.testing.assert_allclose(
    remeasured, np.asarray(corrector.v_target), rtol=1.0e-9, atol=1.0e-6
)
