#!/usr/bin/env python3
"""Two driven EB electrodes in RZ that are separate bodies: a coax with a ring.

The other multi-electrode fixtures in this directory drive one sphere whose
surface is split at the equator into two patches. That exercises the corrector's
plumbing, but it is not a physical configuration: with no resolved insulating
gap the two patches are short-circuited, and its capacitance matrix comes out
badly non-reciprocal.

This test uses the arrangement the feature is for -- a cathode rod on the axis,
a separate ring electrode held at its own potential, and the grounded vessel as
the Dirichlet wall -- and asserts the three properties that make such a
geometry a well-posed two-degree-of-freedom problem:

1. **Reciprocity.** ``C[j,k] = C[k,j]`` is a theorem for any set of conductors.
   The split-sphere geometry violates it by tens of percent; separate bodies do
   not.

2. **Not a Faraday cage.** If one electrode enclosed the other, the column sums
   of ``C`` would vanish and the enclosed electrode's voltage error would solve
   to identically zero -- the second actuator would be idle whatever the plasma
   did, and the 2x2 path would never be loaded. Non-zero column sums are what
   make this a genuine two-electrode test.

3. **The loop closes**: at the last correction the observer reads both
   electrodes within a small fraction of the bias of their targets.

No conductor surface coincides with a domain boundary: both bodies keep vacuum
to the wall. ``r = 0`` is the regularity axis rather than a wall, so the rod may
sit on it -- no inner vacuum is required.
"""

import numpy as np

from pywarpx import picmi
from pywarpx.callbacks import installafterEsolve
from pywarpx.multi_electrode_corrector import MultiElectrodeBiasCorrector

total_steps = 8
correction_interval = 2
assert total_steps % correction_interval == 0

r_wall, nr = 6.0e-2, 24
z_half, nz = 8.0e-2, 48
# radii kept off an integer cell count: an exactly integer-cell radius makes the
# capacitance come back NaN
rod_r, rod_z = 0.62e-2, 4.1e-2
ring_in, ring_out, ring_z = 3.1e-2, 4.1e-2, 0.7e-2
bias = -1.0e5

grid = picmi.CylindricalGrid(
    number_of_cells=[nr, nz],
    n_azimuthal_modes=1,
    lower_bound=[0.0, -z_half],
    upper_bound=[r_wall, z_half],
    lower_boundary_conditions=["none", "dirichlet"],
    upper_boundary_conditions=["dirichlet", "dirichlet"],
    lower_boundary_conditions_particles=["none", "absorbing"],
    upper_boundary_conditions_particles=["absorbing", "absorbing"],
    warpx_blocking_factor=8,
    warpx_max_grid_size=32,
)
solver = picmi.ElectromagneticSolver(grid=grid, method="Yee", cfl=0.9)

rod = f"min({rod_r}*{rod_r}-(x*x+y*y), {rod_z}*{rod_z}-z*z)"
ring = (
    f"min(min((x*x+y*y)-{ring_in}*{ring_in},"
    f" {ring_out}*{ring_out}-(x*x+y*y)), {ring_z}*{ring_z}-z*z)"
)
eb = picmi.EmbeddedBoundary(
    implicit_function=f"max({rod}, {ring})",
    potential=0.0,
)

# A handful of macroparticles, enough to induce charge on both electrodes
# without making the test slow.
distribution = picmi.ParticleListDistribution(
    x=[0.0141, 0.0213, 0.0177],
    y=[0.0, 0.0, 0.0],
    z=[0.0037, -0.0211, 0.0263],
    ux=[0.0] * 3,
    uy=[0.0] * 3,
    uz=[0.0] * 3,
    weight=[1.2e9, 0.9e9, 1.5e9],
)
electrons = picmi.Species(
    name="electrons", particle_type="electron", initial_distribution=distribution
)

sim = picmi.Simulation(
    solver=solver,
    max_steps=total_steps,
    particle_shape="linear",
    warpx_embedded_boundary=eb,
    warpx_use_filter=False,
    verbose=0,
)
sim.add_species(
    electrons,
    layout=picmi.GriddedLayout(n_macroparticle_per_cell=[0, 0], grid=grid),
)

# "region" selects the electrode's surface and sets its potential; "volume"
# encloses it for the charge integral, with its boundary in live vacuum. Roughly
# two cells of clearance is the measured requirement, and these have several.
electrodes = [
    {
        "name": "cathode",
        "region": f"(x*x+y*y<{rod_r}*{rod_r}*2.25)",
        "volume": "(x*x+y*y<0.018*0.018)*(z*z<0.052*0.052)",
        "potential": bias,
    },
    {
        "name": "ring",
        "region": f"(x*x+y*y>{ring_in}*{ring_in}*0.9)",
        "volume": "(x*x+y*y>0.022*0.022)*(x*x+y*y<0.052*0.052)*(z*z<0.018*0.018)",
        "potential": 0.0,
    },
]

corrector = MultiElectrodeBiasCorrector(
    sim=sim,
    correction_interval=correction_interval,
    electrodes=electrodes,
    qg_mode="reciprocity",
    observer="volume",
    verbose=False,
)

installafterEsolve(corrector.correct_field)

# afterInitEsolve does not fire in an electromagnetic run, so the setup solves
# are triggered explicitly, exactly as the other clamp inputs here do.
sim.initialize_inputs()
sim.initialize_warpx()
corrector.setup_after_init()

sim.step(total_steps)

setup = corrector.setup_state()
cap = np.asarray(setup["capacitance_matrix"])
assert cap.shape == (2, 2), f"expected a 2x2 capacitance matrix, got {cap.shape}"
assert setup["observer"] == "volume"
assert setup["qg_mode"] == "reciprocity"

# 1. reciprocity
off_scale = max(abs(cap[0, 1]), abs(cap[1, 0]))
asymmetry = abs(cap[0, 1] - cap[1, 0]) / off_scale
assert asymmetry < 5.0e-3, (
    f"capacitance matrix is not reciprocal: C01={cap[0, 1]:.6e}, "
    f"C10={cap[1, 0]:.6e}, relative asymmetry {asymmetry:.3e}. Reciprocity is a "
    "theorem for a set of conductors, so this means the two electrodes are not "
    "separate bodies."
)

# 2. not a Faraday cage: each electrode must return charge to the grounded wall
column_ratio = np.sum(cap, axis=0) / np.diag(cap)
assert np.all(np.abs(column_ratio) > 0.05), (
    f"column sums {column_ratio} are near zero, so one electrode encloses the "
    "other. The enclosed electrode's voltage error then solves to identically "
    "zero and its actuator is idle by construction: this would not be a "
    "two-degree-of-freedom test."
)

# 3. both electrodes reach their targets, as the corrector sees them
state = corrector.last_correction_state()
assert state is not None, "the corrector never ran a correction"
# voltage_before is what the observer read at the last correction, i.e. the
# error the loop had to remove; after a few corrections it must be close to the
# target, or the loop is not closing.
reached = np.asarray(state["voltage_before"])
targets = np.asarray(setup["target_voltages"])
scale = np.max(np.abs(targets[targets != 0.0]))
assert np.all(np.abs(reached - targets) < 0.15 * scale), (
    f"the loop is not closing: the observer read {reached} at the last "
    f"correction against targets {targets}"
)

# The independent check -- a line integral of E_r from the rod out to the
# grounded wall, which the corrector never uses -- is deliberately left to the
# research probe `probe_rz_coax_ring.py` rather than reimplemented here. It needs
# a box-by-box gather that has no place in a CI input, and the three assertions
# above are the ones that make this geometry a well-posed two-electrode problem.

print(
    f"coax+ring: asymmetry {asymmetry:.2e}, column ratios {column_ratio}, "
    f"observer read {reached} V against targets {targets} V"
)
