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

``--decomposed`` additionally exercises cut-free regular and covered boxes. The
translated sphere contains a whole box, including its one-cell Yee EB halo,
while distant boxes do not intersect it. This guards every cut-centroid access
in the adjoint assembly and finalization, not just convergence of the solve.
"""

import argparse

import numpy as np

from pywarpx import picmi
from pywarpx.multi_electrode_corrector import MultiElectrodeBiasCorrector

parser = argparse.ArgumentParser()
parser.add_argument("--decomposed", action="store_true")
args = parser.parse_args()

nx = ny = nz = 32 if args.decomposed else 24
center = np.full(3, -0.2) if args.decomposed else np.zeros(3)
radius = 0.48 if args.decomposed else 0.22
grid = picmi.Cartesian3DGrid(
    number_of_cells=[nx, ny, nz],
    lower_bound=[-0.8, -0.8, -0.8],
    upper_bound=[0.8, 0.8, 0.8],
    lower_boundary_conditions=["dirichlet"] * 3,
    upper_boundary_conditions=["dirichlet"] * 3,
    lower_boundary_conditions_particles=["absorbing"] * 3,
    upper_boundary_conditions_particles=["absorbing"] * 3,
    warpx_blocking_factor=8,
    warpx_max_grid_size=8 if args.decomposed else 24,
)
solver = picmi.ElectromagneticSolver(grid=grid, method="Yee", cfl=0.9)
eb = picmi.EmbeddedBoundary(
    implicit_function="-((x-xc)**2+(y-yc)**2+(z-zc)**2-radius*radius)",
    potential=0.0,
    radius=radius,
    xc=center[0],
    yc=center[1],
    zc=center[2],
)

# deliberately off-node, spread over all octants
positions = np.array(
    [
        [0.3123, 0.2051, 0.2637],
        [-0.4271, 0.3311, -0.3173],
        [0.1317, -0.4247, 0.1179],
        [-0.2213, 0.0189, -0.4121],
    ]
)
if args.decomposed:
    positions = center + np.array(
        [
            [0.4013, 0.3021, 0.3517],
            [-0.3817, 0.3511, -0.3413],
            [0.3617, -0.3947, 0.3079],
            [-0.3613, 0.3189, -0.4021],
        ]
    )

# The complete CIC support stays outside the conductor and inside the domain.
dx = 1.6 / nx
assert np.all(np.linalg.norm(positions - center, axis=1) > radius + np.sqrt(3) * dx)
assert np.all(np.abs(positions) + dx < 0.8)
distribution = picmi.ParticleListDistribution(
    x=positions[:, 0],
    y=positions[:, 1],
    z=positions[:, 2],
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
        {"name": "upper", "region": f"(z>{center[2]})", "potential": +250.0},
        {"name": "lower", "region": f"(z<={center[2]})", "potential": -400.0},
    ],
    qg_mode="reciprocity",
    adjoint_tolerance=2.0e-10,
    adjoint_max_iterations=200,
    verbose=True,
)

sim.initialize_inputs()
sim.initialize_warpx()
warpx = corrector._warpx()

if args.decomposed:
    # Check the actual partition geometrically. For a sphere, nearest/farthest
    # points of each box give exact regular/covered bounds; include the one-cell
    # EB halo used by the Yee solver when determining whether cut data exist.
    boxes = warpx.boxArray(0)
    assert boxes.size > 1, "the decomposed test requires multiple boxes"
    n_regular = n_covered = 0
    for box in boxes:
        lo = -0.8 + dx * (np.array([box.small_end[d] for d in range(3)]) - 1)
        hi = -0.8 + dx * (np.array([box.big_end[d] for d in range(3)]) + 2)
        nearest = np.maximum(np.maximum(lo - center, center - hi), 0.0)
        farthest = np.maximum(np.abs(lo - center), np.abs(hi - center))
        n_regular += np.linalg.norm(nearest) > radius
        n_covered += np.linalg.norm(farthest) < radius
    print(
        f"3D partition: {boxes.size} boxes, "
        f"{n_regular} geometrically regular, {n_covered} geometrically covered"
    )
    assert n_regular > 0, "the fixture needs cut-free regular boxes"
    assert n_covered > 0, "the fixture needs cut-free covered boxes"

corrector.setup_after_init()

# --- the discrete adjoint identity for each grounded hemisphere ------------
warpx.set_potential_on_eb("0.0")
warpx.solve_poisson_efield()
q_solve = np.array(
    [
        warpx.compute_eb_charge(weighting=region, field="Efield_fp")
        for region in corrector.regions
    ]
)
q_adjoint = np.asarray(
    warpx.grounded_charge_from_adjoint(psi_fields=corrector._psi_names, lev=0)
)

for name, grounded, adjoint in zip(corrector.names, q_solve, q_adjoint):
    scale = max(abs(grounded), abs(adjoint), 1.0e-30)
    rel_error = abs(adjoint - grounded) / scale
    print(
        f"3D adjoint reciprocity ({name}): "
        f"grounded_solve={grounded:+.16e} C, "
        f"adjoint={adjoint:+.16e} C, rel_error={rel_error:.3e}"
    )
    assert np.signbit(adjoint) == np.signbit(grounded), (
        f"3D adjoint and grounded solve disagree in sign for {name}: "
        f"{adjoint:+.16e} vs {grounded:+.16e} C"
    )
    assert rel_error < 2.0e-6, (
        f"3D discrete adjoint identity failed for {name}: "
        f"relative error {rel_error:.3e} >= 2e-6"
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
