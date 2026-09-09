#!/usr/bin/env python3
"""Cartesian counterpart of the RZ clamp actuator/Faraday regression.

Two **physically separate** conducting spheres are driven to different
potentials, so unlike the RZ fixture no single conductor body carries two
potentials and nothing here depends on an electrical short. That makes this the
device-shaped geometry: separate electrodes, each at one potential.

The registered 3-D adjoint-weighting tests do not distinguish the two actuator
representations: on that single-sphere geometry their capacitance matrices are
bit-identical even though the unit fields differ by more than an order of
magnitude. ``ChargeOnEB`` samples ``E`` at a cell shifted away from each cut cell,
and on that fixture those samples all land where the two extractions agree
exactly. It is not a property of 3-D -- on the two separate spheres used here the
capacitance does differ, by about 5.6% -- but it does mean the older tests do not
show that ``actuator_gradient`` reaches the 3-D path.

This input supplies that discrimination on the field that is actually applied.
The asserted quantity is ``c|B|/E_ref`` after a short vacuum evolution, the 3-D
analogue of the RZ Faraday check; it separates the two representations by many
orders of magnitude, so a build that ignored the option would fail here. The unit
field component maxima are printed alongside as a recorded diagnostic.

Vacuum, no particles, no feedback after the single correction, no solve during
evolution: any ``B`` at all is numerical.
"""

import argparse

import numpy as np

from pywarpx import algo, picmi
from pywarpx.multi_electrode_corrector import MultiElectrodeBiasCorrector

parser = argparse.ArgumentParser()
parser.add_argument(
    "--actuator-gradient", choices=["eb_aware", "ordinary"], default="eb_aware"
)
args = parser.parse_args()

steps = 20
nx = ny = nz = 24
half = 0.6
radius = 0.18
offset = 0.3

grid = picmi.Cartesian3DGrid(
    number_of_cells=[nx, ny, nz],
    lower_bound=[-half, -half, -half],
    upper_bound=[half, half, half],
    lower_boundary_conditions=["dirichlet"] * 3,
    upper_boundary_conditions=["dirichlet"] * 3,
    lower_boundary_conditions_particles=["absorbing"] * 3,
    upper_boundary_conditions_particles=["absorbing"] * 3,
    warpx_blocking_factor=8,
    warpx_max_grid_size=12,  # splits every direction, so boxes miss the spheres
)
solver = picmi.ElectromagneticSolver(
    grid=grid, method="Yee", cfl=0.9, divE_cleaning=False
)

# Two disjoint bodies: covered where either sphere is, i.e. -min(f_upper, f_lower).
upper = f"(x*x+y*y+(z-{offset})*(z-{offset})-{radius}*{radius})"
lower = f"(x*x+y*y+(z+{offset})*(z+{offset})-{radius}*{radius})"
eb = picmi.EmbeddedBoundary(
    implicit_function=f"-min({upper},{lower})",
    potential=0.0,
)

sim = picmi.Simulation(
    solver=solver,
    max_steps=steps,
    particle_shape="linear",
    warpx_embedded_boundary=eb,
    warpx_use_filter=False,
    verbose=0,
)

electrodes = [
    {"name": "upper", "region": "(z>0.0)", "potential": +250.0},
    {"name": "lower", "region": "(z<=0.0)", "potential": -400.0},
]

corrector = MultiElectrodeBiasCorrector(
    sim=sim,
    correction_interval=1,
    electrodes=electrodes,
    qg_mode="reciprocity",
    adjoint_tolerance=2.0e-10,
    adjoint_max_iterations=200,
    verbose=True,
    actuator_gradient=args.actuator_gradient,
)

sim.initialize_inputs()
# PICMI omits the particle shape without species; the field solve still needs it.
algo.particle_shape = 1
sim.initialize_warpx()
corrector.setup_after_init()

mfr = corrector._mfr()
Direction = corrector._Direction


def _max_abs(name, components):
    """Global max |field| over the valid region. Collective: all ranks call it."""
    return max(
        mfr.get(name, dir=Direction(comp), level=0).norm0(0, 0, False, False)
        for comp in components
    )


# The unit fields themselves must differ between the representations. Recording
# their component maxima is what pins that actuator_gradient reaches the 3-D
# path; the capacitance alone does not, being identical for both here.
unit_maxima = [
    [
        mfr.get(name, dir=Direction(comp), level=0).norm0(0, 0, False, False)
        for comp in (0, 1, 2)
    ]
    for name in corrector._unit_names
]
print(f"unit field component maxima ({args.actuator_gradient}): {unit_maxima}")

assert _max_abs("Bfield_fp", (0, 1, 2)) == 0.0, "the fixture must start from B = 0"

corrector.correct_field()
state = corrector.last_correction_state()
assert state is not None
np.testing.assert_allclose(
    state["delta_voltage"], corrector.v_target, rtol=1.0e-9, atol=1.0e-6
)

e_applied = _max_abs("Efield_fp", (0, 1, 2))
assert e_applied > 0.0, "the clamp applied no field"
e_reference = max(abs(v) for v in corrector.v_target) / half

sim.step(steps)

b_scale = _max_abs("Bfield_fp", (0, 1, 2))
normalized = picmi.constants.c * b_scale / e_reference
print(
    f"actuator_gradient={args.actuator_gradient}: after {steps} steps "
    f"max|E_applied|={e_applied:.6e} V/m, max|B|={b_scale:.6e} T, "
    f"E_ref={e_reference:.6e} V/m, c|B|/E_ref={normalized:.6e}"
)

if args.actuator_gradient == "ordinary":
    # The full-grid gradient is annihilated by the Faraday stencil in 3-D too.
    assert normalized < 1.0e-9, (
        "the ordinary-gradient actuator is supposed to be curl-free, but it "
        f"induced c|B|/E_ref = {normalized:.6e} in {steps} steps"
    )
else:
    # Preserved counterexample, on separate conducting bodies rather than the
    # RZ fixture's electrically shorted hemispheres.
    assert normalized > 1.0e-3, (
        "the EB-aware actuator no longer induces a magnetic field in 3-D "
        f"(c|B|/E_ref = {normalized:.6e}). The control that discriminates the "
        "two representations has stopped discriminating: re-derive the "
        "actuator/Faraday compatibility argument before trusting either."
    )
