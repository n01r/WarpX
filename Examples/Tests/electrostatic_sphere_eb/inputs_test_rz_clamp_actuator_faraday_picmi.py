#!/usr/bin/env python3
"""Regression test: is the voltage clamp's actuator field a stationary EM state?

The clamp adds a linear combination of precomputed unit fields directly to
``Efield_fp``. Nothing in the charge algebra requires that field to be
compatible with the Yee Faraday update, and by default it is not: the unit
fields come from the EB-aware Poisson gradient, which uses the shortened fluid
length on cut edges, while ``EvolveB`` differences ordinary full-grid edges. The
two representations do not cancel, so an otherwise static vacuum grows a
boundary ``B`` for as long as the bias is applied. That is a real field a
particle gathers, not unused storage.

This runs a vacuum fixture -- no particles, no feedback after the single
correction, no solve during evolution -- so any ``B`` at all is numerical. It
applies the clamp bias through the production corrector and evolves:

* ``--actuator-gradient ordinary`` must leave ``B`` at roundoff. That is the
  property the option exists to provide.
* ``--actuator-gradient eb_aware`` (the default) must **not**. The counterexample
  is asserted rather than merely documented so that the two representations stay
  distinguishable: if this direction ever stops growing ``B``, either the
  actuator or the Faraday update changed and the ordinary-gradient option needs
  to be re-justified.

The domain is split in both directions so the unit fields are also built under a
decomposition. On a staggered (Yee) grid ``computeE`` writes each face from the
two ``phi`` nodes bounding it, which are both valid data, so it needs no ghost
exchange; the serial and two-rank results here are bit-identical, which is the
evidence for that. The collocated branch of ``computeE`` uses centred
differences and does read ``phi(i+/-1)``, and is not exercised by this test.

``B`` is normalized by a fixed drive scale rather than by ``max|E|``: with two
driven regions on one conductor body, the ordinary gradient's largest value sits
on edges interior to the conductor, where the parser's region boundary puts a
one-cell potential jump between covered nodes. That makes ``max|E|`` differ
between the two modes for a reason unrelated to what is being asserted here.
"""

import argparse

import numpy as np

from pywarpx import algo, picmi
from pywarpx.multi_electrode_corrector import MultiElectrodeBiasCorrector

parser = argparse.ArgumentParser()
parser.add_argument(
    "--actuator-gradient", choices=["eb_aware", "ordinary"], default="eb_aware"
)
args, _ = parser.parse_known_args()

steps = 20
radius = 0.18
nr, nz = 24, 48

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
    # splits in r as well as z, so boxes off the conductor also carry unit field
    warpx_max_grid_size=16,
)
solver = picmi.ElectromagneticSolver(
    grid=grid, method="Yee", cfl=0.9, divE_cleaning=False
)
eb = picmi.EmbeddedBoundary(
    implicit_function="-(x*x+y*y+z*z-radius*radius)",
    potential=0.0,
    radius=radius,
)

sim = picmi.Simulation(
    solver=solver,
    max_steps=steps,
    particle_shape="linear",
    warpx_embedded_boundary=eb,
    warpx_use_filter=False,
    verbose=0,
)

# Two independently driven electrodes at unequal potentials, so the applied
# field is a genuine combination of both unit columns rather than one of them.
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
# PICMI omits the particle shape without species; the field solve still needs it.
algo.particle_shape = 1
sim.initialize_warpx()
corrector.setup_after_init()

warpx = corrector._warpx()
mfr = warpx.multifab_register()
Direction = corrector._Direction


def _max_abs(name, components):
    """Global max |field| over the valid region. Collective: all ranks call it."""
    return max(
        # norm0(comp, nghost, local, ignore_covered): reduces across ranks
        mfr.get(name, dir=Direction(comp), level=0).norm0(0, 0, False, False)
        for comp in components
    )


# RZ m=0 keeps Er, Btheta and Ez; the remaining components stay zero.
# Vacuum: nothing has driven any field yet.
assert _max_abs("Bfield_fp", (0, 1, 2)) == 0.0, "the fixture must start from B = 0"

# One correction from the zero state applies the full target bias.
corrector.correct_field()
state = corrector.last_correction_state()
assert state is not None
np.testing.assert_allclose(
    state["delta_voltage"], corrector.v_target, rtol=1.0e-9, atol=1.0e-6
)

e_applied = _max_abs("Efield_fp", (0, 2))
assert e_applied > 0.0, "the clamp applied no field"

# A fixed drive scale: the largest target voltage over the radial extent of the
# domain. It depends only on the fixture, so the two modes are normalized
# identically and their rows are directly comparable.
e_reference = max(abs(v) for v in corrector.v_target) / 0.8

sim.step(steps)

# c |B| / E_ref is the dimensionless size of the spurious magnetic field. A
# genuinely static electric state leaves it at roundoff for as long as it is held.
b_scale = _max_abs("Bfield_fp", (0, 1, 2))
normalized = picmi.constants.c * b_scale / e_reference
print(
    f"actuator_gradient={args.actuator_gradient}: after {steps} steps "
    f"max|E_applied|={e_applied:.6e} V/m, max|B|={b_scale:.6e} T, "
    f"E_ref={e_reference:.6e} V/m, c|B|/E_ref={normalized:.6e}"
)

if args.actuator_gradient == "ordinary":
    # The full-grid gradient is annihilated by the Faraday stencil, so this is
    # bounded by accumulated roundoff over the steps taken, not by the mesh.
    assert normalized < 1.0e-9, (
        "the ordinary-gradient actuator is supposed to be curl-free, but it "
        f"induced c|B|/E_ref = {normalized:.6e} in {steps} steps"
    )
else:
    # Preserved counterexample. The measured value is far above this bound; the
    # threshold only has to separate the two representations.
    assert normalized > 1.0e-3, (
        "the EB-aware actuator no longer induces a boundary magnetic field "
        f"(c|B|/E_ref = {normalized:.6e}). The discriminating control that "
        "motivates actuator_gradient='ordinary' has stopped discriminating: "
        "re-derive the actuator/Faraday compatibility argument before trusting "
        "either representation."
    )
