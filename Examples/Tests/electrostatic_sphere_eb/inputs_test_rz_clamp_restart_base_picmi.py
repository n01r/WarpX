#!/usr/bin/env python3
"""Base run for the clamp checkpoint/restart pair.

WarpX does not call ``afterInitEsolve`` on a restart, so the clamp registers its
setup on ``afterInitatRestart`` as well. Nothing in the corrector is written to
the checkpoint: the unit fields, the capacitance matrix and the adjoints are all
rebuilt from the restored state. This pair tests that the rebuild reproduces the
base run rather than assuming it does, and that registering setup on both hooks
runs it exactly once.

This run evolves a few steps with the clamp installed as a real callback and
writes a checkpoint plus ``clamp_setup.json``. The restart run
(``inputs_test_rz_clamp_restart_picmi.py``) picks both up.
"""

import json

import numpy as np

from pywarpx import picmi
from pywarpx.callbacks import (
    installafterEsolve,
    installafterInitatRestart,
    installafterInitEsolve,
)
from pywarpx.multi_electrode_corrector import MultiElectrodeBiasCorrector
from pywarpx.multi_electrode_logger import MultiElectrodeClampTelemetry

steps = 10
grid = picmi.CylindricalGrid(
    number_of_cells=[24, 48],
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
    x=[0.0123, 0.2871, 0.4317],
    y=[0.0, 0.0311, -0.0247],
    z=[0.2637, -0.3173, 0.1179],
    ux=[0.0] * 3,
    uy=[0.0] * 3,
    uz=[0.0] * 3,
    weight=[1.0e9, 1.7e9, 0.8e9],
)
electrons = picmi.Species(
    name="electrons", particle_type="electron", initial_distribution=distribution
)

sim = picmi.Simulation(
    solver=solver,
    max_steps=steps,
    particle_shape="linear",
    warpx_embedded_boundary=eb,
    warpx_use_filter=False,
    verbose=0,
)
sim.add_species(
    electrons,
    layout=picmi.GriddedLayout(n_macroparticle_per_cell=[0, 0, 0], grid=grid),
)
sim.add_diagnostic(picmi.Checkpoint(period=steps, name="chk"))

electrodes = [
    {"name": "upper", "region": "(z>0.0)", "potential": +250.0},
    {"name": "lower", "region": "(z<=0.0)", "potential": -400.0},
]

corrector = MultiElectrodeBiasCorrector(
    sim=sim,
    correction_interval=2,
    electrodes=electrodes,
    qg_mode="reciprocity",
    adjoint_tolerance=2.0e-10,
)
telemetry = MultiElectrodeClampTelemetry(
    corrector, out_csv="clamp_telemetry.csv", setup_json="clamp_setup.json"
)

setup_calls = []


def combined_setup():
    """One function on both init hooks, as the production input must also do."""
    setup_calls.append(1)
    corrector.setup_after_init()
    telemetry.setup()


installafterInitEsolve(combined_setup)
installafterInitatRestart(combined_setup)
installafterEsolve(corrector.correct_field)
# Log after the correction, so each row is the state the clamp left behind.
# This is also what makes a duplicated setup registration observable: the
# telemetry row series is the side effect the guide's warning is about.
installafterEsolve(telemetry.log)

sim.step(steps)

# Exactly one of the two init hooks fires in a normal run.
assert len(setup_calls) == 1, f"setup ran {len(setup_calls)} times, expected once"

state = corrector.last_correction_state()
assert state is not None, "the clamp never corrected during the base run"
with open("clamp_setup.json", encoding="utf-8") as stream:
    saved_setup = json.load(stream)
print(
    "base run capacitance: "
    f"{np.asarray(saved_setup['capacitance_matrix']).tolist()}, "
    f"actuator_gradient={saved_setup['actuator_gradient']}"
)
