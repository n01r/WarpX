#!/usr/bin/env python3
"""Restart half of the clamp checkpoint/restart pair.

Depends on ``inputs_test_rz_clamp_restart_base_picmi.py``, whose checkpoint and
``clamp_setup.json`` this run reads from the base test's directory.

``afterInitEsolve`` does not fire on a restart, which is why the clamp must also
be registered on ``afterInitatRestart``. Registering the same function on both is
what the production input does, so the first thing checked here is that it still
runs exactly once.

The corrector stores nothing in the checkpoint: the unit fields, the capacitance
and the adjoints are rebuilt from the restored state. So the calibration must
reproduce the base run's to solve precision, and a correction must still drive
the electrodes to target. Both are asserted rather than assumed.
"""

import csv
import json
import pathlib

import numpy as np

from pywarpx import picmi
from pywarpx.callbacks import (
    installafterEsolve,
    installafterInitatRestart,
    installafterInitEsolve,
)
from pywarpx.multi_electrode_corrector import MultiElectrodeBiasCorrector
from pywarpx.multi_electrode_logger import (
    MultiElectrodeClampTelemetry,
    _mpi_rank,
)

base_dir = "../test_rz_clamp_restart_base_picmi"
total_steps = 16
# The step the base run checkpointed at, and so the last step it corrected.
checkpoint_step = 10
correction_interval = 2
# The closing check re-measures the voltage against the target, so the run has
# to end on a step the clamp actually corrected; otherwise it measures drift
# accumulated since the last correction and not whether the loop closed.
assert total_steps % correction_interval == 0
assert checkpoint_step % correction_interval == 0

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
    max_steps=total_steps,
    particle_shape="linear",
    warpx_embedded_boundary=eb,
    warpx_use_filter=False,
    verbose=0,
    warpx_amr_restart=f"{base_dir}/diags/chk{checkpoint_step:06d}",
)
sim.add_species(
    electrons,
    layout=picmi.GriddedLayout(n_macroparticle_per_cell=[0, 0, 0], grid=grid),
)

electrodes = [
    {"name": "upper", "region": "(z>0.0)", "potential": +250.0},
    {"name": "lower", "region": "(z<=0.0)", "potential": -400.0},
]

corrector = MultiElectrodeBiasCorrector(
    sim=sim,
    correction_interval=correction_interval,
    electrodes=electrodes,
    qg_mode="reciprocity",
    adjoint_tolerance=2.0e-10,
)
# The telemetry preserves an existing CSV whenever the run starts at a positive
# step, so that a restart continues the base run's series rather than truncating
# it. This test writes into its own directory, which CTest reuses between
# invocations, so a leftover file from a previous run would be appended to and
# the series would no longer be a record of this run alone. Start it clean.
# Only rank 0 ever writes this file, so only rank 0 clears it: letting every
# rank delete it races against rank 0's own write with no collective in between.
if _mpi_rank() == 0:
    pathlib.Path("clamp_telemetry.csv").unlink(missing_ok=True)

telemetry = MultiElectrodeClampTelemetry(
    corrector, out_csv="clamp_telemetry.csv", setup_json="clamp_setup.json"
)

setup_calls = []


def combined_setup():
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

sim.step(total_steps - checkpoint_step)

# afterInitEsolve does not fire on a restart and afterInitatRestart does, so
# registering one function on both hooks must still set up exactly once.
assert len(setup_calls) == 1, (
    f"setup ran {len(setup_calls)} times after restart, expected exactly once"
)

with open(f"{base_dir}/clamp_setup.json", encoding="utf-8") as stream:
    base_setup = json.load(stream)
restart_setup = corrector.setup_state()

assert restart_setup["actuator_gradient"] == base_setup["actuator_gradient"]
assert restart_setup["electrode_names"] == base_setup["electrode_names"]

base_capacitance = np.asarray(base_setup["capacitance_matrix"])
restart_capacitance = np.asarray(restart_setup["capacitance_matrix"])
relative_difference = np.max(
    np.abs(restart_capacitance - base_capacitance) / np.abs(base_capacitance)
)
print(f"base capacitance:    {base_capacitance.tolist()}")
print(f"restart capacitance: {restart_capacitance.tolist()}")
print(f"capacitance rebuild max relative difference: {relative_difference:.3e}")

# Not a bitwise comparison: the unit fields are rebuilt by fresh MLMG solves that
# start from a different initial guess than the base run's, so the two iterates
# differ within the solve tolerance. The claim being tested is that nothing about
# the calibration depends on state that the checkpoint does not carry, and the
# bound is far tighter than the 2e-10 solve tolerance that limits it.
assert relative_difference < 1.0e-12, (
    "the restarted run rebuilt a different capacitance matrix: max relative "
    f"difference {relative_difference:.3e}"
)


# The guide warns that registering the setup on both hooks can double-register
# the runtime callbacks it installs. Counting setup calls only proves a function
# ran once; the row series is the side effect that would actually be corrupted.
# A restart appends to its own file, so its first logged step must lie strictly
# after the checkpoint step: a re-corrected step would show up as a repeat.
def _steps(path):
    with open(path, encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return [int(row["step"]) for row in rows]


restart_steps = _steps("clamp_telemetry.csv")
base_steps = _steps(f"{base_dir}/clamp_telemetry.csv")
print(f"base logged steps={base_steps}")
print(f"restart logged steps={restart_steps}")

assert restart_steps, "the restarted run logged no telemetry rows"
assert len(set(restart_steps)) == len(restart_steps), (
    f"a step was logged more than once after the restart: {restart_steps}"
)
assert restart_steps == sorted(restart_steps), (
    f"restart telemetry steps are not monotonic: {restart_steps}"
)
assert min(restart_steps) > checkpoint_step, (
    f"the restart re-logged step {min(restart_steps)}, at or before the "
    f"checkpoint step {checkpoint_step}: the clamp corrected an already "
    "corrected step"
)
assert not set(base_steps) & set(restart_steps), (
    "base and restart runs logged overlapping steps: "
    f"{sorted(set(base_steps) & set(restart_steps))}"
)

state = corrector.last_correction_state()
assert state is not None, "the clamp never corrected after the restart"

# The loop must still close on the restored state.
remeasured_state = corrector.measure_voltage_state()
target = np.asarray(corrector.v_target)
print(
    f"target={target}, re-measured after restart correction={remeasured_state['voltage']}"
)
np.testing.assert_allclose(
    remeasured_state["voltage"], target, rtol=1.0e-9, atol=1.0e-6
)
