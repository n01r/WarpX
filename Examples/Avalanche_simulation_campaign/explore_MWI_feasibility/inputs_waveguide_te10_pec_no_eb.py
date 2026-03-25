#!/usr/bin/env python3
"""
3D WarpX vacuum simulation with TE10 mode injection in straight WR-15 waveguide
Microwave frequency: 67 GHz
PEC boundaries in x/y (waveguide walls), PML in z (no embedded boundary)
"""

import numpy as np
from pywarpx import picmi

# Physical constants
c = picmi.constants.c
mu0 = picmi.constants.mu0
ep0 = picmi.constants.ep0

# Microwave parameters
freq = 67e9  # Hz
omega = 2 * np.pi * freq
wavelength = c / freq  # ~4.48 mm
k0 = 2 * np.pi / wavelength

# Time step (omega * dt <= 0.2 for stability)
dt = 0.2 / omega  # ~4.75e-13 s

# Horn aperture dimensions (for TE10 mode)
# These define the waveguide mode at the emitting horn
# Compare with https://www.everythingrf.com/tech-resources/waveguides-sizes/wr15
a_horn = 3.7592e-3  # Horn width in x - determines TE10 cutoff
b_horn = 1.8796e-3  # Horn height in y

# Domain dimensions (in meters)
# x/y = waveguide cross-section (PEC walls act as waveguide boundaries)
x_size = a_horn  # 3.759 mm (exact WR-15 width)
y_size = b_horn  # 1.880 mm (exact WR-15 height)
z_size = 0.280   # 280 mm (same as EB version for comparison)

# Grid resolution: 20 cells/wavelength gives ~8 cells across WR15 narrow dim (b=1.88mm)
cells_per_wavelength = 16
dx = wavelength / cells_per_wavelength
dy = wavelength / cells_per_wavelength
dz = wavelength / cells_per_wavelength

# Calculate initial number of cells
nx_raw = int(x_size / dx)
ny_raw = int(y_size / dy)
nz_raw = int(z_size / dz)

# Round to nearest multiple of 8 for GPU efficiency and AMReX compatibility
def round_to_multiple(n, multiple=8):
    """Round to nearest multiple, ensuring at least 'multiple' cells."""
    return max(multiple, int(np.ceil(n / multiple) * multiple))

nx = round_to_multiple(nx_raw, 8)
ny = round_to_multiple(ny_raw, 8)
nz = round_to_multiple(nz_raw, 8)

# Adjust cell sizes to match rounded dimensions
dx = x_size / nx
dy = y_size / ny
dz = z_size / nz

print(f"Grid: {nx} x {ny} x {nz} = {nx*ny*nz:.3e} cells")
print(f"Cell size: dx={dx:.3e} m, dy={dy:.3e} m, dz={dz:.3e} m")
print(f"Wavelength: {wavelength*1e3:.3f} mm")
print(f"Time step: {dt:.3e} s (omega*dt = {omega*dt:.3f})")

# PML parameters
pml_ncells = 10

# Create 3D Cartesian grid with PML boundaries
# Note: PICMI uses 'open' for PML, not 'pml'
# Grid dimensions are divisible by 8 for GPU efficiency
grid = picmi.Cartesian3DGrid(
    number_of_cells=[nx, ny, nz],
    lower_bound=[-x_size/2, -y_size/2, -z_size/2],
    upper_bound=[x_size/2, y_size/2, z_size/2],
    lower_boundary_conditions=['dirichlet', 'dirichlet', 'open'],
    upper_boundary_conditions=['dirichlet', 'dirichlet', 'open'],
    lower_boundary_conditions_particles=['absorbing', 'absorbing', 'absorbing'],
    upper_boundary_conditions_particles=['absorbing', 'absorbing', 'absorbing'],
)

# Yee solver with PML
cfl = 0.995 #c * dt / min(dx, dy, dz)
print(f"CFL number: {cfl:.3f}")

solver = picmi.ElectromagneticSolver(
    grid=grid,
    method='Yee',
    cfl=cfl,
    warpx_pml_ncell=pml_ncells,
)

# TE10 mode parameters for emitting horn aperture
# TE10 has variation sin(pi*x/a) in the x-direction (wider dimension)
# For TE10: cutoff wavelength = 2*a, cutoff freq = c/(2*a)
a = a_horn  # Horn aperture width in x (16 cm)
b = b_horn  # Horn aperture height in y (14 cm)

# Emitting horn position and aperture center
# ADJUST THESE to match your STL horn positions!
emit_horn_x_center = 0.0  # Center x position of emitting horn
emit_horn_y_center = 0.0  # Center y position of emitting horn
emit_horn_z_pos = -12.8e-2  # z position inside rectangular waveguide

# Receiving diagnostic plane position
# ADJUST THIS to match your receiving horn aperture in the STL
recv_plane_x_center = 0.0  # Center x position of receiving plane
recv_plane_y_center = 0.0  # Center y position of receiving plane
recv_plane_z_pos = -emit_horn_z_pos  # symmetric setup, measure inside receiving horn

# Check if frequency is above cutoff
fc_te10 = c / (2 * a)
print(f"TE10 cutoff frequency: {fc_te10*1e-9:.2f} GHz")
if freq < fc_te10:
    print(f"WARNING: Operating frequency ({freq*1e-9:.2f} GHz) is below TE10 cutoff!")
    print("TE10 mode will be evanescent.")

# TE10 mode field amplitude (E0 in V/m)
E0 = 1e6

# Horn aperture half-widths
horn_half_width_x = a / 2
horn_half_width_y = b / 2

# Thickness of source region in z (a few cells)
source_thickness_z = 2 * dz

# TE10 mode expressions for antenna inside emitting horn (propagating in +z)
# E_y = E0 * sin(pi*x/a) * cos(omega*t)
# H_x = -(E0/(Z_TE10)) * sin(pi*x/a) * cos(omega*t)
# where Z_TE10 = omega*mu0/beta, beta = sqrt(k0^2 - (pi/a)^2)

beta = np.sqrt(k0**2 - (np.pi/a)**2)
Z_TE10 = omega * mu0 / beta
H0 = E0 / Z_TE10

print(f"TE10 guide wavelength: {2*np.pi/beta*1e2:.2f} cm")
print(f"TE10 impedance: {Z_TE10:.1f} Ohm")
print(f"Horn aperture: {a*1e2:.2f} x {b*1e2:.2f} cm")
print(f"Emitting plane z-position: {emit_horn_z_pos*1e2:.2f} cm")
print(f"Receiving plane z-position: {recv_plane_z_pos*1e2:.2f} cm")

# Define TE10 mode as a continuous source using binary file laser.
# NOTE: AnalyticLaser (parse_field_function) crashes on GPU in this build.
# Workaround: pre-compute the TE10 field E(x,y,t) and write a binary file.
# The laser antenna continuously injects fields at the antenna plane.
# Antenna-local coordinates: X is along x, Y is along y (for prop in z, pol in y).
# TE10 transverse profile: sin(pi*(X + a/2)/a) gated to the aperture.

import struct
import os

# Generate binary file for laser injection
# Binary format: flag(1B) + nt(4B) + nx(4B) + ny(4B) + t_range(2*8B) + x_range(2*8B) + y_range(2*8B) + data(nt*nx*ny*8B)
# Field values are normalized by e_max, so we set e_max=E0 and store shape only.

binary_laser_file = 'te10_laser_profile.bin'

# Grid for the laser profile (antenna-local coordinates)
# Cover the full transverse domain so laser particles span the antenna plane
laser_nx = 256
laser_ny = 256
# Use enough time samples to resolve the Gaussian ramp-up
ramp_periods = 10  # Ramp up over 10 RF periods
T_rf = 1.0 / freq
ramp_time = ramp_periods * T_rf
laser_nt = 256  # Enough samples to resolve the Gaussian envelope

# Transverse extent: cover the full domain
laser_x_min = -x_size / 2
laser_x_max = x_size / 2
laser_y_min = -y_size / 2
laser_y_max = y_size / 2

# Time range: span the full simulation
laser_t_min = 0.0
laser_t_max = 2000 * dt  # Well beyond max_steps * dt

laser_x = np.linspace(laser_x_min, laser_x_max, laser_nx)
laser_y = np.linspace(laser_y_min, laser_y_max, laser_ny)
laser_t = np.linspace(laser_t_min, laser_t_max, laser_nt)

# Build the TE10 spatial profile: sin(pi*(x + a/2)/a) gated to aperture
profile_xy = np.zeros((laser_nx, laser_ny), dtype=np.float64)
for ix, xv in enumerate(laser_x):
    for iy, yv in enumerate(laser_y):
        if abs(xv) <= horn_half_width_x and abs(yv) <= horn_half_width_y:
            profile_xy[ix, iy] = np.sin(np.pi * (xv + a/2) / a)

# Gaussian ramp-up envelope: 1 - exp(-(t/tau)^2), reaches ~1 after ramp_time
# tau chosen so envelope is ~0.99 at t = ramp_time
tau = ramp_time / 2.15  # 2.15 sigma gives ~99% amplitude
envelope = np.where(laser_t < ramp_time,
                    1.0 - np.exp(-(laser_t / tau)**2),
                    1.0)

# Combined profile: spatial × temporal envelope
# Normalized by E0 since e_max = E0 in the laser definition
profile = np.zeros((laser_nt, laser_nx, laser_ny), dtype=np.float64)
for it in range(laser_nt):
    profile[it, :, :] = envelope[it] * profile_xy

print(f"Gaussian ramp-up: {ramp_periods} RF periods ({ramp_time*1e12:.1f} ps)")
print(f"Laser binary file: {laser_nx} x {laser_ny} x {laser_nt}, max profile = {profile.max():.4f}")

# Write binary file
with open(binary_laser_file, 'wb') as f:
    f.write(struct.pack('B', 1))  # flag: uniform grid
    f.write(struct.pack('I', laser_nt))
    f.write(struct.pack('I', laser_nx))
    f.write(struct.pack('I', laser_ny))
    f.write(struct.pack('dd', laser_t_min, laser_t_max))
    f.write(struct.pack('dd', laser_x_min, laser_x_max))
    f.write(struct.pack('dd', laser_y_min, laser_y_max))
    f.write(profile.tobytes())  # nt is slowest, then nx, then ny

print(f"Wrote {binary_laser_file} ({os.path.getsize(binary_laser_file)} bytes)")

# We'll configure the laser via low-level pywarpx after initialize_inputs
# since PICMI doesn't directly support from_file laser profiles.
emit_antenna_position = [emit_horn_x_center, emit_horn_y_center, emit_horn_z_pos]

# Diagnostic plane at receiving horn location (Full 3D diagnostic)
# Records E and B fields at EVERY time step (period=1)
recv_plane_diag = picmi.FieldDiagnostic(
    name='receiving_plane',
    grid=grid,
    period=1,  # Every time step!
    data_list=['Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz'],
    write_dir='./diags',
    warpx_format='openpmd',
    warpx_openpmd_backend='bp5',
    # Limit diagnostic to receiving plane region
    lower_bound=[recv_plane_x_center - horn_half_width_x,
                 recv_plane_y_center - horn_half_width_y,
                 recv_plane_z_pos],
    upper_bound=[recv_plane_x_center + horn_half_width_x,
                 recv_plane_y_center + horn_half_width_y,
                 recv_plane_z_pos],
)

# FieldProbe plane diagnostic (for performance comparison)
# Samples fields on a 2D plane using FieldProbe reduced diagnostic
recv_plane_probe = picmi.ReducedDiagnostic(
    diag_type="FieldProbe",
    name="recv_probe",
    period=1,  # Every time step like full diagnostic
    path="diags/",
    extension="dat",
    probe_geometry="Plane",
    resolution=50,  # Number of points along each edge
    x_probe=recv_plane_x_center,
    y_probe=recv_plane_y_center,
    z_probe=recv_plane_z_pos,
    detector_radius=min(horn_half_width_x, horn_half_width_y),  # Half edge length
    target_normal_x=0.0,
    target_normal_y=0.0,
    target_normal_z=1.0,  # Normal pointing in +z direction
    target_up_x=0.0,
    target_up_y=1.0,  # Up direction in y
    target_up_z=0.0,
)

# Field diagnostics (full domain with embedded boundary visualization)
field_diag = picmi.FieldDiagnostic(
    name='fields',
    grid=grid,
    period=100,
    data_list=['E', 'B'],
    write_dir='./diags',
    warpx_format='openpmd',
    warpx_openpmd_backend='h5',
)

# Reduced diagnostics for field monitoring
field_max_diag = picmi.ReducedDiagnostic(
    diag_type='FieldMaximum',
    name='field_max',
    period=10,
)

field_energy_diag = picmi.ReducedDiagnostic(
    diag_type='FieldEnergy',
    name='field_energy',
    period=10,
)

# No embedded boundary needed - PEC domain boundaries act as waveguide walls

# Create simulation
sim = picmi.Simulation(
    solver=solver,
    #time_step_size=dt,
    max_steps=1000,
    verbose=1,
)

# Add diagnostics
sim.add_diagnostic(field_diag)  # Full domain, periodic (with eb_covered)
sim.add_diagnostic(recv_plane_diag)  # Receiving plane full 3D, every timestep
sim.add_diagnostic(recv_plane_probe)  # Receiving plane FieldProbe, every timestep
sim.add_diagnostic(field_max_diag)  # Reduced diagnostic
sim.add_diagnostic(field_energy_diag)  # Reduced diagnostic

# Write input file or run simulation
if __name__ == "__main__":
    import sys
    import pywarpx
    
    # FieldProbe requires algo.particle_shape for interpolation.
    pywarpx.algo.particle_shape = 1

    pywarpx.amrex.the_arena_is_managed = 0
    pywarpx.amrex.the_arena_init_size = 0  # Let AMReX use all available GPU memory

    # Configure laser directly via pywarpx (bypass PICMI AnalyticLaser GPU bug)
    pywarpx.lasers.names = ['laser1']
    laser = pywarpx.Lasers.newlaser('laser1')
    laser.profile = 'from_file'
    laser.position = emit_antenna_position
    laser.direction = [0, 0, 1]
    laser.polarization = [0, 1, 0]
    laser.e_max = E0
    laser.wavelength = wavelength
    laser.binary_file_name = binary_laser_file
    laser.do_continuous_injection = 0

    if '--write-inputs' in sys.argv:
        sim.write_input_file(file_name='inputs_from_picmi')
        print("\nInput file written to 'inputs_from_picmi'")
    else:
        print("\nInitializing inputs...", flush=True)
        sim.initialize_inputs()

        # Enable ADIOS2 BP5 asynchronous writing for receiving plane diagnostic
        # so simulation doesn't block on I/O. Uses host RAM as buffer.
        recv_bucket = pywarpx.diagnostics._diagnostics_dict['receiving_plane']
        recv_bucket.add_new_group_attr('adios2_engine', 'parameters.AsyncWrite', 'on')
        recv_bucket.add_new_group_attr('adios2_engine', 'parameters.BufferChunkSize', str(32*1024*1024*1024))

        print("\nInitializing WarpX...", flush=True)
        sim.initialize_warpx()
        print("\nStarting simulation...", flush=True)
        sim.step(sim.max_steps)
        print("Simulation complete!", flush=True)
