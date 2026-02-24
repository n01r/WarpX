#!/usr/bin/env python3
"""
3D WarpX vacuum simulation with TE10 mode injection
Microwave frequency: 67 GHz
PML boundaries with embedded boundary from STL file
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

# Domain dimensions (in meters) - FULL simulation box
x_size = 0.16  # 16 cm
y_size = 0.14  # 14 cm
z_size = 0.30  # 30 cm

# Horn aperture dimensions (for TE10 mode)
# These define the waveguide mode at the emitting horn
a_horn = 0.16  # Horn width in x (16 cm) - determines TE10 cutoff
b_horn = 0.14  # Horn height in y (14 cm)

# Grid resolution (aim for ~10-15 cells per wavelength)
cells_per_wavelength = 12
dx = wavelength / cells_per_wavelength
dy = wavelength / cells_per_wavelength
dz = wavelength / cells_per_wavelength

nx = int(x_size / dx)
ny = int(y_size / dy)
nz = int(z_size / dz)

# Adjust to make dimensions exact
dx = x_size / nx
dy = y_size / ny
dz = z_size / nz

print(f"Grid: {nx} x {ny} x {nz} = {nx*ny*nz} cells")
print(f"Cell size: dx={dx*1e3:.3f} mm, dy={dy*1e3:.3f} mm, dz={dz*1e3:.3f} mm")
print(f"Wavelength: {wavelength*1e3:.3f} mm")
print(f"Time step: {dt*1e15:.3f} fs (omega*dt = {omega*dt:.3f})")

# PML parameters
pml_ncells = 10

# Create 3D Cartesian grid with PML boundaries
grid = picmi.Cartesian3DGrid(
    number_of_cells=[nx, ny, nz],
    lower_bound=[-x_size/2, -y_size/2, -z_size/2],
    upper_bound=[x_size/2, y_size/2, z_size/2],
    lower_boundary_conditions=['open', 'open', 'open'],
    upper_boundary_conditions=['open', 'open', 'open'],
    warpx_max_grid_size=128,
    warpx_blocking_factor=32,
)

# Yee solver with PML
cfl = c * dt / min(dx, dy, dz)
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
emit_horn_z_pos = -z_size/2 + (pml_ncells + 2) * dz  # z position (near -z boundary)

# Receiving diagnostic plane position
# ADJUST THIS to match your receiving horn aperture in the STL
recv_plane_x_center = 0.0  # Center x position of receiving plane
recv_plane_y_center = 0.0  # Center y position of receiving plane
recv_plane_z_pos = z_size/2 - (pml_ncells + 2) * dz  # z position (near +z boundary)

# Check if frequency is above cutoff
fc_te10 = c / (2 * a)
print(f"TE10 cutoff frequency: {fc_te10*1e-9:.2f} GHz")
if freq < fc_te10:
    print(f"WARNING: Operating frequency ({freq*1e-9:.2f} GHz) is below TE10 cutoff!")
    print("TE10 mode will be evanescent.")

# TE10 mode field amplitude (E0 in V/m)
E0 = 1e6  # 1 MV/m - adjust as needed

# Horn aperture half-widths
horn_half_width_x = a / 2
horn_half_width_y = b / 2

# Thickness of source region in z (a few cells)
source_thickness_z = 2 * dz

# TE10 mode expressions for antenna at -z boundary (propagating in +z)
# E_y = E0 * sin(pi*x/a) * cos(omega*t)
# H_x = -(E0/(Z_TE10)) * sin(pi*x/a) * cos(omega*t)
# where Z_TE10 = omega*mu0/beta, beta = sqrt(k0^2 - (pi/a)^2)

beta = np.sqrt(k0**2 - (np.pi/a)**2)
Z_TE10 = omega * mu0 / beta
H0 = E0 / Z_TE10

print(f"TE10 guide wavelength: {2*np.pi/beta*1e2:.2f} cm")
print(f"TE10 impedance: {Z_TE10:.1f} Ohm")
print(f"Horn aperture: {a*1e2:.2f} x {b*1e2:.2f} cm")
print(f"Emitting horn z-position: {emit_horn_z_pos*1e2:.2f} cm")
print(f"Receiving plane z-position: {recv_plane_z_pos*1e2:.2f} cm")

# Define TE10 mode source at emitting horn (propagating +z)
# Spatial gating ensures fields only exist within horn aperture
# TE10: E_y ~ sin(pi*(x-x_center)/a + pi/2) for centered horn
emit_antenna = picmi.AnalyticAppliedField(
    Ex_expression="0",
    Ey_expression=f"({E0}) * sin(pi*(x - ({emit_horn_x_center}))/{a} + pi/2) * cos({omega}*t) * "
                  f"(abs(x - ({emit_horn_x_center})) <= {horn_half_width_x}) * "
                  f"(abs(y - ({emit_horn_y_center})) <= {horn_half_width_y}) * "
                  f"(abs(z - ({emit_horn_z_pos})) < {source_thickness_z})",
    Ez_expression="0",
    Bx_expression=f"({-mu0*H0}) * sin(pi*(x - ({emit_horn_x_center}))/{a} + pi/2) * cos({omega}*t) * "
                  f"(abs(x - ({emit_horn_x_center})) <= {horn_half_width_x}) * "
                  f"(abs(y - ({emit_horn_y_center})) <= {horn_half_width_y}) * "
                  f"(abs(z - ({emit_horn_z_pos})) < {source_thickness_z})",
    By_expression="0",
    Bz_expression="0",
)

# Diagnostic plane at receiving horn location
# Records E and B fields at EVERY time step (period=1)
recv_plane_diag = picmi.FieldDiagnostic(
    name='receiving_plane',
    grid=grid,
    period=1,  # Every time step!
    data_list=['Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz'],
    write_dir='./diags',
    warpx_format='openpmd',
    warpx_openpmd_backend='h5',
    # Limit diagnostic to receiving plane region
    lower_bound=[recv_plane_x_center - horn_half_width_x,
                 recv_plane_y_center - horn_half_width_y,
                 recv_plane_z_pos - dz],
    upper_bound=[recv_plane_x_center + horn_half_width_x,
                 recv_plane_y_center + horn_half_width_y,
                 recv_plane_z_pos + dz],
)

# Field diagnostics
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

# Create simulation
sim = picmi.Simulation(
    solver=solver,
    time_step_size=dt,
    max_steps=5000,
    verbose=1,
    warpx_embedded_boundary=dict(
        geom_type='stl',
        stl_file='embedded_object.stl',  # Replace with your STL file path
    ),
)

# Add emitting antenna source
sim.add_applied_field(emit_antenna)

# Add diagnostics
sim.add_diagnostic(field_diag)  # Full domain, periodic
sim.add_diagnostic(recv_plane_diag)  # Receiving plane, every timestep
sim.add_diagnostic(field_max_diag)  # Reduced diagnostic
sim.add_diagnostic(field_energy_diag)  # Reduced diagnostic

# Write input file or run simulation
if __name__ == "__main__":
    import sys
    
    if '--write-inputs' in sys.argv:
        # Write input file for compiled WarpX
        sim.write_input_file(file_name='inputs_from_picmi')
        print("\nInput file written to 'inputs_from_picmi'")
    else:
        # Run with libwarpx
        sim.initialize_inputs()
        sim.initialize_warpx()
        print("\nStarting simulation...")
        sim.step(sim.max_steps)
        print("Simulation complete!")
