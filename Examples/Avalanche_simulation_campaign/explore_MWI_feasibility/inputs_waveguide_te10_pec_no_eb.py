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
cells_per_wavelength = 64
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
    warpx_max_grid_size_x=nx,
    warpx_max_grid_size_y=ny,
    warpx_max_grid_size_z=nz
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

# Number of guide wavelengths to propagate through the waveguide
n_guide_wavelengths = 20

# Guide wavelength and group velocity
wavelength_guide = 2 * np.pi / beta
vg = c * np.sqrt(1 - (fc_te10 / freq)**2)

# Time for n guide wavelengths to traverse at group velocity
propagation_time = n_guide_wavelengths * wavelength_guide / vg

# Convert to steps (using CFL-determined dt)
dt_cfl = cfl / (c * np.sqrt(1/dx**2 + 1/dy**2 + 1/dz**2))
max_steps = int(np.ceil(propagation_time / dt_cfl))
print(f"Time step: {dt_cfl:.3e} s (omega*dt = {omega*dt_cfl:.3f})")

print(f"Guide wavelength: {wavelength_guide*1e3:.3f} mm")
print(f"Group velocity: {vg/c:.3f} c")
print(f"Propagation time for {n_guide_wavelengths} λg: {propagation_time*1e9:.3f} ns")
print(f"max_steps: {max_steps}")

print(f"TE10 guide wavelength: {2*np.pi/beta*1e2:.2f} cm")
print(f"TE10 impedance: {Z_TE10:.1f} Ohm")
print(f"Horn aperture: {a*1e2:.2f} x {b*1e2:.2f} cm")
print(f"Emitting plane z-position: {emit_horn_z_pos*1e2:.2f} cm")
print(f"Receiving plane z-position: {recv_plane_z_pos*1e2:.2f} cm")

# Generate LASY file for laser injection
# LASY stores the complex envelope; WarpX adds the carrier oscillation cos(omega*t)
# automatically using the wavelength parameter. This means we only need to resolve
# the smooth ramp-up envelope, not the 67 GHz oscillation.

from lasy.laser import Laser
from lasy.profiles.profile import Profile

# Gaussian ramp-up parameters
ramp_periods = 20  # Ramp up over 20 RF periods
T_rf = 1.0 / freq
ramp_time = ramp_periods * T_rf
tau = ramp_time / 2.15  # ~99% amplitude at t = ramp_time

class TE10Profile(Profile):
    """TE10 waveguide mode profile with Gaussian ramp-up envelope."""
    def __init__(self, wavelength, pol, E0, a_wg, ramp_time, tau):
        super().__init__(wavelength, pol)
        self.E0 = E0
        self.a_wg = a_wg
        self.ramp_time = ramp_time
        self.tau = tau

    def evaluate(self, x, y, t):
        # Spatial: TE10 profile cos(pi*x/a), centered at x=0
        spatial = self.E0 * np.cos(np.pi * x / self.a_wg)
        # Temporal: Gaussian ramp-up envelope
        envelope = np.where(t < self.ramp_time,
                            1.0 - np.exp(-(t / self.tau)**2),
                            1.0)
        return (spatial * envelope).astype(complex)

te10_profile = TE10Profile(
    wavelength=wavelength,
    pol=(0, 1),  # Ey polarization
    E0=E0,
    a_wg=a,
    ramp_time=ramp_time,
    tau=tau,
)

# LASY grid: match simulation transverse resolution, time covers ramp then CW
laser_nx = nx
laser_ny = ny
# Time axis: ramp-up region needs ~10 samples per ramp_period for smooth envelope,
# then a single CW value after. Total time extends to end of simulation.
laser_t_max = max_steps * dt_cfl * 1.1  # 10% margin beyond simulation end
samples_during_ramp = max(64, int(ramp_periods * 10))  # >=10 samples per ramp period
laser_nt = samples_during_ramp + 2  # +2 for CW portion (start and end of flat region)

lasy_laser = Laser(
    dim='xyt',
    lo=[-x_size/2, -y_size/2, 0.0],
    hi=[x_size/2, y_size/2, laser_t_max],
    npoints=(laser_nx, laser_ny, laser_nt),
    profile=te10_profile,
)

lasy_file = 'te10_laser_profile'
lasy_laser.write_to_file(file_prefix=lasy_file, file_format='h5')
lasy_file_path = f'diags/{lasy_file}_00000.h5'
print(f"Wrote LASY file: {lasy_file_path}")
print(f"Gaussian ramp-up: {ramp_periods} RF periods ({ramp_time*1e12:.1f} ps)")
print(f"LASY grid: {laser_nx} x {laser_ny} x {laser_nt}, t_max = {laser_t_max*1e9:.3f} ns")

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
    warpx_openpmd_backend='bp5',
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
    max_steps=max_steps,
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

    # Configure laser via low-level pywarpx (bypasses AnalyticLaser GPU bug)
    # Uses LASY file: stores complex envelope, WarpX adds carrier oscillation
    pywarpx.lasers.names = ['laser1']
    laser = pywarpx.Lasers.newlaser('laser1')
    laser.profile = 'from_file'
    laser.position = emit_antenna_position
    laser.direction = [0, 0, 1]
    laser.polarization = [0, 1, 0]
    laser.e_max = E0
    laser.wavelength = wavelength
    laser.lasy_file_name = lasy_file_path
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
