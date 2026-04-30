#!/usr/bin/env python3
"""
3D WarpX vacuum simulation with TE10 mode injection in a straight WR-15 waveguide.
Microwave frequency: 67 GHz.

Tuned EB configuration
----------------------
This version is tuned for *straight-waveguide validation* rather than for an
open-ended aperture test:

- X/Y use Dirichlet boundaries so only Z uses PML.
- The embedded-boundary waveguide extends through the full Z domain.
- The EB walls are aligned to cell faces by choosing a grid that fits the
  WR-15 dimensions exactly: 16 cells across the broad wall and 8 across the
  narrow wall.
- Only a modest covered-cell buffer is kept outside the waveguide walls,
  instead of several free-space wavelengths.  This cuts memory use sharply.
- The Z PML is made thicker than in the original file.

Why this helps
--------------
The original EB file allocated a very large X/Y domain and used open
boundaries on all six faces, but the implicit function actually made the
region outside the WR-15 cross-section conducting throughout the domain.
That means the extra transverse size was mostly covered EB conductor, not
useful vacuum.  It also meant the PML had to coexist with EB-covered regions.
This tuned setup makes the EB case much closer to the clean PEC reference,
while still exercising the EB machinery.

If you *do* want a true open-ended waveguide aperture that radiates into
vacuum before the PML, see the commented alternative finite-length EB
function near the waveguide definition.  In that case some reflection is
physical and cannot be removed entirely by PML tuning.
"""

import numpy as np
from pywarpx import picmi

c = picmi.constants.c
mu0 = picmi.constants.mu0
ep0 = picmi.constants.ep0

# =========================================================================
#                        USER PARAMETERS
# =========================================================================

# ---- Microwave source ----
freq = 67e9           # Hz
E0 = 1e6              # V/m -- peak Ey amplitude of the TE10 drive

# ---- Excitation mode:  "cw"  or  "pulse" ----
excitation_mode = "pulse"

# CW parameters
cw_ramp_periods = 20
cw_n_guide_wl = 20

# Gaussian-pulse parameters
pulse_t_peak = 0.13e-9
pulse_fwhm = 0.05e-9

# ---- Simulation duration override ----
t_sim_override = 10e-9

# ---- Waveguide geometry (WR15) ----
# Propagation along +Z. Broad wall (a) along X, narrow wall (b) along Y.
a_wg = 3.7592e-3
b_wg = 1.8796e-3

# ---- Waveguide centre ----
waveguide_x_center = 0.0
waveguide_y_center = 0.0

# ---- Injection / receiving planes ----
emit_z = -0.13000     # m -- LASY antenna position
inject_z = -0.12272   # m -- injection diagnostic plane
recv_z = +0.12591     # m -- receiving diagnostic plane

# ---- Grid / EB tuning ----
# Choose the guide resolution so the WR-15 walls land on cell faces.
guide_nx = 16         # cells across a_wg  -> dx = a_wg / 16
# Because b_wg = a_wg / 2, using 8 cells gives dy = dx.
guide_ny = 8          # cells across b_wg

# Covered-cell conductor buffer outside the guide walls in X/Y.
# 8 cells on each side keeps the EB comfortably away from the outer boundary
# while remaining far smaller than the original 3-lambda padding.
eb_buffer_ncells_xy = 8

# Use cubic cells in Z as well.
dz_match_xy = True

# ---- Z-domain buffers ----
# Keep some straight guide between the source/receiver planes and the PML.
z_buffer_before_emit = 10e-3   # m
z_buffer_after_recv = 10e-3    # m

# ---- PML ----
pml_ncells = 16

# ---- CFL ----
cfl = 0.995

# ---- FieldProbe resolution ----
fieldprobe_resolution = 50

# ---- Diagnostic output intervals ----
field_snapshot_dt = 0.15e-9
field_snapshot_t_start = 0.0
field_snapshot_t_end = None
field_slice_snapshot_dt = 0.005e-9
port_diag_period = 1
reduced_diag_period = 10

# =========================================================================
#                     END OF USER PARAMETERS
# =========================================================================

omega = 2 * np.pi * freq
wavelength = c / freq
k0 = 2 * np.pi / wavelength

wg_half_a = a_wg / 2
wg_half_b = b_wg / 2

# Grid chosen so the EB walls align to cell faces.
dx = a_wg / guide_nx
dy = b_wg / guide_ny
if not np.isclose(dx, dy, rtol=0.0, atol=1e-15):
    raise ValueError("guide_nx and guide_ny should be chosen so dx == dy for this setup")

dz = dx if dz_match_xy else wavelength / 20.0

# Total transverse cells = guide cells + covered-cell EB buffer on both sides.
nx = guide_nx + 2 * eb_buffer_ncells_xy
ny = guide_ny + 2 * eb_buffer_ncells_xy

x_size = nx * dx
y_size = ny * dy

# Z extent includes straight-guide buffer + PML thickness at each end.
pml_thickness = pml_ncells * dz
z_lo = emit_z - z_buffer_before_emit - pml_thickness
z_hi = recv_z + z_buffer_after_recv + pml_thickness
z_size = z_hi - z_lo

# Round nz up to a multiple of 8 without changing dx=dy=dz.
def _round8(n):
    return max(8, int(np.ceil(n / 8) * 8))

nz = _round8(int(np.ceil(z_size / dz)))
z_size = nz * dz
z_center = 0.5 * (z_lo + z_hi)
z_lo = z_center - 0.5 * z_size
z_hi = z_center + 0.5 * z_size

dt_cfl = cfl / (c * np.sqrt(1 / dx**2 + 1 / dy**2 + 1 / dz**2))

fc_te10 = c / (2 * a_wg)
beta = np.sqrt(k0**2 - (np.pi / a_wg) ** 2)
Z_TE10 = omega * mu0 / beta
H0 = E0 / Z_TE10
wavelength_guide = 2 * np.pi / beta
vg = c * np.sqrt(1 - (fc_te10 / freq) ** 2)

emitter_to_receiver = abs(recv_z - emit_z)
t_propagation = emitter_to_receiver / vg

if excitation_mode == "cw":
    T_rf = 1.0 / freq
    ramp_time = cw_ramp_periods * T_rf
    ramp_tau = ramp_time / 2.15
    pulse_tau = None
elif excitation_mode == "pulse":
    pulse_tau = pulse_fwhm / (2 * np.sqrt(np.log(2)))
    ramp_time = None
    ramp_tau = None
else:
    raise ValueError(f"Unknown excitation_mode '{excitation_mode}'. Use 'cw' or 'pulse'.")

if t_sim_override is not None and t_sim_override > 0:
    t_sim = t_sim_override
    max_steps = int(np.ceil(t_sim / dt_cfl))
    duration_source = f"user override ({t_sim*1e9:.3f} ns)"
elif excitation_mode == "cw":
    t_sim = cw_n_guide_wl * wavelength_guide / vg
    max_steps = int(np.ceil(t_sim / dt_cfl))
    duration_source = f"auto CW: {cw_n_guide_wl} lambda_g"
elif excitation_mode == "pulse":
    t_sim = pulse_t_peak + t_propagation + 6 * pulse_tau
    max_steps = int(np.ceil(t_sim / dt_cfl))
    duration_source = "auto pulse: t_peak + propagation + 6tau"

if field_snapshot_dt is not None and field_snapshot_dt > 0:
    field_snapshot_period = max(1, int(round(field_snapshot_dt / dt_cfl)))
    field_snapshot_min_step = max(0, int(round(field_snapshot_t_start / dt_cfl)))
    field_snapshot_max_step = (
        int(round(field_snapshot_t_end / dt_cfl))
        if field_snapshot_t_end is not None else max_steps
    )
    field_snapshot_intervals = (
        f"{field_snapshot_min_step}:{field_snapshot_max_step}:{field_snapshot_period}"
    )
else:
    field_snapshot_period = 9999999999
    field_snapshot_intervals = None

field_slice_snapshot_period = (
    max(1, int(round(field_slice_snapshot_dt / dt_cfl)))
    if field_slice_snapshot_dt else 9999999999
)

# Embedded boundary: straight WR-15 waveguide extending through the full domain.
# implicit_function < 0 = vacuum, > 0 = conductor.
# Because dx and dy were chosen to fit the WR-15 dimensions exactly, the EB walls
# lie on cell faces, minimizing geometric mismatch versus the PEC reference case.
waveguide_function = (
    f"max(abs(x) - {wg_half_a}, abs(y) - {wg_half_b})"
)

# Alternative finite-length open-aperture guide, if you later want the mode to
# radiate into vacuum before the PML.  Keep in mind that this introduces a real,
# physical mismatch at the open end, so some reflection is unavoidable.
#
# open_z_buffer = 2.0 * wavelength
# wg_z_lo_open = emit_z - open_z_buffer
# wg_z_hi_open = recv_z + open_z_buffer
# waveguide_function = (
#     f"min(max(abs(x) - {wg_half_a}, abs(y) - {wg_half_b}), "
#     f"min(z - ({wg_z_lo_open}), ({wg_z_hi_open}) - z))"
# )

print("Tuned EB waveguide configuration")
print(f"Domain: x={x_size*1e3:.3f} mm, y={y_size*1e3:.3f} mm, z={z_size*1e3:.3f} mm")
print(f"Grid: {nx} x {ny} x {nz} = {nx*ny*nz:.3e} cells")
print(f"Cell size: dx={dx:.3e}, dy={dy:.3e}, dz={dz:.3e} m")
print(f"PML cells (z only): {pml_ncells} -> thickness = {pml_thickness*1e3:.3f} mm")
print(f"Wavelength: {wavelength*1e3:.3f} mm | CFL dt = {dt_cfl:.3e} s")
print(
    f"TE10 cutoff: {fc_te10*1e-9:.2f} GHz | guide lambda = {wavelength_guide*1e2:.2f} cm | vg = {vg/c:.3f} c"
)
print(f"Injection  z = {inject_z*1e2:.5f} cm")
print(f"Receiving  z = {recv_z*1e2:.5f} cm")
print(f"Propagation: {emitter_to_receiver*1e2:.3f} cm ({t_propagation*1e9:.3f} ns)")
if excitation_mode == "cw":
    print(f"Excitation: CW ramp {cw_ramp_periods} periods ({ramp_time*1e12:.1f} ps)")
else:
    print(
        f"Excitation: pulse t_peak={pulse_t_peak*1e9:.3f} ns "
        f"FWHM={pulse_fwhm*1e12:.1f} ps tau={pulse_tau*1e12:.2f} ps"
    )
print(f"Simulation: {t_sim*1e9:.3f} ns = {max_steps} steps [{duration_source}]")

# =========================================================================
# LASY profile classes
# =========================================================================
from lasy.laser import Laser
from lasy.profiles.profile import Profile


class TE10CWProfile(Profile):
    """TE10 mode: Gaussian ramp-up -> CW."""

    def __init__(self, wavelength, pol, E0, a_wg, ramp_time, tau):
        super().__init__(wavelength, pol)
        self.E0 = E0
        self.a_wg = a_wg
        self.ramp_time = ramp_time
        self.tau = tau

    def evaluate(self, x, y, t):
        spatial = self.E0 * np.cos(np.pi * x / self.a_wg)
        envelope = np.where(t < self.ramp_time, 1.0 - np.exp(-(t / self.tau) ** 2), 1.0)
        return (spatial * envelope).astype(complex)


class TE10PulseProfile(Profile):
    """TE10 mode: Gaussian pulse."""

    def __init__(self, wavelength, pol, E0, a_wg, t_peak, tau):
        super().__init__(wavelength, pol)
        self.E0 = E0
        self.a_wg = a_wg
        self.t_peak = t_peak
        self.tau = tau

    def evaluate(self, x, y, t):
        spatial = self.E0 * np.cos(np.pi * x / self.a_wg)
        envelope = np.exp(-((t - self.t_peak) / self.tau) ** 2)
        return (spatial * envelope).astype(complex)


# =========================================================================
# LASY laser profile
# =========================================================================
laser_n_transverse = 64

if excitation_mode == "cw":
    te10_profile = TE10CWProfile(
        wavelength=wavelength,
        pol=(0, 1),
        E0=E0,
        a_wg=a_wg,
        ramp_time=ramp_time,
        tau=ramp_tau,
    )
    laser_t_min = 0.0
    laser_t_max = max_steps * dt_cfl * 1.1
    laser_nt = max(64, int(cw_ramp_periods * 10)) + 2
else:
    te10_profile = TE10PulseProfile(
        wavelength=wavelength,
        pol=(0, 1),
        E0=E0,
        a_wg=a_wg,
        t_peak=pulse_t_peak,
        tau=pulse_tau,
    )
    laser_t_min = max(0.0, pulse_t_peak - 5 * pulse_tau)
    laser_t_max = pulse_t_peak + 5 * pulse_tau
    laser_nt = max(64, int((laser_t_max - laser_t_min) * freq * 10))

lasy_laser = Laser(
    dim='xyt',
    lo=[-(a_wg / 2 + dx), -(b_wg / 2 + dy), laser_t_min],
    hi=[(a_wg / 2 + dx), (b_wg / 2 + dy), laser_t_max],
    npoints=(laser_n_transverse, laser_n_transverse, laser_nt),
    profile=te10_profile,
)
lasy_file = 'te10_laser_profile'
lasy_laser.write_to_file(file_prefix=lasy_file, file_format='h5')
lasy_file_path = f'diags/{lasy_file}_00000.h5'
print(f"LASY: {lasy_file_path} ({laser_n_transverse}x{laser_n_transverse}x{laser_nt})")

# =========================================================================
# WarpX objects
# =========================================================================

grid = picmi.Cartesian3DGrid(
    number_of_cells=[nx, ny, nz],
    lower_bound=[-x_size / 2, -y_size / 2, z_lo],
    upper_bound=[x_size / 2, y_size / 2, z_hi],
    lower_boundary_conditions=['dirichlet', 'dirichlet', 'open'],
    upper_boundary_conditions=['dirichlet', 'dirichlet', 'open'],
    lower_boundary_conditions_particles=['absorbing', 'absorbing', 'absorbing'],
    upper_boundary_conditions_particles=['absorbing', 'absorbing', 'absorbing'],
    warpx_max_grid_size_x=nx,
    warpx_max_grid_size_y=ny,
    warpx_max_grid_size_z=nz,
)

solver = picmi.ElectromagneticSolver(
    grid=grid,
    method='Yee',
    cfl=cfl,
    warpx_pml_ncell=pml_ncells,
)

embedded_boundary = picmi.EmbeddedBoundary(implicit_function=waveguide_function)

emit_antenna_position = [waveguide_x_center, waveguide_y_center, emit_z]

field_diag = picmi.FieldDiagnostic(
    name='fields',
    grid=grid,
    period=field_snapshot_period,
    data_list=['E', 'B', 'eb_covered'],
    write_dir='./diags',
    warpx_format='openpmd',
    warpx_openpmd_backend='bp5',
)
if field_snapshot_intervals is not None:
    field_diag.intervals = field_snapshot_intervals

field_slice_diag = picmi.FieldDiagnostic(
    name='fields_sliced',
    grid=grid,
    period=field_slice_snapshot_period,
    data_list=['E', 'B', 'eb_covered'],
    write_dir='./diags',
    warpx_format='openpmd',
    warpx_openpmd_backend='bp5',
    lower_bound=[-x_size / 2, waveguide_y_center, z_lo],
    upper_bound=[x_size / 2, waveguide_y_center, z_hi],
    warpx_openpmd_encoding='v',
)

inject_plane_diag = picmi.FieldDiagnostic(
    name='injection_plane',
    grid=grid,
    period=port_diag_period,
    data_list=['Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz'],
    write_dir='./diags',
    warpx_format='openpmd',
    warpx_openpmd_backend='bp5',
    lower_bound=[waveguide_x_center - wg_half_a - dx,
                 waveguide_y_center - wg_half_b - dy,
                 inject_z],
    upper_bound=[waveguide_x_center + wg_half_a + dx,
                 waveguide_y_center + wg_half_b + dy,
                 inject_z],
    warpx_openpmd_encoding='v',
)

inject_plane_probe = picmi.ReducedDiagnostic(
    diag_type='FieldProbe',
    name='inject_probe',
    period=port_diag_period,
    path='diags/',
    extension='dat',
    probe_geometry='Plane',
    resolution=fieldprobe_resolution,
    x_probe=waveguide_x_center,
    y_probe=waveguide_y_center,
    z_probe=inject_z,
    detector_radius=wg_half_a,
    target_normal_x=0.0,
    target_normal_y=0.0,
    target_normal_z=1.0,
    target_up_x=0.0,
    target_up_y=1.0,
    target_up_z=0.0,
)

recv_plane_diag = picmi.FieldDiagnostic(
    name='receiving_plane',
    grid=grid,
    period=port_diag_period,
    data_list=['Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz'],
    write_dir='./diags',
    warpx_format='openpmd',
    warpx_openpmd_backend='bp5',
    lower_bound=[waveguide_x_center - wg_half_a - dx,
                 waveguide_y_center - wg_half_b - dy,
                 recv_z],
    upper_bound=[waveguide_x_center + wg_half_a + dx,
                 waveguide_y_center + wg_half_b + dy,
                 recv_z],
    warpx_openpmd_encoding='v',
)

recv_plane_probe = picmi.ReducedDiagnostic(
    diag_type='FieldProbe',
    name='recv_probe',
    period=port_diag_period,
    path='diags/',
    extension='dat',
    probe_geometry='Plane',
    resolution=fieldprobe_resolution,
    x_probe=waveguide_x_center,
    y_probe=waveguide_y_center,
    z_probe=recv_z,
    detector_radius=wg_half_a,
    target_normal_x=0.0,
    target_normal_y=0.0,
    target_normal_z=1.0,
    target_up_x=0.0,
    target_up_y=1.0,
    target_up_z=0.0,
)

field_max_diag = picmi.ReducedDiagnostic(
    diag_type='FieldMaximum',
    name='field_max',
    period=reduced_diag_period,
)
field_energy_diag = picmi.ReducedDiagnostic(
    diag_type='FieldEnergy',
    name='field_energy',
    period=reduced_diag_period,
)

sim = picmi.Simulation(
    solver=solver,
    max_steps=max_steps,
    verbose=1,
    warpx_embedded_boundary=embedded_boundary,
)
sim.add_diagnostic(field_diag)
sim.add_diagnostic(field_slice_diag)
sim.add_diagnostic(inject_plane_diag)
sim.add_diagnostic(inject_plane_probe)
sim.add_diagnostic(recv_plane_diag)
sim.add_diagnostic(recv_plane_probe)
sim.add_diagnostic(field_max_diag)
sim.add_diagnostic(field_energy_diag)

# =========================================================================
# Run
# =========================================================================
if __name__ == '__main__':
    import sys
    import pywarpx

    pywarpx.algo.particle_shape = 2
    pywarpx.amrex.the_arena_is_managed = 0
    pywarpx.amrex.the_arena_init_size = 0

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
    else:
        print("\nInitializing inputs...", flush=True)
        sim.initialize_inputs()

        _buf = str(512 * 1024 * 1024)
        for _name in ('injection_plane', 'receiving_plane', 'fields_sliced'):
            _b = pywarpx.diagnostics._diagnostics_dict[_name]
            _b.add_new_group_attr('adios2_engine', 'parameters.AsyncWrite', 'on')
            _b.add_new_group_attr('adios2_engine', 'parameters.BufferChunkSize', _buf)

        print("\nInitializing WarpX...", flush=True)
        sim.initialize_warpx()
        print("\nStarting simulation...", flush=True)
        sim.step(sim.max_steps)
        print("Simulation complete!", flush=True)