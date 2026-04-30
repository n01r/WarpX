#!/usr/bin/env python3
"""
3D WarpX vacuum simulation with TE10 mode injection.
Microwave frequency: 67 GHz.  PML boundaries with embedded boundary from STL.

Supports two excitation modes:
  - "cw":    Gaussian ramp-up → constant emission
  - "pulse": Gaussian pulse (for broadband S21, VIAS3D comparison)

All user-configurable parameters are in the section below.
"""

import numpy as np
from pywarpx import picmi

# Physical constants
c   = picmi.constants.c
mu0 = picmi.constants.mu0
ep0 = picmi.constants.ep0

# #########################################################################
#                        USER PARAMETERS
# All tuneable settings are in this section.  Nothing below needs editing
# for routine use.
# #########################################################################

# ---- Microwave source ----
freq = 67e9           # Hz — carrier frequency
E0   = 1e6            # V/m — peak Ey amplitude of the TE10 drive

# ---- Excitation mode:  "cw"  or  "pulse" ----
excitation_mode = "pulse"

# CW parameters (used when excitation_mode == "cw")
cw_ramp_periods  = 20     # ramp-up duration in RF periods
cw_n_guide_wl    = 100    # simulate this many guide wavelengths (fallback duration)

# Gaussian-pulse parameters (used when excitation_mode == "pulse")
pulse_t_peak = 0.13e-9    # s — pulse centre (matching VIAS3D)
pulse_fwhm   = 0.05e-9    # s — full width at half maximum

# ---- Simulation duration override ----
# Set to a positive value (in seconds) to force the total simulation time.
# Set to None to let the script compute it automatically.
t_sim_override = 2.2e-9      # e.g. 1.5e-9 for 1.5 ns

# ---- Domain ----
x_size = 0.030    # m (30 mm)
y_size = 0.030    # m (30 mm)
z_size = 0.280    # m (280 mm, STL spans ±130 mm)

# ---- Grid resolution ----
cells_per_wavelength = 20  # ~8 cells across WR15 narrow dim at 20 cells/λ
cfl = 0.995

# ---- PML ----
pml_ncells = 10

# ---- Waveguide / horn geometry (WR15) ----
# See https://www.everythingrf.com/tech-resources/waveguides-sizes/wr15
a_horn = 3.7592e-3    # m — broad-wall width (x), sets TE10 cutoff
b_horn = 1.8796e-3    # m — narrow-wall height (y)

# ---- Emitting horn position ----
emit_horn_x_center = 0.0      # m
emit_horn_y_center = 0.0      # m
emit_horn_z_pos    = -13.0e-2  # m — z position inside rectangular waveguide

# ---- Receiving diagnostic plane position ----
recv_plane_x_center = 0.0
recv_plane_y_center = 0.0
recv_plane_z_pos    = -emit_horn_z_pos  # symmetric, inside receiving horn

# ---- FieldProbe plane diagnostic ----
fieldprobe_resolution = 50   # N×N grid on the receiving plane

# ---- Diagnostic output intervals ----
# Full-domain field snapshot interval.  Set in seconds (SI).
# Set to None to use a default of 100 time steps.
field_snapshot_dt = 0.005e-9      # e.g. 0.1e-9 for every 0.1 ns

# Receiving-plane diagnostic interval (in time steps)
recv_plane_period = 1         # every time step

# Reduced diagnostics (field max, field energy) period in steps
reduced_diag_period = 10

# ---- Embedded boundary ----
stl_file_path = './stl_input/cleaned_mwi_horns.STL'

# #########################################################################
#                     END OF USER PARAMETERS
# #########################################################################


# =========================================================================
# Derived quantities  (do not edit below unless extending the script)
# =========================================================================

omega      = 2 * np.pi * freq
wavelength = c / freq
k0         = 2 * np.pi / wavelength

# Grid cell sizes and rounding to multiple of 8 for GPU efficiency
def _round8(n):
    return max(8, int(np.ceil(n / 8) * 8))

dx = wavelength / cells_per_wavelength
dy = wavelength / cells_per_wavelength
dz = wavelength / cells_per_wavelength

nx = _round8(int(x_size / dx))
ny = _round8(int(y_size / dy))
nz = _round8(int(z_size / dz))

dx = x_size / nx
dy = y_size / ny
dz = z_size / nz

# CFL time step
dt_cfl = cfl / (c * np.sqrt(1/dx**2 + 1/dy**2 + 1/dz**2))

# Waveguide parameters
a = a_horn
b = b_horn
horn_half_width_x = a / 2
horn_half_width_y = b / 2

fc_te10 = c / (2 * a)
beta    = np.sqrt(k0**2 - (np.pi / a)**2)
Z_TE10  = omega * mu0 / beta
H0      = E0 / Z_TE10
wavelength_guide = 2 * np.pi / beta
vg = c * np.sqrt(1 - (fc_te10 / freq)**2)

# Propagation time from emitter to receiver
emitter_to_receiver = abs(recv_plane_z_pos - emit_horn_z_pos)
t_propagation = emitter_to_receiver / vg

# ---- Mode-specific derived parameters (needed by both duration logic and LASY) ----
if excitation_mode == "cw":
    T_rf = 1.0 / freq
    ramp_time = cw_ramp_periods * T_rf
    ramp_tau  = ramp_time / 2.15
    pulse_tau = None
elif excitation_mode == "pulse":
    pulse_tau = pulse_fwhm / (2 * np.sqrt(np.log(2)))
    ramp_time = None
    ramp_tau  = None
else:
    raise ValueError(f"Unknown excitation_mode '{excitation_mode}'. "
                     "Use 'cw' or 'pulse'.")

# ---- Simulation duration ----
if t_sim_override is not None and t_sim_override > 0:
    t_sim = t_sim_override
    max_steps = int(np.ceil(t_sim / dt_cfl))
    duration_source = f"user override ({t_sim*1e9:.3f} ns)"
elif excitation_mode == "cw":
    t_sim = cw_n_guide_wl * wavelength_guide / vg
    max_steps = int(np.ceil(t_sim / dt_cfl))
    duration_source = f"auto CW: {cw_n_guide_wl} λg"
elif excitation_mode == "pulse":
    t_sim = pulse_t_peak + t_propagation + 6 * pulse_tau
    max_steps = int(np.ceil(t_sim / dt_cfl))
    duration_source = "auto pulse: t_peak + propagation + 6τ"

# ---- Diagnostic periods ----
if field_snapshot_dt is not None and field_snapshot_dt > 0:
    field_snapshot_period = max(1, int(round(field_snapshot_dt / dt_cfl)))
else:
    field_snapshot_period = 100  # default fallback

# =========================================================================
# Print summary
# =========================================================================
print(f"Grid: {nx} x {ny} x {nz} = {nx*ny*nz:.3e} cells")
print(f"Cell size: dx={dx:.3e}, dy={dy:.3e}, dz={dz:.3e} m")
print(f"Wavelength: {wavelength*1e3:.3f} mm")
print(f"CFL: {cfl:.3f},  dt = {dt_cfl:.3e} s  (ω·dt = {omega*dt_cfl:.3f})")
print(f"TE10 cutoff: {fc_te10*1e-9:.2f} GHz")
if freq < fc_te10:
    print(f"  WARNING: f = {freq*1e-9:.2f} GHz is below cutoff — evanescent!")
print(f"TE10 guide wavelength: {wavelength_guide*1e2:.2f} cm")
print(f"TE10 impedance: {Z_TE10:.1f} Ω")
print(f"Group velocity: {vg/c:.3f} c")
print(f"Horn aperture: {a*1e3:.3f} x {b*1e3:.3f} mm")
print(f"Emitter z = {emit_horn_z_pos*1e2:.2f} cm,  Receiver z = {recv_plane_z_pos*1e2:.2f} cm")
print(f"Propagation time (emitter→receiver): {t_propagation*1e9:.3f} ns")
if excitation_mode == "cw":
    print(f"Excitation: CW  (ramp {cw_ramp_periods} periods = {ramp_time*1e12:.1f} ps)")
elif excitation_mode == "pulse":
    print(f"Excitation: Gaussian pulse  "
          f"(t_peak = {pulse_t_peak*1e9:.3f} ns, "
          f"FWHM = {pulse_fwhm*1e12:.1f} ps, τ = {pulse_tau*1e12:.2f} ps)")
print(f"Simulation: {t_sim*1e9:.3f} ns = {max_steps} steps  [{duration_source}]")
print(f"Field snapshot every {field_snapshot_period} steps "
      f"({field_snapshot_period * dt_cfl * 1e9:.4f} ns)")

# =========================================================================
# LASY profile classes
# =========================================================================
from lasy.laser import Laser
from lasy.profiles.profile import Profile


class TE10CWProfile(Profile):
    """TE10 mode: Gaussian ramp-up → constant (CW) envelope."""
    def __init__(self, wavelength, pol, E0, a_wg, ramp_time, tau):
        super().__init__(wavelength, pol)
        self.E0 = E0;  self.a_wg = a_wg
        self.ramp_time = ramp_time;  self.tau = tau

    def evaluate(self, x, y, t):
        spatial  = self.E0 * np.cos(np.pi * x / self.a_wg)
        envelope = np.where(t < self.ramp_time,
                            1.0 - np.exp(-(t / self.tau)**2), 1.0)
        return (spatial * envelope).astype(complex)


class TE10PulseProfile(Profile):
    """TE10 mode: Gaussian pulse envelope."""
    def __init__(self, wavelength, pol, E0, a_wg, t_peak, tau):
        super().__init__(wavelength, pol)
        self.E0 = E0;  self.a_wg = a_wg
        self.t_peak = t_peak;  self.tau = tau

    def evaluate(self, x, y, t):
        spatial  = self.E0 * np.cos(np.pi * x / self.a_wg)
        envelope = np.exp(-((t - self.t_peak) / self.tau) ** 2)
        return (spatial * envelope).astype(complex)


# =========================================================================
# Build LASY laser profile
# =========================================================================
laser_nx = nx
laser_ny = ny

if excitation_mode == "cw":
    te10_profile = TE10CWProfile(
        wavelength=wavelength, pol=(0, 1), E0=E0, a_wg=a,
        ramp_time=ramp_time, tau=ramp_tau)
    laser_t_min = 0.0
    laser_t_max = max_steps * dt_cfl * 1.1
    laser_nt = max(64, int(cw_ramp_periods * 10)) + 2
else:
    te10_profile = TE10PulseProfile(
        wavelength=wavelength, pol=(0, 1), E0=E0, a_wg=a,
        t_peak=pulse_t_peak, tau=pulse_tau)
    laser_t_min = max(0.0, pulse_t_peak - 5 * pulse_tau)
    laser_t_max = pulse_t_peak + 5 * pulse_tau
    laser_nt = max(64, int((laser_t_max - laser_t_min) * freq * 10))

lasy_laser = Laser(
    dim='xyt',
    lo=[-(a/2 + dx), -(b/2 + dy), laser_t_min],
    hi=[ (a/2 + dx),  (b/2 + dy), laser_t_max],
    npoints=(laser_nx, laser_ny, laser_nt),
    profile=te10_profile,
)

lasy_file = 'te10_laser_profile'
lasy_laser.write_to_file(file_prefix=lasy_file, file_format='h5')
lasy_file_path = f'diags/{lasy_file}_00000.h5'
print(f"Wrote LASY file: {lasy_file_path}")
print(f"  LASY grid: {laser_nx} x {laser_ny} x {laser_nt}, "
      f"t = [{laser_t_min*1e12:.1f}, {laser_t_max*1e12:.1f}] ps")

# =========================================================================
# Build WarpX simulation objects
# =========================================================================

grid = picmi.Cartesian3DGrid(
    number_of_cells=[nx, ny, nz],
    lower_bound=[-x_size/2, -y_size/2, -z_size/2],
    upper_bound=[ x_size/2,  y_size/2,  z_size/2],
    lower_boundary_conditions=['open', 'open', 'open'],
    upper_boundary_conditions=['open', 'open', 'open'],
    warpx_max_grid_size_x=nx,
    warpx_max_grid_size_y=ny,
    warpx_max_grid_size_z=nz,
)

solver = picmi.ElectromagneticSolver(
    grid=grid, method='Yee', cfl=cfl, warpx_pml_ncell=pml_ncells,
)

embedded_boundary = picmi.EmbeddedBoundary(
    stl_file=stl_file_path, cover_multiple_cuts=True,
)

emit_antenna_position = [emit_horn_x_center, emit_horn_y_center, emit_horn_z_pos]

# --- Diagnostics ---

# Full-domain field snapshot
field_slice_diag = picmi.FieldDiagnostic(
    name='fields_sliced', grid=grid, period=field_snapshot_period,
    data_list=['E', 'B', 'eb_covered'],
    write_dir='./diags', warpx_format='openpmd', warpx_openpmd_backend='bp5',
    lower_bound=[-0.5*x_size,
                 0.0,
                 -0.5*z_size],
    upper_bound=[0.5*x_size,
                 0.0,
                 0.5*z_size],
    warpx_openpmd_encoding='v'
)

# --- Assemble simulation ---
sim = picmi.Simulation(
    solver=solver, max_steps=max_steps, verbose=1,
    warpx_embedded_boundary=embedded_boundary,
)
sim.add_diagnostic(field_slice_diag)

# =========================================================================
# Run
# =========================================================================
if __name__ == "__main__":
    import sys
    import pywarpx

    pywarpx.algo.particle_shape = 1
    pywarpx.amrex.the_arena_is_managed = 0
    pywarpx.amrex.the_arena_init_size = 0

    # Laser injection via low-level pywarpx (bypasses AnalyticLaser GPU bug)
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

        recv_bucket = pywarpx.diagnostics._diagnostics_dict['fields_sliced']
        recv_bucket.add_new_group_attr(
            'adios2_engine', 'parameters.AsyncWrite', 'on')
        recv_bucket.add_new_group_attr(
            'adios2_engine', 'parameters.BufferChunkSize',
            str(32 * 1024 * 1024 * 1024))

        print("\nInitializing WarpX...", flush=True)
        sim.initialize_warpx()
        print("\nStarting simulation...", flush=True)
        sim.step(sim.max_steps)
        print("Simulation complete!", flush=True)
