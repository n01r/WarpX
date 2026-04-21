#!/usr/bin/env python3
"""
3D WarpX vacuum simulation with TE10 mode injection.
Microwave frequency: 67 GHz.  PML boundaries with embedded boundary from STL.
Geometry: horn-to-horn assembly, propagation along +Z.
TE10 broad wall (a) along X, narrow wall (b) along Y.
E-field polarised along Y.

Supports two excitation modes:
  - "cw":    Gaussian ramp-up → constant emission
  - "pulse": Gaussian pulse (for broadband S21, VIAS3D comparison)
"""

import numpy as np
from pywarpx import picmi

c   = picmi.constants.c
mu0 = picmi.constants.mu0
ep0 = picmi.constants.ep0

# =========================================================================
#                        USER PARAMETERS
# =========================================================================

# ---- Microwave source ----
freq = 67e9           # Hz — carrier frequency
E0   = 1e6            # V/m — peak Ey amplitude of the TE10 drive

# ---- Excitation mode:  "cw"  or  "pulse" ----
excitation_mode = "pulse"

# CW parameters (used when excitation_mode == "cw")
cw_ramp_periods  = 20
cw_n_guide_wl    = 100

# Gaussian-pulse parameters (used when excitation_mode == "pulse")
pulse_t_peak = 0.13e-9    # s
pulse_fwhm   = 0.05e-9    # s

# ---- Simulation duration override ----
t_sim_override = 15e-9    # s  (None = auto)

# ---- Domain ----
x_size = 0.030    # m
y_size = 0.030    # m
z_size = 0.280    # m  (STL spans ±130 mm)

# ---- Grid resolution ----
cells_per_wavelength = 20
cfl = 0.995

# ---- PML ----
pml_ncells = 10

# ---- Waveguide geometry (WR15) ----
# Propagation along +Z.  Broad wall (a) along X, narrow wall (b) along Y.
a_wg = 3.7592e-3     # m — broad-wall width (X), sets TE10 cutoff
b_wg = 1.8796e-3     # m — narrow-wall height (Y)

# ---- Waveguide centre (X-Y coordinates, shared by both ports) ----
waveguide_x_center = 0.0    # m
waveguide_y_center = 0.0    # m

# ---- Injection port (X-Y plane at Z = -0.12272 m) ----
# The LASY antenna fires further back at emit_z = -0.13000 m so the
# mode has room to establish itself before the diagnostic plane.
emit_z   = -0.13000   # m — LASY antenna position
inject_z = -0.12272   # m — injection diagnostic plane

# ---- Receiving port (X-Y plane at Z = +0.12591 m) ----
recv_z = +0.12591     # m

# ---- FieldProbe resolution ----
fieldprobe_resolution = 50

# ---- Diagnostic output intervals ----
field_snapshot_dt      = 0.15e-9    # s — full-domain snapshot cadence
field_snapshot_t_start = 0.0
field_snapshot_t_end   = None       # None = no limit

# XZ slice (at waveguide_y_center) snapshot cadence
field_slice_snapshot_dt = 0.005e-9  # s

# Port diagnostic interval (every step for FFT-based S21)
port_diag_period = 1

# Reduced diagnostics period (steps)
reduced_diag_period = 10

# ---- Embedded boundary ----
stl_file_path = './stl_input/cleaned_mwi_horns.STL'

# =========================================================================
#                     END OF USER PARAMETERS
# =========================================================================

omega      = 2 * np.pi * freq
wavelength = c / freq
k0         = 2 * np.pi / wavelength

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

dt_cfl = cfl / (c * np.sqrt(1/dx**2 + 1/dy**2 + 1/dz**2))

wg_half_a = a_wg / 2
wg_half_b = b_wg / 2

fc_te10          = c / (2 * a_wg)
beta             = np.sqrt(k0**2 - (np.pi / a_wg)**2)
Z_TE10           = omega * mu0 / beta
H0               = E0 / Z_TE10
wavelength_guide = 2 * np.pi / beta
vg               = c * np.sqrt(1 - (fc_te10 / freq)**2)

emitter_to_receiver = abs(recv_z - emit_z)
t_propagation       = emitter_to_receiver / vg

if excitation_mode == "cw":
    T_rf      = 1.0 / freq
    ramp_time = cw_ramp_periods * T_rf
    ramp_tau  = ramp_time / 2.15
    pulse_tau = None
elif excitation_mode == "pulse":
    pulse_tau = pulse_fwhm / (2 * np.sqrt(np.log(2)))
    ramp_time = None
    ramp_tau  = None
else:
    raise ValueError(f"Unknown excitation_mode '{excitation_mode}'. Use 'cw' or 'pulse'.")

if t_sim_override is not None and t_sim_override > 0:
    t_sim           = t_sim_override
    max_steps       = int(np.ceil(t_sim / dt_cfl))
    duration_source = f"user override ({t_sim*1e9:.3f} ns)"
elif excitation_mode == "cw":
    t_sim           = cw_n_guide_wl * wavelength_guide / vg
    max_steps       = int(np.ceil(t_sim / dt_cfl))
    duration_source = f"auto CW: {cw_n_guide_wl} λg"
elif excitation_mode == "pulse":
    t_sim           = pulse_t_peak + t_propagation + 6 * pulse_tau
    max_steps       = int(np.ceil(t_sim / dt_cfl))
    duration_source = "auto pulse: t_peak + propagation + 6τ"

if field_snapshot_dt is not None and field_snapshot_dt > 0:
    field_snapshot_period   = max(1, int(round(field_snapshot_dt / dt_cfl)))
    field_snapshot_min_step = max(0, int(round(field_snapshot_t_start / dt_cfl)))
    field_snapshot_max_step = (int(round(field_snapshot_t_end / dt_cfl))
                               if field_snapshot_t_end is not None else max_steps)
    field_snapshot_intervals = (f"{field_snapshot_min_step}:"
                                f"{field_snapshot_max_step}:{field_snapshot_period}")
else:
    field_snapshot_period    = 9999999999
    field_snapshot_intervals = None

field_slice_snapshot_period = (max(1, int(round(field_slice_snapshot_dt / dt_cfl)))
                                if field_slice_snapshot_dt else 9999999999)

print(f"Grid: {nx} x {ny} x {nz} = {nx*ny*nz:.3e} cells")
print(f"Cell size: dx={dx:.3e}, dy={dy:.3e}, dz={dz:.3e} m")
print(f"Wavelength: {wavelength*1e3:.3f} mm  |  CFL dt = {dt_cfl:.3e} s")
print(f"TE10 cutoff: {fc_te10*1e-9:.2f} GHz  |  guide λ = {wavelength_guide*1e2:.2f} cm  |  vg = {vg/c:.3f} c")
print(f"Injection  z = {inject_z*1e2:.5f} cm")
print(f"Receiving  z = {recv_z*1e2:.5f} cm")
print(f"Propagation: {emitter_to_receiver*1e2:.3f} cm  ({t_propagation*1e9:.3f} ns)")
if excitation_mode == "cw":
    print(f"Excitation: CW ramp {cw_ramp_periods} periods ({ramp_time*1e12:.1f} ps)")
else:
    print(f"Excitation: pulse  t_peak={pulse_t_peak*1e9:.3f} ns  "
          f"FWHM={pulse_fwhm*1e12:.1f} ps  τ={pulse_tau*1e12:.2f} ps")
print(f"Simulation: {t_sim*1e9:.3f} ns = {max_steps} steps  [{duration_source}]")

# =========================================================================
# LASY profile classes
# =========================================================================
from lasy.laser import Laser
from lasy.profiles.profile import Profile


class TE10CWProfile(Profile):
    """TE10 mode: Gaussian ramp-up → CW.
    LASY x → sim X (broad wall), LASY y → sim Y (narrow wall).
    """
    def __init__(self, wavelength, pol, E0, a_wg, ramp_time, tau):
        super().__init__(wavelength, pol)
        self.E0 = E0; self.a_wg = a_wg
        self.ramp_time = ramp_time; self.tau = tau

    def evaluate(self, x, y, t):
        spatial  = self.E0 * np.cos(np.pi * x / self.a_wg)
        envelope = np.where(t < self.ramp_time,
                            1.0 - np.exp(-(t / self.tau)**2), 1.0)
        return (spatial * envelope).astype(complex)


class TE10PulseProfile(Profile):
    """TE10 mode: Gaussian pulse.
    LASY x → sim X (broad wall), LASY y → sim Y (narrow wall).
    """
    def __init__(self, wavelength, pol, E0, a_wg, t_peak, tau):
        super().__init__(wavelength, pol)
        self.E0 = E0; self.a_wg = a_wg
        self.t_peak = t_peak; self.tau = tau

    def evaluate(self, x, y, t):
        spatial  = self.E0 * np.cos(np.pi * x / self.a_wg)
        envelope = np.exp(-((t - self.t_peak) / self.tau) ** 2)
        return (spatial * envelope).astype(complex)


# =========================================================================
# LASY laser profile
# =========================================================================
laser_n_transverse = 64

if excitation_mode == "cw":
    te10_profile = TE10CWProfile(
        wavelength=wavelength, pol=(0, 1), E0=E0, a_wg=a_wg,
        ramp_time=ramp_time, tau=ramp_tau)
    laser_t_min = 0.0
    laser_t_max = max_steps * dt_cfl * 1.1
    laser_nt    = max(64, int(cw_ramp_periods * 10)) + 2
else:
    te10_profile = TE10PulseProfile(
        wavelength=wavelength, pol=(0, 1), E0=E0, a_wg=a_wg,
        t_peak=pulse_t_peak, tau=pulse_tau)
    laser_t_min = max(0.0, pulse_t_peak - 5 * pulse_tau)
    laser_t_max = pulse_t_peak + 5 * pulse_tau
    laser_nt    = max(64, int((laser_t_max - laser_t_min) * freq * 10))

lasy_laser = Laser(
    dim='xyt',
    lo=[-(a_wg/2 + dx), -(b_wg/2 + dy), laser_t_min],
    hi=[ (a_wg/2 + dx),  (b_wg/2 + dy), laser_t_max],
    npoints=(laser_n_transverse, laser_n_transverse, laser_nt),
    profile=te10_profile,
)
lasy_file      = 'te10_laser_profile'
lasy_laser.write_to_file(file_prefix=lasy_file, file_format='h5')
lasy_file_path = f'diags/{lasy_file}_00000.h5'
print(f"LASY: {lasy_file_path}  ({laser_n_transverse}×{laser_n_transverse}×{laser_nt})")

# =========================================================================
# WarpX objects
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
    grid=grid, method='Yee', cfl=cfl, warpx_pml_ncell=pml_ncells)

embedded_boundary = picmi.EmbeddedBoundary(
    stl_file=stl_file_path, cover_multiple_cuts=True)

emit_antenna_position = [waveguide_x_center, waveguide_y_center, emit_z]

# ---- Full-domain snapshot ----
field_diag = picmi.FieldDiagnostic(
    name='fields', grid=grid, period=field_snapshot_period,
    data_list=['E', 'B', 'eb_covered'],
    write_dir='./diags', warpx_format='openpmd', warpx_openpmd_backend='bp5',
)
if field_snapshot_intervals is not None:
    field_diag.intervals = field_snapshot_intervals

# ---- XZ slice at waveguide_y_center (variable-based) ----
field_slice_diag = picmi.FieldDiagnostic(
    name='fields_sliced', grid=grid, period=field_slice_snapshot_period,
    data_list=['E', 'B', 'eb_covered'],
    write_dir='./diags', warpx_format='openpmd', warpx_openpmd_backend='bp5',
    lower_bound=[-x_size/2, waveguide_y_center, -z_size/2],
    upper_bound=[ x_size/2, waveguide_y_center,  z_size/2],
    warpx_openpmd_encoding='v',
)

# ---- Injection-plane diagnostic (X-Y plane at inject_z) ----
inject_plane_diag = picmi.FieldDiagnostic(
    name='injection_plane', grid=grid, period=port_diag_period,
    data_list=['Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz'],
    write_dir='./diags', warpx_format='openpmd', warpx_openpmd_backend='bp5',
    lower_bound=[waveguide_x_center - wg_half_a - dx,
                 waveguide_y_center - wg_half_b - dy,
                 inject_z],
    upper_bound=[waveguide_x_center + wg_half_a + dx,
                 waveguide_y_center + wg_half_b + dy,
                 inject_z],
    warpx_openpmd_encoding='v',
)

inject_plane_probe = picmi.ReducedDiagnostic(
    diag_type="FieldProbe", name="inject_probe",
    period=port_diag_period,
    path="diags/", extension="dat",
    probe_geometry="Plane", resolution=fieldprobe_resolution,
    x_probe=waveguide_x_center,
    y_probe=waveguide_y_center,
    z_probe=inject_z,
    detector_radius=wg_half_a * np.sqrt(2),
    target_normal_x=0.0, target_normal_y=0.0, target_normal_z=1.0,
    target_up_x=0.0,     target_up_y=1.0,     target_up_z=0.0,
)

# ---- Receiving-plane diagnostic (X-Y plane at recv_z) ----
recv_plane_diag = picmi.FieldDiagnostic(
    name='receiving_plane', grid=grid, period=port_diag_period,
    data_list=['Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz'],
    write_dir='./diags', warpx_format='openpmd', warpx_openpmd_backend='bp5',
    lower_bound=[waveguide_x_center - wg_half_a - dx,
                 waveguide_y_center - wg_half_b - dy,
                 recv_z],
    upper_bound=[waveguide_x_center + wg_half_a + dx,
                 waveguide_y_center + wg_half_b + dy,
                 recv_z],
    warpx_openpmd_encoding='v',
)

recv_plane_probe = picmi.ReducedDiagnostic(
    diag_type="FieldProbe", name="recv_probe",
    period=port_diag_period,
    path="diags/", extension="dat",
    probe_geometry="Plane", resolution=fieldprobe_resolution,
    x_probe=waveguide_x_center,
    y_probe=waveguide_y_center,
    z_probe=recv_z,
    detector_radius=wg_half_a * np.sqrt(2),
    target_normal_x=0.0, target_normal_y=0.0, target_normal_z=1.0,
    target_up_x=0.0,     target_up_y=1.0,     target_up_z=0.0,
)

field_max_diag    = picmi.ReducedDiagnostic(
    diag_type='FieldMaximum', name='field_max', period=reduced_diag_period)
field_energy_diag = picmi.ReducedDiagnostic(
    diag_type='FieldEnergy', name='field_energy', period=reduced_diag_period)

sim = picmi.Simulation(
    solver=solver, max_steps=max_steps, verbose=1,
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
if __name__ == "__main__":
    import sys
    import pywarpx

    pywarpx.algo.particle_shape = 2
    pywarpx.amrex.the_arena_is_managed = 0
    pywarpx.amrex.the_arena_init_size = 0

    pywarpx.lasers.names = ['laser1']
    laser = pywarpx.Lasers.newlaser('laser1')
    laser.profile               = 'from_file'
    laser.position              = emit_antenna_position
    laser.direction             = [0, 0, 1]    # propagation along +Z
    laser.polarization          = [0, 1, 0]    # Ey polarisation
    laser.e_max                 = E0
    laser.wavelength            = wavelength
    laser.lasy_file_name        = lasy_file_path
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
