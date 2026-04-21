#!/usr/bin/env python3
"""
3D WarpX vacuum simulation with TE10 mode injection in a straight WR-15 waveguide.
Microwave frequency: 67 GHz.

Boundary conditions
-------------------
X/Y/Z: PML (open) — all six domain faces use absorbing boundaries.
Embedded boundary: implicit function defining a straight WR-15 rectangular
     waveguide.  The EB runs from wg_z_lo to wg_z_hi, which extends
     ~1 free-space wavelength (5 mm) beyond both the antenna and the
     receiving diagnostic plane.  Outside this range the waveguide walls
     are absent so the field can expand into vacuum and be absorbed by the
     PML before reaching the domain boundary.

The antenna position (emit_z), injection diagnostic (inject_z), and
receiving diagnostic (recv_z) are identical to the horn-to-horn scripts,
allowing direct comparison of S21 with and without the horn geometry.

This is the most basic test case for the emitter–receiver geometry.
Use it to validate mode injection, port diagnostics, and S21 reconstruction
before adding the horn geometry or scooter section.
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
freq = 67e9           # Hz
E0   = 1e6            # V/m — peak Ey amplitude of the TE10 drive

# ---- Excitation mode:  "cw"  or  "pulse" ----
excitation_mode = "cw"

# CW parameters
cw_ramp_periods  = 20
cw_n_guide_wl    = 20

# Gaussian-pulse parameters
pulse_t_peak = 0.13e-9
pulse_fwhm   = 0.05e-9

# ---- Simulation duration override ----
t_sim_override = None

# ---- Domain ----
# Z: waveguide runs from emit_z - wg_margin to recv_z + wg_margin
#    (~1 free-space λ margin each side), plus PML padding.
# X/Y: waveguide cross-section plus 3 free-space λ on each side for
#      evanescent decay and PML absorption.
wg_margin = 5e-3     # m — ~1 free-space wavelength at 67 GHz

# Computed below after waveguide geometry is defined; these are overridden.
# Set placeholders here so user section is self-contained.
x_size = None   # computed from a_wg + lateral_margin
y_size = None   # computed from b_wg + lateral_margin
z_size = None   # computed from port positions + wg_margin + PML

# ---- Grid resolution ----
cells_per_wavelength = 16
cfl = 0.995

# ---- PML ----
pml_ncells = 10

# ---- Waveguide geometry (WR15) ----
# Propagation along +Z.  Broad wall (a) along X, narrow wall (b) along Y.
a_wg = 3.7592e-3
b_wg = 1.8796e-3

# ---- Waveguide centre ----
waveguide_x_center = 0.0
waveguide_y_center = 0.0

# ---- Injection port (X-Y plane at Z = -0.12272 m) ----
# The LASY antenna fires further back at emit_z = -0.13000 m so the
# mode has room to establish itself before the diagnostic plane.
emit_z   = -0.13000   # m — LASY antenna position
inject_z = -0.12272   # m — injection diagnostic plane

# ---- Receiving port (X-Y plane at Z = +0.12591 m) ----
recv_z = +0.12591

# ---- FieldProbe resolution ----
fieldprobe_resolution = 50

# ---- Diagnostic output intervals ----
field_snapshot_dt      = 0.15e-9
field_snapshot_t_start = 0.0
field_snapshot_t_end   = None

# XZ slice at waveguide_y_center
field_slice_snapshot_dt = 0.005e-9

# Port diagnostic interval
port_diag_period = 1

# Reduced diagnostics period
reduced_diag_period = 10

# =========================================================================
#                     END OF USER PARAMETERS
# =========================================================================

omega      = 2 * np.pi * freq
wavelength = c / freq
k0         = 2 * np.pi / wavelength

# ---- Waveguide EB extent ----
# The EB waveguide runs from wg_z_lo to wg_z_hi, enclosing the antenna,
# injection diagnostic, and receiving diagnostic with ~1 λ margin each side.
wg_z_lo = emit_z   - wg_margin   # ~ -0.135 m
wg_z_hi = recv_z   + wg_margin   # ~ +0.131 m

# ---- Domain sizes ----
# Z: waveguide extent + PML padding (estimated conservatively as 15 cells × λ/cells_per_λ)
pml_margin = 15 * (wavelength / cells_per_wavelength)
z_lo = wg_z_lo - pml_margin
z_hi = wg_z_hi + pml_margin
z_size = z_hi - z_lo

# X/Y: waveguide cross-section + 3 free-space λ lateral margin for evanescent
#      field decay + PML absorption.  Fields outside the waveguide walls decay
#      exponentially, so 3 λ is more than sufficient before the PML.
lateral_margin = 3 * wavelength
x_size = a_wg + 2 * lateral_margin
y_size = b_wg + 2 * lateral_margin

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

# Embedded boundary: straight WR-15 waveguide via implicit CSG function.
# The conductor occupies everything outside the rectangular cross-section
# for z >= wg_start_z (5 guide wavelengths before the antenna).
# implicit_function < 0 = vacuum, > 0 = conductor.
wg_start_z = inject_z - 5 * wavelength_guide
print(f"Waveguide EB starts at z = {wg_start_z*1e2:.2f} cm (5 λg before antenna)")

waveguide_function = (f"min(max(abs(x) - {wg_half_a}, abs(y) - {wg_half_b}), "
                      f"z - ({wg_start_z}))")

print(f"Waveguide EB: z = [{wg_z_lo*1e2:.3f}, {wg_z_hi*1e2:.3f}] cm  "
      f"(margin = {wg_margin*1e3:.1f} mm each side)")
print(f"Domain: x={x_size*1e3:.1f} mm, y={y_size*1e3:.1f} mm, z={z_size*1e3:.1f} mm")
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
    lower_bound=[-x_size/2, -y_size/2, z_lo],
    upper_bound=[ x_size/2,  y_size/2, z_hi],
    lower_boundary_conditions=['open', 'open', 'open'],
    upper_boundary_conditions=['open', 'open', 'open'],
    warpx_max_grid_size_x=nx,
    warpx_max_grid_size_y=ny,
    warpx_max_grid_size_z=nz,
)

solver = picmi.ElectromagneticSolver(
    grid=grid, method='Yee', cfl=cfl, warpx_pml_ncell=pml_ncells)

embedded_boundary = picmi.EmbeddedBoundary(implicit_function=waveguide_function)

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
    lower_bound=[-x_size/2, waveguide_y_center, z_lo],
    upper_bound=[ x_size/2, waveguide_y_center, z_hi],
    warpx_openpmd_encoding='v',
)

# ---- Injection-plane diagnostic ----
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

# ---- Receiving-plane diagnostic ----
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
    laser.direction             = [0, 0, 1]
    laser.polarization          = [0, 1, 0]
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
