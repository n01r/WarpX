#!/usr/bin/env python3
"""
3D WarpX vacuum simulation with TE10 mode injection in a straight WR-15 waveguide.
Microwave frequency: 67 GHz.

Boundary conditions
-------------------
X/Y: Dirichlet (PEC) — the simulation domain walls act directly as the
     conducting waveguide walls.  No embedded boundary or STL is needed.
     Only the cross-section of the WR-15 aperture is simulated, so there
     is no free vacuum outside the waveguide.
Z:   PML (open) — absorbs outgoing waves at both ends of the waveguide,
     preventing reflections from trapping energy in the domain.

Maxwell solver: CKC (Cole-Karkkainen-Cowan)
--------------------------------------------
CKC is a modified Yee stencil with fourth-order spatial accuracy in the
dispersion relation along the Cartesian axes, compared to second-order for
standard Yee.  Numerical phase velocity error scales as O(dx^4) rather than
O(dx^2), so halving the resolution increases the phase error by 16x instead
of 4x.  CKC is compatible with PEC (Dirichlet) boundary conditions, unlike
PSATD which does not support PEC boundaries in WarpX.

At 20 cells/wavelength the phase error with CKC is expected to be roughly
(2/4)^2 = 1/4 of the Yee error at the same resolution, i.e. ~7 deg max
over the 25 cm propagation distance instead of ~27 deg with Yee.

Reference: Cole-Karkkainen-Cowan stencil (https://doi.org/10.1103/PhysRevSTAB.16.041303)

This is the most basic test case for the emitter-receiver geometry.
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
freq = 67e9           # Hz — carrier frequency
E0   = 1e6            # V/m — peak Ey amplitude of the TE10 drive

# ---- Excitation mode:  "cw"  or  "pulse" ----
excitation_mode = "pulse"

# CW parameters (used when excitation_mode == "cw")
cw_ramp_periods  = 20
cw_n_guide_wl    = 20

# Gaussian-pulse parameters (used when excitation_mode == "pulse")
pulse_t_peak = 0.13e-9    # s
pulse_fwhm   = 0.05e-9    # s

# ---- Simulation duration override ----
t_sim_override = 10.e-9   # s  (None = auto)

# ---- Waveguide geometry (WR15) ----
# Propagation along +Z.  Broad wall (a) along X, narrow wall (b) along Y.
# Domain X/Y is set exactly to the waveguide cross-section; PEC domain
# walls replace the conducting waveguide walls.
a_wg = 3.7592e-3     # m — broad wall (X), sets TE10 cutoff
b_wg = 1.8796e-3     # m — narrow wall (Y)

# Z extent — same as horn-to-horn scripts for direct comparison
z_size = 0.280       # m

# ---- Grid resolution ----
cells_per_wavelength = 20
cfl = 0.995

# ---- PML ----
pml_ncells = 10

# ---- Waveguide centre (X-Y; both zero by construction for PEC case) ----
waveguide_x_center = 0.0
waveguide_y_center = 0.0

# ---- Injection port (X-Y plane at Z = -0.12272 m) ----
# The LASY antenna fires further back at emit_z = -0.13000 m so the
# mode has room to establish itself before the diagnostic plane.
emit_z   = -0.13000   # m — LASY antenna position
inject_z = -0.12272   # m — injection diagnostic plane

# ---- Receiving port (X-Y plane at Z = +0.12591 m) ----
recv_z = +0.12591     # m

# ---- Diagnostic output intervals ----
field_snapshot_dt      = 0.15e-9    # s
field_snapshot_t_start = 1.50e-9
field_snapshot_t_end   = None

# XZ slice (at waveguide_y_center = 0) snapshot cadence
field_slice_snapshot_dt = 0.005e-9  # s

# Port diagnostic interval (every step for FFT-based S21)
port_diag_period = 1

# Reduced diagnostics period (steps)
reduced_diag_period = 10

# =========================================================================
#                     END OF USER PARAMETERS
# =========================================================================

omega      = 2 * np.pi * freq
wavelength = c / freq
k0         = 2 * np.pi / wavelength

# Domain is exactly the waveguide cross-section in X/Y
x_size = a_wg
y_size = b_wg

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
print(f"Domain X/Y = WR15 cross-section ({a_wg*1e3:.3f} x {b_wg*1e3:.3f} mm), Z = {z_size*1e3:.0f} mm")
print(f"Wavelength: {wavelength*1e3:.3f} mm  |  CFL dt = {dt_cfl:.3e} s")
print(f"TE10 cutoff: {fc_te10*1e-9:.2f} GHz  |  guide λ = {wavelength_guide*1e2:.2f} cm  |  vg = {vg/c:.3f} c")
print(f"Boundaries: PEC in X/Y (waveguide walls), PML in Z (open ends)")
print(f"Solver: CKC (O(dx^4) dispersion, compatible with PEC boundaries)")
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
laser_n_transverse = nx   # match simulation resolution exactly

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
    lo=[-x_size/2, -y_size/2, laser_t_min],
    hi=[ x_size/2,  y_size/2, laser_t_max],
    npoints=(laser_n_transverse, ny, laser_nt),
    profile=te10_profile,
)
lasy_file      = 'te10_laser_profile'
lasy_laser.write_to_file(file_prefix=lasy_file, file_format='h5')
lasy_file_path = f'diags/{lasy_file}_00000.h5'
print(f"LASY: {lasy_file_path}  ({laser_n_transverse}×{ny}×{laser_nt})")

# =========================================================================
# WarpX objects
# =========================================================================

grid = picmi.Cartesian3DGrid(
    number_of_cells=[nx, ny, nz],
    lower_bound=[-x_size/2, -y_size/2, -z_size/2],
    upper_bound=[ x_size/2,  y_size/2,  z_size/2],
    lower_boundary_conditions=['dirichlet', 'dirichlet', 'open'],
    upper_boundary_conditions=['dirichlet', 'dirichlet', 'open'],
    lower_boundary_conditions_particles=['absorbing', 'absorbing', 'absorbing'],
    upper_boundary_conditions_particles=['absorbing', 'absorbing', 'absorbing'],
    warpx_max_grid_size_x=nx,
    warpx_max_grid_size_y=ny,
    warpx_max_grid_size_z=nz,
)

# CKC: Cole-Karkkainen-Cowan solver.  O(dx^4) dispersion, Courant-limited
# like Yee (same CFL formula applies), fully compatible with PEC boundaries.
solver = picmi.ElectromagneticSolver(
    grid=grid, method='CKC', cfl=cfl, warpx_pml_ncell=pml_ncells)

emit_antenna_position = [waveguide_x_center, waveguide_y_center, emit_z]

# ---- Full-domain snapshot ----
field_diag = picmi.FieldDiagnostic(
    name='fields', grid=grid, period=field_snapshot_period,
    data_list=['E', 'B'],
    write_dir='./diags', warpx_format='openpmd', warpx_openpmd_backend='bp5',
)
if field_snapshot_intervals is not None:
    field_diag.intervals = field_snapshot_intervals

# ---- XZ slice at waveguide_y_center = 0 (variable-based) ----
# Slice through the broad-wall midplane, showing propagation along Z
# and the TE10 cosine variation along X.
field_slice_diag = picmi.FieldDiagnostic(
    name='fields_sliced', grid=grid, period=field_slice_snapshot_period,
    data_list=['E', 'B'],
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
    lower_bound=[-wg_half_a - dx, -wg_half_b - dy, inject_z],
    upper_bound=[ wg_half_a + dx,  wg_half_b + dy, inject_z],
    warpx_openpmd_encoding='v',
)

# ---- Receiving-plane diagnostic (X-Y plane at recv_z) ----
recv_plane_diag = picmi.FieldDiagnostic(
    name='receiving_plane', grid=grid, period=port_diag_period,
    data_list=['Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz'],
    write_dir='./diags', warpx_format='openpmd', warpx_openpmd_backend='bp5',
    lower_bound=[-wg_half_a - dx, -wg_half_b - dy, recv_z],
    upper_bound=[ wg_half_a + dx,  wg_half_b + dy, recv_z],
    warpx_openpmd_encoding='v',
)

field_max_diag    = picmi.ReducedDiagnostic(
    diag_type='FieldMaximum', name='field_max', period=reduced_diag_period)
field_energy_diag = picmi.ReducedDiagnostic(
    diag_type='FieldEnergy', name='field_energy', period=reduced_diag_period)

# No embedded boundary — PEC domain walls serve as waveguide conductor.
sim = picmi.Simulation(
    solver=solver, max_steps=max_steps, verbose=1,
)
sim.add_diagnostic(field_diag)
sim.add_diagnostic(field_slice_diag)
sim.add_diagnostic(inject_plane_diag)
sim.add_diagnostic(recv_plane_diag)
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