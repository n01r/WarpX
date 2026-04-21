#!/usr/bin/env python3
"""
analyze_waveguide_s21.py
========================
General S21 analysis script for all four WarpX WR-15 waveguide cases:

  - inputs_waveguide_te10_pec_no_eb.py              (PEC domain boundaries, Z propagation)
  - inputs_waveguide_te10_pec_ckc.py                (PEC domain boundaries, CKC solver, Z propagation)
  - inputs_simple_waveguide_te10.py                 (EB implicit function,  Z propagation)
  - inputs_microwave_vacuum_te10_short_pulse.py     (STL horn-to-horn,      Z propagation)
  - inputs_microwave_vacuum_te10_scooter_section.py (STL scooter,           X propagation)

The propagation axis and aperture plane orientation are inferred automatically
from the openPMD mesh.axis_labels metadata.  The thin axis of the slab
diagnostic (smallest physical span = n_cells * cell_size) is the propagation
axis and is averaged over.  The two remaining axes span the aperture.

Forward/backward wave decomposition
-----------------------------------
To remove reflections from the S21 measurement, the script also loads the
transverse B component paired with the chosen E component and projects both
onto the same TE10 mode shape.  For the `exp(-i*beta*z + i*omega*t)`
convention used here, a pure forward-propagating TE10 mode satisfies

    B_transverse = -E_transverse / v_p(f)

and a pure backward-propagating mode satisfies

    B_transverse = +E_transverse / v_p(f),   v_p(f) = omega / beta(f).

At each plane we form, in the frequency domain,

    A_+(f) = 0.5 * ( A_E(f) - v_p(f) * A_B(f) )     # forward-travelling
    A_-(f) = 0.5 * ( A_E(f) + v_p(f) * A_B(f) )     # backward-travelling

so that S21(f) = A_+_recv(f) / A_+_inj(f) is contaminated only by what leaks
through before the backward-moving reflection arrives at the injection plane.
As a by-product we also get S11(f) = A_-_inj(f) / A_+_inj(f) (reflection seen
at the input) and an analogous forward/backward ratio at the receiver.

Pairing of E and B by propagation axis:
    Z propagation (broad wall along X):  Ey  <-> Bx
    X propagation (broad wall along Y):  Ez  <-> By

Workflow
--------
1. Load injection_plane and receiving_plane openPMD time series.  Both the
   file-based encoding (one file per iteration, e.g. openpmd_00000010.bp5)
   and the variable-based encoding (a single openpmd.bp5 holding all
   iterations) are detected automatically.
2. Detect propagation axis and aperture axes from mesh.axis_labels.
3. Detect broad-wall axis from aperture coordinate spans.
4. Project E and the paired B onto the TE10 mode shape -> modal coefficients
   a_E(t), a_B(t).
5. Apply window + zero-padding FFT to both time series.
6. Split each spectrum into forward (A_+) and backward (A_-) components and
   compute the directional S21(f) = A_+_recv(f) / A_+_inj(f).
7. Compare against analytic solution S21(f) = exp(-j*beta(f)*L).
8. Plot and print residuals, plus reflection diagnostics (A_+ vs A_-, S11).

Convention: forward-travelling wave ~ exp(-i*beta*z + i*omega*t), so
    S21_analytic = exp(-i*beta*L),  L = |recv_pos - inject_pos|
    confirmed by simulation output.

Usage
-----
    python analyze_waveguide_s21.py [--diag-dir ./diags] [--no-plot]
"""

import argparse
import glob
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import openpmd_api as io
from scipy.integrate import simpson

# ── Physical constants ────────────────────────────────────────────────────────
c   = 2.99792458e8      # m/s
mu0 = 4e-7 * np.pi     # H/m

# ── Simulation parameters (must match input scripts) ─────────────────────────
a_wg = 3.7592e-3    # m — WR-15 broad wall
b_wg = 1.8796e-3    # m — WR-15 narrow wall

# Port positions for Z-propagation cases (PEC, EB waveguide, horn-to-horn)
inject_z = -0.12272     # m
recv_z   =  0.12591     # m

# Port positions for X-propagation case (scooter)
inject_x = -0.11665     # m
recv_x   =  0.13468     # m

# Frequency band of interest
f_lo = 60e9             # Hz
f_hi = 74e9             # Hz

# ── FFT / windowing parameters ────────────────────────────────────────────────
# Zero-padding factor: interpolates the frequency axis (does not increase
# spectral resolution, which is set by 1/T_sim).
ZERO_PAD_FACTOR = 64

# Window options:
#   'none'   — no window; safe when the pulse decays well before end of series
#   'hann'   — full symmetric Hann; avoid when the pulse is not centred in time
#   'tukey'  — one-sided cosine taper at t=0 only, flat elsewhere;
#              best for pulse sims where signal is gone long before t_end
WINDOW_TYPE   = 'none'
TUKEY_TAPER_S = 0.5e-9   # seconds to taper at the start (Tukey window only)

# ── Derived waveguide quantities ──────────────────────────────────────────────
fc_te10 = c / (2 * a_wg)


def beta(f):
    """TE10 phase constant beta(f) [rad/m]. Returns NaN below cutoff."""
    arg = (2 * np.pi * f / c)**2 - (np.pi / a_wg)**2
    return np.where(arg > 0, np.sqrt(arg), np.nan)


def S21_analytic(f, L):
    """
    Analytic S21 = exp(-j*beta*L) for a lossless waveguide.
    Convention: forward-travelling wave is exp(-i*beta*z + i*omega*t),
    confirmed by simulation output.
    """
    return np.exp(-1j * beta(f) * L)


# ── Window function ───────────────────────────────────────────────────────────

# Super-Gaussian window parameters (only used when WINDOW_TYPE == 'supergauss').
# These are intended to isolate the direct pulse arrival at each plane and
# suppress later reflections.  They should be chosen *after* inspecting the
# first run's a_E(t) traces, which is why they are intentionally left disabled
# (zero width) here.
#
# A super-Gaussian of the form exp(-((t-t0)/tau)**(2*n)) has a near-flat top
# over ~0.9*tau for n >= 4 and smooth edges, so it barely attenuates the main
# pulse while killing the late reflection train.  A plain Gaussian (n=1) would
# visibly bite into a wide pulse, and a hard rectangular gate causes spectral
# ringing; order 4-8 avoids both.
#
# If the injection and receiving planes need different window centres (because
# the direct pulse arrives at very different times), set the *_RECV versions.
# They default to the injection values when left as None.
SUPERGAUSS_ORDER      = 6
SUPERGAUSS_T_CENTER   = 0.0    # [s] centre of window at injection plane
SUPERGAUSS_TAU        = 0.0    # [s] half-width; 0 disables windowing entirely
SUPERGAUSS_T_CENTER_RECV = None
SUPERGAUSS_TAU_RECV      = None


def make_window(times, plane='inj'):
    """
    Build a time-domain window based on WINDOW_TYPE.

      'none'       : rectangular (no windowing)
      'hann'       : symmetric Hann — only appropriate when the pulse is
                     centred in the time series
      'tukey'      : one-sided cosine taper over the first TUKEY_TAPER_S seconds,
                     flat everywhere else.  Suppresses leakage from the hard
                     turn-on at t=0 without distorting pulse amplitudes.
      'supergauss' : exp(-((t-t0)/tau)**(2*n)).  Flat-top window centred on
                     the direct pulse arrival; used to gate out reflections.

    `plane` selects which set of super-Gaussian parameters to use:
        'inj'  -> SUPERGAUSS_T_CENTER, SUPERGAUSS_TAU
        'recv' -> SUPERGAUSS_T_CENTER_RECV, SUPERGAUSS_TAU_RECV (fallback to inj)
    """
    N = len(times)
    if WINDOW_TYPE == 'none':
        return np.ones(N)
    elif WINDOW_TYPE == 'hann':
        return np.hanning(N)
    elif WINDOW_TYPE == 'tukey':
        dt      = float(np.mean(np.diff(times)))
        n_taper = max(1, int(round(TUKEY_TAPER_S / dt)))
        win     = np.ones(N)
        ramp    = 0.5 * (1 - np.cos(np.pi * np.arange(n_taper) / n_taper))
        win[:n_taper] = ramp
        return win
    elif WINDOW_TYPE == 'supergauss':
        if plane == 'recv':
            t0  = (SUPERGAUSS_T_CENTER_RECV
                   if SUPERGAUSS_T_CENTER_RECV is not None
                   else SUPERGAUSS_T_CENTER)
            tau = (SUPERGAUSS_TAU_RECV
                   if SUPERGAUSS_TAU_RECV is not None
                   else SUPERGAUSS_TAU)
        else:
            t0  = SUPERGAUSS_T_CENTER
            tau = SUPERGAUSS_TAU
        if tau <= 0:
            # Unset parameters -> no windowing (equivalent to 'none').
            return np.ones(N)
        return np.exp(-(((times - t0) / tau) ** (2 * SUPERGAUSS_ORDER)))
    else:
        raise ValueError(f"Unknown WINDOW_TYPE '{WINDOW_TYPE}'. "
                         "Use 'none', 'hann', 'tukey', or 'supergauss'.")


# ── openPMD loader ────────────────────────────────────────────────────────────

# Pairing of E component with transverse B component per propagation axis:
#   Z propagation (Ey) -> Bx    (forward TE10:  Bx = -Ey / v_p)
#   X propagation (Ez) -> By    (forward TE10:  By = -Ez / v_p)
# Y propagation is not used in any of the current cases but included for symmetry.
_E_COMP_BY_PROP_AXIS = {0: 'z', 1: 'z', 2: 'y'}
_B_COMP_BY_PROP_AXIS = {0: 'y', 1: 'x', 2: 'x'}

# openPMD backends WarpX may have produced, in order of preference.
_OPENPMD_EXTS = ("bp5", "bp4", "bp", "h5")


def _detect_openpmd_series_path(diag_dir, diag_name):
    """
    Locate the openPMD series for ``diag_name`` under ``diag_dir`` and return
    the path string that should be passed to ``openpmd_api.Series`` together
    with a short human-readable description of the detected encoding, the
    ``openpmd_api.Access`` mode that should be used to open it, and a JSON
    options string for the ``openpmd_api.Series`` constructor.

    Two encodings are supported:

      - Variable-based (warpx_openpmd_encoding='v' or 'variableBased'): a
        single ``openpmd.<ext>`` file (or directory, for ADIOS BP*) that
        contains every iteration.  Opened with ``Access.read_linear``
        because BP5 variable-based iterations are only streamed in order.
        Empty options string (default).
      - File-based (default in older WarpX runs): one
        ``openpmd_<iter>.<ext>`` per iteration, addressed through the
        openPMD ``%T`` placeholder.  Opened with ``Access.read_only`` so
        openpmd-api can index iterations on demand, together with the
        ``defer_iteration_parsing`` JSON option -- otherwise
        ``Series(..)`` construction alone eagerly parses every iteration
        header, which takes tens of minutes for 10k+ files.  Iterations
        are still opened transparently by ``series.read_iterations()``
        below as each one is consumed.

    Returns
    -------
    path     : str   openPMD-api path, e.g. ``diags/recv/openpmd.bp5`` or
                     ``diags/recv/openpmd_%T.bp5``
    encoding : str   short label describing the detected encoding
    access   : io.Access   access mode to pass to ``openpmd_api.Series``
    options  : str   JSON options string to pass to ``openpmd_api.Series``
    """
    base = os.path.join(diag_dir, diag_name)
    if not os.path.isdir(base):
        raise FileNotFoundError(
            f"Diagnostic directory not found: {base}"
        )

    # Variable-based: single openpmd.<ext> entry (file or ADIOS directory).
    for ext in _OPENPMD_EXTS:
        candidate = os.path.join(base, f"openpmd.{ext}")
        if os.path.exists(candidate):
            return (candidate,
                    f"variable-based (.{ext})",
                    io.Access.read_linear,
                    "")

    # File-based: one openpmd_<digits>.<ext> entry per iteration, addressed
    # via the openPMD %T placeholder.  Pick whichever extension matches and
    # enforce a digit-only iteration suffix so we don't accidentally pick up
    # e.g. openpmd_backup files.
    iter_pat = re.compile(r"^openpmd_(\d+)$")
    for ext in _OPENPMD_EXTS:
        matches = [
            m for m in glob.glob(os.path.join(base, f"openpmd_*.{ext}"))
            if iter_pat.match(os.path.splitext(os.path.basename(m))[0])
        ]
        if matches:
            # defer_iteration_parsing: do NOT parse every iteration header
            # up-front when the Series is constructed.  Required for any
            # reasonable open time when there are tens of thousands of
            # per-iteration files.
            file_opts = '{"defer_iteration_parsing": true}'
            return (os.path.join(base, f"openpmd_%T.{ext}"),
                    f"file-based (.{ext}, {len(matches)} iterations found)",
                    io.Access.read_only,
                    file_opts)

    raise FileNotFoundError(
        f"No openPMD series found under {base}. Expected one of:\n"
        f"  {base}/openpmd.{{bp5,bp4,bp,h5}}        (variable-based)\n"
        f"  {base}/openpmd_<iter>.{{bp5,bp4,bp,h5}} (file-based)"
    )


def load_plane_timeseries(diag_dir, diag_name):
    """
    Load a thin-slab FieldDiagnostic time series.  Both the variable-based
    encoding (``warpx_openpmd_encoding='v'``, single ``openpmd.<ext>`` entry)
    and the older file-based encoding (one ``openpmd_<iter>.<ext>`` per
    iteration) are accepted; the loader picks whichever is present under
    ``{diag_dir}/{diag_name}/``.

    Uses mesh.axis_labels to map array dimensions to physical axes (x, y, z).
    The thin axis (smallest physical span = n_cells * cell_spacing) is identified
    as the propagation axis and averaged over.  The two remaining axes are the
    aperture axes.

    The TE10 E-field component is selected based on the propagation axis:
      - Z propagation (PEC, EB, horn-to-horn): Ey  (broad wall along X)
      - X propagation (scooter):               Ez  (broad wall along Y)

    The transverse B component paired with E (used for the forward/backward
    wave decomposition) is also loaded:
      - Z propagation (Ey):  Bx
      - X propagation (Ez):  By

    Returns
    -------
    times    : (N,)         simulation time at each step  [s]
    E_ts     : (N, Na, Nb)  TE10 E-field on the aperture grid  [V/m]
    B_ts     : (N, Na, Nb)  paired transverse B-field on the aperture grid [T]
    coord_a  : (Na,)        coordinate along first aperture array dimension  [m]
    coord_b  : (Nb,)        coordinate along second aperture array dimension  [m]
    d_a, d_b : float        cell sizes along aperture dimensions  [m]
    prop_axis: int          physical propagation axis index (0=X, 1=Y, 2=Z)
    E_comp   : str          E-field component used ('y' or 'z')
    B_comp   : str          paired B-field component used ('x' or 'y')
    """
    path, encoding, access, options = _detect_openpmd_series_path(
        diag_dir, diag_name)
    print(f"  openPMD series: {path}")
    print(f"  openPMD encoding: {encoding}")
    print(f"  openPMD access mode: {access.name}")
    if options:
        print(f"  openPMD options: {options}")
    # read_linear is needed for variable-based BP5 (iterations only available
    # in order via streaming); read_only is used for file-based series, with
    # defer_iteration_parsing so construction does not eagerly parse tens of
    # thousands of per-iteration file headers.
    series = io.Series(path, access, options)

    times   = []
    E_ts    = []
    B_ts    = []
    coord_a = coord_b = None
    d_a = d_b = None
    prop_axis = None
    thin_dim  = None
    aper_dims = None
    E_comp    = None
    B_comp    = None

    for it in series.read_iterations():
        mesh_E = it.meshes["E"]
        mesh_B = it.meshes["B"]
        gs   = mesh_E.grid_spacing        # cell sizes indexed by array dimension
        orig = mesh_E.grid_global_offset  # origin indexed by array dimension

        # ── One-time axis detection ───────────────────────────────────────────
        if prop_axis is None:
            # axis_labels maps each array dimension to its physical axis name.
            labels = list(mesh_E.axis_labels)   # e.g. ['z', 'y', 'x']

            # Load one component to get array shape
            raw_probe = mesh_E["y"].load_chunk()
            it.series_flush()
            shape = raw_probe.shape           # (N0, N1, N2)

            # Physical span of each array dimension
            spans = [shape[i] * abs(float(gs[i])) for i in range(3)]

            # The thin dimension is the slab normal (propagation direction)
            thin_dim    = int(np.argmin(spans))
            thin_label  = labels[thin_dim]                 # 'x', 'y', or 'z'
            prop_axis   = 'xyz'.index(thin_label)          # physical axis index

            aper_dims   = [i for i in range(3) if i != thin_dim]
            aper_labels = [labels[i] for i in aper_dims]

            # TE10 polarization depends on propagation axis (see module table).
            E_comp = _E_COMP_BY_PROP_AXIS[prop_axis]
            B_comp = _B_COMP_BY_PROP_AXIS[prop_axis]

            print(f"  axis_labels (array dim -> physical): {labels}")
            print(f"  Physical spans: " +
                  ", ".join(f"{labels[i]}={spans[i]*1e3:.2f} mm" for i in range(3)))
            print(f"  Thin dim: array axis {thin_dim} = physical "
                  f"'{thin_label}' -> propagation axis {'XYZ'[prop_axis]}")
            print(f"  Aperture array dims: {aper_dims} "
                  f"= physical {''.join(l.upper() for l in aper_labels)}")
            print(f"  Using E component: E{E_comp}"
                  f"  paired B component: B{B_comp}")

        # ── Load field components ─────────────────────────────────────────────
        comp_E = mesh_E[E_comp]
        comp_B = mesh_B[B_comp]
        raw_E  = comp_E.load_chunk()
        raw_B  = comp_B.load_chunk()
        it.series_flush()
        raw_E  = raw_E * comp_E.unit_SI     # V/m
        raw_B  = raw_B * comp_B.unit_SI     # T

        # Average over the thin slab dimension -> (Na, Nb)
        E_2d = raw_E.mean(axis=thin_dim)
        B_2d = raw_B.mean(axis=thin_dim)

        # ── Build coordinate arrays on first data iteration ───────────────────
        if coord_a is None:
            offs = comp_E.position        # sub-cell offset fractions per array dim
            na, nb = E_2d.shape
            dim0, dim1 = aper_dims
            coord_a = (orig[dim0] + offs[dim0] * gs[dim0]
                       + np.arange(na) * gs[dim0])
            coord_b = (orig[dim1] + offs[dim1] * gs[dim1]
                       + np.arange(nb) * gs[dim1])
            d_a, d_b = float(gs[dim0]), float(gs[dim1])

        times.append(float(it.time))
        E_ts.append(E_2d)
        B_ts.append(B_2d)

    del series
    return (np.array(times),
            np.stack(E_ts, axis=0),
            np.stack(B_ts, axis=0),
            coord_a, coord_b, d_a, d_b,
            prop_axis, E_comp, B_comp)


# ── TE10 mode shape and overlap integral ──────────────────────────────────────

def te10_mode_2d(coord_a, coord_b):
    """
    TE10 mode shape phi(u) = cos(pi*u/a) on the (Na, Nb) aperture grid.

    The broad wall (dimension a_wg) is identified as the aperture axis with
    the larger coordinate span.  phi varies as a cosine across the broad wall
    and is uniform across the narrow wall.  Zero outside the aperture rectangle.

    Returns
    -------
    phi : (Na, Nb) array, peak = 1
    """
    span_a = coord_a[-1] - coord_a[0]
    span_b = coord_b[-1] - coord_b[0]

    AA, BB = np.meshgrid(coord_a, coord_b, indexing='ij')

    broad_grid  = AA if span_a >= span_b else BB
    narrow_grid = BB if span_a >= span_b else AA

    broad_rel  = broad_grid  + a_wg / 2    # shift so aperture runs [0, a]
    narrow_rel = narrow_grid + b_wg / 2

    in_ap = ((broad_rel  >= 0) & (broad_rel  <= a_wg) &
             (narrow_rel >= 0) & (narrow_rel <= b_wg))

    return np.where(in_ap, np.cos(np.pi * broad_rel / a_wg), 0.0)


def project_mode(E_stack, coord_a, coord_b, d_a, d_b):
    """
    Normalized TE10 overlap integral for every time step:

        a(t) = (2 / (a*b)) * integral_a integral_b  E(a,b,t) * phi(a,b)  da db

    Normalization N = a*b/2 follows from:
        integral cos^2(pi*u/a) du from -a/2 to a/2  =  a/2
        integral dv from -b/2 to b/2                =  b
    so a pure TE10 mode at amplitude E0 returns E0.

    Returns
    -------
    modal : (N,) array of modal coefficients in V/m
    """
    phi  = te10_mode_2d(coord_a, coord_b)   # (Na, Nb)
    norm = a_wg * b_wg / 2.0               # N = ab/2
    return np.array([
        simpson(simpson(E_stack[i] * phi, dx=d_b, axis=1), dx=d_a) / norm
        for i in range(E_stack.shape[0])
    ])


# ── FFT ──────────────────────────────────────────────────────────────────────────────

def modal_to_spectrum(times, modal, plane='inj'):
    """
    Apply window + zero-padding and return one-sided FFT.
    Window type is controlled by WINDOW_TYPE (see make_window).

    `plane` is forwarded to `make_window` so the per-plane super-Gaussian
    window parameters (if any) are picked up correctly.

    Returns
    -------
    freqs : (M,) frequency array [Hz]
    spec  : (M,) complex spectrum
    """
    N        = len(times)
    dt       = float(np.mean(np.diff(times)))
    win      = make_window(times, plane=plane)
    win_norm = win.sum() / N                    # amplitude correction
    nfft     = N * ZERO_PAD_FACTOR
    spec     = np.fft.rfft(modal * win, n=nfft) / (N * win_norm)
    freqs    = np.fft.rfftfreq(nfft, d=dt)
    return freqs, spec


# ── Forward/backward decomposition ──────────────────────────────────────────

def phase_velocity(freqs):
    """
    TE10 phase velocity v_p(f) = omega / beta(f) [m/s]. NaN at/below cutoff.
    """
    b = beta(freqs)
    omega = 2 * np.pi * freqs
    return np.where(np.isfinite(b) & (b > 0), omega / b, np.nan)


def split_forward_backward(freqs, A_E, A_B):
    """
    Decompose the TE10 spectra of a single plane into forward (A_+) and
    backward (A_-) travelling components using the paired transverse
    B-field spectrum.

    With the `exp(-i*beta*z + i*omega*t)` convention,
      forward:  B_t = -E_t / v_p(f)
      backward: B_t = +E_t / v_p(f)
    so that
      A_+(f) = 0.5 * ( A_E - v_p * A_B )
      A_-(f) = 0.5 * ( A_E + v_p * A_B )

    Returns NaN below TE10 cutoff where v_p is not defined.

    Parameters
    ----------
    freqs : (M,)  frequency array [Hz]
    A_E   : (M,)  complex spectrum of the TE10 E modal coefficient
    A_B   : (M,)  complex spectrum of the paired TE10 B modal coefficient

    Returns
    -------
    A_plus  : (M,) complex forward  spectrum (same units as A_E)
    A_minus : (M,) complex backward spectrum (same units as A_E)
    """
    vp = phase_velocity(freqs)
    A_plus  = 0.5 * (A_E - vp * A_B)
    A_minus = 0.5 * (A_E + vp * A_B)
    return A_plus, A_minus


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Validate WR-15 waveguide S21 against analytic solution.")
    parser.add_argument("--diag-dir", default="./diags",
                        help="Path to WarpX diags directory")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--analytic-ref", action="store_true",
                        help="Use analytical Gaussian pulse as incident reference "
                             "instead of injection_plane diagnostic.  Requires "
                             "--pulse-t-peak, --pulse-fwhm, and --emit-pos.")
    parser.add_argument("--pulse-t-peak", type=float, default=0.13e-9,
                        help="Pulse peak time [s] (default: 0.13e-9)")
    parser.add_argument("--pulse-fwhm",   type=float, default=0.05e-9,
                        help="Pulse FWHM [s] (default: 0.05e-9)")
    parser.add_argument("--emit-pos",     type=float, default=None,
                        help="Emitter position along the propagation axis [m]. "
                             "Used with --analytic-ref to compute arrival time "
                             "at the receiving plane.")
    parser.add_argument("--E0",           type=float, default=1e6,
                        help="Peak injected E-field amplitude [V/m] (default: 1e6)")
    args = parser.parse_args()

    # ── Load receiving plane ───────────────────────────────────────────────────────
    print("Loading receiving plane ...", flush=True)
    (t_rec, E_rec, B_rec,
     coord_a, coord_b, d_a, d_b,
     prop_axis, E_comp, B_comp) = load_plane_timeseries(
         args.diag_dir, "receiving_plane")

    aE_rec = project_mode(E_rec, coord_a, coord_b, d_a, d_b)
    aB_rec = project_mode(B_rec, coord_a, coord_b, d_a, d_b)
    print(f"  {len(t_rec)} steps,  dt = {np.mean(np.diff(t_rec)):.4e} s,  "
          f"t ∈ [{t_rec[0]*1e9:.4f}, {t_rec[-1]*1e9:.4f}] ns")
    print(f"  peak |a_E_rec| = {np.max(np.abs(aE_rec)):.4e} V/m,  "
          f"peak |a_B_rec| = {np.max(np.abs(aB_rec)):.4e} T")

    # ── Incident reference: injection plane or analytical ─────────────────────
    if args.analytic_ref:
        # Build analytical Gaussian pulse envelope as incident reference.
        # A pure forward TE10 pulse with peak E0 has
        #   a_E_inc(t) = E0 * gaussian(t - t_arrival)
        #   a_B_inc(t) = -a_E_inc(t) / v_p(f0)   (at the carrier, used for v_p)
        # where t_arrival = t_peak + |recv_pos - emit_pos| / vg
        if args.emit_pos is None:
            parser.error("--emit-pos is required when using --analytic-ref")

        recv_pos  = recv_x if prop_axis == 0 else recv_z
        L_emit    = abs(recv_pos - args.emit_pos)
        vg_ref    = c * np.sqrt(max(1 - (fc_te10 / 67e9)**2, 0))
        vp_ref    = c / np.sqrt(max(1 - (fc_te10 / 67e9)**2, 1e-30))
        t_arrival = args.pulse_t_peak + L_emit / vg_ref
        pulse_tau = args.pulse_fwhm / (2 * np.sqrt(np.log(2)))

        print(f"\nUsing analytical Gaussian reference:")
        print(f"  E0 = {args.E0:.3e} V/m,  t_peak = {args.pulse_t_peak*1e9:.4f} ns")
        print(f"  FWHM = {args.pulse_fwhm*1e12:.1f} ps,  τ = {pulse_tau*1e12:.2f} ps")
        print(f"  emit_pos = {args.emit_pos*1e2:.5f} cm,  recv_pos = {recv_pos*1e2:.5f} cm")
        print(f"  vg = {vg_ref/c:.4f} c,  v_p = {vp_ref/c:.4f} c,  "
              f"t_arrival = {t_arrival*1e9:.4f} ns")

        pulse_env = args.E0 * np.exp(-(((t_rec - t_arrival) / pulse_tau) ** 2))
        aE_inj = pulse_env
        aB_inj = -pulse_env / vp_ref     # pure forward wave at the carrier
        t_inj  = t_rec

    else:
        print("Loading injection plane ...", flush=True)
        (t_inj, E_inj, B_inj,
         coord_a, coord_b, d_a, d_b,
         prop_axis, E_comp, B_comp) = load_plane_timeseries(
             args.diag_dir, "injection_plane")
        aE_inj = project_mode(E_inj, coord_a, coord_b, d_a, d_b)
        aB_inj = project_mode(B_inj, coord_a, coord_b, d_a, d_b)
        print(f"  {len(t_inj)} steps,  dt = {np.mean(np.diff(t_inj)):.4e} s,  "
              f"t ∈ [{t_inj[0]*1e9:.4f}, {t_inj[-1]*1e9:.4f}] ns")
        print(f"  peak |a_E_inj| = {np.max(np.abs(aE_inj)):.4e} V/m,  "
              f"peak |a_B_inj| = {np.max(np.abs(aB_inj)):.4e} T")

    # Sanity check: zero signal means the pulse has not yet reached the receiver
    if np.max(np.abs(aE_rec)) < 1e-20:
        L_check   = abs(recv_x - inject_x) if prop_axis == 0 else abs(recv_z - inject_z)
        vg_approx = c * np.sqrt(max(1 - (fc_te10 / 67e9)**2, 0))
        t_arr     = L_check / vg_approx if vg_approx > 0 else float('inf')
        print(f"\n  WARNING: receiving plane signal is zero.")
        print(f"  Estimated pulse arrival: {t_arr*1e9:.3f} ns, "
              f"but simulation ended at {t_rec[-1]*1e9:.4f} ns.")
        print(f"  Re-run with excitation_mode='pulse' or "
              f"t_sim_override >= {(t_arr + 0.5)*1e9:.1f}e-9")
        return

    # Trim to common length
    n      = min(len(t_inj), len(t_rec))
    t_inj  = t_inj[:n];   aE_inj = aE_inj[:n];  aB_inj = aB_inj[:n]
    t_rec  = t_rec[:n];   aE_rec = aE_rec[:n];  aB_rec = aB_rec[:n]

    # ── Spectra ────────────────────────────────────────────────────────────
    freqs, A_E_inj = modal_to_spectrum(t_inj, aE_inj, plane='inj')
    _,     A_B_inj = modal_to_spectrum(t_inj, aB_inj, plane='inj')
    _,     A_E_rec = modal_to_spectrum(t_rec, aE_rec, plane='recv')
    _,     A_B_rec = modal_to_spectrum(t_rec, aB_rec, plane='recv')

    # Forward/backward split at each plane.
    A_plus_inj, A_minus_inj = split_forward_backward(freqs, A_E_inj, A_B_inj)
    A_plus_rec, A_minus_rec = split_forward_backward(freqs, A_E_rec, A_B_rec)

    mask = (freqs >= f_lo) & (freqs <= f_hi) & (freqs > fc_te10)

    # ── S-parameters from directional components ────────────────────────────
    # S21 is now the forward-only transmission, immune (to first order) to the
    # backward reflection that otherwise contaminates A_E_inj.
    eps_plus = np.nanmax(np.abs(A_plus_inj)) * 1e-6
    S21_sim = np.where(np.abs(A_plus_inj) > eps_plus,
                       A_plus_rec / A_plus_inj, np.nan + 0j)

    # Reflection diagnostics: S11 at injection plane is the ratio of the
    # measured backward wave to the measured forward wave there.  S22-like at
    # the receiving plane tells us how much the far-end boundary is returning.
    S11_sim = np.where(np.abs(A_plus_inj) > eps_plus,
                       A_minus_inj / A_plus_inj, np.nan + 0j)
    eps_plus_rec = np.nanmax(np.abs(A_plus_rec)) * 1e-6
    S22_sim = np.where(np.abs(A_plus_rec) > eps_plus_rec,
                       A_minus_rec / A_plus_rec, np.nan + 0j)

    # Also compute the naive E-only S21 for comparison (what the old script did).
    eps_E        = np.nanmax(np.abs(A_E_inj)) * 1e-6
    S21_sim_Eonly = np.where(np.abs(A_E_inj) > eps_E,
                             A_E_rec / A_E_inj, np.nan + 0j)

    L = abs(recv_x - inject_x) if prop_axis == 0 else abs(recv_z - inject_z)
    S21_ana = S21_analytic(freqs, L)

    # ── Residuals ────────────────────────────────────────────────────────────
    f_b     = freqs[mask]
    mag_sim = np.abs(S21_sim[mask])
    mag_ana = np.abs(S21_ana[mask])
    phs_sim = np.unwrap(np.angle(S21_sim[mask]))
    phs_ana = np.unwrap(np.angle(S21_ana[mask]))
    phs_err = np.degrees(phs_sim - phs_ana)
    mag_err = mag_sim - mag_ana

    mag_sim_Eonly = np.abs(S21_sim_Eonly[mask])
    mag_err_Eonly = mag_sim_Eonly - mag_ana

    print(f"\nPropagation axis: {'XYZ'[prop_axis]},  L = {L*1e2:.5f} cm")
    print(f"TE10 cutoff = {fc_te10*1e-9:.3f} GHz")
    print(f"Window: {WINDOW_TYPE}")
    print(f"\nDirectional S21 = A_+_recv / A_+_inj, residuals in "
          f"[{f_lo*1e-9:.0f}, {f_hi*1e-9:.0f}] GHz ({mask.sum()} freq points):")
    print(f"  |S21| error  max = {np.nanmax(np.abs(mag_err)):.2e}  "
          f"RMS = {np.sqrt(np.nanmean(mag_err**2)):.2e}")
    print(f"  Phase error  max = {np.nanmax(np.abs(phs_err)):.4f} deg  "
          f"RMS = {np.sqrt(np.nanmean(phs_err**2)):.4f} deg")
    print(f"\nNaive E-only S21 for comparison:")
    print(f"  |S21| error  max = {np.nanmax(np.abs(mag_err_Eonly)):.2e}  "
          f"RMS = {np.sqrt(np.nanmean(mag_err_Eonly**2)):.2e}")

    with np.errstate(invalid='ignore'):
        s11_band = np.abs(S11_sim[mask])
        s22_band = np.abs(S22_sim[mask])
    print(f"\nReflection diagnostics (from directional split):")
    print(f"  |S11|_inj  mean = {np.nanmean(s11_band):.3e}  "
          f"max = {np.nanmax(s11_band):.3e}")
    print(f"  |S22|_recv mean = {np.nanmean(s22_band):.3e}  "
          f"max = {np.nanmax(s22_band):.3e}")

    if args.no_plot:
        return

    # ── Plots ──────────────────────────────────────────────────────────────────
    prop_name = 'XYZ'[prop_axis]
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle(
        f"WR-15 waveguide — directional S21 validation\n"
        f"Propagation along {prop_name},  E{E_comp} / B{B_comp},  "
        f"L = {L*1e2:.3f} cm,  window = {WINDOW_TYPE}",
        fontsize=12)

    # Top-left: time-domain modal coefficients of E (direct check that pulses look sane).
    ax = axes[0, 0]
    ax.plot(t_inj*1e9, aE_inj, label="injection plane", lw=0.8)
    ax.plot(t_rec*1e9, aE_rec, label="receiving plane", lw=0.8, ls='--')
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("a_E(t)  (V/m)")
    ax.set_title("TE10 E modal coefficient a_E(t)")
    ax.legend(); ax.grid(True)

    # Top-middle: forward vs backward amplitude at each plane.
    ax = axes[0, 1]
    ax.semilogy(f_b*1e-9, np.abs(A_plus_inj[mask]),
                label="|A_+| injection",  lw=1.3)
    ax.semilogy(f_b*1e-9, np.abs(A_minus_inj[mask]),
                label="|A_-| injection",  lw=1.3, ls='--')
    ax.semilogy(f_b*1e-9, np.abs(A_plus_rec[mask]),
                label="|A_+| receiving",  lw=1.3)
    ax.semilogy(f_b*1e-9, np.abs(A_minus_rec[mask]),
                label="|A_-| receiving",  lw=1.3, ls='--')
    ax.axvline(fc_te10*1e-9, color='k', ls=':', lw=1,
               label=f"f_c = {fc_te10*1e-9:.2f} GHz")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Spectral amplitude (V/m·s)")
    ax.set_title("Forward (A_+) vs backward (A_-) spectra")
    ax.legend(fontsize=8); ax.grid(True, which='both')

    # Top-right: |S11| and |S22| (reflection levels).
    ax = axes[0, 2]
    ax.plot(f_b*1e-9, np.abs(S11_sim[mask]),
            label="|S11|  (injection-plane reflection)", lw=1.4)
    ax.plot(f_b*1e-9, np.abs(S22_sim[mask]),
            label="|S22|-like (receiver backward / forward)",
            lw=1.4, ls='--')
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Reflection coefficient magnitude")
    ax.set_title("Reflection diagnostics")
    ax.set_ylim(bottom=0.0)
    ax.legend(fontsize=9); ax.grid(True)

    # Bottom-left: |S21| directional vs E-only vs analytic.
    ax = axes[1, 0]
    ax.plot(f_b*1e-9, mag_sim,        label="|S21| directional (A_+/A_+)", lw=1.5)
    ax.plot(f_b*1e-9, mag_sim_Eonly,  label="|S21| naive (A_E/A_E)",
            lw=1.0, ls=':')
    ax.plot(f_b*1e-9, mag_ana,        label="|S21| analytic = 1",
            lw=1.5, ls='--', color='k')
    ax.set_ylim(0, 1.2)
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("|S21|")
    ax.set_title("|S21| magnitude")
    ax.legend(fontsize=9); ax.grid(True)

    # Bottom-middle: directional S21 phase vs analytic.
    ax = axes[1, 1]
    ax.plot(f_b*1e-9, np.degrees(phs_sim), label="Phase directional", lw=1.5)
    ax.plot(f_b*1e-9, np.degrees(phs_ana), label="Phase analytic (−βL)",
            lw=1.5, ls='--', color='k')
    ax2 = ax.twinx()
    ax2.plot(f_b*1e-9, phs_err, color='red', alpha=0.6, lw=0.8,
             label="Error (deg)")
    ax2.set_ylabel("Phase error (deg)", color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Phase (deg)")
    ax.set_title("S21 phase vs analytic −βL")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    ax.grid(True)

    # Bottom-right: B modal coefficients in the time domain — useful to verify
    # that the B traces look sensible and match the expected -a_E/v_p scaling
    # during the direct pulse, and that reflections have the opposite sign.
    ax = axes[1, 2]
    ax.plot(t_inj*1e9, aB_inj, label="injection plane", lw=0.8)
    ax.plot(t_rec*1e9, aB_rec, label="receiving plane", lw=0.8, ls='--')
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("a_B(t)  (T)")
    ax.set_title(f"TE10 B modal coefficient a_B(t)  (B{B_comp})")
    ax.legend(); ax.grid(True)

    fig.tight_layout()
    outfile = "s21_validation_directional_splitting.png"
    fig.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved: {outfile}")
    plt.show()


if __name__ == "__main__":
    main()
