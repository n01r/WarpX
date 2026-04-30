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
import time as _time
import numpy as np
import matplotlib.pyplot as plt
import openpmd_api as io
from scipy.integrate import simpson
from scipy.signal    import hilbert, butter, sosfiltfilt

# Print a progress line every PROGRESS_EVERY iterations during series read.
# Variable-based BP5 with ~15 k+ steps is fundamentally slow per-step (a few ms
# of Python + ADIOS2 metadata work each), so without this it looks frozen.
PROGRESS_EVERY = 500

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
    """TE10 phase constant beta(f) [rad/m]. Returns NaN below cutoff.

    np.where evaluates both branches unconditionally, so feeding sqrt a
    negative argument raises a cosmetic RuntimeWarning even though the
    negative branch is discarded by the where.  Clamp with np.maximum so
    the sqrt always sees a non-negative operand, then select on arg > 0.
    """
    f = np.asarray(f, dtype=float)
    arg = (2 * np.pi * f / c)**2 - (np.pi / a_wg)**2
    return np.where(arg > 0, np.sqrt(np.maximum(arg, 0.0)), np.nan)


def safe_unwrap(phases):
    """np.unwrap that does not poison the whole array when a NaN is present.

    np.unwrap accumulates 2*pi corrections along the array; as soon as it
    hits a NaN all subsequent samples become NaN, wiping out the phase plot.
    Here we unwrap only the contiguous valid region(s) and leave NaN gaps
    alone.
    """
    phases = np.asarray(phases, dtype=float)
    out    = phases.copy()
    valid  = ~np.isnan(phases)
    if not np.any(valid):
        return out
    # Unwrap in contiguous valid runs so a NaN block does not force a jump.
    idx   = np.flatnonzero(valid)
    splits = np.flatnonzero(np.diff(idx) > 1) + 1
    for run in np.split(idx, splits):
        out[run] = np.unwrap(phases[run])
    return out


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


def _detect_openpmd_series_path(diag_dir, diag_name, access_override=None):
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
    # Access mode default is read_linear (streaming iterator).  For a *completed*
    # series this works but can hang indefinitely on a series that ended with an
    # incomplete final iteration (e.g. the WarpX run crashed mid-write) because
    # streaming mode waits for more data that will never arrive.  In that case
    # the user can pass access_override=io.Access.read_only, which returns
    # everything currently on disk and stops, accepting that the last iteration
    # may be missing or partial.
    for ext in _OPENPMD_EXTS:
        candidate = os.path.join(base, f"openpmd.{ext}")
        if os.path.exists(candidate):
            mode = access_override if access_override is not None \
                                   else io.Access.read_linear
            return (candidate,
                    f"variable-based (.{ext})",
                    mode,
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
        f"  {base}/openpmd_<iter>.{{bp5,bp4,bp,h5}} (file-based)\n"
        f"\n"
        f"If '{os.path.basename(base)}' is the injection_plane and your input\n"
        f"script doesn't write one, rerun this script with\n"
        f"    --analytic-ref --emit-pos <z>  --pulse-t-peak <s>  --pulse-fwhm <s>\n"
        f"to substitute an analytic Gaussian incident reference."
    )


def load_plane_timeseries(diag_dir, diag_name, access_override=None,
                          max_iterations=None):
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
        diag_dir, diag_name, access_override=access_override)
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

    iter_count = 0
    t_start_wall = _time.monotonic()
    if max_iterations is not None:
        print(f"  loader will stop after at most {max_iterations} iterations "
              f"(--max-iterations)")
    try:
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
            iter_count += 1

            # Progress log: first iteration (so the user immediately sees we
            # are alive), then every PROGRESS_EVERY iterations after that.
            if iter_count == 1 or iter_count % PROGRESS_EVERY == 0:
                elapsed = _time.monotonic() - t_start_wall
                rate = iter_count / max(elapsed, 1e-6)
                print(f"    read {iter_count} iterations  "
                      f"(t = {float(it.time)*1e9:.3f} ns, "
                      f"elapsed {elapsed:.1f} s, "
                      f"{rate:.0f} iter/s)", flush=True)

            # Early stop: cap the number of iterations to skip a problematic
            # trailing tail of the series.  Some BP5 variable-based series
            # written by long jobs hang in read_linear at the very end
            # because the streaming end-of-stream marker was never written;
            # cutting one or two trailing iterations is harmless for FFT/CW
            # analysis and avoids the hang.
            if max_iterations is not None and iter_count >= max_iterations:
                print(f"    reached --max-iterations = {max_iterations}, "
                      f"stopping series read early", flush=True)
                break
    except Exception as exc:
        # This commonly fires on a series whose final iteration was never
        # fully flushed (WarpX crashed mid-write), especially with
        # Access.read_linear.  Keep whatever iterations we already consumed
        # and warn.  The analysis pipeline downstream can work with a
        # truncated series; it just covers a shorter time interval.
        print(f"  WARNING: series iteration raised {type(exc).__name__}: {exc}")
        print(f"           Proceeding with the {iter_count} iterations read so far.")
        print(f"           If this series was produced by a WarpX job that")
        print(f"           crashed or was killed mid-run, try rerunning with")
        print(f"           --openpmd-access-mode read_only to skip the")
        print(f"           streaming-wait behaviour.")

    if iter_count == 0:
        raise RuntimeError(
            f"No iterations successfully read from {path}.  "
            f"Check that the series is readable and non-empty.")

    elapsed_total = _time.monotonic() - t_start_wall
    rate_total    = iter_count / max(elapsed_total, 1e-6)
    print(f"  done: {iter_count} iterations read in {elapsed_total:.1f} s "
          f"({rate_total:.0f} iter/s)")

    del series
    return (np.array(times),
            np.stack(E_ts, axis=0),
            np.stack(B_ts, axis=0),
            coord_a, coord_b, d_a, d_b,
            prop_axis, E_comp, B_comp)


# ── FieldProbe loader ──────────────────────────────────────────────────────────

# WarpX FieldProbe .dat column layout, one row per probe point per step:
#   step(0)  time(1)  x(2) y(3) z(4)
#   Ex(5)    Ey(6)    Ez(7)
#   Bx(8)    By(9)    Bz(10)   S(11)
_FP_USE_COLS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# Positional indices into the DataFrame after usecols selection.
_FP_I_TIME = 1
_FP_I_XYZ  = (2, 3, 4)
_FP_I_E    = {'x': 5, 'y': 6, 'z': 7}
_FP_I_B    = {'x': 8, 'y': 9, 'z': 10}


def _count_dat_header_lines(filepath):
    """Count leading comment / header lines (starting with '#' or '[')."""
    n = 0
    with open(filepath) as f:
        for line in f:
            s = line.lstrip()
            if s.startswith("#") or s.startswith("["):
                n += 1
            else:
                break
    return n


def load_fieldprobe_plane_timeseries(filepath, resolution,
                                     t_start=None, t_end=None):
    """
    Load a WarpX FieldProbe ``Plane`` diagnostic .dat file and return data in
    the SAME shape as :func:`load_plane_timeseries` so the rest of the
    pipeline (TE10 projection, FFT, forward/backward split, plotting) is
    source-agnostic.

    The ``Plane`` probe is a square NxN grid (N = resolution) of side length
    2*detector_radius set in the input script.  When the radius is larger
    than a_wg/2 or b_wg/2, probe points fall OUTSIDE the WR-15 cross-section
    -- inside the EB-conductor area in the EB-waveguide cases.  Those points
    are zeroed out below, matching the openPMD slab which is sized to the
    WR-15 cross-section plus one cell on each side.

    Parameters
    ----------
    filepath : str
        Path to the FieldProbe .dat file (e.g. ``diags/recv_probe.dat``).
    resolution : int
        N such that the plane has N x N probe points.  MUST match the
        ``resolution`` argument of the WarpX FieldProbe diagnostic.
    t_start, t_end : float, optional
        Restrict the loader to the time window ``[t_start, t_end]`` [s].
        Useful for very large CW runs where only the steady-state part is
        wanted.  Default: read the entire file.

    Returns
    -------
    Same tuple as :func:`load_plane_timeseries`:
        times, E_ts, B_ts, coord_a, coord_b, d_a, d_b,
        prop_axis, E_comp, B_comp
    """
    # Deferred import so the module still loads in environments without
    # pandas (only the FieldProbe path needs it).
    import pandas as pd

    n_pts = resolution * resolution
    n_header = _count_dat_header_lines(filepath)

    print(f"  FieldProbe file: {filepath}")
    print(f"  resolution: {resolution} x {resolution} = {n_pts} probe points/step")

    # Peek at the first two blocks to grab dt and per-point coordinates.
    peek = pd.read_csv(
        filepath, sep=r"\s+", header=None, skiprows=n_header,
        usecols=_FP_USE_COLS, nrows=2 * n_pts,
        dtype=np.float64, engine="c",
    )
    if len(peek) < n_pts:
        raise RuntimeError(
            f"FieldProbe file {filepath} has fewer than {n_pts} rows in its "
            f"first block; resolution probably does not match the input "
            f"script.  Pass --fieldprobe-resolution N explicitly.")
    t0 = float(peek.iat[0,     _FP_I_TIME])
    t1 = (float(peek.iat[n_pts, _FP_I_TIME])
          if len(peek) >= 2 * n_pts else t0)
    dt = (t1 - t0) if t1 > t0 else 0.0

    # Time-window skip / cap (only meaningful when dt > 0).
    skip_data_rows = 0
    max_data_rows  = None
    if t_start is not None and dt > 0 and t_start > t0:
        skip_steps     = max(0, int(np.floor((t_start - t0) / dt)))
        skip_data_rows = skip_steps * n_pts
    if t_end is not None and dt > 0:
        start_step    = skip_data_rows // n_pts
        end_step      = int(np.ceil((t_end - t0) / dt)) + 1
        max_data_rows = max(n_pts, (end_step - start_step) * n_pts)

    wall0 = _time.monotonic()
    df = pd.read_csv(
        filepath, sep=r"\s+", header=None,
        skiprows=n_header + skip_data_rows,
        usecols=_FP_USE_COLS, nrows=max_data_rows,
        dtype=np.float64, engine="c",
    )
    elapsed = _time.monotonic() - wall0

    n_rows = (len(df) // n_pts) * n_pts   # trim incomplete trailing block
    if n_rows == 0:
        raise RuntimeError(
            f"No complete FieldProbe time-step blocks in {filepath} for the "
            f"requested window.")
    df = df.iloc[:n_rows]
    n_steps = n_rows // n_pts
    print(f"  read {n_rows:,} rows = {n_steps} steps in {elapsed:.1f} s")

    # ---- Detect propagation axis from the (~constant) coordinate column.
    coord_block = [df.iloc[:n_pts, _FP_I_XYZ[k]].values for k in range(3)]
    ranges = [float(c.max() - c.min()) for c in coord_block]
    prop_axis = int(np.argmin(ranges))
    aper_axes = [k for k in range(3) if k != prop_axis]
    print(f"  Coordinate ranges: x={ranges[0]*1e3:.2f} mm, "
          f"y={ranges[1]*1e3:.2f} mm, z={ranges[2]*1e3:.2f} mm")
    print(f"  Propagation axis (smallest range): {'XYZ'[prop_axis]}")

    E_comp = _E_COMP_BY_PROP_AXIS[prop_axis]
    B_comp = _B_COMP_BY_PROP_AXIS[prop_axis]
    E_col  = _FP_I_E[E_comp]
    B_col  = _FP_I_B[B_comp]
    print(f"  Using E component: E{E_comp},  paired B component: B{B_comp}")

    # ---- Aperture coordinate axes.  WarpX writes probe points in row-major
    # order; whichever aperture axis varies along the FIRST grid axis depends
    # on WarpX's internal loop order, so detect empirically by which slice is
    # constant vs. varying.
    a_grid = coord_block[aper_axes[0]].reshape(resolution, resolution)
    b_grid = coord_block[aper_axes[1]].reshape(resolution, resolution)
    a_along_axis0 = float(np.ptp(a_grid[:, 0])) > float(np.ptp(a_grid[0, :]))
    if a_along_axis0:
        coord_a = a_grid[:, 0]
        coord_b = b_grid[0, :]
        transpose_data = False
    else:
        coord_a = a_grid[0, :]
        coord_b = b_grid[:, 0]
        transpose_data = True
    d_a = float(np.mean(np.diff(coord_a))) if coord_a.size > 1 else 0.0
    d_b = float(np.mean(np.diff(coord_b))) if coord_b.size > 1 else 0.0
    print(f"  Aperture grid: {coord_a.size} x {coord_b.size},  "
          f"d_a={d_a*1e3:.4f} mm,  d_b={d_b*1e3:.4f} mm")

    # ---- Reshape field data to (n_steps, res, res), then transpose to align
    # with (coord_a, coord_b) if necessary.
    E_ts = df.iloc[:, E_col].values.reshape(n_steps, resolution, resolution)
    B_ts = df.iloc[:, B_col].values.reshape(n_steps, resolution, resolution)
    if transpose_data:
        E_ts = np.swapaxes(E_ts, 1, 2)
        B_ts = np.swapaxes(B_ts, 1, 2)

    # ---- Crop to the WR-15 cross-section + 1 cell, matching the openPMD
    # slab.  Probe samples landing inside the EB conductor read interpolator
    # noise rather than exactly zero, so leaving them in would contaminate
    # the TE10 modal projection at the ~1e-6 level.  We zero them outright,
    # which is exactly what the in-rectangle ``phi(u, v)`` from
    # ``te10_mode_2d`` does anyway -- this just makes it explicit and makes
    # any debug plots of E_ts(t) directly comparable to the openPMD slice.
    span_a = coord_a[-1] - coord_a[0]
    span_b = coord_b[-1] - coord_b[0]
    a_is_broad = span_a >= span_b
    # Centre on the slab midpoint so absolute probe coordinates (e.g. WG
    # placed at y ~ 28 mm, z ~ 37 mm in the scooter geometry) compare
    # against a/2 and b/2 correctly.
    a_center = 0.5 * (coord_a[0] + coord_a[-1])
    b_center = 0.5 * (coord_b[0] + coord_b[-1])
    AA, BB = np.meshgrid(coord_a - a_center, coord_b - b_center,
                         indexing='ij')
    broad_grid  = AA if a_is_broad else BB
    narrow_grid = BB if a_is_broad else AA
    crop_mask = ((np.abs(broad_grid)  <= a_wg / 2 + d_a) &
                 (np.abs(narrow_grid) <= b_wg / 2 + d_b))
    inside = int(np.count_nonzero(crop_mask))
    total  = int(crop_mask.size)
    print(f"  Crop to WR-15 + 1 cell: {inside}/{total} samples kept "
          f"({100.0*inside/total:.1f}%)")
    E_ts = E_ts * crop_mask[None, :, :]
    B_ts = B_ts * crop_mask[None, :, :]

    times = df.iloc[::n_pts, _FP_I_TIME].values

    return (times, E_ts, B_ts,
            coord_a, coord_b, d_a, d_b,
            prop_axis, E_comp, B_comp)


# ── TE10 mode shape and overlap integral ───────────────────────────────────────

def te10_mode_2d(coord_a, coord_b):
    """
    TE10 mode shape phi(u) = cos(pi*u/a) on the (Na, Nb) aperture grid.

    The broad wall (dimension a_wg) is identified as the aperture axis with
    the larger coordinate span.  phi varies as a cosine across the broad
    wall and is uniform across the narrow wall.  Zero outside the aperture
    rectangle.

    The slab coordinates produced by the openPMD diagnostic are absolute
    positions in the simulation frame, NOT centred on the waveguide axis
    (e.g. for the scooter geometry the slab sits at y ~ 30 mm, z ~ 37 mm).
    We auto-centre by subtracting the midpoint of each coordinate range
    before computing phi -- the diagnostic slab is, by construction, sized
    to the waveguide cross-section + 1 cell margin on each side, so its
    geometric midpoint coincides with the waveguide axis.

    Returns
    -------
    phi : (Na, Nb) array, peak = 1
    """
    span_a = coord_a[-1] - coord_a[0]
    span_b = coord_b[-1] - coord_b[0]

    a_center = 0.5 * (coord_a[0] + coord_a[-1])
    b_center = 0.5 * (coord_b[0] + coord_b[-1])
    coord_a_c = coord_a - a_center
    coord_b_c = coord_b - b_center

    AA, BB = np.meshgrid(coord_a_c, coord_b_c, indexing='ij')

    broad_grid  = AA if span_a >= span_b else BB
    narrow_grid = BB if span_a >= span_b else AA

    broad_rel  = broad_grid  + a_wg / 2    # shift so aperture runs [0, a]
    narrow_rel = narrow_grid + b_wg / 2

    in_ap = ((broad_rel  >= 0) & (broad_rel  <= a_wg) &
             (narrow_rel >= 0) & (narrow_rel <= b_wg))

    # TE10 mode shape in the shifted [0, a] frame is sin(pi * x' / a),
    # which equals cos(pi * x / a) in the centred [-a/2, +a/2] frame:
    # sin(pi * (x + a/2) / a) = sin(pi*x/a + pi/2) = cos(pi*x/a).
    # Using cos(pi * broad_rel / a) here would instead give -sin(pi*x/a),
    # which is antisymmetric and orthogonal to the actual TE10 field --
    # the resulting modal projection then collapses to grid-asymmetry
    # residuals, ~3 orders of magnitude below the true mode amplitude.
    return np.where(in_ap, np.sin(np.pi * broad_rel / a_wg), 0.0)


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


# ── I/Q demodulation ──────────────────────────────────────────────────────────

def iq_demodulate(t, a, f0, lpf_cutoff_hz, lpf_order=6):
    """Extract the slowly-varying complex envelope of a carrier-modulated signal.

    The input a(t) is assumed to be of the form
        a(t) = A(t) * cos(2*pi*f0*t + phi(t))
    where A(t) and phi(t) vary on timescales much slower than 1/f0.
    Multiplying by 2*exp(-i*2*pi*f0*t) gives
        A*cos(phi) + i*A*sin(phi)  +  terms at  2*f0
    and a low-pass filter with cutoff well below 2*f0 but well above the
    modulation bandwidth removes the 2*f0 image, leaving
        z(t) = A(t) * exp(i*phi(t)).

    Then |z(t)| is the instantaneous amplitude and arg(z(t)) the phase,
    both as continuous functions of time.  This is the standard technique
    used by experimental microwave interferometers.

    Parameters
    ----------
    t : (N,) array
        Uniformly sampled time stamps [s].
    a : (N,) array
        Real-valued modal coefficient trace.
    f0 : float
        Carrier frequency [Hz].  Must match the WarpX laser carrier.
    lpf_cutoff_hz : float
        Low-pass cutoff [Hz].  Must satisfy
            plasma_bandwidth < lpf_cutoff_hz < 2 * f0.
        Rule of thumb: 0.1 * f0 works for everything we care about
        (plasma dynamics slower than ~0.1 * f0 ≈ 6.7 GHz).
    lpf_order : int
        Butterworth order (default 6).

    Returns
    -------
    z : (N,) complex array
        Baseband complex envelope, z(t) = A(t) * exp(i*phi(t)).
    """
    dt = float(np.mean(np.diff(t)))
    fs = 1.0 / dt
    nyq = 0.5 * fs
    if lpf_cutoff_hz >= nyq:
        raise ValueError(
            f"LPF cutoff {lpf_cutoff_hz*1e-9:.2f} GHz must be below the "
            f"Nyquist frequency {nyq*1e-9:.2f} GHz.  The sim time step "
            f"dt = {dt*1e12:.2f} ps is too coarse for this carrier/cutoff "
            f"combination.")
    if lpf_cutoff_hz <= 0:
        raise ValueError("LPF cutoff must be positive.")

    # Baseband mix: multiply by 2*exp(-i*omega0*t) so |z| recovers A(t).
    mixed = 2.0 * a * np.exp(-1j * 2.0 * np.pi * f0 * t)

    # Butterworth SOS low-pass, applied with sosfiltfilt for zero group delay
    # (filters forward and backward; edge transients bleed in from both ends
    # but the centre of the trace is clean).
    sos = butter(lpf_order, lpf_cutoff_hz, btype='low', fs=fs, output='sos')
    z_real = sosfiltfilt(sos, mixed.real)
    z_imag = sosfiltfilt(sos, mixed.imag)
    return z_real + 1j * z_imag


# ── CW analysis ──────────────────────────────────────────────────────────────

def cw_analysis(args, t_rec, aE_rec, aB_rec, prop_axis, E_comp, B_comp, is_complex):
    """Time-resolved |S21|(t) and phase(t) by I/Q demodulation at args.freq.

    Builds an analytic CW reference (steady-state cosine at the carrier) when
    --analytic-ref is set, or uses the injection_plane diagnostic otherwise.
    Applies I/Q demodulation to both reference and received signals, takes the
    ratio, and reports |S21|(t), phase(t), and the steady-state averaged
    values.  Also reports a line-integrated electron density estimate from
    the phase under the low-density approximation, as a sanity-check output
    for plasma runs.
    """
    f0 = args.freq
    c  = 2.99792458e8

    # ── Derived physical quantities ──────────────────────────────────────────
    vg_f0 = c * np.sqrt(max(1 - (fc_te10 / f0) ** 2, 0.0))
    vp_f0 = c / np.sqrt(max(1 - (fc_te10 / f0) ** 2, 1e-30))
    L     = abs(recv_x - inject_x) if prop_axis == 0 else abs(recv_z - inject_z)

    # ── Pick transient-skip time ─────────────────────────────────────────────
    # Rough rule: allow two round-trips of direct-path propagation for the
    # ramp + first-bounce standing-wave pattern to settle.
    if args.cw_t_start is not None:
        t_start = args.cw_t_start
    else:
        t_start = max(3.0 * L / vg_f0, 0.5e-9)
        print(f"  auto cw-t-start = 3*L/v_g = {t_start*1e9:.3f} ns")

    # ── LPF cutoff ───────────────────────────────────────────────────────────
    if args.cw_lpf_cutoff is not None:
        lpf_cut = args.cw_lpf_cutoff
    else:
        lpf_cut = 0.1 * f0
        print(f"  auto cw-lpf-cutoff = 0.1 * f0 = {lpf_cut*1e-9:.3f} GHz")

    # ── Build / load reference ──────────────────────────────────────────────
    # In CW mode the "reference" should be a steady-state carrier at f0 with
    # no ramp.  If the user set --analytic-ref, construct one; otherwise load
    # the injection_plane diagnostic and demodulate that too.
    if args.analytic_ref:
        if args.emit_pos is None:
            raise SystemExit("--emit-pos is required with --analytic-ref "
                             "even in CW mode (sets the zero-phase reference "
                             "plane).")
        recv_pos = recv_x if prop_axis == 0 else recv_z
        # The reference is steady-state CW at the carrier, phase-referenced
        # to the EMITTER plane (NOT time-shifted to receiver arrival).  A
        # time-shift τ_shift = L/v_g would multiply A_inj(f) by exp(-iωτ),
        # injecting a spurious phase offset equal to -β(f0)L into the ratio
        # S21 = z_rec/z_inj and flipping the sign of any real phase shift
        # sitting on top.  Leaving the reference at the emitter frame makes
        # arg(S21_vacuum) = -βL directly, and any plasma-induced phase is a
        # small perturbation on top of that with the correct sign.
        aE_inj  = args.E0 * np.cos(2.0 * np.pi * f0 * t_rec)
        aB_inj  = -aE_inj / vp_f0
        t_inj   = t_rec
        ref_desc = (f"analytic CW at emitter frame  "
                    f"(E0={args.E0:.2e} V/m, f0={f0*1e-9:.2f} GHz)")
    else:
        print("Loading injection plane ...", flush=True)
        (t_inj, E_inj, B_inj, _, _, _, _,
         _, _, _) = load_plane_timeseries(args.diag_dir, "injection_plane")
        aE_inj = project_mode(E_inj, *_cw_coord_args(args))
        aB_inj = project_mode(B_inj, *_cw_coord_args(args))
        ref_desc = "measured injection_plane"

    # Trim to common length
    n      = min(len(t_inj), len(t_rec))
    t_inj, aE_inj, aB_inj = t_inj[:n], aE_inj[:n], aB_inj[:n]
    t_rec, aE_rec, aB_rec = t_rec[:n], aE_rec[:n], aB_rec[:n]

    # ── I/Q demodulate both signals ──────────────────────────────────────────
    print(f"\nI/Q demodulating at f0 = {f0*1e-9:.3f} GHz, "
          f"LPF cutoff {lpf_cut*1e-9:.3f} GHz, order {args.cw_lpf_order}")
    z_inj = iq_demodulate(t_inj, aE_inj, f0, lpf_cut, args.cw_lpf_order)
    z_rec = iq_demodulate(t_rec, aE_rec, f0, lpf_cut, args.cw_lpf_order)

    # ── S21(t) = z_rec(t) / z_inj(t) ─────────────────────────────────────────
    eps = np.max(np.abs(z_inj)) * 1e-6
    with np.errstate(divide='ignore', invalid='ignore'):
        S21_t = np.where(np.abs(z_inj) > eps, z_rec / z_inj, np.nan + 0j)

    mag_t = np.abs(S21_t)
    phi_t = safe_unwrap(np.angle(S21_t))      # radians, unwrapped

    # ── Steady-state window: t > t_start, and avoid the end edge of the
    #    filter transient.  sosfiltfilt bleeds ~ 3 * filter-group-delay from
    #    both ends; use 2 carrier periods as a conservative buffer.
    edge_buffer = 2.0 / f0
    ss_mask = (t_rec >= t_start) & (t_rec <= t_rec[-1] - edge_buffer)
    if not np.any(ss_mask):
        print("WARNING: steady-state window is empty.  Check --cw-t-start "
              "and the simulation duration.")
        ss_mask = np.ones_like(t_rec, dtype=bool)

    mag_ss  = np.nanmean(mag_t[ss_mask])
    mag_std = np.nanstd(mag_t[ss_mask])
    phi_ss  = np.nanmean(phi_t[ss_mask])
    phi_std = np.nanstd(phi_t[ss_mask])

    print(f"\nSteady-state window: t ∈ [{t_start*1e9:.3f}, "
          f"{(t_rec[-1]-edge_buffer)*1e9:.3f}] ns  "
          f"({ss_mask.sum()} samples)")
    print(f"Reference: {ref_desc}")
    # Wrapped phase in [-180, +180] -- this is what a real plasma
    # interferometer reads off the lock-in.  The unwrapped value above
    # accumulates 360-degree turns from `safe_unwrap`; the wrapped value
    # is the unique observable.
    phi_ss_deg          = float(np.degrees(phi_ss))
    phi_ss_wrapped_deg  = ((phi_ss_deg + 180.0) % 360.0) - 180.0
    n_turns_unwrap      = int(round((phi_ss_deg - phi_ss_wrapped_deg) / 360.0))

    print(f"\n|S21|(steady-state)  = {mag_ss:.4e} ± {mag_std:.2e}  "
          f"({20.0 * np.log10(max(mag_ss, 1e-30)):.2f} dB)")
    print(f"Δφ  (steady-state, unwrapped) = {phi_ss_deg:+.3f} ± "
          f"{np.degrees(phi_std):.3f} deg")
    print(f"Δφ  (steady-state, wrapped)   = {phi_ss_wrapped_deg:+.3f} deg "
          f"(unwrapped − {n_turns_unwrap}·360°;  experimentally observable)")

    # ── Line-integrated density estimate ─────────────────────────────────────
    # Under the low-density approximation for TE10 in a uniform-cross-section
    # waveguide filled with a plasma of length d (here we assume d ~ L):
    #   Δφ ≈ (ω / (2*c*n_crit)) * ∫ n_e dz
    # so
    #   <n_e>*d ≈ 2*c*n_crit*|Δφ| / ω
    # with n_crit = eps0*m_e*omega^2 / e^2.  This is a first-order estimate
    # that ignores waveguide dispersion corrections; see e.g. Heald & Wharton
    # "Plasma Diagnostics with Microwaves" for the exact TE10-in-plasma form.
    #
    # IMPORTANT: in a *vacuum* simulation Δφ should be zero.  Any non-zero
    # value from a vacuum run is the reflection/drift noise floor for your
    # geometry, and is the quantity you need to subtract when you swap in
    # a plasma.
    eps0 = 8.8541878128e-12
    me   = 9.1093837015e-31
    qe   = 1.602176634e-19
    omega0 = 2.0 * np.pi * f0
    n_crit = eps0 * me * omega0**2 / qe**2
    nL_est = 2.0 * c * n_crit * abs(phi_ss) / omega0
    print(f"\nCarrier ω0 = 2π·{f0*1e-9:.2f} GHz,  "
          f"n_crit = {n_crit:.3e} m⁻³")
    print(f"Low-density line-integrated n_e estimate from Δφ:")
    print(f"  <n_e>·d ≈ {nL_est:.3e} m⁻²     (vacuum run: should be ~0; any "
          f"non-zero value is the reflection/standing-wave noise floor).")

    if args.no_plot:
        return

    # ── Plots ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    prop_name = "XYZ"[prop_axis]
    geom_tag  = (" [geometry='complex']" if is_complex else "")
    fig.suptitle(
        f"WR-15 waveguide — CW I/Q demodulation{geom_tag}\n"
        f"Propagation along {prop_name},  E{E_comp} / B{B_comp},  "
        f"L = {L*1e2:.3f} cm,  f0 = {f0*1e-9:.3f} GHz,  "
        f"LPF = {lpf_cut*1e-9:.2f} GHz, order {args.cw_lpf_order}",
        fontsize=12)

    # Top-left: raw a_E(t) for both (log |Hilbert envelope| so they're both visible).
    ax = axes[0, 0]
    with np.errstate(invalid='ignore'):
        env_inj = np.abs(hilbert(aE_inj))
        env_rec = np.abs(hilbert(aE_rec))
    ax.semilogy(t_inj*1e9, env_inj,
                label=f"injection  (peak {env_inj.max():.2e} V/m)", lw=0.8)
    ax.semilogy(t_rec*1e9, env_rec,
                label=f"receiving  (peak {env_rec.max():.2e} V/m)",
                lw=0.8, ls='--')
    ax.axvline(t_start*1e9, color='k', ls=':', lw=1,
               label=f"t_start = {t_start*1e9:.2f} ns")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("|envelope a_E(t)|  (V/m, log)")
    ax.set_title("TE10 E modal coefficient — Hilbert envelope")
    ax.legend(fontsize=8); ax.grid(True, which='both')

    # Top-middle: |z_inj(t)| and |z_rec(t)| — the I/Q amplitudes.
    ax = axes[0, 1]
    ax.plot(t_inj*1e9, np.abs(z_inj),
            label="|z_inj(t)|  = A_inj(t)", lw=1.0)
    ax.plot(t_rec*1e9, np.abs(z_rec),
            label="|z_rec(t)|  = A_rec(t)", lw=1.0, ls='--')
    ax.axvspan(t_start*1e9, (t_rec[-1]-edge_buffer)*1e9,
               color='0.6', alpha=0.18, label="steady-state window")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Amplitude (V/m)")
    ax.set_yscale('log')
    ax.set_title("I/Q amplitudes A(t) = |z(t)|")
    ax.legend(fontsize=9); ax.grid(True, which='both')

    # Top-right: arg(z_inj) and arg(z_rec) — the I/Q phases (unwrapped).
    ax = axes[0, 2]
    phi_inj = safe_unwrap(np.angle(z_inj))
    phi_rec = safe_unwrap(np.angle(z_rec))
    ax.plot(t_inj*1e9, np.degrees(phi_inj),
            label="arg(z_inj)  = φ_inj(t)", lw=1.0)
    ax.plot(t_rec*1e9, np.degrees(phi_rec),
            label="arg(z_rec)  = φ_rec(t)", lw=1.0, ls='--')
    ax.axvspan(t_start*1e9, (t_rec[-1]-edge_buffer)*1e9,
               color='0.6', alpha=0.18)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Phase (deg, unwrapped)")
    ax.set_title("I/Q phases φ(t) = arg(z(t))")
    ax.legend(fontsize=9); ax.grid(True)

    # Bottom-left: |S21|(t) — the primary CW observable.
    ax = axes[1, 0]
    ax.plot(t_rec*1e9, mag_t, lw=1.0, label="|S21|(t)")
    ax.axhline(mag_ss, color='red', ls='--', lw=1.2,
               label=f"steady-state mean = {mag_ss:.3e}")
    ax.axvspan(t_start*1e9, (t_rec[-1]-edge_buffer)*1e9,
               color='0.6', alpha=0.18)
    ax.set_yscale('log')
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("|S21|(t)  (log)")
    ax.set_title("Time-resolved |S21|(t)")
    ax.legend(fontsize=9); ax.grid(True, which='both')

    # Bottom-middle: Δφ(t).
    ax = axes[1, 1]
    ax.plot(t_rec*1e9, np.degrees(phi_t), lw=1.0, label="Δφ(t)")
    ax.axhline(np.degrees(phi_ss), color='red', ls='--', lw=1.2,
               label=f"steady-state mean = {np.degrees(phi_ss):+.2f}°")
    ax.axvspan(t_start*1e9, (t_rec[-1]-edge_buffer)*1e9,
               color='0.6', alpha=0.18)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Δφ(t)  (deg, unwrapped)")
    ax.set_title("Time-resolved phase shift Δφ(t) = arg(S21(t))")
    ax.legend(fontsize=9); ax.grid(True)

    # Bottom-right: IQ trajectory — S21(t) in the complex plane.
    # The mean point gives steady-state |S21| (radius from origin) and
    # arg(S21) (angle from +Re axis).  The shape and motion of the trajectory
    # around that mean reveal residual physics:
    #   tight blob          → clean steady state, low standing wave
    #   small arc / ellipse → standing wave at f0 (radial = |S21| ripple,
    #                         tangential = phase ripple)
    #   smooth arc          → slow phase drift (plasma density growth, or
    #                         carrier-frequency mismatch between sim and f0)
    #   spiral inward       → increasing absorption with time
    #   spiral outward      → build-up / instability
    # Steady-state samples are colour-coded by time so you can see the
    # direction of motion at a glance.
    ax = axes[1, 2]
    ss = ss_mask
    re_ss_mean = float(np.nanmean(S21_t[ss].real))
    im_ss_mean = float(np.nanmean(S21_t[ss].imag))

    # Transient cloud (faint).
    ax.plot(S21_t[~ss].real, S21_t[~ss].imag,
            '.', ms=1.0, alpha=0.15, color='0.5', label="transient")
    # Steady-state samples coloured by time.
    t_ss = t_rec[ss]
    sc = ax.scatter(S21_t[ss].real, S21_t[ss].imag,
                    c=t_ss * 1e9, cmap='viridis',
                    s=4, alpha=0.55,
                    label="steady-state samples")
    cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("time (ns)", fontsize=8)

    # Reference circles at |S21|_ss and 2*|S21|_ss to gauge ripple amplitude.
    theta = np.linspace(0, 2*np.pi, 361)
    for r_mult, ls, alpha in [(1.0, '--', 0.6), (2.0, ':', 0.35)]:
        ax.plot(r_mult * mag_ss * np.cos(theta),
                r_mult * mag_ss * np.sin(theta),
                color='0.4', ls=ls, lw=0.8, alpha=alpha,
                label=(f"|S21| = {r_mult:g} · mean" if r_mult != 1 else None))
    # Radial line from origin through the mean (zero-phase reference is +Re).
    ax.plot([0, re_ss_mean], [0, im_ss_mean],
            color='red', lw=0.7, alpha=0.7, ls='-')
    # Mean point marker.  Show both the unwrapped phase used by the printout
    # AND the wrapped phase in [-180, +180] (what a real lock-in would read).
    ax.plot(re_ss_mean, im_ss_mean,
            'rx', ms=10, mew=2,
            label=(f"mean = {mag_ss:.2e}\n"
                   f"Δφ unwrapped = {phi_ss_deg:+.1f}°\n"
                   f"Δφ wrapped   = {phi_ss_wrapped_deg:+.2f}°"))

    ax.axhline(0, color='0.7', lw=0.5)
    ax.axvline(0, color='0.7', lw=0.5)
    ax.set_aspect('equal')
    # Force scientific notation on both axes so |S21| ~ 1e-4 doesn't print as
    # "0.00015" labels stacked on top of each other.  set_powerlimits=(0, 0)
    # always uses scientific form; the shared exponent appears in the corner.
    ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0),
                        useMathText=True)
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    ax.set_xlabel("I = Re(S21)  =  (A_rec/A_inj)·cos(Δφ)")
    ax.set_ylabel("Q = Im(S21)  =  (A_rec/A_inj)·sin(Δφ)")
    ax.set_title("S21(t) in the complex plane\n"
                 "(radius = |S21|, angle = Δφ;  colour = time)")
    ax.legend(fontsize=7, loc='best'); ax.grid(True, alpha=0.4)

    fig.tight_layout()
    outfile = "s21_cw_iq_demodulation.png"
    fig.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved: {outfile}")
    plt.show()


def _cw_coord_args(_args):
    """Placeholder hook: in CW mode with a measured injection_plane we'd
    normally cache coord_a/coord_b/d_a/d_b from the receiver load.  This
    helper exists only so the injection-plane load path above reads cleanly;
    the actual coords are threaded through in the main caller."""
    raise NotImplementedError(
        "CW mode with measured injection_plane requires threading the "
        "aperture coords from the receiver load.  Use --analytic-ref for "
        "now, or extend this helper to cache the coords.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Validate WR-15 waveguide S21 against analytic solution.")
    parser.add_argument("--diag-dir", default="./diags",
                        help="Path to WarpX diags directory")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--max-iterations", type=int, default=None,
                        help="Stop the series loader after at most N "
                             "iterations.  Useful when a BP5 variable-based "
                             "series with read_linear hangs in the very last "
                             "step or two (no end-of-stream marker), or when "
                             "you only need a prefix of the series.  Default: "
                             "read every iteration.")
    parser.add_argument("--openpmd-access-mode", default="auto",
                        choices=["auto", "read_linear", "read_only"],
                        help="openPMD access mode for variable-based BP5 series.\n"
                             "  auto (default): use read_linear, the streaming "
                             "  iterator that handles a writer that's still "
                             "  appending iterations.  Correct for completed runs.\n"
                             "  read_only: open the series in random-access mode "
                             "  and read whatever iterations are present, then "
                             "  stop.  Use this when the WarpX run crashed or was "
                             "  killed mid-write — read_linear will hang waiting "
                             "  for the missing trailing data.\n"
                             "  read_linear: force read_linear even from auto.")
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
    parser.add_argument("--freq",         type=float, default=67e9,
                        help="Carrier frequency of the injected pulse [Hz] "
                             "(default: 67e9).  Used by --analytic-ref to "
                             "build envelope*cos(2*pi*freq*t), to compute "
                             "v_g and v_p at the carrier, and to set the "
                             "pulse arrival time.  MUST match the WarpX "
                             "laser.wavelength setting in your input script.")
    parser.add_argument("--mode",         default="pulse",
                        choices=["pulse", "cw"],
                        help="Analysis mode.\n"
                             "  pulse (default): broadband FFT S21(f) from a "
                             "  short Gaussian pulse.  Use for characterising "
                             "  horn/STL geometries.\n"
                             "  cw: I/Q demodulation of a continuous-wave "
                             "  excitation to produce time-resolved |S21|(t) "
                             "  and phase(t).  Use for plasma density "
                             "  diagnostics and anything where the "
                             "  transmission changes during the simulation.")
    parser.add_argument("--cw-t-start",   type=float, default=None,
                        help="CW mode: skip the first N seconds of the time "
                             "series to let ramp-up + first-bounce transients "
                             "decay before measurement.  Default: auto-"
                             "estimate as 3*L/v_g(f0) plus a small margin.")
    parser.add_argument("--cw-lpf-cutoff", type=float, default=None,
                        help="CW mode: low-pass cutoff [Hz] applied after "
                             "baseband mixing, in units of the carrier "
                             "frequency args.freq.  Use 0.05-0.15 to keep the "
                             "baseband while crushing the 2*f0 image.  "
                             "Default: 0.1 * args.freq.")
    parser.add_argument("--cw-lpf-order", type=int, default=6,
                        help="CW mode: Butterworth low-pass order (default 6). "
                             "Applied with sosfiltfilt for zero group delay.")
    parser.add_argument("--source",       default="openpmd",
                        choices=["openpmd", "fieldprobe"],
                        help="Diagnostic source to read.\n"
                             "  openpmd (default): variable-based or "
                             "  file-based openPMD series under "
                             "  <diag-dir>/{injection,receiving}_plane/.\n"
                             "  fieldprobe: WarpX FieldProbe Plane .dat "
                             "  files.  Useful for older runs that did not "
                             "  write per-plane openPMD slabs, or for "
                             "  cross-checking the openPMD result against a "
                             "  different diagnostic.  Probe samples are "
                             "  cropped to the WR-15 cross-section plus one "
                             "  cell so the modal projection is comparable "
                             "  to the openPMD slab.")
    parser.add_argument("--fieldprobe-resolution", type=int, default=50,
                        help="FieldProbe Plane resolution N (NxN grid). MUST "
                             "match the 'resolution' argument passed to the "
                             "WarpX FieldProbe diagnostic.  Default 50 "
                             "matches the example input scripts.")
    parser.add_argument("--fieldprobe-inj-file", default=None,
                        help="Path to the injection-side FieldProbe .dat "
                             "file.  Default: <diag-dir>/inject_probe.dat.")
    parser.add_argument("--fieldprobe-recv-file", default=None,
                        help="Path to the receiving-side FieldProbe .dat "
                             "file.  Default: <diag-dir>/recv_probe.dat.")
    parser.add_argument("--fieldprobe-t-start", type=float, default=None,
                        help="FieldProbe only: skip data before this time [s].")
    parser.add_argument("--fieldprobe-t-end",   type=float, default=None,
                        help="FieldProbe only: stop reading after this time [s].")
    parser.add_argument("--geometry",     default="straight",
                        choices=["straight", "complex"],
                        help="Geometry class, controls interpretation of the "
                             "analytic reference.\n"
                             "  straight (default): straight waveguide with "
                             "  identical cross-section at both ports.  The "
                             "  lossless analytic exp(-j*beta*L) IS the "
                             "  expected answer; |S21| axis is linear in "
                             "  [0, 1.2] and residuals are printed as errors.\n"
                             "  complex: horn-to-horn, scooter, or any other "
                             "  geometry where the straight-guide analytic is "
                             "  NOT the expected result.  The analytic curve "
                             "  is still drawn, but styled lightly and labelled "
                             "  'straight-guide reference'; |S21| is plotted "
                             "  on a log axis with auto limits; the printed "
                             "  figures are labelled as deviations from the "
                             "  reference, not errors.")
    args = parser.parse_args()

    is_complex = (args.geometry == "complex")

    # Resolve the openPMD access-mode override.
    if args.openpmd_access_mode == "auto":
        access_override = None
    elif args.openpmd_access_mode == "read_only":
        access_override = io.Access.read_only
    elif args.openpmd_access_mode == "read_linear":
        access_override = io.Access.read_linear
    else:
        raise SystemExit(f"Unknown --openpmd-access-mode '{args.openpmd_access_mode}'")

    # FieldProbe: in CW mode the injection plane is loaded inside cw_analysis
    # which still goes through the openPMD path; surface the limitation up
    # front rather than failing inside cw_analysis.
    if args.source == "fieldprobe" and args.mode == "cw" and not args.analytic_ref:
        raise SystemExit(
            "--source fieldprobe in CW mode currently requires --analytic-ref "
            "(the CW path does not yet load FieldProbe injection traces).")

    def _load_plane(kind):
        """Dispatch to openPMD or FieldProbe loader for ``kind`` in
        {'injection', 'receiving'}.  Returns the same 10-tuple in both cases.
        """
        if args.source == "fieldprobe":
            if kind == "injection":
                fp_path = (args.fieldprobe_inj_file
                           if args.fieldprobe_inj_file is not None
                           else os.path.join(args.diag_dir, "inject_probe.dat"))
            elif kind == "receiving":
                fp_path = (args.fieldprobe_recv_file
                           if args.fieldprobe_recv_file is not None
                           else os.path.join(args.diag_dir, "recv_probe.dat"))
            else:
                raise ValueError(f"Unknown plane kind '{kind}'")
            return load_fieldprobe_plane_timeseries(
                fp_path, args.fieldprobe_resolution,
                t_start=args.fieldprobe_t_start,
                t_end=args.fieldprobe_t_end)
        else:
            diag_name = ("injection_plane" if kind == "injection"
                         else "receiving_plane")
            return load_plane_timeseries(args.diag_dir, diag_name,
                                         access_override=access_override,
                                         max_iterations=args.max_iterations)

    # ── Load receiving plane ───────────────────────────────────────────────────────
    print(f"Loading receiving plane (source = {args.source}) ...", flush=True)
    (t_rec, E_rec, B_rec,
     coord_a, coord_b, d_a, d_b,
     prop_axis, E_comp, B_comp) = _load_plane("receiving")

    aE_rec = project_mode(E_rec, coord_a, coord_b, d_a, d_b)
    aB_rec = project_mode(B_rec, coord_a, coord_b, d_a, d_b)
    print(f"  {len(t_rec)} steps,  dt = {np.mean(np.diff(t_rec)):.4e} s,  "
          f"t ∈ [{t_rec[0]*1e9:.4f}, {t_rec[-1]*1e9:.4f}] ns")
    print(f"  peak |a_E_rec| = {np.max(np.abs(aE_rec)):.4e} V/m,  "
          f"peak |a_B_rec| = {np.max(np.abs(aB_rec)):.4e} T")

    # ── CW mode dispatch ─────────────────────────────────────────────────────
    # In CW mode the broadband-pulse FFT pipeline below is meaningless: the
    # signal is a continuous carrier, no time-gating separates direct wave
    # from reflections, and the quantity you want is time-resolved |S21|(t)
    # and Δφ(t) at a single frequency.  Delegate to cw_analysis and stop.
    if args.mode == "cw":
        print(f"\n=== CW mode ===")
        cw_analysis(args, t_rec, aE_rec, aB_rec,
                    prop_axis, E_comp, B_comp, is_complex)
        return

    # ── Incident reference: injection plane or analytical ─────────────────────
    if args.analytic_ref:
        # Build analytical Gaussian pulse envelope as incident reference.
        # A pure forward TE10 pulse at carrier f0 with peak E0 has
        #   a_E_inc(t) = E0 * envelope(t - t_arrival) * cos(2*pi*f0*(t - t_arrival))
        #   a_B_inc(t) = -a_E_inc(t) / v_p(f0)
        # where t_arrival = t_peak + |recv_pos - emit_pos| / v_g(f0).
        #
        # The carrier cos(2*pi*f0*t) is NOT optional: WarpX injects
        # envelope*carrier at the antenna (laser.wavelength sets the carrier),
        # so its Fourier content lives near f0.  If the analytic reference is
        # the envelope alone, its spectrum sits at DC and the [f_lo, f_hi]
        # analysis band sees essentially zero, which drives every S21 bin to
        # NaN and hides the actual simulated transmission.
        if args.emit_pos is None:
            parser.error("--emit-pos is required when using --analytic-ref")

        recv_pos  = recv_x if prop_axis == 0 else recv_z
        L_emit    = abs(recv_pos - args.emit_pos)
        vg_ref    = c * np.sqrt(max(1 - (fc_te10 / args.freq)**2, 0))
        vp_ref    = c / np.sqrt(max(1 - (fc_te10 / args.freq)**2, 1e-30))
        t_arrival = args.pulse_t_peak + L_emit / vg_ref
        pulse_tau = args.pulse_fwhm / (2 * np.sqrt(np.log(2)))

        print(f"\nUsing analytical Gaussian reference:")
        print(f"  E0 = {args.E0:.3e} V/m,  carrier f0 = {args.freq*1e-9:.3f} GHz")
        print(f"  t_peak = {args.pulse_t_peak*1e9:.4f} ns  (emitter-frame "
              f"peak time — NOT shifted to receiver)")
        print(f"  FWHM = {args.pulse_fwhm*1e12:.1f} ps,  τ = {pulse_tau*1e12:.2f} ps")
        print(f"  emit_pos = {args.emit_pos*1e2:.5f} cm,  "
              f"recv_pos = {recv_pos*1e2:.5f} cm")
        print(f"  vg(f0) = {vg_ref/c:.4f} c,  v_p(f0) = {vp_ref/c:.4f} c")
        print(f"  Expected straight-guide arrival at receiver: "
              f"t = {t_arrival*1e9:.4f} ns  (informational only)")

        # IMPORTANT: build the reference at the EMITTER frame (t_peak), NOT
        # time-shifted to the receiver arrival.  A time-shift by τ_shift in
        # the time domain corresponds to multiplication by exp(-iωτ_shift) in
        # the frequency domain.  If we shifted the reference to τ_arrival =
        # L/v_g(f0), then  arg(A_inj) = -ωτ_arrival  and the ratio
        # S21 = A_rec/A_inj would pick up an extra  +iωτ_arrival  phase ramp
        # which has the OPPOSITE sign to the physically correct  -β(f)L  that
        # the simulated A_rec carries.  The consequence was a wrapped S21
        # phase with positive dφ/df ("negative group delay") — cosmetic but
        # confusing.  Leaving the reference at t_peak makes
        # S21 = A_rec/A_inj ≈ S21_geom(f)·exp(-iβ(f)L)
        # so the wrapped phase tracks the analytic -βL directly and the
        # residual is genuine geometry contribution.
        t_shifted = t_rec - args.pulse_t_peak
        pulse_env = args.E0 * np.exp(-(t_shifted / pulse_tau) ** 2)
        carrier   = np.cos(2.0 * np.pi * args.freq * t_shifted)
        aE_inj = pulse_env * carrier
        # Pure forward TE10 at carrier:  B_x = -E_y / v_p(f0).  The ratio is
        # evaluated at the carrier only, so off-carrier bins will pick up a
        # small spurious backward-wave component of order (v_p(f)/v_p(f0) - 1);
        # across [60, 74] GHz with f0 = 67 GHz this is at most ~8%.
        aB_inj = -aE_inj / vp_ref
        t_inj  = t_rec
        print(f"  peak |a_E_inj| = {np.max(np.abs(aE_inj)):.4e} V/m  "
              f"(envelope * carrier)")
        print(f"  peak |a_B_inj| = {np.max(np.abs(aB_inj)):.4e} T")

    else:
        print(f"Loading injection plane (source = {args.source}) ...",
              flush=True)
        (t_inj, E_inj, B_inj,
         coord_a, coord_b, d_a, d_b,
         prop_axis, E_comp, B_comp) = _load_plane("injection")
        aE_inj = project_mode(E_inj, coord_a, coord_b, d_a, d_b)
        aB_inj = project_mode(B_inj, coord_a, coord_b, d_a, d_b)
        print(f"  {len(t_inj)} steps,  dt = {np.mean(np.diff(t_inj)):.4e} s,  "
              f"t ∈ [{t_inj[0]*1e9:.4f}, {t_inj[-1]*1e9:.4f}] ns")
        print(f"  peak |a_E_inj| = {np.max(np.abs(aE_inj)):.4e} V/m,  "
              f"peak |a_B_inj| = {np.max(np.abs(aB_inj)):.4e} T")

    # Sanity check: zero signal means the pulse has not yet reached the receiver
    if np.max(np.abs(aE_rec)) < 1e-20:
        L_check   = abs(recv_x - inject_x) if prop_axis == 0 else abs(recv_z - inject_z)
        vg_approx = c * np.sqrt(max(1 - (fc_te10 / args.freq)**2, 0))
        t_arr     = L_check / vg_approx if vg_approx > 0 else float('inf')
        print(f"\n  WARNING: receiving plane signal is zero.")
        print(f"  Estimated pulse arrival: {t_arr*1e9:.3f} ns, "
              f"but simulation ended at {t_rec[-1]*1e9:.4f} ns.")
        print(f"  Re-run with excitation_mode='pulse' or "
              f"t_sim_override >= {(t_arr + 0.5)*1e9:.1f}e-9")
        return

    # ── Consistency check: are the two planes from the SAME simulation? ──
    # A common failure mode is a leftover ./diags/injection_plane/ directory
    # from an earlier run that happens to share the diags/ directory with the
    # current one.  The loader will happily open it and produce plausible-but-
    # meaningless results.  Catch that here before we go any further.
    if not args.analytic_ref:
        n_inj, n_rec = len(t_inj), len(t_rec)
        t_end_inj = float(t_inj[-1]) if n_inj else 0.0
        t_end_rec = float(t_rec[-1]) if n_rec else 0.0
        dt_inj    = float(np.mean(np.diff(t_inj))) if n_inj > 1 else 0.0
        dt_rec    = float(np.mean(np.diff(t_rec))) if n_rec > 1 else 0.0
        mismatch  = (
            abs(t_end_inj - t_end_rec) > 1e-12 or
            abs(dt_inj - dt_rec)       > 1e-15 or
            abs(n_inj - n_rec)         > 1
        )
        if mismatch:
            print("\n" + "=" * 72)
            print("WARNING: injection_plane and receiving_plane do not look like")
            print("         they come from the SAME simulation run.")
            print(f"  injection_plane: {n_inj} iterations, "
                  f"t_end = {t_end_inj*1e9:.4f} ns, dt = {dt_inj:.4e} s")
            print(f"  receiving_plane: {n_rec} iterations, "
                  f"t_end = {t_end_rec*1e9:.4f} ns, dt = {dt_rec:.4e} s")
            print("  This typically means ./diags/injection_plane/ contains STALE")
            print("  data from a previous simulation.  Any S21/S11 below is")
            print("  comparing apples and oranges.  Recommended actions:")
            print("    (1) rm -rf ./diags/injection_plane/  and either")
            print("        re-run the sim with an injection_plane diagnostic, or")
            print("    (2) rerun this script with  --analytic-ref --emit-pos <z>")
            print("        to use an analytic Gaussian reference instead.")
            print("=" * 72 + "\n")

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
    # backward reflection that otherwise contaminates A_E_inj.  The np.where
    # masks already route zero-denominator bins to NaN, so silence the
    # cosmetic warnings the division itself still emits.
    with np.errstate(divide='ignore', invalid='ignore'):
        eps_plus = np.nanmax(np.abs(A_plus_inj)) * 1e-6
        S21_sim = np.where(np.abs(A_plus_inj) > eps_plus,
                           A_plus_rec / A_plus_inj, np.nan + 0j)

        # Reflection diagnostics: S11 at injection plane is the ratio of the
        # measured backward wave to the measured forward wave there.  S22-like
        # at the receiving plane tells us how much the far-end boundary is
        # returning.
        S11_sim = np.where(np.abs(A_plus_inj) > eps_plus,
                           A_minus_inj / A_plus_inj, np.nan + 0j)
        eps_plus_rec = np.nanmax(np.abs(A_plus_rec)) * 1e-6
        S22_sim = np.where(np.abs(A_plus_rec) > eps_plus_rec,
                           A_minus_rec / A_plus_rec, np.nan + 0j)

        # Also compute the naive E-only S21 for comparison (what the old
        # script did before the directional split was introduced).
        eps_E         = np.nanmax(np.abs(A_E_inj)) * 1e-6
        S21_sim_Eonly = np.where(np.abs(A_E_inj) > eps_E,
                                 A_E_rec / A_E_inj, np.nan + 0j)

    L = abs(recv_x - inject_x) if prop_axis == 0 else abs(recv_z - inject_z)
    S21_ana = S21_analytic(freqs, L)

    # ── Residuals ────────────────────────────────────────────────────────────
    f_b     = freqs[mask]
    mag_sim = np.abs(S21_sim[mask])
    mag_ana = np.abs(S21_ana[mask])
    # NaN-safe unwrap so a single bad bin does not wipe out the whole phase
    # plot.  np.unwrap propagates NaNs to every subsequent sample.
    phs_sim = safe_unwrap(np.angle(S21_sim[mask]))
    phs_ana = safe_unwrap(np.angle(S21_ana[mask]))

    # Remove the ambiguous 2*pi*k branch offset that np.unwrap cannot resolve.
    # At 67 GHz and L = 25 cm the analytic phase has accumulated ~50 full
    # turns, so sim and ana unwraps anchor on different integer multiples of
    # 2*pi.  We align them by snapping the constant offset between their
    # first valid samples to the nearest multiple of 2*pi; the *residual* is
    # then the genuine geometry-induced deviation, not an unwrap artifact.
    # (The sign convention is already correct:  S21_analytic = exp(-i*beta*L)
    # has been verified against an FFT of a delayed pulse in a straight
    # guide.)
    with np.errstate(invalid='ignore'):
        valid_both = np.isfinite(phs_sim) & np.isfinite(phs_ana)
        if np.any(valid_both):
            i0 = int(np.flatnonzero(valid_both)[0])
            offset_rad = 2.0 * np.pi * np.round(
                (phs_sim[i0] - phs_ana[i0]) / (2.0 * np.pi))
            phs_sim_aligned = phs_sim - offset_rad
        else:
            phs_sim_aligned = phs_sim.copy()
            offset_rad = 0.0
    phs_err = np.degrees(phs_sim_aligned - phs_ana)
    mag_err = mag_sim - mag_ana

    mag_sim_Eonly = np.abs(S21_sim_Eonly[mask])
    mag_err_Eonly = mag_sim_Eonly - mag_ana

    print(f"\nPropagation axis: {'XYZ'[prop_axis]},  L = {L*1e2:.5f} cm")
    print(f"TE10 cutoff = {fc_te10*1e-9:.3f} GHz")
    print(f"Window: {WINDOW_TYPE}")
    print(f"Geometry: {args.geometry}"
          + ("  (straight-guide analytic IS the expected answer)"
             if not is_complex else
             "  (straight-guide analytic is a REFERENCE only)"))
    if is_complex:
        print("  NOTE: numbers reported below as '|S21| deviation' are the")
        print("        difference between simulation and the straight-guide")
        print("        reference and are NOT expected to be small.")
    residual_label = "deviation" if is_complex else "error"
    n_turns_removed_print = int(round(offset_rad / (2.0 * np.pi)))
    print(f"\nDirectional S21 = A_+_recv / A_+_inj,  "
          f"{residual_label}s in "
          f"[{f_lo*1e-9:.0f}, {f_hi*1e-9:.0f}] GHz ({mask.sum()} freq points):")
    if n_turns_removed_print != 0:
        # beta*L at the carrier gives the expected number of accumulated turns.
        f_mid = 0.5 * (f_lo + f_hi)
        expected_turns = float(beta(f_mid)) * L / (2.0 * np.pi)
        print(f"  (aligned phase branches by removing a "
              f"{n_turns_removed_print}·360° offset;  "
              f"β(f_mid)·L/2π ≈ {expected_turns:.1f} turns expected)")
    print(f"  |S21| {residual_label}  max = {np.nanmax(np.abs(mag_err)):.2e}  "
          f"RMS = {np.sqrt(np.nanmean(mag_err**2)):.2e}")
    print(f"  Phase {residual_label}  max = {np.nanmax(np.abs(phs_err)):.4f} deg  "
          f"RMS = {np.sqrt(np.nanmean(phs_err**2)):.4f} deg")
    print(f"\nNaive E-only S21 for comparison:")
    print(f"  |S21| {residual_label}  max = {np.nanmax(np.abs(mag_err_Eonly)):.2e}  "
          f"RMS = {np.sqrt(np.nanmean(mag_err_Eonly**2)):.2e}")

    if is_complex:
        # Headline numbers that actually mean something for a lossy geometry.
        with np.errstate(invalid='ignore'):
            mag_sim_finite = mag_sim[np.isfinite(mag_sim)]
        if mag_sim_finite.size:
            print(f"\nSimulated |S21| in [{f_lo*1e-9:.0f}, {f_hi*1e-9:.0f}] GHz:")
            print(f"  mean = {mag_sim_finite.mean():.3e}  "
                  f"peak = {mag_sim_finite.max():.3e}  "
                  f"min  = {mag_sim_finite.min():.3e}")
            # Convert mean to dB for intuition.
            with np.errstate(divide='ignore'):
                mean_db = 20.0 * np.log10(mag_sim_finite.mean())
            print(f"  mean |S21| in dB = {mean_db:.2f} dB")

    with np.errstate(invalid='ignore'):
        s11_band = np.abs(S11_sim[mask])
        s22_band = np.abs(S22_sim[mask])
    print(f"\nReflection diagnostics (from directional split):")
    print(f"  |S11|_inj  mean = {np.nanmean(s11_band):.3e}  "
          f"max = {np.nanmax(s11_band):.3e}")
    print(f"  |S22|_recv mean = {np.nanmean(s22_band):.3e}  "
          f"max = {np.nanmax(s22_band):.3e}")

    # ── S21 at the carrier frequency f0 = args.freq ─────────────────────────
    # The FFT grid is dense (ZERO_PAD_FACTOR = 64) so the nearest bin sits
    # within sub-MHz of args.freq at 67 GHz.  Report the complex S21 there
    # together with the wrapped phase (the experimentally observable quantity
    # for a plasma interferometer) and its difference from the straight-guide
    # analytic reference.
    f0 = args.freq
    if not (freqs[0] <= f0 <= freqs[-1]):
        print(f"\nCarrier f0 = {f0*1e-9:.3f} GHz is outside the FFT grid "
              f"[{freqs[0]*1e-9:.3f}, {freqs[-1]*1e-9:.3f}] GHz; "
              f"not reporting S21(f0).")
    else:
        i_f0 = int(np.argmin(np.abs(freqs - f0)))
        f0_bin    = float(freqs[i_f0])
        S21_f0    = S21_sim[i_f0]
        S21_anaf0 = S21_ana[i_f0]
        S11_f0    = S11_sim[i_f0]
        S22_f0    = S22_sim[i_f0]

        mag_f0 = float(np.abs(S21_f0))
        with np.errstate(divide='ignore'):
            mag_db_f0 = 20.0 * np.log10(max(mag_f0, 1e-30))

        # Wrapped phase (in [-180, +180] deg) -- this is what a plasma
        # interferometer actually reads, independent of any 2*pi*k unwrap
        # branch choice.
        wrapped_sim_deg = float(np.degrees(np.angle(S21_f0)))
        wrapped_ana_deg = float(np.degrees(np.angle(S21_anaf0)))
        # Wrap the sim-minus-ana phase back into [-180, +180] so it reports
        # as a compact angle even when sim and ana land on different sides
        # of +-pi.
        dphi_wrapped_deg = float(np.degrees(
            np.angle(S21_f0 / S21_anaf0)))

        print(f"\n=== S21 at carrier f0 = {f0*1e-9:.3f} GHz "
              f"(nearest bin {f0_bin*1e-9:.4f} GHz, "
              f"offset {(f0_bin-f0)*1e-6:+.3f} MHz) ===")
        print(f"  |S21|(f0)         = {mag_f0:.6e}  ({mag_db_f0:+.3f} dB)")
        print(f"  arg(S21)(f0)      = {wrapped_sim_deg:+.4f} deg   "
              f"(wrapped to [-180, +180])")
        print(f"  arg(S21_ref)(f0)  = {wrapped_ana_deg:+.4f} deg   "
              f"(straight-guide analytic reference)")
        print(f"  Δφ_wrapped(f0)    = {dphi_wrapped_deg:+.4f} deg   "
              f"(arg(S21_sim / S21_ref), wrapped)")
        print(f"  |S11|(f0)_inj     = {float(np.abs(S11_f0)):.4e}")
        print(f"  |S22|(f0)_recv    = {float(np.abs(S22_f0)):.4e}")

    if args.no_plot:
        return

    # ── Plots ──────────────────────────────────────────────────────────────────
    prop_name = 'XYZ'[prop_axis]
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    geom_tag = (" [geometry='complex' — analytic is REFERENCE only]"
                if is_complex else "")
    fig.suptitle(
        f"WR-15 waveguide — directional S21 validation{geom_tag}\n"
        f"Propagation along {prop_name},  E{E_comp} / B{B_comp},  "
        f"L = {L*1e2:.3f} cm,  window = {WINDOW_TYPE}",
        fontsize=12)

    # Top-left: time-domain modal coefficients of E.
    # Straight mode: plot a_E(t) directly, linear axis.
    # Complex mode:  plot |Hilbert envelope| on a log axis so injection
    # (peak E0 ~ 1e6) and receiving (often 1e3-1e4 x smaller) are both
    # visible.  Hilbert envelope strips the 67 GHz carrier oscillations
    # that would otherwise produce near-zero samples and picket-fence
    # behaviour on a log axis.  In complex mode the a_B envelopes share
    # the same x-axis on a twin right axis, so we can eyeball the
    # B_t = -E_t / v_p forward-wave relation at the direct-pulse arrival.
    inj_label = "injection (analytic)" if args.analytic_ref else "injection plane"
    ax = axes[0, 0]
    if is_complex:
        # hilbert() on a real signal returns the analytic signal; its magnitude
        # is the envelope.  Safe for both the modulated analytic reference and
        # the measured receiver trace.
        with np.errstate(invalid='ignore'):
            env_inj = np.abs(hilbert(aE_inj))
            env_rec = np.abs(hilbert(aE_rec))
            env_Binj = np.abs(hilbert(aB_inj))
            env_Brec = np.abs(hilbert(aB_rec))
        l1, = ax.semilogy(t_inj*1e9, env_inj, color='C0', lw=0.9,
                          label=f"E  {inj_label}  (peak {env_inj.max():.2e} V/m)")
        l2, = ax.semilogy(t_rec*1e9, env_rec, color='C1', lw=0.9, ls='--',
                          label=f"E  receiving  (peak {env_rec.max():.2e} V/m)")
        ax.set_ylabel("|envelope a_E(t)|  (V/m, log)")
        # Twin axis for the paired B envelope — different units, same x.
        axB = ax.twinx()
        l3, = axB.semilogy(t_inj*1e9, env_Binj, color='C2', lw=0.7, alpha=0.7,
                           label=f"B{B_comp}  {inj_label}  (peak {env_Binj.max():.2e} T)")
        l4, = axB.semilogy(t_rec*1e9, env_Brec, color='C3', lw=0.7, alpha=0.7,
                           ls='--',
                           label=f"B{B_comp}  receiving  (peak {env_Brec.max():.2e} T)")
        axB.set_ylabel(f"|envelope a_B{B_comp}(t)|  (T, log)", color='0.35')
        axB.tick_params(axis='y', labelcolor='0.35')
        ax.set_title("TE10 modal-coefficient Hilbert envelopes (E left, B right)")
        # Combined legend from both axes
        ax.legend(handles=[l1, l2, l3, l4], fontsize=7, loc='upper right')
    else:
        ax.plot(t_inj*1e9, aE_inj, label=inj_label, lw=0.8)
        ax.plot(t_rec*1e9, aE_rec, label="receiving plane", lw=0.8, ls='--')
        ax.set_ylabel("a_E(t)  (V/m)")
        ax.set_title("TE10 E modal coefficient a_E(t)")
        ax.legend(fontsize=8)
    ax.set_xlabel("Time (ns)")
    ax.grid(True, which='both')

    # Top-middle: forward vs backward amplitude at each plane, FULL spectrum.
    # Plotting only the narrow analysis band hides everything useful for
    # debugging (noise floor, DC content, pulse envelope, spurious peaks).
    # Show the whole positive-frequency spectrum; shade the analysis band.
    ax = axes[0, 1]
    f_plot_hi = min(3.0 * f_hi, float(freqs[-1]))   # generous upper limit
    plot_mask = (freqs > 0) & (freqs <= f_plot_hi)
    fp = freqs[plot_mask] * 1e-9
    inj_tag = "inj (analytic)" if args.analytic_ref else "injection"
    ax.semilogy(fp, np.abs(A_plus_inj[plot_mask]),
                label=f"|A_+| {inj_tag}", lw=1.0)
    ax.semilogy(fp, np.abs(A_minus_inj[plot_mask]),
                label=f"|A_-| {inj_tag}", lw=1.0, ls='--')
    ax.semilogy(fp, np.abs(A_plus_rec[plot_mask]),
                label="|A_+| receiving", lw=1.0)
    ax.semilogy(fp, np.abs(A_minus_rec[plot_mask]),
                label="|A_-| receiving", lw=1.0, ls='--')
    ax.axvspan(f_lo*1e-9, f_hi*1e-9, color='0.6', alpha=0.18,
               label=f"analysis band [{f_lo*1e-9:.0f}, {f_hi*1e-9:.0f}] GHz")
    ax.axvline(fc_te10*1e-9, color='k', ls=':', lw=1,
               label=f"f_c = {fc_te10*1e-9:.2f} GHz")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Spectral amplitude (V/m·s)")
    ax.set_title("Forward (A_+) vs backward (A_-) spectra")
    ax.legend(fontsize=8, loc='lower left'); ax.grid(True, which='both')

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
    # Straight-guide geometry: linear 0..1.2 axis, analytic is the truth (black).
    # Complex geometry:         log axis with auto limits so |S21| ~ 1e-4 is
    # visible, and the analytic is drawn as a muted grey reference curve.
    ax = axes[1, 0]
    if is_complex:
        # Use a positive floor for safety on the log axis.
        finite_plot = np.concatenate([mag_sim[np.isfinite(mag_sim)],
                                      mag_sim_Eonly[np.isfinite(mag_sim_Eonly)]])
        ax.semilogy(f_b*1e-9, mag_sim,
                    label="|S21| directional (A_+/A_+)", lw=1.5)
        ax.semilogy(f_b*1e-9, mag_sim_Eonly,
                    label="|S21| naive (A_E/A_E)", lw=1.0, ls=':')
        ax.semilogy(f_b*1e-9, mag_ana,
                    label="straight-guide reference (|S21|=1)",
                    lw=1.0, ls='--', color='0.5', alpha=0.7)
        if finite_plot.size:
            lo = max(finite_plot.min() / 3.0, 1e-8)
            ax.set_ylim(lo, 2.0)
        ax.set_ylabel("|S21|  (log)")
        ax.set_title("|S21| magnitude  (log axis for lossy geometry)")
    else:
        ax.plot(f_b*1e-9, mag_sim,
                label="|S21| directional (A_+/A_+)", lw=1.5)
        ax.plot(f_b*1e-9, mag_sim_Eonly,
                label="|S21| naive (A_E/A_E)", lw=1.0, ls=':')
        ax.plot(f_b*1e-9, mag_ana,
                label="|S21| analytic = 1",
                lw=1.5, ls='--', color='k')
        ax.set_ylim(0, 1.2)
        ax.set_ylabel("|S21|")
        ax.set_title("|S21| magnitude")
    # Mark the carrier and annotate the |S21|(f0) value so the number in
    # the printout is also visible on the figure.
    if f_b[0] <= args.freq <= f_b[-1]:
        i_f0_band = int(np.argmin(np.abs(f_b - args.freq)))
        mag_at_f0 = float(mag_sim[i_f0_band])
        ax.axvline(args.freq*1e-9, color='purple', ls='-.', lw=1.2,
                   alpha=0.8, label=f"f0 = {args.freq*1e-9:.2f} GHz")
        if np.isfinite(mag_at_f0):
            ax.annotate(f"|S21|(f0) = {mag_at_f0:.3e}",
                        xy=(args.freq*1e-9, mag_at_f0),
                        xytext=(6, 6), textcoords='offset points',
                        fontsize=8, color='purple')
    ax.set_xlabel("Frequency (GHz)")
    ax.legend(fontsize=9); ax.grid(True, which='both')

    # Bottom-middle: directional S21 phase vs analytic.
    # Both curves are plotted on the *same* 2*pi branch by subtracting an
    # integer-multiple-of-2*pi offset (see residual computation above).
    # Without this alignment the two would often differ by many thousands of
    # degrees — a pure unwrap-branch ambiguity, not real physics.  In complex
    # mode the analytic −βL is only a reference; de-emphasise it and relabel
    # the difference as a deviation, not an error.
    n_turns_removed = int(round(offset_rad / (2.0 * np.pi)))
    ax = axes[1, 1]
    sim_label = (f"Phase directional (offset −{n_turns_removed} × 360°)"
                 if n_turns_removed != 0 else "Phase directional")
    ax.plot(f_b*1e-9, np.degrees(phs_sim_aligned), label=sim_label, lw=1.5)
    if is_complex:
        ax.plot(f_b*1e-9, np.degrees(phs_ana),
                label="straight-guide ref (−βL)",
                lw=1.0, ls='--', color='0.5', alpha=0.7)
    else:
        ax.plot(f_b*1e-9, np.degrees(phs_ana),
                label="Phase analytic (−βL)",
                lw=1.5, ls='--', color='k')
    if f_b[0] <= args.freq <= f_b[-1]:
        ax.axvline(args.freq*1e-9, color='purple', ls='-.', lw=1.2,
                   alpha=0.8, label=f"f0 = {args.freq*1e-9:.2f} GHz")
    ax2 = ax.twinx()
    diff_label = "Deviation (deg)" if is_complex else "Error (deg)"
    ax2.plot(f_b*1e-9, phs_err, color='red', alpha=0.6, lw=0.8,
             label=diff_label)
    ax2.set_ylabel(diff_label, color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Phase (deg)")
    ax.set_title("S21 phase"
                 + ("  vs straight-guide ref (−βL)"
                    if is_complex else "  vs analytic −βL"))
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    ax.grid(True)

    # Bottom-right: wrapped S21 phase diagnostic.
    # The raw np.angle(S21) stays in [-180, +180] and is unambiguous -- no
    # unwrap-branch choice, no 2*pi*k alignment.  A lossless guide gives
    # dense sawteeth (360 deg jumps spaced 1/(beta'(f)*L) apart in
    # frequency); horn/STL geometries add slow deviations on top of those
    # sawteeth.  Bins where |S21| drops to the noise floor show random
    # scatter here, which tells you exactly where the spectrum is not
    # trustworthy -- the unwrapped plot hides that by smoothing through it.
    ax = axes[1, 2]
    phs_sim_wrapped = np.degrees(np.angle(S21_sim[mask]))
    phs_ana_wrapped = np.degrees(np.angle(S21_ana[mask]))
    ax.plot(f_b*1e-9, phs_sim_wrapped, lw=0.9,
            label="Simulated (wrapped)", color='C0')
    ana_style = dict(lw=0.9, ls='--',
                     color=('0.5' if is_complex else 'k'),
                     alpha=(0.7 if is_complex else 1.0))
    ax.plot(f_b*1e-9, phs_ana_wrapped,
            label=("straight-guide ref (wrapped)" if is_complex
                   else "Analytic (wrapped)"),
            **ana_style)
    ax.axhline(+180, color='0.8', lw=0.5)
    ax.axhline(-180, color='0.8', lw=0.5)
    ax.axhline(0,    color='0.8', lw=0.5)
    # Mark the carrier and annotate the wrapped-phase difference at f0,
    # which is the experimentally observable quantity for a plasma
    # interferometer.
    if f_b[0] <= args.freq <= f_b[-1]:
        i_f0_band = int(np.argmin(np.abs(f_b - args.freq)))
        dphi_at_f0 = float(phs_sim_wrapped[i_f0_band] - phs_ana_wrapped[i_f0_band])
        # Re-wrap into [-180, +180] so the annotation is compact.
        dphi_at_f0 = ((dphi_at_f0 + 180.0) % 360.0) - 180.0
        ax.axvline(args.freq*1e-9, color='purple', ls='-.', lw=1.2,
                   alpha=0.8,
                   label=(f"f0 = {args.freq*1e-9:.2f} GHz, "
                          f"Δφ_wrap = {dphi_at_f0:+.2f}°"))
    ax.set_ylim(-190, 190)
    ax.set_yticks([-180, -90, 0, 90, 180])
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Wrapped phase (deg)")
    ax.set_title("S21 phase wrapped to [−180°, +180°]")
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, which='both', alpha=0.4)

    fig.tight_layout()
    # Source-aware output filename so an openPMD run and a FieldProbe run
    # of the same simulation don't overwrite each other's plots.
    src_suffix = ("_fieldprobe" if args.source == "fieldprobe" else "")
    outfile = f"s21_validation_directional_splitting{src_suffix}.png"
    fig.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved: {outfile}")
    plt.show()


if __name__ == "__main__":
    main()