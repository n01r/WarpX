#!/usr/bin/env python3
"""
Extract S21 scattering parameter (magnitude and phase) from WarpX FieldProbe
plane diagnostics for a TE10 waveguide mode.

Two analysis modes:
  - lock-in (default): multiply the TE10 modal coefficient by cos(wt) and
    sin(wt) at the drive frequency and average to get amplitude and phase.
    Only needs a few steady-state RF cycles -- ideal for CW excitation.
  - fft: compute the full spectrum S21(f) -- use when plasma effects may
    cause frequency shifts, harmonic generation, etc.

Both modes use pandas for fast I/O (5-10x faster than np.loadtxt) and
support a time window (--t-start / --t-end) to skip transients and read
only the relevant portion of large files.

Usage:
    # Lock-in (CW vacuum):
    python fieldprobe_s21.py recv_probe.dat --waveguide-a 3.7592e-3 \\
        --freq 67e9 --amplitude 1e6 --resolution 50 \\
        --t-start 1.5e-9 --t-end 2.0e-9 --plot

    # FFT (broadband / plasma):
    python fieldprobe_s21.py recv_probe.dat --waveguide-a 3.7592e-3 \\
        --freq 67e9 --amplitude 1e6 --mode fft \\
        --t-start 1.5e-9 --plot
"""

import argparse
import sys
import time as timer
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Column layout of the FieldProbe .dat file
# step(0) time(1) x(2) y(3) z(4) Ex(5) Ey(6) Ez(7) Bx(8) By(9) Bz(10) S(11)
# ---------------------------------------------------------------------------
_COL_STEP = 0
_COL_TIME = 1
_COL_X    = 2
_COL_Y    = 3
_COL_EY   = 6
_USE_COLS = [_COL_STEP, _COL_TIME, _COL_X, _COL_Y, _COL_EY]
# positional indices within _USE_COLS (after usecols selection):
_I_TIME = 1
_I_X    = 2
_I_Y    = 3
_I_EY   = 4


def _count_header_lines(filepath):
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


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_fieldprobe(filepath, resolution, t_start=None, t_end=None):
    """
    Parse a WarpX FieldProbe plane .dat file with pandas.

    Parameters
    ----------
    filepath : str
    resolution : int
        N such that the plane has N x N probe points.
    t_start, t_end : float or None
        Only load data within this time window (seconds).

    Returns
    -------
    times  : ndarray (n_steps,)
    coords : ndarray (resolution, resolution, 2)   -- (x, y) from step 0
    Ey     : ndarray (n_steps, resolution, resolution)
    """
    n_pts = resolution * resolution
    n_header = _count_header_lines(filepath)

    # --- Peek at first two blocks to get dt and grid coordinates ---
    peek = pd.read_csv(
        filepath, sep=r"\s+", header=None, skiprows=n_header,
        usecols=_USE_COLS, nrows=2 * n_pts, dtype=np.float64, engine="c",
    )
    t0 = float(peek.iat[0,      _I_TIME])
    t1 = float(peek.iat[n_pts,  _I_TIME])
    dt = t1 - t0

    x0 = peek.iloc[:n_pts, _I_X].values.reshape(resolution, resolution)
    y0 = peek.iloc[:n_pts, _I_Y].values.reshape(resolution, resolution)
    coords = np.stack([x0, y0], axis=-1)

    # --- Decide which rows to read ---
    skip_data_rows = 0
    max_data_rows  = None

    if t_start is not None and t_start > t0:
        skip_steps     = max(0, int(np.floor((t_start - t0) / dt)))
        skip_data_rows = skip_steps * n_pts

    if t_end is not None:
        start_step    = skip_data_rows // n_pts
        end_step      = int(np.ceil((t_end - t0) / dt)) + 1
        max_data_rows = max(n_pts, (end_step - start_step) * n_pts)

    t_read_start = t0 + (skip_data_rows // n_pts) * dt
    print(f"  Reading from t = {t_read_start:.4e} s"
          + (f"  ({max_data_rows // n_pts} steps)" if max_data_rows else " (all remaining steps)")
          + " ...")

    wall0 = timer.monotonic()
    df = pd.read_csv(
        filepath, sep=r"\s+", header=None,
        skiprows=n_header + skip_data_rows,
        usecols=_USE_COLS, nrows=max_data_rows,
        dtype=np.float64, engine="c",
    )
    elapsed = timer.monotonic() - wall0

    n_rows = len(df)
    n_rows = (n_rows // n_pts) * n_pts   # trim incomplete trailing block
    if n_rows == 0:
        raise ValueError("No complete timestep blocks found in the requested time window.")
    df = df.iloc[:n_rows]

    print(f"  Read {n_rows:,} rows in {elapsed:.1f} s")

    n_steps = n_rows // n_pts
    times   = df.iloc[::n_pts, _I_TIME].values
    Ey      = df.iloc[:,        _I_EY ].values.reshape(n_steps, resolution, resolution)

    return times, coords, Ey


# ---------------------------------------------------------------------------
# TE10 modal projection
# ---------------------------------------------------------------------------

def te10_mode_shape(x_grid, x_center, a_wg):
    """
    phi(x) = cos(pi * (x - x_center) / a_wg)

    Centred form of the TE10 transverse profile (x_center = waveguide axis).
    Equivalent to sin(pi*x'/a) when x' is measured from the conducting wall.
    """
    if a_wg <= 0:
        raise ValueError("Waveguide width a_wg must be positive")
    return np.cos(np.pi * (x_grid - x_center) / a_wg)


def project_te10_timeseries(Ey_3d, coords, x_center, a_wg):
    """
    Vectorised overlap integral of Ey with the TE10 mode shape for all steps.

    Returns
    -------
    modal : ndarray (n_steps,)
    """
    phi  = te10_mode_shape(coords[..., 0], x_center, a_wg)  # (res, res)
    norm = np.sum(phi ** 2)
    if norm == 0:
        return np.zeros(Ey_3d.shape[0])
    overlap = np.tensordot(Ey_3d, phi, axes=([1, 2], [0, 1]))  # (n_steps,)
    return overlap / norm


# ---------------------------------------------------------------------------
# Lock-in (heterodyne) S21 -- single frequency, CW
# ---------------------------------------------------------------------------

def lockin_s21(times, modal, drive_freq, drive_amplitude=1.0):
    """
    Extract complex S21 at *drive_freq* using lock-in detection.

        a(t) ≈ A cos(ωt + φ)  in steady state
        I = <2 a(t) cos(ωt)> = A cos(φ)
        Q = <2 a(t) sin(ωt)> = -A sin(φ)
        phasor = I - jQ = A exp(jφ)
        S21 = phasor / drive_amplitude

    Parameters
    ----------
    drive_amplitude : float
        Expected peak modal coefficient of the input (V/m), used for
        normalisation.  For a TE10 mode driven with E0, this is E0 times
        the spatial overlap of the mode with itself (≈ E0/2 for a full
        cosine across the waveguide, but depends on the probe coverage).
        Pass the known E0 and interpret |S21| as a relative insertion loss.
    """
    omega   = 2 * np.pi * drive_freq
    cos_ref = np.cos(omega * times)
    sin_ref = np.sin(omega * times)
    I =  2.0 * np.mean(modal * cos_ref)
    Q =  2.0 * np.mean(modal * sin_ref)
    return (I - 1j * Q) / drive_amplitude


# ---------------------------------------------------------------------------
# FFT-based S21 -- broadband / nonlinear plasma case
# ---------------------------------------------------------------------------

def compute_s21_fft(times, modal_recv, modal_emit=None,
                    drive_freq=None, drive_amplitude=1.0):
    """
    Compute S21(f) = A_recv(f) / A_ref(f).

    Reference priority: modal_emit > synthetic sinusoid at drive_freq > raw.
    """
    N  = len(times)
    dt = np.mean(np.diff(times))
    if dt <= 0:
        raise ValueError("Time array is not monotonically increasing")

    nfft   = max(N, 2 ** (int(np.ceil(np.log2(N))) + 1))
    window = np.hanning(N)
    recv_fft = np.fft.rfft(modal_recv * window, n=nfft)
    freqs    = np.fft.rfftfreq(nfft, d=dt)

    if modal_emit is not None:
        emit_fft = np.fft.rfft(modal_emit * window, n=nfft)
        eps = np.max(np.abs(emit_fft)) * 1e-10
        S21 = recv_fft / np.where(np.abs(emit_fft) > eps, emit_fft, eps + 0j)
    elif drive_freq is not None:
        ref = drive_amplitude * np.cos(2 * np.pi * drive_freq * times)
        ref_fft = np.fft.rfft(ref * window, n=nfft)
        eps = np.max(np.abs(ref_fft)) * 1e-10
        S21 = recv_fft / np.where(np.abs(ref_fft) > eps, ref_fft, eps + 0j)
    else:
        S21 = recv_fft   # raw spectrum, normalise externally

    return freqs, S21


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_modal_timeseries(times, modal, drive_freq=None, output_prefix="s21"):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(times * 1e9, modal, lw=0.5)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("TE10 modal coeff. (V/m)")
    ax.set_title("Receiving plane — TE10 modal coefficient a(t)")
    ax.grid(True)
    if drive_freq is not None:
        T_rf = 1 / drive_freq
        # Zoom in on the last 5 RF cycles
        ax.set_xlim(times[-1] * 1e9 - 5 * T_rf * 1e9, times[-1] * 1e9)
    fig.tight_layout()
    fname = f"{output_prefix}_modal.png"
    fig.savefig(fname, dpi=150)
    print(f"  Plot saved to {fname}")
    plt.show()


def plot_fft_results(freqs, S21, drive_freq=None, output_prefix="s21"):
    import matplotlib.pyplot as plt

    mag_dB    = 20 * np.log10(np.abs(S21) + 1e-30)
    phase_deg = np.degrees(np.unwrap(np.angle(S21)))

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    ax = axes[0]
    ax.plot(freqs * 1e-9, mag_dB)
    ax.set_ylabel("|S21| (dB)")
    ax.set_title("S21 — TE10 modal analysis")
    ax.grid(True)
    if drive_freq is not None:
        ax.axvline(drive_freq * 1e-9, color="r", ls="--",
                   label=f"f₀ = {drive_freq*1e-9:.3f} GHz")
        ax.legend()

    ax = axes[1]
    ax.plot(freqs * 1e-9, phase_deg)
    ax.set_ylabel("∠S21 (deg)")
    ax.grid(True)
    if drive_freq is not None:
        ax.axvline(drive_freq * 1e-9, color="r", ls="--")

    ax = axes[2]
    ax.plot(freqs * 1e-9, S21.real, label="Re(S21)")
    ax.plot(freqs * 1e-9, S21.imag, label="Im(S21)")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("S21")
    ax.legend()
    ax.grid(True)
    if drive_freq is not None:
        ax.axvline(drive_freq * 1e-9, color="r", ls="--")

    fig.tight_layout()
    fname = f"{output_prefix}_fft.png"
    fig.savefig(fname, dpi=150)
    print(f"  Plot saved to {fname}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract S21 from WarpX FieldProbe plane data (TE10 mode).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("recv_file",
                        help="Path to receiving-plane .dat file")
    parser.add_argument("--emit", default=None,
                        help="Path to emitting-plane .dat file (FFT mode only)")
    parser.add_argument("--freq", type=float, default=None,
                        help="Drive frequency in Hz")
    parser.add_argument("--amplitude", type=float, default=1.0,
                        help="Peak Ey amplitude of the TE10 drive (V/m), "
                             "used for normalisation")
    parser.add_argument("--resolution", type=int, default=50,
                        help="FieldProbe plane resolution (N for N×N grid)")
    parser.add_argument("--waveguide-a", type=float, required=True,
                        help="Waveguide broad-wall width (m), e.g. 3.7592e-3 for WR15")
    parser.add_argument("--x-center", type=float, default=None,
                        help="Waveguide centre x (m); default = midpoint of probe extent")
    parser.add_argument("--t-start", type=float, default=None,
                        help="Start of analysis window (s) — skip ramp-up transient")
    parser.add_argument("--t-end",   type=float, default=None,
                        help="End of analysis window (s)")
    parser.add_argument("--mode", choices=["lockin", "fft"], default="lockin",
                        help="'lockin': single-freq CW (fast). "
                             "'fft': full spectrum (needed for plasma frequency shifts)")
    parser.add_argument("--plot", action="store_true",
                        help="Show and save plots")
    parser.add_argument("--output", default="s21.npz",
                        help="Output .npz file")
    args = parser.parse_args()

    if args.mode == "lockin" and args.freq is None:
        parser.error("--freq is required for lock-in mode")

    # --- Parse receiving plane ---
    print(f"Parsing receiving probe: {args.recv_file}")
    times_r, coords_r, Ey_r = parse_fieldprobe(
        args.recv_file, args.resolution,
        t_start=args.t_start, t_end=args.t_end,
    )
    n_steps = len(times_r)
    dt_r    = float(np.mean(np.diff(times_r))) if n_steps > 1 else 0.0
    print(f"  {n_steps} steps, dt = {dt_r:.4e} s, "
          f"t ∈ [{times_r[0]:.4e}, {times_r[-1]:.4e}] s")
    print(f"  x ∈ [{coords_r[...,0].min():.4e}, {coords_r[...,0].max():.4e}] m")
    print(f"  y ∈ [{coords_r[...,1].min():.4e}, {coords_r[...,1].max():.4e}] m")

    a_wg     = args.waveguide_a
    x_center = args.x_center if args.x_center is not None else \
               float(0.5 * (coords_r[..., 0].min() + coords_r[..., 0].max()))
    print(f"  Waveguide: a = {a_wg:.4e} m, centre x = {x_center:.4e} m")

    if args.freq is not None:
        T_rf     = 1.0 / args.freq
        n_cycles = (times_r[-1] - times_r[0]) / T_rf
        print(f"  Window = {n_cycles:.1f} RF cycles at {args.freq*1e-9:.2f} GHz")

    # --- TE10 modal projection (vectorised) ---
    print("Computing TE10 modal projection ...")
    w0 = timer.monotonic()
    modal_recv = project_te10_timeseries(Ey_r, coords_r, x_center, a_wg)
    print(f"  Done in {timer.monotonic()-w0:.1f} s, "
          f"peak |a| = {np.max(np.abs(modal_recv)):.4e} V/m")

    # --- Emitting plane (optional) ---
    modal_emit = None
    if args.emit is not None:
        if args.mode == "lockin":
            print("Note: --emit is ignored in lock-in mode (reference is analytic)")
        else:
            print(f"Parsing emitting probe: {args.emit}")
            times_e, coords_e, Ey_e = parse_fieldprobe(
                args.emit, args.resolution,
                t_start=args.t_start, t_end=args.t_end,
            )
            n_min = min(len(times_e), n_steps)
            times_r    = times_r[:n_min]
            modal_recv = modal_recv[:n_min]
            n_steps    = n_min
            modal_emit = project_te10_timeseries(
                Ey_e[:n_min], coords_e, x_center, a_wg)
            print(f"  Peak |a_emit| = {np.max(np.abs(modal_emit)):.4e} V/m")

    # --- S21 ---
    if args.mode == "lockin":
        print(f"\nLock-in at {args.freq*1e-9:.4f} GHz ...")
        s21_val  = lockin_s21(times_r, modal_recv, args.freq, args.amplitude)
        mag      = abs(s21_val)
        mag_dB   = 20 * np.log10(mag + 1e-30)
        phase_deg = np.degrees(np.angle(s21_val))
        print(f"  S21        = {s21_val.real:+.6f} {s21_val.imag:+.6f}j")
        print(f"  |S21|      = {mag:.6f}  ({mag_dB:.2f} dB)")
        print(f"  phase(S21) = {phase_deg:.2f} deg")

        np.savez(args.output,
                 freq=args.freq,
                 S21=s21_val,
                 S21_mag=mag,
                 S21_phase_deg=phase_deg,
                 times=times_r,
                 modal_recv=modal_recv)

        if args.plot:
            plot_modal_timeseries(times_r, modal_recv, args.freq,
                                  Path(args.output).stem)

    else:  # fft
        print("\nComputing FFT-based S21 ...")
        freqs, S21 = compute_s21_fft(
            times_r, modal_recv, modal_emit,
            drive_freq=args.freq, drive_amplitude=args.amplitude,
        )
        if args.freq is not None:
            idx = np.argmin(np.abs(freqs - args.freq))
            s  = S21[idx]
            print(f"  S21 at {freqs[idx]*1e-9:.4f} GHz: "
                  f"|S21| = {abs(s):.4f} ({20*np.log10(abs(s)+1e-30):.2f} dB), "
                  f"phase = {np.degrees(np.angle(s)):.2f} deg")

        np.savez(args.output,
                 freqs=freqs, S21=S21,
                 times=times_r,
                 modal_recv=modal_recv,
                 modal_emit=modal_emit)

        if args.plot:
            prefix = Path(args.output).stem
            plot_modal_timeseries(times_r, modal_recv, args.freq, prefix)
            plot_fft_results(freqs, S21, args.freq, prefix)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
