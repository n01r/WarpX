# domain_boundary_waveguide / v2 — Yee, with injection plane and pulse mode

Same PEC straight-guide geometry as v1, but heavily restructured.  This
is the version that defined the user-parameter / diagnostic layout used
by every later case.

## Differences vs. v1
- **Resolution dropped to 20 cells / wavelength** (v1 used 64).
- **Pulse mode added** (`excitation_mode = "pulse"`, t_peak / FWHM
  matching VIAS3D).  CW with ramp-up still selectable.
- **Injection-plane diagnostic added** (`name='injection_plane'` at
  `inject_z = -12.272 cm`), so S21 is computed from the simulated
  incident wave rather than an analytic reference.
- **Antenna shifted upstream** to `emit_z = -13.000 cm` so the TE10 mode
  has room to settle before crossing the injection plane.  Receiver at
  `recv_z = +12.591 cm`.
- Adds an `XZ` `fields_sliced` diagnostic (variable-based BP5) for
  visualization, plus a windowed full-domain `fields` snapshot
  (`field_snapshot_t_start = 1.5 ns`).
- Particle shape bumped from 1 → 2.
- Async ADIOS2 BP5 buffering enabled for `injection_plane`,
  `receiving_plane`, and `fields_sliced`.

## Run status
- Completed (`AMReX … finalized`).  See `output_51563934 (1).log`.

## Reproducing the analysis

```bash
python ../../analysis/analyze_s21_directional_splitting.py \
    --diag-dir ./diags \
    --emit-pos -0.13000 \
    --recv-pos  0.12591 \
    --pulse-t-peak 0.13e-9 --pulse-fwhm 0.05e-9 \
    --geometry straight \
    --mode pulse
```
