# horn_2_horn / v1 — short pulse, recv-only, runs to completion

Same STL geometry as v0 (30 × 30 × 280 mm vacuum, PML on all faces, MWI
horn STL).  Reorganised input deck following the
`../../domain_boundary_waveguide/v2/` template.

## Differences vs. v0
- **Pulse excitation by default** (`pulse_t_peak = 0.13 ns`,
  `pulse_fwhm = 50 ps`); CW path retained.
- All knobs lifted into a single user-parameter block (resolution, PML,
  domain, port positions, diagnostic cadence, STL path, etc.).
- `t_sim_override = 15 ns` (replaces v0's `1000 λ_g` open-ended
  duration).
- Antenna nudged upstream from `-12.8 cm` to `emit_horn_z_pos =
  -13.0 cm` so the TE10 mode has more room to settle; receiver kept at
  `+13.0 cm`.
- Diagnostics: full-domain `fields` snapshot every 0.1 ns, plus
  `receiving_plane` slab + `recv_probe` FieldProbe.  Still **no
  injection plane** in this case (analysis uses `--analytic-ref`).

## Run status
- Completed at STEP 35364, `TIME = 15.00 ns` (`AMReX … finalized` in
  `output.txt`).

## Reproducing the analysis

Use `--geometry complex` because the analytic straight-guide curve is
not the expected answer here (horn taper, free-space propagation, etc.).

```bash
python ../../analysis/analyze_s21_directional_splitting.py \
    --diag-dir ./diags \
    --analytic-ref \
    --emit-pos -0.130 \
    --pulse-t-peak 0.13e-9 --pulse-fwhm 0.05e-9 \
    --geometry complex \
    --mode pulse
```
