# embedded_boundary_waveguide / v3 — pml=32, unstable (kept as a warning)

Identical script to v2 except for the PML / Z-buffer tuning.  Kept in
the campaign because the failure mode is informative.

## Differences vs. v2
- `pml_ncells = 16` → **`32`** (doubled).
- Z buffer between each port and the PML increased from `10 mm` →
  `50 mm` so the direct pulse and the first PML reflection are
  separated in time at both diagnostic planes.
- `inject_z` and `recv_z` are unchanged, so the S21 propagation length
  matches v2.

## Run status
- Reached its `t_sim` (`AMReX … finalized` in `output_51648400.log`),
  **but the simulation was unstable**: the field energy grew
  exponentially.  Root cause: too many PML cells for this setup.  Use
  v2 for any quantitative comparison; this case is retained as a
  cautionary data point.

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

