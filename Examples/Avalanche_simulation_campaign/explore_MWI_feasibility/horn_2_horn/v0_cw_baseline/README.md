# horn_2_horn / v0 — STL horn-to-horn baseline (CW)

First end-to-end horn-to-horn case: full STL of the cleaned MWI horn
pair (`./stl_input/cleaned_mwi_horns.STL`) embedded in a 30 × 30 × 280 mm
vacuum domain with PML on all six faces.

## Key parameters
- Maxwell solver: Yee
- Resolution: 20 cells / wavelength
- Propagation axis: **+Z**
- Excitation: CW with 20-period Gaussian ramp-up, `n_guide_wavelengths =
  1000` (i.e. effectively unbounded simulation duration).
- Antenna position: `emit_horn_z_pos = -12.8 cm` (inside the rectangular
  feed); receiving plane at the symmetric `+12.8 cm` location inside
  the receive horn.
- Diagnostics: `receiving_plane` slab + `recv_probe` FieldProbe.  No
  injection plane.
- Run status: not preserved (no run output committed).

## Differences vs. later versions

This is the baseline.  See `../v1/README.md` for the cleanup pass that
introduced pulse mode and made all parameters configurable.

## Reproducing the analysis

```bash
python ../../analysis/analyze_s21_directional_splitting.py \
    --diag-dir ./diags \
    --analytic-ref \
    --emit-pos -0.128 \
    --pulse-t-peak 0.13e-9 --pulse-fwhm 0.05e-9 \
    --geometry complex \
    --mode pulse
```
