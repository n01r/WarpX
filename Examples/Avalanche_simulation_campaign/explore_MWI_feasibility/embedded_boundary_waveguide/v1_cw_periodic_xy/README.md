# embedded_boundary_waveguide / v1 — naive EB tube, periodic X/Y

First attempt at modelling a WR-15 waveguide with an embedded boundary
defined by an *implicit function* (no STL).  The EB conductor is the
exterior of a rectangular tube starting 5 guide-wavelengths upstream of
the antenna and extending to +Z infinity.

## Key parameters
- Maxwell solver: Yee
- Resolution: 20 cells / wavelength
- Domain X×Y = 10 × 6 mm with **periodic** transverse boundaries (the EB
  tube is what confines the mode, not the domain edges)
- Propagation axis: **+Z**
- Excitation: CW with 20-period Gaussian ramp-up
- Antenna position: `emit_z = -12.8 cm`
- Diagnostics: `receiving_plane` slab + `recv_probe` FieldProbe at
  `+12.8 cm`.  No injection_plane.
- `n_guide_wavelengths = 1000` → effectively unbounded run length
  (simulation duration set this way was unrealistic and was reduced in v2).

## Why this case exists / known issues
- Demonstrates that the EB-tube approach works but wastes a lot of cells
  on EB-covered conductor outside the guide.
- No injection plane → analysis must use `--analytic-ref`.
- No run output committed.

## Reproducing the analysis

```bash
python ../../analysis/analyze_s21_directional_splitting.py \
    --diag-dir ./diags \
    --analytic-ref \
    --emit-pos -0.128 \
    --pulse-t-peak 0.13e-9 --pulse-fwhm 0.05e-9 \
    --geometry straight \
    --mode pulse
```
