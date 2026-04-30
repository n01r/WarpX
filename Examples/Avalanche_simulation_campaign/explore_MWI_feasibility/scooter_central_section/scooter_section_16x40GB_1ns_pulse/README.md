# scooter_section_16x40GB_1ns — short-pulse, runs to completion

First scooter-section case: the central scooter STL
(`./stl_input/scooter_section.STL`, X-bbox ±13.7 cm) with **propagation
along +X** and the WR-15 broad wall along Y (Ez polarisation).
Configured for 16 GPUs × 40 GB.

## Key parameters
- Maxwell solver: Yee
- Resolution: 20 cells / wavelength (`dx ≈ 224 μm`)
- Propagation axis: **+X** (broad wall along Y, narrow wall along Z)
- Excitation: pulse (t_peak = 0.13 ns, FWHM = 50 ps)
- Domain: X [-0.150, +0.150], Y [-0.112, +0.112], Z [-0.026, +0.100] m;
  PML on all six faces (the EB touches the Z PMLs).
- Antenna position: `emit_x = -13.72 cm`, port centre `(y, z) =
  (3.0 cm, 3.7 cm)`; receiver port symmetric at `recv_x = +13.72 cm`.
- Diagnostics:
  - `receiving_plane` slab (Y-Z plane at `recv_x`, slab thickness =
    one cell along X)
  - `recv_probe` FieldProbe on the same plane
  - **No injection_plane** diag (analysis uses `--analytic-ref`)
  - `fields_sliced` XY slice at `Z = 0` for visualization
  - Full-domain `fields` snapshot every 0.15 ns, between 0 and 20 ns

## Run status
- Completed at STEP 2347, `TIME = 1.000 ns` (`t_sim_override = 1 ns`,
  `AMReX … finalized` in `output_51239814.txt`).

## Reproducing the analysis

```bash
python ../../analysis/analyze_s21_directional_splitting.py \
    --diag-dir ./diags \
    --analytic-ref \
    --emit-pos -0.137189 \
    --pulse-t-peak 0.13e-9 --pulse-fwhm 0.05e-9 \
    --geometry complex \
    --mode pulse
```
