# embedded_boundary_waveguide / v2 — tuned EB, face-aligned grid

The "good" embedded-boundary case.  Wholly rewritten relative to v1 to
make the EB approach competitive (in cell count and accuracy) with the
PEC reference in `../../domain_boundary_waveguide/v2/`.

## Differences vs. v1
- **Cell-face-aligned EB walls**: 16 cells across the broad wall, 8
  across the narrow wall (so `dx == dy`, walls land on cell faces).
- **Compact transverse domain**: 8-cell EB conductor buffer outside the
  walls instead of a full multi-wavelength padding (much less wasted
  EB-covered volume).
- **Dirichlet** in X/Y instead of periodic (legitimate now that the EB
  tube spans the whole Z extent).
- **Pulse mode added** (Gaussian, t_peak=0.13 ns, FWHM=50 ps), CW path
  retained.
- Antenna `emit_z = -13.000 cm`; injection plane at `inject_z = -12.272 cm`;
  receiver at `recv_z = +12.591 cm`.
- Both **`injection_plane`** and **`receiving_plane`** openPMD slabs +
  matching FieldProbe planes.
- Adds `XZ` `fields_sliced` for visualization, plus a windowed
  full-domain `fields` snapshot.
- PML thickened to `pml_ncells = 16`, Z buffer of 10 mm between each
  port and the PML.
- `t_sim_override = 10 ns`.

## Run status
- Completed (`AMReX … finalized`).  See `output_51648400.log`.

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
