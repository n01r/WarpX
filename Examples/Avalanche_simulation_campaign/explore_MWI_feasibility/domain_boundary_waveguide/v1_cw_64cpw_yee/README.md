# domain_boundary_waveguide / v1 — first PEC straight-guide CW pass

Geometry: straight WR-15 waveguide, no STL, no embedded boundary.  Domain
X/Y is exactly the WR-15 cross-section so the PEC (`dirichlet`) domain
walls *are* the waveguide walls.  PML on Z only.

## What this case is

Initial proof-of-concept for the TE10 mode-injection / port-diagnostic
chain, before any horn / STL geometry was introduced.

## Key parameters
- Maxwell solver: Yee
- Resolution: **64 cells / wavelength** (deliberately over-resolved for a
  first pass; later versions drop to 20 cpw)
- Propagation axis: **+Z**
- Excitation: CW with 20-period Gaussian ramp-up (no pulse mode yet)
- Antenna position: `emit_z = -12.8 cm`
- Diagnostics:
  - `receiving_plane` openPMD slab at `+12.8 cm`
  - `recv_probe` FieldProbe on the same plane
  - **No injection_plane diag** — the analysis run for this case has to
    use `--analytic-ref` to substitute an analytic Gaussian incident
    reference at the emitter port.
- Run status: not preserved (no run log committed).

## Differences vs. later versions

This is the baseline.  See `../v2/README.md` for the cleanup pass that
followed.

## Reproducing the analysis

Use the top-level unified script
`Examples/Avalanche_simulation_campaign/explore_MWI_feasibility/analysis/analyze_s21_directional_splitting.py`.
The local copy in `analysis/` (if any) is older.

```bash
python ../../analysis/analyze_s21_directional_splitting.py \
    --diag-dir ./diags \
    --analytic-ref \
    --emit-pos -0.128 \
    --pulse-t-peak 0.13e-9 --pulse-fwhm 0.05e-9 \
    --geometry straight \
    --mode pulse
```
