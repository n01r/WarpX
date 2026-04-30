# domain_boundary_waveguide / v2_ckc — same as v2 but with CKC solver

Sibling of `../v2/`.  All physics, geometry, resolution, antenna /
injection / receiver positions, and diagnostics are identical.  Only the
Maxwell solver changes:

- `method='Yee'` → **`method='CKC'`** (Cole–Kärkkäinen–Cowan).

CKC has fourth-order phase-velocity error along the Cartesian axes
(`O(dx^4)` vs. `O(dx^2)` for Yee), so it is the cheap-resolution
benchmark for whether numerical dispersion is biasing the S21 phase.
CKC is compatible with the PEC (`dirichlet`) boundaries used here.

## Differences vs. v2
- Solver: Yee → CKC (everything else unchanged).

## Run status
- Completed (`AMReX … finalized`).  See `output_51563934.log`.

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
