# scooter_section_16x40GB_1ns_continuous_wave — CW twin of the 1 ns pulse case

Same script as `../scooter_section_16x40GB_1ns/`, with one parameter
change.

## Differences vs. scooter_section_16x40GB_1ns
- `excitation_mode = "pulse"` → **`"cw"`** (Gaussian-ramp continuous
  wave at 67 GHz; everything else identical).

Resolution, solver, geometry, port positions, diagnostics, and
`t_sim_override = 1 ns` are unchanged.

## Run status
- Completed at STEP 2347, `TIME = 1.000 ns`
  (`output_51240016.txt`).  Note: 1 ns is too short to see any S21
  steady state; this case is mostly a sanity check on CW injection.

## Reproducing the analysis

```bash
python ../../analysis/analyze_s21_directional_splitting.py \
    --diag-dir ./diags \
    --analytic-ref \
    --emit-pos -0.137189 \
    --geometry complex \
    --mode cw
```
