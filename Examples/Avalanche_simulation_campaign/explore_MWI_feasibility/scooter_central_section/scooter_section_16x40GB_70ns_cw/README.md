# scooter_section_16x40GB_70ns_continuous_wave — long CW run, did not finish

CW twin of `../scooter_section_16x40GB_70ns/`.  Same scooter STL, same
domain, same diagnostics, same 70 ns target.  Intended for the I/Q
demodulation analysis path (`--mode cw`).

## Differences vs. scooter_section_16x40GB_70ns
- `excitation_mode = "pulse"` → **`"cw"`** (Gaussian-ramp continuous
  wave; everything else identical).

## Run status
- **Did not run to completion.**  Killed by Slurm `TASK FAILURE` at
  STEP 14951, `TIME ≈ 6.37 ns` (`output_51246074.txt`,
  `WarpX.e51246074`).  Of the planned 70 ns, only the first ~6.4 ns
  were written.
- The CW ramp-up is `cw_ramp_periods = 20` RF periods ≈ 0.30 ns, so the
  drive is fully on for most of the recorded interval — but the run
  stopped well before any S21 steady state could be established.

## Reproducing the analysis

```bash
python ../../analysis/analyze_s21_directional_splitting.py \
    --diag-dir ./diags \
    --analytic-ref \
    --emit-pos -0.137189 \
    --geometry complex \
    --mode cw \
    --openpmd-access-mode read_only
```
