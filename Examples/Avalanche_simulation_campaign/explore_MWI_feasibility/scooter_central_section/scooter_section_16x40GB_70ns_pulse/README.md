# scooter_section_16x40GB_70ns — long pulse run, did not finish

Sister of `../scooter_section_16x40GB_1ns/`: same scooter STL, +X
propagation, 16 GPUs × 40 GB layout.  Configured to run for 70 ns so we
can see the pulse arrive at the receiver and look at multi-bounce
behaviour.

## Differences vs. scooter_section_16x40GB_1ns
- `t_sim_override`: 1 ns → **70 ns**.
- `receiving_plane` slab thickened by ±1 cell along Y and Z
  (`±(wg_half_a + dy)` / `±(wg_half_b + dz)` instead of `±wg_half_*`)
  and switched to **variable-based BP5 encoding (`warpx_openpmd_encoding='v'`)**
  so that all 14k+ iterations land in a single `openpmd.bp5` file
  rather than tens of thousands of per-step files.
- `fields_sliced` slice plane moved from `Z = 0` to `Z =
  recv_z_center` so the visualization tracks the propagation channel.

## Run status
- **Did not run to completion.**  `t_sim_override = 70 ns` corresponds
  to ~164 k steps; both attempts were killed well before that:
  - `output_51244411.txt` / `WarpX.e51244411`: SIGTERM at STEP 8654,
    `TIME ≈ 3.69 ns`.
  - `output_51245689.txt` / `WarpX.e51245689`: Slurm `TIME LIMIT` at
    STEP 14984, `TIME ≈ 6.39 ns`.
- The available data covers only the first few ns of the intended
  70 ns simulation.  Quantitative S21 from this run is therefore
  preliminary; treat the analysis output as exploratory.

## Reproducing the analysis

The analysis run for this case is what `analysis_run.log` captures.
Use `--max-iterations` if `read_linear` hangs at the end of the
truncated series, and `--openpmd-access-mode read_only` if the
streaming reader cannot find an end-of-stream marker.

```bash
python ../../analysis/analyze_s21_directional_splitting.py \
    --diag-dir ./diags \
    --analytic-ref \
    --emit-pos -0.137189 \
    --pulse-t-peak 0.13e-9 --pulse-fwhm 0.05e-9 \
    --geometry complex \
    --mode pulse \
    --openpmd-access-mode read_only
```
