# horn_2_horn / v1_more_slice_diag_outputs — visualization-only XZ slice run

Same input script as `../v1/`, repurposed as a movie-generation run:
keep the STL/horn geometry, the pulse, and 20 cells/wavelength, but
**replace the recv-plane diagnostics with a high-cadence XZ slice** so
we can render the pulse propagating through the horns.

## Differences vs. v1
- `t_sim_override = 15 ns` → **`2.2 ns`** (cut to just the pulse
  transit; the actual run was eventually carried further before being
  interrupted).
- `field_snapshot_dt = 0.1 ns` → **`0.005 ns`** (every ~12 steps; many
  more frames for the movie).
- `receiving_plane` slab and `recv_probe` FieldProbe **removed**.
- Replaced by a single `fields_sliced` full-domain XZ slice at `Y = 0`
  (variable-based BP5 encoding, single file across all iterations).

## Run status
- Interrupted with SIGINT at STEP 9324, `TIME ≈ 3.95 ns` (see
  `output.txt`).  This was a deliberate kill once enough movie frames
  had been written; the run was not driven to a "finished" state.

## Reproducing

This case does **not** produce S21 data (no port slabs).  It is
visualization-only; see `laser_propagation.ipynb` and the analysis
notebooks in `analysis/` for the movie pipeline.
