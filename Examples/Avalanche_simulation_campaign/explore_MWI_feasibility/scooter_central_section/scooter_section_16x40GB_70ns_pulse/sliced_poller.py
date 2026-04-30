"""
sliced_poller.py
Run in a SEPARATE terminal alongside the notebook, NOT inside Jupyter.
Reads fields_sliced with read_linear (the only safe access mode for
variable-based BP5), caches each iteration as a .npz file in CACHE_DIR,
and writes a small JSON manifest so the notebook knows what's available.

Usage:
    python sliced_poller.py
"""

import json
import os
import time
import numpy as np
import openpmd_api as io

DIAG_PATH  = "./diags/fields_sliced/openpmd.bp5/"
CACHE_DIR  = "./diags/fields_sliced_cache/"
MANIFEST   = os.path.join(CACHE_DIR, "manifest.json")
SLICE_AX   = 0    # WarpX writes (Nz, Ny, Nx) — axis 0 is the thin Z slab

os.makedirs(CACHE_DIR, exist_ok=True)

print(f"Polling {DIAG_PATH}")
print(f"Writing cache to {CACHE_DIR}")
print("Ctrl-C to stop.\n")

manifest = {}

# read_linear is a blocking forward-only pass — it will sit here
# and yield iterations as WarpX writes them, until the sim ends.
series  = io.Series(DIAG_PATH, io.Access.read_linear)
eb_done = False

for it in series.read_iterations():
    i = it.iteration_index

    mesh   = it.meshes["E"]
    comp   = mesh["z"]
    gs     = mesh.grid_spacing
    offset = mesh.grid_global_offset

    raw  = comp.load_chunk()
    it.series_flush()
    raw  = raw * comp.unit_SI
    arr  = raw.mean(axis=SLICE_AX)          # (Nx, Ny)

    axes = [a for a in range(raw.ndim) if a != SLICE_AX]
    ax0, ax1 = axes
    x0 = offset[ax0]; x1 = x0 + gs[ax0] * arr.shape[0]
    y0 = offset[ax1]; y1 = y0 + gs[ax1] * arr.shape[1]
    t  = float(it.time)

    # eb_covered — static, save once
    eb_path = os.path.join(CACHE_DIR, "eb_covered.npy")
    if not eb_done and "eb_covered" in it.meshes:
        eb_comp = it.meshes["eb_covered"][io.Mesh_Record_Component.SCALAR]
        eb_raw  = eb_comp.load_chunk()
        it.series_flush()
        eb_arr  = eb_raw.mean(axis=SLICE_AX)
        np.save(eb_path, eb_arr)
        eb_done = True

    # Save field as .npz
    cache_file = os.path.join(CACHE_DIR, f"iter_{i:06d}.npz")
    np.savez(cache_file, arr=arr, extent=[x0, x1, y0, y1], t=t)

    # Update manifest atomically
    manifest[str(i)] = {"file": cache_file, "t": t}
    tmp = MANIFEST + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f)
    os.replace(tmp, MANIFEST)   # atomic on POSIX

    print(f"  iter {i:6d}  t = {t*1e9:.4f} ns", flush=True)

del series
print("Sim finished — all iterations cached.")