"""
Utilities for reading and plotting variable-based WarpX/openPMD ADIOS2 output
in linear mode.

Designed for crash-truncated variable-based ADIOS2 output where random access
and "read until EOF" are unsafe.

Typical Jupyter usage
---------------------

from warpx_openpmd_movie import (
    read_field_linear_capped,
    plot_warpx_slice_imshow,
    save_series_frames_linear,
)

arr, meta = read_field_linear_capped(
    "diags/diag1/openpmd.bp",
    target_iteration=14928,
    field="E/z",
    max_linear_steps=1245,
    verbose=False,
)

fig, ax, cax, im = plot_warpx_slice_imshow(
    arr,
    meta,
    slice_axis=0,
    slice_reduction="mean",
    autoscale_percentile=99.5,
    colorbar_unit="V/m",
)

n_saved = save_series_frames_linear(
    "diags/diag1/openpmd.bp",
    field="E/z",
    outdir="movie_frames_Ez",
    max_linear_steps=1245,
    every=1,
    slice_axis=0,
    slice_reduction="mean",
    autoscale_percentile=99.5,
    colorbar_unit="V/m",
    verbose=True,
)
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import openpmd_api as io


Meta = Dict[str, Any]
__module_version__ = "v6-derived-magnitude-parser-fix"



def _read_linear_access():
    """
    Support openPMD-api enum spelling differences across versions.
    """
    for enum_name in ("Access_Type", "Access"):
        enum = getattr(io, enum_name, None)
        if enum is not None and hasattr(enum, "read_linear"):
            return enum.read_linear
    raise RuntimeError("Could not find openPMD-api read_linear access enum.")


def _get_attr_or_call(obj: Any, names: Iterable[str], default: Any = None) -> Any:
    """
    Robustly support Python-style properties and C++-style method names.
    """
    for name in names:
        if not hasattr(obj, name):
            continue
        value = getattr(obj, name)
        try:
            return value() if callable(value) else value
        except Exception:
            pass
    return default


def _as_tuple_or_default(value: Any, n: int, default: float) -> Tuple[float, ...]:
    """
    Convert scalar or iterable metadata to a tuple of length n.
    """
    if value is None:
        return tuple(float(default) for _ in range(n))

    if np.isscalar(value):
        return tuple(float(value) for _ in range(n))

    try:
        value = tuple(value)
    except TypeError:
        return tuple(float(value) for _ in range(n))

    if len(value) == 1 and n > 1:
        return tuple(float(value[0]) for _ in range(n))

    if len(value) != n:
        raise ValueError(f"Expected metadata length {n}, got {len(value)}")

    return tuple(float(x) for x in value)



def _parse_vector_magnitude_request(field: str) -> Optional[str]:
    """
    Recognize derived vector-magnitude field requests and return the base record.

    Accepted examples:
      - "E_mag"
      - "E_abs"
      - "E_magnitude"
      - "E/abs"
      - "E/mag"
      - "E/magnitude"
      - "abs(E)"
      - "norm(E)"
      - "|E|"

    Returns None for stored fields such as "E/z".
    """
    token = str(field).strip()

    if token.startswith("|") and token.endswith("|") and len(token) > 2:
        base = token[1:-1].strip()
        return base or None

    for func in ("abs", "norm", "mag", "magnitude"):
        prefix = f"{func}("
        if token.startswith(prefix) and token.endswith(")"):
            base = token[len(prefix):-1].strip()
            return base or None

    if "/" in token:
        base, suffix = token.split("/", 1)
        if suffix.strip() in ("abs", "mag", "magnitude", "norm"):
            base = base.strip()
            return base or None
        return None

    for suffix in ("_mag", "_abs", "_magnitude"):
        if token.endswith(suffix) and len(token) > len(suffix):
            base = token[:-len(suffix)].strip()
            return base or None

    return None


def _field_base_name(field: str) -> str:
    """
    Return the physical base record name for regular or derived fields.
    """
    base = _parse_vector_magnitude_request(field)
    if base is not None:
        return base
    return str(field).split("/", 1)[0].strip()


def _default_display_name(field: str) -> str:
    """
    Nicely formatted default field label.
    """
    base = _parse_vector_magnitude_request(field)
    if base is not None:
        return f"|{base}|"
    return str(field)


def _default_prefix(field: str) -> str:
    """
    Safe filename prefix derived from field spec.
    """
    base = _parse_vector_magnitude_request(field)
    if base is not None:
        return f"{base}mag"

    out = str(field).replace("/", "")
    out = "".join(ch for ch in out if ch.isalnum() or ch in ("_", "-"))
    return out or "frame"


def _try_mesh_component(mesh: Any, component: str):
    """
    Return mesh[component] if it exists, otherwise None.

    Some openPMD-api Python builds expose Mesh as an iterable/mapping,
    while others only support direct indexing such as mesh["x"].
    """
    try:
        return mesh[component]
    except Exception:
        return None


def _iter_mesh_component_names(mesh: Any):
    """
    Return component names from openPMD-api's mesh iteration interface.

    For file-based datasets this is the equivalent of:
        list(it.meshes["E"])
    which commonly returns:
        ["x", "y", "z"]

    This is the preferred way to discover record components when available.
    """
    try:
        return [str(k) for k in list(mesh)]
    except Exception:
        return []


def _keys_mesh_component_names(mesh: Any):
    """
    Return component names from mesh.keys(), for openPMD-api builds that
    expose a mapping-like interface.
    """
    if not hasattr(mesh, "keys"):
        return []

    try:
        return [str(k) for k in mesh.keys()]
    except Exception:
        return []


def _mesh_axis_labels(mesh: Any):
    """
    Return mesh.axis_labels when available.

    These are spatial array-axis labels, e.g. ["z", "y", "x"], not formally
    record component names. They are therefore only used as a fallback hint.
    """
    labels = _get_attr_or_call(mesh, ["axis_labels", "axisLabels"], None)
    if labels is None:
        return []
    return [str(label) for label in labels]


def _dedupe_preserve_order(names):
    out = []
    seen = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def _mesh_component_names(mesh: Any, *, include_axis_label_fallback: bool = True):
    """
    Discover component names for an openPMD Mesh.

    Priority:
      1. list(mesh), e.g. list(it.meshes["E"]) -> ["x", "y", "z"]
      2. mesh.keys(), if the local openPMD-api build supports it
      3. direct probing of common component names
      4. optional axis-label fallback, probed as component names

    axis_labels are spatial dimension labels, not guaranteed component names,
    so they are intentionally below list(mesh)/keys().
    """
    names = _iter_mesh_component_names(mesh)
    if names:
        return names

    names = _keys_mesh_component_names(mesh)
    if names:
        return names

    candidates = [
        "x", "y", "z",
        "r", "t",
        "0", "1", "2",
        "scalar",
    ]

    if include_axis_label_fallback:
        candidates.extend(_mesh_axis_labels(mesh))

    candidates = _dedupe_preserve_order(candidates)

    found = []
    for comp in candidates:
        if _try_mesh_component(mesh, comp) is not None:
            found.append(comp)

    return found


def _resolve_mesh_component(iteration: Any, field: str):
    """
    Resolve regular field strings such as:
      "rho"
      "E/x"
      "E/z"
      "B/y"

    Returns
    -------
    mesh, record_component, mesh_name, component
    """
    if "/" in field:
        mesh_name, component = field.split("/", 1)
    else:
        mesh_name, component = field, None

    mesh = iteration.meshes[mesh_name]

    if component is not None:
        rc = mesh[component]
        return mesh, rc, mesh_name, component

    if hasattr(mesh, "load_chunk") and hasattr(mesh, "shape"):
        return mesh, mesh, mesh_name, None

    comps = _mesh_component_names(mesh)
    if len(comps) == 1:
        component = comps[0]
        rc = mesh[component]
        return mesh, rc, mesh_name, component

    raise ValueError(
        f"Field {mesh_name!r} has components {comps}. "
        f"Use e.g. {mesh_name}/x, {mesh_name}/y, or {mesh_name}/z."
    )


def _resolve_vector_components(iteration: Any, base_field: str):
    """
    Resolve vector components for a magnitude request such as |E|.

    The preferred component discovery is:
        list(iteration.meshes[base_field])

    For E this should return something like ["x", "y", "z"]. If the
    openPMD-api build does not expose iteration over Mesh, we fall back to
    direct probing of common component names.
    """
    mesh = iteration.meshes[base_field]

    components = _mesh_component_names(mesh, include_axis_label_fallback=False)

    # For vector magnitudes, avoid scalar-like components if they appear.
    vector_like = [
        comp for comp in components
        if comp not in ("scalar",)
    ]
    components = vector_like if vector_like else components

    if not components:
        raise ValueError(
            f"Could not discover components for vector field {base_field!r}. "
            f"Try checking list(it.meshes[{base_field!r}]) or reading an explicit "
            f"component such as {base_field}/x first."
        )

    return mesh, components

def _meta_per_dim(meta: Meta, key: str) -> Tuple[float, ...]:
    """
    Return a tuple of length ndim even if meta[key] is scalar.
    """
    ndim = len(meta["axis_labels"])
    return _as_tuple_or_default(meta.get(key, None), ndim, 1.0)


def copy_meta_from_iteration(
    it: Any,
    mesh: Any,
    rc: Any,
    field: str,
    mesh_name: str,
    component: Optional[str],
    offset_use: Tuple[int, ...],
    extent_use: Tuple[int, ...],
) -> Meta:
    """
    Copy all metadata needed for plotting before the iteration is closed.
    """
    ndim = len(tuple(rc.shape))

    axis_labels = _get_attr_or_call(mesh, ["axis_labels", "axisLabels"], None)
    if axis_labels is None:
        axis_labels = tuple(f"dim{d}" for d in range(ndim))
    else:
        axis_labels = tuple(str(x) for x in axis_labels)

    grid_spacing = _get_attr_or_call(mesh, ["grid_spacing", "gridSpacing"], None)
    grid_spacing = _as_tuple_or_default(grid_spacing, ndim, 1.0)

    grid_global_offset = _get_attr_or_call(
        mesh,
        ["grid_global_offset", "gridGlobalOffset"],
        None,
    )
    grid_global_offset = _as_tuple_or_default(grid_global_offset, ndim, 0.0)

    grid_unit_SI = _get_attr_or_call(
        mesh,
        ["grid_unit_SI", "gridUnitSI", "grid_unit_si"],
        1.0,
    )
    # Keep scalar if scalar, but axis helpers handle both scalar and tuple.
    if not np.isscalar(grid_unit_SI):
        grid_unit_SI = tuple(float(x) for x in grid_unit_SI)
    else:
        grid_unit_SI = float(grid_unit_SI)

    position = _get_attr_or_call(rc, ["position"], None)
    position = _as_tuple_or_default(position, ndim, 0.0)

    unit_SI = _get_attr_or_call(rc, ["unit_SI", "unitSI", "unit_si"], 1.0)

    time = _get_attr_or_call(it, ["time"], 0.0)
    time_unit_SI = _get_attr_or_call(
        it,
        ["time_unit_SI", "timeUnitSI", "time_unit_si"],
        1.0,
    )

    return {
        "iteration": int(it.iteration_index),
        "field": field,
        "mesh_name": mesh_name,
        "component": component,
        "shape": tuple(int(x) for x in rc.shape),
        "offset": tuple(int(x) for x in offset_use),
        "extent": tuple(int(x) for x in extent_use),
        "axis_labels": axis_labels,
        "grid_spacing": tuple(float(x) for x in grid_spacing),
        "grid_global_offset": tuple(float(x) for x in grid_global_offset),
        "grid_unit_SI": grid_unit_SI,
        "position": tuple(float(x) for x in position),
        "unit_SI": float(unit_SI),
        "time": float(time),
        "time_unit_SI": float(time_unit_SI),
    }


def axis_centers(meta: Meta, dim: int) -> np.ndarray:
    """
    Physical coordinates of cell centers along one loaded dimension.

    Includes:
      - loaded chunk offset
      - grid_global_offset
      - component position / staggering
      - grid_spacing
      - grid_unit_SI
    """
    spacing = _meta_per_dim(meta, "grid_spacing")[dim]
    global_offset = _meta_per_dim(meta, "grid_global_offset")[dim]
    grid_unit_SI = _meta_per_dim(meta, "grid_unit_SI")[dim]
    position = _meta_per_dim(meta, "position")[dim]

    i0 = int(meta["offset"][dim])
    n = int(meta["extent"][dim])

    centers = global_offset + (i0 + np.arange(n) + position) * spacing
    return centers * grid_unit_SI


def axis_edges(meta: Meta, dim: int) -> np.ndarray:
    """
    Physical coordinates of cell edges along one loaded dimension.
    Useful for pcolormesh and for imshow extent.
    """
    c = axis_centers(meta, dim)

    if c.size == 1:
        spacing = _meta_per_dim(meta, "grid_spacing")[dim]
        grid_unit_SI = _meta_per_dim(meta, "grid_unit_SI")[dim]
        dx = spacing * grid_unit_SI
        return np.array([c[0] - 0.5 * dx, c[0] + 0.5 * dx])

    dx = np.diff(c)
    return np.concatenate((
        [c[0] - 0.5 * dx[0]],
        0.5 * (c[:-1] + c[1:]),
        [c[-1] + 0.5 * dx[-1]],
    ))


def imshow_extent_from_meta(meta: Meta, xdim: int, ydim: int):
    """
    Return imshow extent=[xmin, xmax, ymin, ymax] from metadata.
    """
    x_edges = axis_edges(meta, xdim)
    y_edges = axis_edges(meta, ydim)
    return [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]


def infer_field_unit(field: str) -> str:
    """
    Light convenience mapping for common WarpX SI field units.
    Override by passing colorbar_unit explicitly.
    """
    base = _field_base_name(field)

    if base == "E":
        return "V/m"
    if base == "B":
        return "T"
    if base in ("j", "J"):
        return "A/m$^2$"
    if base in ("rho", "charge_density", "chargeDensity"):
        return "C/m$^3$"

    return ""


def _prepare_offset_extent(shape, offset=None, extent=None):
    """
    Normalize offset/extent for a loaded record component.
    """
    ndim = len(shape)

    if offset is None:
        offset_use = tuple(0 for _ in range(ndim))
    else:
        offset_use = tuple(int(x) for x in offset)

    if extent is None:
        extent_use = tuple(int(x) for x in shape)
    else:
        extent_use = tuple(int(x) for x in extent)

    if len(offset_use) != ndim:
        raise ValueError(f"offset has length {len(offset_use)}, expected {ndim}")

    if len(extent_use) != ndim:
        raise ValueError(f"extent has length {len(extent_use)}, expected {ndim}")

    return offset_use, extent_use


def _load_regular_field(it, series, field, offset=None, extent=None, verbose=False):
    """
    Load a regular stored field component, e.g. E/z.
    """
    mesh, rc, mesh_name, component = _resolve_mesh_component(it, field)

    full_shape = tuple(int(x) for x in rc.shape)
    offset_use, extent_use = _prepare_offset_extent(full_shape, offset, extent)

    if verbose:
        print(f"Reading {field} at iteration {int(it.iteration_index)}")
        print(f"full shape = {full_shape}")
        print(f"offset     = {offset_use}")
        print(f"extent     = {extent_use}")

    data_proxy = rc.load_chunk(offset_use, extent_use)
    series.flush()

    arr = np.array(data_proxy, copy=True)
    del data_proxy

    meta = copy_meta_from_iteration(
        it,
        mesh,
        rc,
        field,
        mesh_name,
        component,
        offset_use,
        extent_use,
    )
    meta["display_name"] = _default_display_name(field)
    meta["nonnegative"] = False
    meta["symmetric_default"] = True

    return arr, meta


def _load_vector_magnitude(it, series, field, offset=None, extent=None, verbose=False):
    """
    Load a derived vector magnitude such as |E| from stored components.

    This does not attempt to access iteration.meshes["|E|"]. Instead it:
      1. parses the base record, e.g. "E"
      2. discovers components, e.g. ["x", "y", "z"]
      3. loads each component
      4. returns sqrt(sum(component**2))
    """
    base_field = _parse_vector_magnitude_request(field)
    if base_field is None:
        raise ValueError(f"Field {field!r} is not a vector magnitude request.")

    mesh, components = _resolve_vector_components(it, base_field)
    if not components:
        raise ValueError(f"No components found for vector field {base_field!r}")

    rc0 = mesh[components[0]]
    full_shape = tuple(int(x) for x in rc0.shape)
    offset_use, extent_use = _prepare_offset_extent(full_shape, offset, extent)

    if verbose:
        print(
            f"Reading |{base_field}| from components {components} "
            f"at iteration {int(it.iteration_index)}"
        )
        print(f"full shape = {full_shape}")
        print(f"offset     = {offset_use}")
        print(f"extent     = {extent_use}")

    # Queue all component reads before flushing.
    proxies = []
    for comp in components:
        rc = mesh[comp]
        comp_shape = tuple(int(x) for x in rc.shape)
        if comp_shape != full_shape:
            raise ValueError(
                f"Component {base_field}/{comp} has shape {comp_shape}, "
                f"but {base_field}/{components[0]} has shape {full_shape}. "
                "Magnitude currently requires same-index components."
            )
        proxies.append(rc.load_chunk(offset_use, extent_use))

    series.flush()

    # Keep memory bounded: accumulate sum of squares and delete each component.
    sumsq = None
    for proxy in proxies:
        comp_arr = np.array(proxy, dtype=np.float32, copy=True)
        if sumsq is None:
            sumsq = comp_arr * comp_arr
        else:
            sumsq += comp_arr * comp_arr
        del comp_arr
    del proxies

    arr = np.sqrt(sumsq).astype(np.float32, copy=False)
    del sumsq

    meta = copy_meta_from_iteration(
        it,
        mesh,
        rc0,
        field,
        base_field,
        "magnitude",
        offset_use,
        extent_use,
    )
    meta.update({
        "field": field,
        "display_name": f"|{base_field}|",
        "mesh_name": base_field,
        "component": "magnitude",
        "derived": "vector_magnitude",
        "derived_from": tuple(f"{base_field}/{comp}" for comp in components),
        "nonnegative": True,
        "symmetric_default": False,
    })

    return arr, meta


def _load_field_data(it, series, field, offset=None, extent=None, verbose=False):
    """
    Dispatch to either a stored field reader or a derived-field reader.
    """
    if _parse_vector_magnitude_request(field) is not None:
        return _load_vector_magnitude(
            it,
            series,
            field,
            offset=offset,
            extent=extent,
            verbose=verbose,
        )

    return _load_regular_field(
        it,
        series,
        field,
        offset=offset,
        extent=extent,
        verbose=verbose,
    )


def read_field_linear_capped(
    series_path: str,
    target_iteration: int,
    field: str,
    *,
    max_linear_steps: int,
    offset: Optional[Iterable[int]] = None,
    extent: Optional[Iterable[int]] = None,
    verbose: bool = True,
):
    """
    Read one field from one iteration using openPMD-api linear mode.

    Field may be either:
      - a stored component such as "E/z"
      - a derived vector magnitude such as "E_mag", "E/abs", "abs(E)", or "|E|"

    Returns
    -------
    arr, meta
        arr is a real NumPy copy. meta is a copied metadata dict.

    Important
    ---------
    Stops after max_linear_steps and does not intentionally read to EOF.
    This is important for crash-truncated variable-based ADIOS2 output.
    """
    series = io.Series(str(series_path), _read_linear_access())

    try:
        inspected = 0

        for it in series.read_iterations():
            inspected += 1
            idx = int(it.iteration_index)

            try:
                if verbose:
                    print(f"step {inspected - 1}: iteration {idx}")

                if idx == int(target_iteration):
                    return _load_field_data(
                        it,
                        series,
                        field,
                        offset=offset,
                        extent=extent,
                        verbose=verbose,
                    )

            finally:
                it.close()

            # Stop here before requesting another ADIOS2 step.
            if inspected >= int(max_linear_steps):
                raise RuntimeError(
                    f"Stopped after max_linear_steps={max_linear_steps} "
                    f"without finding iteration {target_iteration}."
                )

        raise RuntimeError(
            f"read_iterations() ended before finding iteration {target_iteration}."
        )

    finally:
        series.close()

def reduce_warpx_slice(
    arr: np.ndarray,
    meta: Meta,
    *,
    slice_axis: int = 0,
    slice_reduction: str = "mean",
):
    """
    Reduce a 3D WarpX slice output to a 2D image.

    WarpX slices often contain two cells along the sliced direction.

    Parameters
    ----------
    slice_reduction
        "mean"  : average across the sliced direction.
        "first" : take index 0 along the sliced direction.
        "index:N" : take explicit index N along the sliced direction.

    Returns
    -------
    img, xdim, ydim, slice_coord, slice_note
    """
    arr = np.asarray(arr)

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {arr.shape}")

    slice_axis = int(slice_axis)
    if not (0 <= slice_axis < 3):
        raise ValueError("slice_axis must be 0, 1, or 2")

    n_slice = arr.shape[slice_axis]

    if slice_reduction == "mean":
        img = arr.mean(axis=slice_axis)
        slice_note = f"avg over {n_slice} cells"
        slice_coord = float(np.mean(axis_centers(meta, slice_axis)))

    elif slice_reduction == "first":
        slicer = [slice(None)] * 3
        slicer[slice_axis] = 0
        img = arr[tuple(slicer)]
        slice_note = "cell 0"
        slice_coord = float(axis_centers(meta, slice_axis)[0])

    elif isinstance(slice_reduction, str) and slice_reduction.startswith("index:"):
        idx = int(slice_reduction.split(":", 1)[1])
        slicer = [slice(None)] * 3
        slicer[slice_axis] = idx
        img = arr[tuple(slicer)]
        slice_note = f"cell {idx}"
        slice_coord = float(axis_centers(meta, slice_axis)[idx])

    else:
        raise ValueError(
            "slice_reduction must be 'mean', 'first', or 'index:N'"
        )

    remaining_dims = [d for d in range(3) if d != slice_axis]

    # img.shape follows the original dimension order with slice_axis removed.
    # Matplotlib imshow expects img rows -> y, columns -> x.
    ydim, xdim = remaining_dims

    return img, xdim, ydim, slice_coord, slice_note


def symmetric_absmax(
    img: np.ndarray,
    *,
    absmax: Optional[float] = None,
    autoscale_percentile: Optional[float] = None,
) -> float:
    """
    Return an absmax for symmetric color limits [-absmax, +absmax].

    If absmax is supplied, use it.
    Otherwise compute it per-frame from img.
    If autoscale_percentile is supplied, use percentile(abs(img)).
    """
    if absmax is not None:
        out = float(absmax)
    else:
        finite_abs = np.abs(img[np.isfinite(img)])

        if finite_abs.size == 0:
            out = 1.0
        elif autoscale_percentile is None:
            out = float(np.max(finite_abs))
        else:
            out = float(np.percentile(finite_abs, autoscale_percentile))

    if not np.isfinite(out) or out == 0.0:
        out = 1.0

    return out


def plot_warpx_slice_imshow(
    arr: np.ndarray,
    meta: Meta,
    *,
    slice_axis: int = 0,
    slice_reduction: str = "mean",
    absmax: Optional[float] = None,
    autoscale_percentile: Optional[float] = None,
    cmap: Optional[str] = None,
    colorbar_unit: Optional[str] = None,
    use_unit_SI: bool = True,
    plot_dtype: Any = np.float32,
    fig: Any = None,
    ax: Any = None,
    cax: Any = None,
):
    """
    Plot one WarpX slice with fixed axes positions and symmetric color scale.

    If absmax is None, color limits are computed per frame as:
      [-max(abs(img)), +max(abs(img))]
    or, if autoscale_percentile is set:
      [-percentile(abs(img), p), +percentile(abs(img), p)]

    Returns
    -------
    fig, ax, cax, im
    """
    img, xdim, ydim, slice_coord, slice_note = reduce_warpx_slice(
        arr,
        meta,
        slice_axis=slice_axis,
        slice_reduction=slice_reduction,
    )

    if use_unit_SI:
        img = img * float(meta.get("unit_SI", 1.0))

    if plot_dtype is not None:
        img = np.asarray(img, dtype=plot_dtype)
    else:
        img = np.asarray(img)

    if bool(meta.get("nonnegative", False)):
        if absmax is not None:
            vmax = float(absmax)
        else:
            finite = img[np.isfinite(img)]
            if finite.size == 0:
                vmax = 1.0
            elif autoscale_percentile is None:
                vmax = float(np.max(finite))
            else:
                vmax = float(np.percentile(finite, autoscale_percentile))
        if not np.isfinite(vmax) or vmax <= 0.0:
            vmax = 1.0
        vmin = 0.0
    else:
        amax = symmetric_absmax(
            img,
            absmax=absmax,
            autoscale_percentile=autoscale_percentile,
        )
        vmin = -amax
        vmax = +amax

    if cmap is None:
        cmap = "viridis" if bool(meta.get("nonnegative", False)) else "RdBu_r"

    if fig is None:
        fig = plt.figure(figsize=(8.0, 6.0), dpi=150)

    # Fixed layout for movie consistency.
    if ax is None:
        ax = fig.add_axes([0.10, 0.10, 0.68, 0.78])

    if cax is None:
        cax = fig.add_axes([0.82, 0.12, 0.03, 0.74])

    extent = imshow_extent_from_meta(meta, xdim, ydim)

    im = ax.imshow(
        img,
        origin="lower",
        extent=extent,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    cb = fig.colorbar(im, cax=cax)

    if colorbar_unit is None:
        colorbar_unit = infer_field_unit(meta["field"])

    cbar_label = meta["field"]
    if colorbar_unit:
        cbar_label += f" [{colorbar_unit}]"
    cb.set_label(cbar_label)

    xlab = meta["axis_labels"][xdim]
    ylab = meta["axis_labels"][ydim]
    ax.set_xlabel(f"{xlab} [m]")
    ax.set_ylabel(f"{ylab} [m]")

    t_si = float(meta.get("time", 0.0)) * float(meta.get("time_unit_SI", 1.0))
    slice_axis_label = meta["axis_labels"][slice_axis]

    ax.set_title(
        f"{meta['field']}   it={meta['iteration']}   t={t_si:.6e} s\n"
        f"{slice_axis_label}={slice_coord:.6e} m   "
        f"{slice_note}   range=[{vmin:.3e}, {vmax:.3e}]"
    )
    ax.set_aspect(1.0)

    return fig, ax, cax, im


def iter_field_frames_linear(
    series_path: str,
    field: str,
    *,
    max_linear_steps: int,
    offset: Optional[Iterable[int]] = None,
    extent: Optional[Iterable[int]] = None,
    every: int = 1,
    verbose: bool = True,
) -> Generator[Tuple[np.ndarray, Meta], None, None]:
    """
    Yield (arr, meta) for a field in a linear pass.

    Field may be either:
      - a stored component such as "E/z"
      - a derived vector magnitude such as "E_mag", "E/abs", "abs(E)", or "|E|"

    Only data for one iteration are loaded at a time.

    Important
    ---------
    Stops after max_linear_steps and does not intentionally read to EOF.
    """
    series = io.Series(str(series_path), _read_linear_access())

    try:
        inspected = 0

        for it in series.read_iterations():
            inspected += 1
            idx = int(it.iteration_index)
            should_read = ((inspected - 1) % int(every) == 0)

            arr = None
            meta = None

            try:
                if verbose:
                    print(f"step {inspected - 1}: iteration {idx}")

                if should_read:
                    arr, meta = _load_field_data(
                        it,
                        series,
                        field,
                        offset=offset,
                        extent=extent,
                        verbose=False,
                    )

            finally:
                it.close()

            if should_read:
                yield arr, meta
                del arr, meta

            # Stop here before requesting another ADIOS2 step.
            if inspected >= int(max_linear_steps):
                break

    finally:
        series.close()

def save_series_frames_linear(
    series_path: str,
    field: str,
    outdir: str,
    *,
    max_linear_steps: int,
    offset: Optional[Iterable[int]] = None,
    extent: Optional[Iterable[int]] = None,
    every: int = 1,
    slice_axis: int = 0,
    slice_reduction: str = "mean",
    absmax: Optional[float] = None,
    autoscale_percentile: Optional[float] = None,
    cmap: str = "RdBu_r",
    colorbar_unit: Optional[str] = None,
    use_unit_SI: bool = True,
    dpi: int = 150,
    figsize: Tuple[float, float] = (8.0, 6.0),
    axes_rect: Tuple[float, float, float, float] = (0.10, 0.10, 0.68, 0.78),
    cbar_rect: Tuple[float, float, float, float] = (0.82, 0.12, 0.03, 0.74),
    prefix: Optional[str] = None,
    verbose: bool = True,
    gc_every: int = 10,
) -> int:
    """
    Save PNG frames with:
      - one field component loaded at a time
      - imshow instead of pcolormesh
      - fixed subplot and colorbar positions
      - per-frame symmetric colorbar limits, unless absmax is explicitly fixed
      - explicit cleanup after every frame

    Returns
    -------
    n_saved : int
        Number of PNG frames written.
    """
    outpath = Path(outdir).expanduser()
    outpath.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Output directory: {outpath.resolve()}")

    if prefix is None:
        prefix = _default_prefix(field)

    plt.ioff()

    n_saved = 0

    for arr, meta in iter_field_frames_linear(
        series_path,
        field,
        max_linear_steps=max_linear_steps,
        offset=offset,
        extent=extent,
        every=every,
        verbose=verbose,
    ):
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_axes(list(axes_rect))
        cax = fig.add_axes(list(cbar_rect))

        plot_warpx_slice_imshow(
            arr,
            meta,
            slice_axis=slice_axis,
            slice_reduction=slice_reduction,
            absmax=absmax,
            autoscale_percentile=autoscale_percentile,
            cmap=cmap,
            colorbar_unit=colorbar_unit,
            use_unit_SI=use_unit_SI,
            fig=fig,
            ax=ax,
            cax=cax,
        )

        fname = outpath / f"{prefix}_{n_saved:06d}_it{meta['iteration']:08d}.png"
        fig.savefig(fname, dpi=dpi)

        if verbose:
            print(f"wrote {fname}")

        fig.clear()
        plt.close(fig)

        del fig, ax, cax
        del arr, meta

        n_saved += 1

        if gc_every is not None and gc_every > 0 and n_saved % gc_every == 0:
            gc.collect()

    gc.collect()

    if verbose:
        print(f"Saved {n_saved} frame(s) to {outpath.resolve()}")

    return n_saved
