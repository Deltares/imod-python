"""
Assign wells to layers.
"""

from typing import Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
import xugrid as xu

import imod


def compute_vectorized_overlap(bounds_a, bounds_b):
    """
    Vectorized overlap computation.
    Compare with:
    overlap = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    """
    return np.maximum(
        0.0,
        np.minimum(bounds_a[:, 1], bounds_b[:, 1])
        - np.maximum(bounds_a[:, 0], bounds_b[:, 0]),
    )


def compute_point_filter_overlap(bounds_wells, bounds_layers):
    """
    Special case for filters with zero filter length, these are set to layer
    thickness.
    """
    zero_filter_length = bounds_wells[:, 1] == bounds_wells[:, 0]
    layer_thickness = bounds_layers[:, 1] - bounds_layers[:, 0]
    return zero_filter_length.astype(float) * layer_thickness


def compute_overlap(wells, top, bottom):
    # layer bounds shape of (n_well, n_layer, 2)
    layer_bounds_stack = np.stack((bottom, top), axis=-1)
    well_bounds = np.broadcast_to(
        np.stack(
            (wells["bottom"].to_numpy(), wells["top"].to_numpy()),
            axis=-1,
        )[np.newaxis, :, :],
        layer_bounds_stack.shape,
    ).reshape(-1, 2)
    layer_bounds = layer_bounds_stack.reshape(-1, 2)

    # Deal with filters with a nonzero length
    interval_filter_overlap = compute_vectorized_overlap(
        well_bounds,
        layer_bounds,
    )
    # Deal with filters with zero length
    point_filter_overlap = compute_point_filter_overlap(
        well_bounds,
        layer_bounds,
    )
    return np.maximum(interval_filter_overlap, point_filter_overlap)

def locate_wells(
    wells: pd.DataFrame,
    top: Union[xr.DataArray, xu.UgridDataArray],
    bottom: Union[xr.DataArray, xu.UgridDataArray],
    k: Optional[Union[xr.DataArray, xu.UgridDataArray]],
    validate: bool = True,
):
    if not isinstance(top, (xu.UgridDataArray, xr.DataArray)):
        raise TypeError(
            "top and bottom should be DataArray or UgridDataArray, received: "
            f"{type(top).__name__}"
        )

    # Default to a xy_k value of 1.0: weigh every layer equally.
    xy_k = 1.0
    first = wells.groupby("id").first()
    x = first["x"].to_numpy()
    y = first["y"].to_numpy()

    xy_top = imod.select.points_values(top, x=x, y=y, out_of_bounds="ignore")
    xy_bottom = imod.select.points_values(bottom, x=x, y=y, out_of_bounds="ignore")

    # Raise exception if not all wells could be mapped onto the domain
    if validate and len(x) > len(xy_top["index"]):
        inside = imod.select.points_in_bounds(top, x=x, y=y)
        out = np.where(~inside)
        raise ValueError(
            f"well at x = {x[out[0]]} and y = {y[out[0]]} could not be mapped on the grid"
        )

    if k is not None:
        xy_k = imod.select.points_values(k, x=x, y=y, out_of_bounds="ignore")

    # Discard out-of-bounds wells.
    index = xy_top["index"]
    if validate and not np.array_equal(xy_bottom["index"], index):
        raise ValueError("bottom grid does not match top grid")
    if validate and k is not None and not np.array_equal(xy_k["index"], index):  # type: ignore
        raise ValueError("k grid does not match top grid")
    id_in_bounds = first.index[index]

    return id_in_bounds, xy_top, xy_bottom, xy_k


def assign_wells(
    wells: pd.DataFrame,
    top: Union[xr.DataArray, xu.UgridDataArray],
    bottom: Union[xr.DataArray, xu.UgridDataArray],
    k: Optional[Union[xr.DataArray, xu.UgridDataArray]] = None,
    minimum_thickness: Optional[float] = 0.05,
    minimum_k: Optional[float] = 1.0,
    validate: bool = True,
) -> pd.DataFrame:
    """
    Distribute well pumping rate according to filter length when ``k=None``, or
    to transmissivity of the sediments surrounding the filter. Minimum
    thickness and minimum k should be set to avoid placing wells in clay
    layers.

    Wells where well screen_top equals screen_bottom are assigned to the layer
    they are located in, without any subdivision. Wells located outside of the
    grid are removed.

    Parameters
    ----------
    wells: pd.DataFrame
        Should contain columns x, y, id, top, bottom, rate.
    top: xr.DataArray or xu.UgridDataArray
        Top of the model layers.
    bottom: xr.DataArray or xu.UgridDataArray
        Bottom of the model layers.
    k: xr.DataArray or xu.UgridDataArray, optional
        Horizontal conductivity of the model layers.
    minimum_thickness: float, optional, default: 0.01
    minimum_k: float, optional, default: 1.0
        Minimum conductivity
    validate: bool
        raise an excpetion if one of the wells is not in the domain
    Returns
    -------
    placed_wells: pd.DataFrame
        Wells with rate subdivided per layer. Contains the original columns of
        ``wells``, as well as layer, overlap, transmissivity.
    """

    names = {"x", "y", "id", "top", "bottom", "rate"}
    missing = names.difference(wells.columns)
    if missing:
        raise ValueError(f"Columns are missing in wells dataframe: {missing}")

    types = [type(arg) for arg in (top, bottom, k) if arg is not None]
    if len(set(types)) != 1:
        members = ",".join([t.__name__ for t in types])
        raise TypeError(
            "top, bottom, and optionally k should be of the same type, "
            f"received: {members}"
        )

    id_in_bounds, xy_top, xy_bottom, xy_k = locate_wells(
        wells, top, bottom, k, validate
    )
    wells_in_bounds = wells.set_index("id").loc[id_in_bounds].reset_index()
    first = wells_in_bounds.groupby("id").first()
    overlap = compute_overlap(first, xy_top, xy_bottom)

    if k is None:
        k = 1.0
    else:
        k = xy_k.values.ravel()

    # Distribute rate according to transmissivity.
    n_layer, n_well = xy_top.shape
    df = pd.DataFrame(
        index=pd.Index(np.tile(first.index, n_layer), name="id"),
        data={
            "layer": np.repeat(top["layer"], n_well),
            "overlap": overlap,
            "k": k,
            "transmissivity": overlap * k,
        },
    )
    # remove entries
    #   -in very thin layers or when the wellbore penetrates the layer very little
    #   -in low conductivity layers
    df = df.loc[(df["overlap"] >= minimum_thickness) & (df["k"] >= minimum_k)]
    df["rate"] = df["transmissivity"] / df.groupby("id")["transmissivity"].transform(
        "sum"
    )
    # Create a unique index for every id-layer combination.
    df["index"] = np.arange(len(df))
    df = df.reset_index()

    # Get rid of those that are removed because of minimum thickness or
    # transmissivity.
    wells_in_bounds = wells_in_bounds.loc[wells_in_bounds["id"].isin(df["id"].unique())]

    # Use pandas multi-index broadcasting.
    # Maintain all other columns as-is.
    wells_in_bounds["index"] = 1  # N.B. integer!
    wells_in_bounds["overlap"] = 1.0
    wells_in_bounds["k"] = 1.0
    wells_in_bounds["transmissivity"] = 1.0
    columns = list(set(wells_in_bounds.columns).difference(df.columns))

    indexes = ["id"]
    for dim in ["species", "time"]:
        if dim in wells_in_bounds:
            indexes.append(dim)
            columns.remove(dim)

    df[columns] = 1  # N.B. integer!

    assigned = (
        wells_in_bounds.set_index(indexes) * df.set_index(["id", "layer"])
    ).reset_index()
    return assigned
