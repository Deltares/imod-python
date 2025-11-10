"""
Assign wells to layers.
"""

from typing import Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
import xugrid as xu

import imod
from imod.typing import GridDataArray


def compute_vectorized_overlap(
    bounds_a: npt.NDArray[np.float64], bounds_b: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
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


def _is_point_filter_in_layer(
    bounds_wells: npt.NDArray[np.float64], bounds_layers: npt.NDArray[np.float64]
) -> npt.NDArray[np.bool_]:
    # Unwrap for readability
    wells_top = bounds_wells[:, 1]
    wells_bottom = bounds_wells[:, 0]
    layers_top = bounds_layers[:, 1]
    layers_bottom = bounds_layers[:, 0]

    has_zero_filter_length = wells_top == wells_bottom
    in_layer = (layers_top >= wells_top) & (layers_bottom < wells_bottom)

    return has_zero_filter_length & in_layer


def compute_point_filter_overlap(
    bounds_wells: npt.NDArray[np.float64], bounds_layers: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Special case for filters with zero filter length, these are set to layer
    thickness. Filters which are not in a layer or have a nonzero filter length
    are set to zero overlap.
    """
    # Unwrap for readability
    layers_top = bounds_layers[:, 1]
    layers_bottom = bounds_layers[:, 0]
    layer_thickness = layers_top - layers_bottom

    point_filter_in_layer = _is_point_filter_in_layer(bounds_wells, bounds_layers)

    # Multiplication to set any elements not meeting the criteria to zero.
    point_filter_overlap = point_filter_in_layer.astype(float) * layer_thickness
    return point_filter_overlap


def compute_penetration_mismatch_factor(
    well_bounds: npt.NDArray[np.float64], layer_bounds: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    From iMOD5.6.1 manual, page 705:

    Correct any ratio for a mismatch between the centre of the penetrating model
    layer Zc and the vertical midpoint of the well screen segment Fc.

    F = 1 - |Zc - Fc| / (0.5 * D)

    where D is the thickness of the layer.

    This function achieves a similar thing as
    ``imod.prepare.topsystem.conductance._compute_correction_factor``, but then
    for point data instead of grids.
    """
    D = layer_bounds[:, 1] - layer_bounds[:, 0]
    Z_c = (layer_bounds[:, 1] + layer_bounds[:, 0]) / 2.0
    F_c = (
        np.minimum(well_bounds[:, 1], layer_bounds[:, 1])
        + np.maximum(well_bounds[:, 0], layer_bounds[:, 0])
    ) / 2
    return 1.0 - np.abs(Z_c - F_c) / (0.5 * D)


def compute_overlap_and_correction_factor(
    wells: pd.DataFrame, top: GridDataArray, bottom: GridDataArray
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    # layer bounds stack shape of (n_well, n_layer, 2)
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
    F = compute_penetration_mismatch_factor(well_bounds, layer_bounds)
    # Set overlap to layer thickness for point filters
    overlap = np.maximum(interval_filter_overlap, point_filter_overlap)
    # Set F to 1.0 for point filters
    is_point_filter = point_filter_overlap > 0.0
    F_corrected = np.maximum(F, is_point_filter.astype(float))

    return overlap, F_corrected


def locate_wells(
    wells: pd.DataFrame,
    top: GridDataArray,
    bottom: GridDataArray,
    k: Optional[GridDataArray] = None,
    validate: bool = True,
) -> tuple[
    npt.NDArray[np.object_], GridDataArray, GridDataArray, float | GridDataArray
]:
    if not isinstance(top, (xu.UgridDataArray, xr.DataArray)):
        raise TypeError(
            "top and bottom should be DataArray or UgridDataArray, received: "
            f"{type(top).__name__}"
        )

    # Default to a xy_k value of 1.0: weigh every layer equally.
    xy_k: float | GridDataArray = 1.0
    first = wells.groupby("id", sort=False).first()
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


def validate_well_columnnames(
    wells: pd.DataFrame, names: set = {"x", "y", "id"}
) -> None:
    missing = names.difference(wells.columns)
    if missing:
        raise ValueError(f"Columns are missing in wells dataframe: {missing}")


def validate_arg_types_equal(**kwargs):
    types = [type(arg) for arg in (kwargs.values()) if arg is not None]
    if len(set(types)) != 1:
        members = ", ".join([t.__name__ for t in types])
        names = ", ".join(kwargs.keys())
        raise TypeError(f"{names} should be of the same type, received: {members}")


def assign_wells(
    wells: pd.DataFrame,
    top: GridDataArray,
    bottom: GridDataArray,
    k: Optional[GridDataArray] = None,
    minimum_thickness: Optional[float] = 0.0,
    minimum_k: Optional[float] = 0.0,
    validate: bool = True,
) -> pd.DataFrame:
    """
    Distribute well pumping rate according to filter length when ``k=None``, or
    to transmissivity of the sediments surrounding the filter. Minimum thickness
    and minimum k should be set to avoid placing wells in clay layers. Pumping
    rates are adjusted using a correction factor :math:`F` based on the mismatch
    between the depth of the well screen center :math:`F_c` and the cell center
    :math:`Z_c`, equal to iMOD5's correction factor:

    .. math::

        F = 1 - \\frac{|Zc - Fc|}{0.5 * D}

    where D is the thickness of the layer.

    This factor is multiplied with the transmissivity (k-value * well filter
    thickness) to weigh how rates should be distributed over the layers.

    Wells where well screen_top equals screen_bottom are assigned to the layer
    they are located in, without any subdivision. Wells located outside of the
    grid are removed. To try to automatically fix well filter misplacements, see
    the :func:`imod.prepare.cleanup_wel` function.

    Parameters
    ----------
    wells: pandas.DataFrame
        Should contain columns x, y, id, top, bottom, rate.
    top: xarray.DataArray or xugrid.UgridDataArray
        Top of the model layers.
    bottom: xarray.DataArray or xugrid.UgridDataArray
        Bottom of the model layers.
    k: xarray.DataArray or xugrid.UgridDataArray, optional
        Horizontal conductivity of the model layers.
    minimum_thickness: float, optional, default: 0.0
        Minimum thickness, cells with thicknesses smaller than this value will
        be dropped.
    minimum_k: float, optional, default: 0.0
        Minimum horizontal conductivity, cells with horizontal conductivities
        smaller than this value will be dropped.
    validate: bool
        raise an excpetion if one of the wells is not in the domain
    Returns
    -------
    placed_wells: pd.DataFrame
        Wells with rate subdivided per layer. Contains the original columns of
        ``wells``, as well as layer, overlap, transmissivity.
    """
    columnnames = {"x", "y", "id", "top", "bottom", "rate"}
    validate_well_columnnames(wells, columnnames)
    validate_arg_types_equal(top=top, bottom=bottom, k=k)

    id_in_bounds, xy_top, xy_bottom, xy_k = locate_wells(
        wells, top, bottom, k, validate
    )
    wells_in_bounds = wells.set_index("id").loc[id_in_bounds].reset_index()
    first = wells_in_bounds.groupby("id", sort=False).first()
    overlap, F = compute_overlap_and_correction_factor(first, xy_top, xy_bottom)

    if isinstance(xy_k, (xr.DataArray, xu.UgridDataArray)):
        k_for_df = xy_k.values.ravel()
    else:
        k_for_df = xy_k

    # Distribute rate according to transmissivity.
    n_layer, n_well = xy_top.shape
    df_factor = pd.DataFrame(
        index=pd.Index(np.tile(first.index, n_layer), name="id"),
        data={
            "layer": np.repeat(top["layer"], n_well),
            "overlap": overlap,
            "k": k_for_df,
            "transmissivity": overlap * k_for_df * F,
            "F": F,
        },
    )
    # remove entries
    #   -in very thin layers or when the wellbore penetrates the layer very little
    #   -in low conductivity layers
    df_factor = df_factor.loc[
        (df_factor["overlap"] > minimum_thickness) & (df_factor["k"] > minimum_k)
    ]
    df_factor["rate"] = df_factor["transmissivity"] / df_factor.groupby("id")[
        "transmissivity"
    ].transform("sum")
    # Create a unique index for every id-layer combination.
    df_factor["index"] = np.arange(len(df_factor))
    df_factor = df_factor.reset_index()

    # Get rid of those that are removed because of minimum thickness or
    # transmissivity.
    wells_in_bounds = wells_in_bounds.loc[
        wells_in_bounds["id"].isin(df_factor["id"].unique())
    ]

    # Use pandas multi-index broadcasting.
    # Maintain all other columns as-is.
    wells_in_bounds["index"] = 1  # N.B. integer!
    wells_in_bounds["overlap"] = 1.0
    wells_in_bounds["k"] = 1.0
    wells_in_bounds["transmissivity"] = 1.0
    columns = list(set(wells_in_bounds.columns).difference(df_factor.columns))

    indexes = ["id"]
    for dim in ["species", "time"]:
        if dim in wells_in_bounds:
            indexes.append(dim)
            columns.remove(dim)

    df_factor[columns] = 1  # N.B. integer!

    assigned = (
        wells_in_bounds.set_index(indexes) * df_factor.set_index(["id", "layer"])
    ).reset_index()
    return assigned
