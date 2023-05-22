import warnings

import numpy as np
import xarray as xr
import xugrid as xu

import imod


def get_unstructured_cell2d_from_xy(uda, **points):
    # Unstructured grids always require to be tested both on x and y coordinates
    # to see if points are within bounds.
    for coord in ["x", "y"]:
        if coord not in points.keys():
            raise KeyError(
                f"Missing {coord} in point coordinates."
                "Unstructured grids require both an x and y coordinate"
                "to get cell indices."
            )
    xy = np.column_stack([points["x"], points["y"]])
    return uda.ugrid.grid.locate_points(xy)


def __check_and_get_points_shape(points) -> dict:
    """Check whether points have the right shape"""
    shapes = {}
    for coord, value in points.items():
        arr = np.atleast_1d(value)
        points[coord] = arr
        shape = arr.shape
        if not len(shape) == 1:
            raise ValueError(
                f"Coordinate {coord} is not one-dimensional, but has shape: {shape}"
            )
        shapes[coord] = shape
    return shapes


def __check_point_shapes_consistency(shapes):
    if not len(set(shapes.values())) == 1:
        msg = "\n".join([f"{coord}: {shape}" for coord, shape in shapes.items()])
        raise ValueError(f"Shapes of coordinates do match each other:\n{msg}")


def _check_points(points):
    """Check whether points have the right and consistent shape"""

    shapes = __check_and_get_points_shape(points)
    __check_point_shapes_consistency(shapes)


def __arr_like_points(points, fill_value):
    """
    Return array with the same shape as the first array provided in points.
    """
    first_value = next(iter(points.values()))
    shape = np.atleast_1d(first_value).shape

    return np.full(shape, fill_value)


def points_in_bounds(da, **points):
    """
    Returns whether points specified by keyword arguments fall within the bounds
    of ``da``.

    Parameters
    ----------
    da : xr.DataArray
    points : keyword arguments of coordinate=values
        keyword arguments specifying coordinate and values. Please refer to the
        examples.

    Returns
    -------
    in_bounds : np.array of bools

    Examples
    --------
    Create the DataArray, then use the keyword arguments to define along which
    coordinate to check whether the points are within bounds.

    >>> nrow, ncol = 3, 4
    >>> data = np.arange(12.0).reshape(nrow, ncol)
    >>> coords = {"x": [0.5, 1.5, 2.5, 3.5], "y": [2.5, 1.5, 0.5]}
    >>> dims = ("y", "x")
    >>> da = xr.DataArray(data, coords, dims)
    >>> x = [0.4, 2.6]
    >>> points_in_bounds(da, x=x)

    This works for an arbitrary number of coordinates:

    >>> y = [1.3, 2.7]
    >>> points_in_bounds(da, x=x, y=y)

    """

    _check_points(points)

    in_bounds = __arr_like_points(points, True)

    if isinstance(da, xu.UgridDataArray):
        index = get_unstructured_cell2d_from_xy(da, **points)
        # xu.Ugrid2d.locate_points makes cells outside grid -1
        in_bounds = index > 0
        points.pop("x")
        points.pop("y")

    for key, x in points.items():
        da_x = da.coords[key]
        _, xmin, xmax = imod.util.coord_reference(da_x)
        # Inplace bitwise operator
        in_bounds &= (x >= xmin) & (x < xmax)

    return in_bounds


def check_points_in_bounds(da, points, out_of_bounds):
    inside = points_in_bounds(da, **points)
    # Error handling
    msg = "Not all points are located within the bounds of the DataArray"
    if not inside.all():
        if out_of_bounds == "raise":
            raise ValueError(msg)
        elif out_of_bounds == "warn":
            warnings.warn(msg)
        elif out_of_bounds == "ignore":
            points = {dim: x[inside] for dim, x in points.items()}
        else:
            raise ValueError(
                f"Unrecognized option {out_of_bounds} for out_of_bounds, "
                "should be one of: error, warn, ignore."
            )

    return points, inside


def _get_indices_1d(da, coordname, x):
    x = np.atleast_1d(x)
    x_decreasing = da.indexes[coordname].is_monotonic_decreasing
    dx, xmin, _ = imod.util.coord_reference(da.coords[coordname])

    ncell = da[coordname].size
    # Compute edges
    xs = np.full(ncell + 1, xmin)
    # Turn dx into array
    if isinstance(dx, float):
        dx = np.full(ncell, dx)
    # Always increasing
    if x_decreasing:
        xs[1:] += np.abs(dx[::-1]).cumsum()
    else:
        xs[1:] += np.abs(dx).cumsum()

    # From np.searchsorted docstring:
    # Find the indices into a sorted array a such that, if the corresponding
    # elements in v were inserted before the indices, the order of a would
    # be preserved.
    ixs = np.searchsorted(xs, x, side="right")

    # Take care of decreasing coordinates
    if x_decreasing:
        ixs = ncell - ixs
    else:
        ixs = ixs - 1

    return ixs


def points_indices(da, out_of_bounds="raise", **points):
    """
    Get the indices for points as defined by the arrays x and y.

    Not all points may be located in the bounds of the DataArray. By default,
    this function raises an error. This behavior can be controlled with the
    ``out_of_bounds`` argument. If ``out_of_bounds`` is set to "warn" or
    "ignore", out of bounds point are removed. Which points have been removed
    is visible in the ``index`` coordinate of the resulting selection.

    Parameters
    ----------
    da : xr.DataArray
    out_of_bounds : {"raise", "warn", "ignore"}, default: "raise"
        What to do if the points are not located in the bounds of the
        DataArray:
        - "raise": raise an exception
        - "warn": raise a warning, and ignore the missing points
        - "ignore": ignore the missing points
    points : keyword arguments of coordinates and values

    Returns
    -------
    indices : dict of {coordinate: xr.DataArray with indices}

    Examples
    --------

    To extract values:

    >>> x = [1.0, 2.2, 3.0]
    >>> y = [4.0, 5.6, 7.0]
    >>> indices = imod.select.points_indices(da, x=x, y=y)
    >>> ind_y = indices["y"]
    >>> ind_x = indices["x"]
    >>> selection = da.isel(x=ind_x, y=ind_y)

    Or shorter, using dictionary unpacking:

    >>> indices = imod.select.points_indices(da, x=x, y=y)
    >>> selection = da.isel(**indices)

    To set values (in a new array), the following will do the trick:

    >>> empty = xr.full_like(da, np.nan)
    >>> empty.data[indices["y"].values, indices["x"].values] = values_to_set

    Unfortunately, at the time of writing, xarray's .sel method does not
    support setting values yet. The method here works for both numpy and dask
    arrays, but you'll have to manage dimensions yourself!

    The ``imod.select.points_set_values()`` function will take care of the
    dimensions.
    """
    points, inside = check_points_in_bounds(da, points, out_of_bounds)

    indices = {}
    index = np.arange(len(inside))[inside]
    if isinstance(da, xu.UgridDataArray):
        ind_arr = get_unstructured_cell2d_from_xy(da, **points)
        ind_da = xr.DataArray(ind_arr, coords={"index": index}, dims=("index",))
        face_dim = da.ugrid.grid.face_dimension
        indices[face_dim] = ind_da
        points.pop("x")
        points.pop("y")

    for coordname, value in points.items():
        ind_arr = _get_indices_1d(da, coordname, value)
        ind_da = xr.DataArray(ind_arr, coords={"index": index}, dims=("index",))
        indices[coordname] = ind_da

    return indices


def points_values(da, out_of_bounds="error", **points):
    """
    Get values from specified points.

    This function will raise a ValueError if the points fall outside of the
    bounds of the DataArray to avoid ambiguous behavior. Use the
    ``imod.select.points_in_bounds`` function to detect these points.

    Parameters
    ----------
    da : xr.DataArray
    out_of_bounds : {"raise", "warn", "ignore"}, default: "raise"
        What to do if the points are not located in the bounds of the
        DataArray:
        - "raise": raise an exception
        - "warn": raise a warning, and ignore the missing points
        - "ignore": ignore the missing points
    points : keyword arguments of coordinate=values
        keyword arguments specifying coordinate and values.
    Returns
    -------
    selection : xr.DataArray

    Examples
    --------

    >>> x = [1.0, 2.2, 3.0]
    >>> y = [4.0, 5.6, 7.0]
    >>> selection = imod.select.points_values(da, x=x, y=y)

    """
    iterable_points = {}
    for coordname, value in points.items():
        if not isinstance(da, xu.UgridDataArray) and (coordname not in da.coords):
            raise ValueError(f'DataArray has no coordinate "{coordname}"')
        # contents must be iterable
        iterable_points[coordname] = np.atleast_1d(value)

    indices = imod.select.points.points_indices(
        da, out_of_bounds=out_of_bounds, **iterable_points
    )
    selection = da.isel(**indices)

    return selection


def points_set_values(da, values, out_of_bounds="raise", **points):
    """
    Set values at specified points.

    This function will raise a ValueError if the points fall outside of the
    bounds of the DataArray to avoid ambiguous behavior. Use the
    ``imod.select.points_in_bounds`` function to detect these points.

    Parameters
    ----------
    da : xr.DataArray
    values : (int, float) or array of (int, float)
    out_of_bounds : {"raise", "warn", "ignore"}, default: "raise"
        What to do if the points are not located in the bounds of the
        DataArray:
        - "raise": raise an exception
        - "warn": raise a warning, and ignore the missing points
        - "ignore": ignore the missing points
    points : keyword arguments of coordinate=values
        keyword arguments specifying coordinate and values.

    Returns
    -------
    da : xr.DataArray
        DataArray with values set at the point locations.

    Examples
    --------

    >>> x = [1.0, 2.2, 3.0]
    >>> y = [4.0, 5.6, 7.0]
    >>> values = [10.0, 11.0, 12.0]
    >>> out = imod.select.points_set_values(da, values, x=x, y=y)

    """
    points, inside = check_points_in_bounds(da, points, out_of_bounds)
    if not isinstance(values, (bool, float, int, str)):  # then it might be an array
        if len(values) != len(inside):
            raise ValueError(
                "Shape of ``values`` does not match shape of coordinates."
                f"Shape of values: {values.shape}; shape of coordinates: {inside.shape}."
            )
        # Make sure a list or tuple is indexable by inside.
        values = np.atleast_1d(values)[inside]

    # Avoid side-effects just in case
    # Load into memory, so values can be set efficiently via numpy indexing.
    da = da.copy(deep=True).load()

    sel_indices = {}
    for coordname in points.keys():
        if coordname not in da.coords:
            raise ValueError(f'DataArray has no coordinate "{coordname}"')
        underlying_dims = da.coords[coordname].dims
        if len(underlying_dims) != 1:
            raise ValueError(f"Coordinate {coordname} is not one-dimensional")
        # Use the first and only element of underlying_dims
        sel_indices[underlying_dims[0]] = _get_indices_1d(
            da, coordname, points[coordname]
        )

    # Collect indices in the right order
    indices = []
    for dim in da.dims:
        indices.append(sel_indices.get(dim, slice(None, None)))

    # Set values in dask or numpy array
    da.data[tuple(indices)] = values
    return da
