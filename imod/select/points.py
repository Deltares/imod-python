import numpy as np
import pandas as pd
import xarray as xr

import imod


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
    # check sizes
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
    if not len(set(shapes.values())) == 1:
        msg = "\n".join([f"{coord}: {shape}" for coord, shape in shapes.items()])
        raise ValueError(f"Shapes of coordinates do match each other:\n{msg}")

    # Re-use shape state from loop above
    in_bounds = np.full(shape, True)
    for key, x in points.items():
        da_x = da.coords[key]
        _, xmin, xmax = imod.util.coord_reference(da_x)
        # Inplace bitwise operator
        in_bounds &= (x >= xmin) & (x < xmax)

    return in_bounds


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


def points_indices(da, **points):
    """
    Get the indices for points as defined by the arrays x and y.

    This function will raise a ValueError if the points fall outside of the
    bounds of the DataArray to avoid undefined behavior. Use the
    ``imod.select.points_in_bounds`` function to detect these points.

    Parameters
    ----------
    da : xr.DataArray
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
    inside = points_in_bounds(da, **points)
    # Error handling
    if not inside.all():
        raise ValueError(f"Not all points are within the bounds of the DataArray")

    indices = {}
    for coordname, value in points.items():
        ind_da = xr.DataArray(_get_indices_1d(da, coordname, value), dims=["index"])
        ind_da["index"] = np.arange(ind_da.size)
        indices[coordname] = ind_da

    return indices


def points_values(da, **points):
    """
    Get values from specified points.

    This function will raise a ValueError if the points fall outside of the
    bounds of the DataArray to avoid undefined behavior. Use the
    ``imod.select.points_in_bounds`` function to detect these points.

    Parameters
    ----------
    da : xr.DataArray
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
        if coordname not in da.coords:
            raise ValueError(f'DataArray has no coordinate "{coordname}"')
        # contents must be iterable
        iterable_points[coordname] = np.atleast_1d(value)

    indices = imod.select.points.points_indices(da, **iterable_points)
    selection = da.isel(**indices)

    # Fetch a value from the dictionary, try to extract a meaningful index
    sample_dim = next(iter(points.values()))
    if isinstance(sample_dim, pd.Series):
        selection.coords["index"] = sample_dim.index
    else:
        sample_dim = next(iter(iterable_points.values()))
        selection.coords["index"] = np.arange(len(sample_dim))

    return selection


def points_set_values(da, values, **points):
    """
    Set values at specified points.

    This function will raise a ValueError if the points fall outside of the
    bounds of the DataArray to avoid undefined behavior. Use the
    ``imod.select.points_in_bounds`` function to detect these points.

    Parameters
    ----------
    da : xr.DataArray
    values : (int, float) or array of (int, float)

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
    # Avoid side-effects just in case
    # Load into memory, so values can be set efficiently via numpy indexing.
    da = da.copy(deep=True).load()

    inside = points_in_bounds(da, **points)
    # Error handling
    if not inside.all():
        raise ValueError(f"Not all points are within the bounds of the DataArray")
    if not isinstance(values, (int, float)):  # then it might be an array
        if len(values) != len(inside):
            raise ValueError(
                "Shape of ``values`` does not match shape of coordinates."
                f"Shape of values: {values.shape}; shape of coordinates: {inside.shape}."
            )

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
