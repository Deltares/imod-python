import numpy as np
import xarray as xr

import imod


def in_bounds(da, x, y):
    """
    Returns whether points specified by `x` and `y` fall within the bounds of
    `da`.

    Parameters
    ----------
    da : xr.DataArray
    y : np.array of floats
    x : np.array of floats

    Returns
    -------
    in_bounds : np.array of bools
    """
    _, xmin, xmax, _, ymin, ymax = imod.util.spatial_reference(da)
    return (x >= xmin) & (x < xmax) & (y >= ymin) & (y < ymax)


def get_xy_indices(da, x, y):
    """
    Get the indices for points as defined by the arrays x and y.

    Parameters
    ----------
    da : xr.DataArray
    y : float or np.array of floats
    x : float or np.array of floats

    Returns
    -------
    rr, cc : np.array of integers
        row and column indices

    Examples
    --------
    Using the indices that this function provides might be straightforward.
    To get values, the following works:

    >>> x = [1.0, 2.2, 3.0]
    >>> y = [4.0, 5.6, 7.0]
    >>> rr, xx = imod.select.points.get_indices(da, x, y)
    >>> ind_y = xr.DataArray(rr, dims=["index"])
    >>> ind_x = xr.DataArray(cc, dims=["index"])
    >>> selection = da.sel(x=ind_x, y=ind_y)

    To set values (in a new array), the following will do the trick:

    >>> empty = xr.full_like(da, np.nan)
    >>> empty.data[rr, ii] = values

    Unfortunately, at the time of writing, xarray's .sel method does not
    support setting values yet. The method here works for both numpy and dask
    arrays, but you'll have to manage dimensions yourself!
    """
    dx, xmin, _, dy, ymin, _ = imod.util.spatial_reference(da)
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    x_decreasing = da.indexes["x"].is_monotonic_decreasing
    y_decreasing = da.indexes["y"].is_monotonic_decreasing

    inside = in_bounds(da, x, y)
    if not inside.all():
        raise ValueError(
            f"Points with x {x[~inside]} and {y[~inside]} are out of bounds"
        )
    nrow = da.y.size
    ncol = da.x.size
    ys = np.full(nrow + 1, ymin)
    xs = np.full(ncol + 1, xmin)
    # Always increasing
    if x_decreasing:
        xs[1:] += np.abs(dx[::-1]).cumsum()
    else:
        xs[1:] += np.abs(dx).cumsum()
    if y_decreasing:
        ys[1:] += np.abs(dy[::-1]).cumsum()
    else:
        ys[1:] += np.abs(dy).cumsum()

    # From np.searchsorted docstring:
    # Find the indices into a sorted array a such that, if the corresponding
    # elements in v were inserted before the indices, the order of a would
    # be preserved.
    iys = np.searchsorted(ys, y, side="right")
    ixs = np.searchsorted(xs, x, side="right")

    # Take care of decreasing coordinates
    if x_decreasing:
        cc = ncol - 1
    else:
        cc = ixs - 1
    if y_decreasing:
        rr = nrow - iys
    else:
        rr = iys - 1

    return rr, cc


def set_xy_values(da, x, y, values):
    """
    Set values at specified x and y coordinates.

    Parameters
    ----------
    da : xr.DataArray
    x : np.array of length N
    y : np.array of length N
    values : np.array of length N

    Returns
    -------
    da : xr.DataArray
    """
    if not da.dims[-2:] == ("y", "x"):
        raise ValueError('Last two dimensions of DataArray must be ("y", "x")')
    rr, cc = get_xy_indices(da, x, y)
    da.data[..., rr, cc] = values
    return da


def get_xy_values(da, x, y):
    """
    Get values from specified x and y coordinates.
    Out of bounds values are not returned.

    Parameters
    ----------
    da : xr.DataArray
    x : np.array
    y : np.array

    Returns
    -------
    selection : xr.DataArray
    """
    rr, cc = get_xy_indices(da, x, y)
    ind_y = xr.DataArray(rr, dims=["index"])
    ind_x = xr.DataArray(cc, dims=["index"])
    selection = da.isel(x=ind_x, y=ind_y)
    selection.coords["index"] = np.arange(rr.size)
    return selection
