import dask
import dask.array
import numba
import numpy as np
import scipy.ndimage
import xarray as xr

DIM_Z = 0
DIM_Y = 1
DIM_X = 2
LOWER = 3
UPPER = 4
FRONT = 5
BACK = 6
RIGHT = 7
LEFT = 8


def _outer_edge(da):
    data = da.values.copy()
    from_edge = scipy.ndimage.morphology.binary_erosion(data)
    is_edge = (data == 1) & (from_edge == 0)
    return xr.full_like(da, is_edge, dtype=np.bool)


@numba.njit
def _face_indices(face, budgetzone):
    shape = (int(face.sum()), 9)
    indices = np.zeros(shape, dtype=np.int32)
    # Loop over cells
    nlay, nrow, ncol = budgetzone.shape
    count = 0
    for k in range(nlay):
        for i in range(nrow):
            for j in range(ncol):
                if face[k, i, j] == 1:
                    # Store indices
                    indices[count, DIM_Z] = k
                    indices[count, DIM_Y] = i
                    indices[count, DIM_X] = j
                    # Default value: part of domain (1) for edges
                    lower = front = right = upper = back = left = 1
                    if k > 0:
                        upper = budgetzone[k - 1, i, j]
                    if k < (nlay - 1):
                        lower = budgetzone[k + 1, i, j]
                    if i > 0:
                        back = budgetzone[k, i - 1, j]
                    if i < (nrow - 1):
                        front = budgetzone[k, i + 1, j]
                    if j > 0:
                        left = budgetzone[k, i, j - 1]
                    if j < (ncol - 1):
                        right = budgetzone[k, i, j + 1]

                    # Test if cell is a control surface cell for the direction
                    if lower == 0:
                        indices[count, LOWER] = 1
                    if upper == 0:
                        indices[count, UPPER] = 1
                    if front == 0:
                        indices[count, FRONT] = 1
                    if back == 0:
                        indices[count, BACK] = 1
                    if right == 0:
                        indices[count, RIGHT] = 1
                    if left == 0:
                        indices[count, LEFT] = 1

                    # Incrementer counter
                    count += 1
    return indices


@numba.njit
def _collect_flowfront(indices, flow):
    result = np.zeros(flow.shape)
    nface = indices.shape[0]
    for count in range(nface):
        k = indices[count, DIM_Z]
        i = indices[count, DIM_Y]
        j = indices[count, DIM_X]
        if indices[count, FRONT]:
            result[k, i, j] -= flow[k, i, j]
        if indices[count, BACK]:
            result[k, i, j] += flow[k, i - 1, j]
    return result


@numba.njit
def _collect_flowlower(indices, flow):
    result = np.zeros(flow.shape)
    nface = indices.shape[0]
    for count in range(nface):
        k = indices[count, DIM_Z]
        i = indices[count, DIM_Y]
        j = indices[count, DIM_X]
        if indices[count, LOWER]:
            print(LOWER)
            result[k, i, j] += flow[k, i, j]
        if indices[count, UPPER]:
            print(LOWER)
            result[k, i, j] -= flow[k - 1, i, j]
    return result


@numba.njit
def _collect_flowright(indices, flow):
    result = np.zeros(flow.shape)
    nface = indices.shape[0]
    for count in range(nface):
        k = indices[count, DIM_Z]
        i = indices[count, DIM_Y]
        j = indices[count, DIM_X]
        if indices[count, RIGHT]:
            result[k, i, j] -= flow[k, i, j]
        if indices[count, LEFT]:
            result[k, i, j] += flow[k, i, j - 1]
    return result


def delayed_collect(indices, front, lower, right):
    result_front = dask.delayed(_collect_flowfront, nout=1)(indices, front.values)
    result_lower = dask.delayed(_collect_flowlower, nout=1)(indices, lower.values)
    result_right = dask.delayed(_collect_flowright, nout=1)(indices, right.values)
    dask_front = dask.array.from_delayed(result_front, front.shape, dtype=np.float)
    dask_lower = dask.array.from_delayed(result_lower, lower.shape, dtype=np.float)
    dask_right = dask.array.from_delayed(result_right, right.shape, dtype=np.float)
    return dask_front, dask_lower, dask_right


def facebudget(budgetzone, front=None, lower=None, right=None, netflow=True):
    """
    Computes net face flow into a control volume, as defined by ``budgetzone``.

    Returns a three dimensional DataArray with in- and outgoing flows for every
    cell that is located on the edge of the control volume, thereby calculating
    the flow through the control surface of the control volume.

    Front, lower, and right arguments refer to iMOD face flow budgets, in cubic
    meters per day. In terms of flow direction these are defined as:

    * ``front``: positive with ``y`` (negative with row index)
    * ``lower``: positive with ``layer`` (positive with layer index)
    * ``right``: negative with ``x`` (negative with column index)

    Only a single face flow has to be defined, for example, if only vertical
    fluxes (``lower``) are to be considered.

    Note that you generally should not define a budgetzone that is only one cell
    wide. In that case, flow will both enter and leave the cell, resulting in a
    net flow of zero (given there are no boundary conditions present).

    The output of this function defines ingoing flow as positive, and outgoing
    flow as negative. The output is a 3D array with net flows for every control
    surface cell. You can select specific parts of the control surface, for
    example only the east-facing side of the control volume. Please refer to the
    examples.

    Parameters
    ----------
    budgetzone: xr.DataAray
        Binary array defining zone (``1`` or ``True``) and outside of zone 
        (``0`` or ``False``). Dimensions must be exactly ``("layer", "y", "x")``.
    front: xr.DataArray of floats, optional
        Dimensions must be exactly ``("layer", "y", "x")``.
    lower: xr.DataArray of floats, optional
        Dimensions must be exactly ``("layer", "y", "x")``.
    right: xr.DataArray of floats, optional
        Dimensions must be exactly ``("layer", "y", "x")``.
    netflow : bool, optional
        Whether to split flows by direction (front, lower, right).
        True: sum all flows. False: return individual directions.

    Returns
    -------
    facebudget_front, facebudget_lower, face_budget_right : xr.DataArray of floats
        Only returned if `netflow` is False.
    facebudget_net : xr.DataArray of floats
        Only returned if `netflow` is True.

    Examples
    --------
    Load the face flows, and select the last time using index selection:

    >>> import imod
    >>> lower = imod.idf.open("bdgflf*.idf").isel(time=-1)
    >>> right = imod.idf.open("bdgfrf*.idf").isel(time=-1)
    >>> front = imod.idf.open("bdgfff*.idf").isel(time=-1)

    Define the zone of interest, e.g. via rasterizing a shapefile:

    >>> import geopandas as gpd
    >>> gdf = gpd.read_file("zone_of_interest.shp")
    >>> zone2D = imod.prepare.rasterize(gdf, like=lower.isel(layer=0))

    Broadcast it to three dimensions:

    >>> zone = xr.full_like(flow, zone2D, dtype=np.bool)

    Compute net flow through the (control) surface of the budget zone:

    >>> flow = imod.evaluate.facebudget(
    >>>     budgetzone=zone, front=front, lower=lower, right=right
    >>> )

    Or evaluate only horizontal fluxes:

    >>> flow = imod.evaluate.facebudget(
    >>>     budgetzone=zone, front=front, right=right
    >>> )

    Extract the net flow, only on the right side of the zone, for example as 
    defined by x > 10000:

    >>> netflow_right = flow.where(flow["x"] > 10_000.0).sum()

    Extract the net flow for only the first four layers:

    >>> netflow_layers = netflow_right.sel(layer=slice(1, 4)).sum()

    Extract the net flow to the right of an arbitrary diagonal in ``x`` and
    ``y`` is simple using the equation for a straight line:

    >>> netflow_right_of_diagonal = flow.where(
    >>>    flow["y"] < (line_slope * flow["x"] + line_intercept)
    >>> )

    There are many ways to extract flows for a certain part of the zone of
    interest. The most flexible one with regards to the ``x`` and ``y``
    dimensions is by drawing a vector file, rasterizing it, and using it to
    select with ``xarray.where()``.

    To get the flows per direction, pass ``netflow=False``.

    >>> flowfront, flowlower, flowright = imod.evaluate.facebudget(
    >>>    budgetzone=zone, front=front, lower=lower, right=right, netflow=False
    >>> )

    """
    # Error handling
    if front is None and lower is None and right is None:
        raise ValueError("Atleast a single flow budget DataArray must be given")
    if tuple(budgetzone.dims) != ("layer", "y", "x"):
        raise ValueError('Dimensions of budgetzone must be exactly ("layer", "y", "x")')
    shape = budgetzone.shape
    for da, daname in zip((front, lower, right), ("front", "lower", "right")):
        if da is not None:
            dims = da.dims
            coords = da.coords
            if "time" in dims:
                da_shape = da.shape[1:]
            else:
                da_shape = da.shape
            if da_shape != shape:
                raise ValueError(f"Shape of {daname} does not match budgetzone")

    # Create dummy arrays for skipped values, allocate just once
    if front is None:
        f = xr.full_like(budgetzone, 0.0, dtype=np.float)
    if lower is None:
        l = xr.full_like(budgetzone, 0.0, dtype=np.float)
    if right is None:
        r = xr.full_like(budgetzone, 0.0, dtype=np.float)

    # Determine control surface
    # TODO: check for nans?
    # TODO: loop over time if present?
    face = _outer_edge(budgetzone)
    indices = _face_indices(budgetzone.values, face.values)

    results_front = []
    results_lower = []
    results_right = []

    if "time" in dims:
        for itime in range(front.dims.size):
            if front is not None:
                f = front.isel(time=itime)
            if lower is not None:
                l = lower.isel(time=itime)
            if right is not None:
                r = right.isel(time=itime)
            # collect dask arrays
            df, dl, dr = delayed_collect(indices, f, l, r, netflow)
            # append
            results_front.append(df)
            results_lower.append(dl)
            results_right.append(dr)

        # Concatenate over time dimension
        dask_front = dask.array.concatenate(results_front, axis=0)
        dask_lower = dask.array.concatenate(results_lower, axis=0)
        dask_right = dask.array.concatenate(results_right, axis=0)
    else:
        if front is not None:
            f = front
        if lower is not None:
            l = lower
        if right is not None:
            r = right
        dask_front, dask_lower, dask_right = delayed_collect(indices, f, l, r)

    if netflow:
        return xr.DataArray(dask_front + dask_lower + dask_right, coords, dims)
    else:
        return (
            xr.DataArray(dask_front, coords, dims),
            xr.DataArray(dask_lower, coords, dims),
            xr.DataArray(dask_right, coords, dims),
        )
