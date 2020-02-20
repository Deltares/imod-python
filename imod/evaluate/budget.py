import numba
import numpy as np
import scipy.ndimage
import xarray as xr


def _outer_edge(da):
    data = da.values.copy()
    from_edge = scipy.ndimage.morphology.binary_erosion(data)
    is_edge = (data == 1) & (from_edge == 0)
    return xr.full_like(da, is_edge, dtype=np.bool)


@numba.njit
def _facebudget(
    budgetzone,
    face,
    flowfront,
    flowlower,
    flowright,
    result_front,
    result_lower,
    result_right,
):
    nlay, nrow, ncol = budgetzone.shape
    for k in range(nlay):
        for i in range(nrow):
            for j in range(ncol):
                # Initialize accumulator
                if face[k, i, j] == 1:
                    # Default value: part of domain (1) for edges
                    lower = front = right = upper = back = left = 1
                    if k > 0:
                        upper = budgetzone[k - 1, i, j]
                    if k < (nlay - 1):
                        lower = budgetzone[k + 1, i, j]
                    if i < (nrow - 1):
                        front = budgetzone[k, i + 1, j]
                    if i > 0:
                        back = budgetzone[k, i - 1, j]
                    if j < (ncol - 1):
                        right = budgetzone[k, i, j + 1]
                    if j > 0:
                        left = budgetzone[k, i, j - 1]

                    # Test if cell is a control surface cell
                    if upper == 1 and lower == 0:
                        result_lower[k, i, j] += flowlower[k, i, j]
                    if upper == 0 and lower == 1:
                        result_lower[k, i, j] -= flowlower[k - 1, i, j]
                    if front == 1 and back == 0:
                        result_front[k, i, j] -= flowlower[k, i - 1, j]
                    if front == 0 and back == 1:
                        result_front[k, i, j] += flowlower[k, i, j]
                    if right == 1 and left == 0:
                        result_right[k, i, j] -= flowright[k, i, j - 1]
                    if right == 0 and left == 1:
                        result_right[k, i, j] += flowright[k, i, j]


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
    for da, daname in zip((front, lower, right), ("front", "lower", "right")):
        if da is not None:
            if da.shape != budgetzone.shape:
                raise ValueError(f"Shape of {daname} does not match budgetzone")
    # Create dummy arrays for skipped values
    if front is None:
        front = xr.full_like(budgetzone, 0.0, dtype=np.float)
    if lower is None:
        lower = xr.full_like(budgetzone, 0.0, dtype=np.float)
    if right is None:
        right = xr.full_like(budgetzone, 0.0, dtype=np.float)
    # Determine control surface
    # TODO: check for nans?
    # TODO: loop over time if present?
    face = _outer_edge(budgetzone)
    shape = budgetzone.shape
    # In case of netflow, only a single accumulator is necessary
    # The different arrays are just aliases
    if netflow:
        result_front = result_lower = result_right = np.zeros(shape)
    else:
        result_front = np.zeros(shape)
        result_lower = np.zeros(shape)
        result_right = np.zeros(shape)

    _facebudget(
        budgetzone.values,
        face.values,
        front.values,
        lower.values,
        right.values,
        result_front,
        result_lower,
        result_right,
    )
    if netflow:
        return xr.full_like(budgetzone, result_front)
    else:
        return (
            xr.full_like(budgetzone, result_front),
            xr.full_like(budgetzone, result_lower),
            xr.full_like(budgetzone, result_right)
        )
