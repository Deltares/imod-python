import numba
import numpy as np
import xarray as xr

import imod
from imod.util.imports import MissingOptionalModule

try:
    import shapely.geometry as sg
except ImportError:
    sg = MissingOptionalModule("shapely")


@numba.njit
def _index(edges, coord, coef):
    if coef < 0:
        return np.searchsorted(edges, coord, side="left") - 1
    else:
        return np.searchsorted(edges, coord, side="right") - 1


@numba.njit
def _increment(x0, x1):
    if x1 > x0:
        return 1
    else:
        return -1


@numba.njit
def _draw_line(xs, ys, x0, x1, y0, y1, xmin, xmax, ymin, ymax):
    """
    Generate line cell coordinates.

    Based on "A Fast Voxel Traversal Algorithm for Ray Tracing" by Amanatides
    & Woo, 1987.

    Note: out of bound values are marked with -1. This might be slightly
    misleading, since -1 is valid indexing value in Python. However, it is
    actually desirable in this case, since the values will be taken from a
    DataArray/Dataset. In this case, the section will automatically have the
    right dimension size, even with skipped parts at the start and end of the
    cross section.

    Parameters
    ----------
    xs : np.array
        x coordinates of cell edges
    ys : np.array
        y coordinates of cell edges
    x0, y0 : float
        starting point coordinates
    x1, y1 : float
        end point coordinates
    xmin, xmax, ymin, ymax : float
        grid bounds

    Returns
    -------
    ixs : np.array
        column indices, out of bound values marked with -1.
    iys : np.array
        row indices, out of bound values marked with -1.
    segment_length : np.array
        length of segment per sampled cell
    dxs : np.array
        length along column of segment per sampled cell
    dys : np.array
        length along row of segment per sampled cell
    """
    # Vector equations given by:
    # x = x0 + a_x * t
    # y = y0 + a_y * t
    # where t is the vector length; t_x and t_y are the inverse of a_x and a_y,
    # respectively.
    #
    # The algorithm works cell by cell. It computes the distance to cross the
    # cell boundary, in both x and y. We express this distance in the vector
    # space, tmax_x to the next x boundary, and tmax_y to the next y boundary.
    # We compare tmax_x and tmax_y and move the shortest distance.
    # We move by updating the indices, ix and iy, and taking a step along t.
    #
    # If we're outside the grid, we initialize the starting position on a cell
    # boundary.
    #
    # Addtionally, there's some logic to deal with start and end points that
    # fall outside of the grid bounding box.

    dx = x1 - x0
    dy = y1 - y0
    length = np.sqrt(dx**2 + dy**2)
    a_x = dx / length
    a_y = dy / length
    no_dx = dx == 0.0
    no_dy = dy == 0.0

    # Avoid ZeroDivision
    if no_dx:
        t_x = 0.0
    else:
        t_x = 1.0 / a_x

    if no_dy:
        t_y = 0.0
    else:
        t_y = 1.0 / a_y

    # Vector equations
    def x(t):
        return x0 + a_x * t

    def y(t):
        return y0 + a_y * t

    # Set increments; 1 or -1
    x_increment = _increment(x0, x1)
    y_increment = _increment(y0, y1)

    # Initialize start position of t
    if x0 < xmin:
        t = t_x * (xmin - x0)
    elif x0 > xmax:
        t = t_x * (xmax - x0)
    elif y0 < ymin:
        t = t_y * (ymin - y0)
    elif y0 > ymax:
        t = t_y * (ymax - y0)
    else:  # within grid bounds
        t = 0.0

    # Initialize end position of t
    if x1 < xmin:
        t_end = t_x * (xmin - x0)
    elif x1 > xmax:
        t_end = t_x * (xmax - x0)
    elif y1 < ymin:
        t_end = t_y * (ymin - y0)
    elif y1 > ymax:
        t_end = t_y * (ymax - y0)
    else:  # within grid bounds
        t_end = length

    # Collection of results
    ixs = []
    iys = []
    segment_length = []
    dxs = []
    dys = []

    # Store how much of the cross-section has no data
    skipped_start = t
    skipped_end = length - t_end
    if skipped_start > 0.0:
        ixs.append(-1)
        iys.append(-1)
        segment_length.append(skipped_start)
        dxs.append(x(skipped_start) - x(0))
        dys.append(y(skipped_start) - y(0))

    # Arbitrarily large number so it's always the largest one
    if no_dx:
        tmax_x = 1.0e20
    if no_dy:
        tmax_y = 1.0e20

    # First step
    ix = _index(xs, x(t), a_x)
    iy = _index(ys, y(t), a_y)
    ixs.append(ix)
    iys.append(iy)

    # Main loop, move through grid
    ncol = xs.size - 1
    nrow = ys.size - 1
    tstep = 0.0
    while ix < ncol and iy < nrow:
        # Compute distance to cell boundary
        # We need the start of the cell if we're moving in negative direction.
        if x_increment == -1:
            cellboundary_x = xs[ix]
        else:
            cellboundary_x = xs[ix + x_increment]

        if y_increment == -1:
            cellboundary_y = ys[iy]
        else:
            cellboundary_y = ys[iy + y_increment]

        # Compute max distance to move along t.
        # Checks for infinite slopes
        if not no_dx:
            # dx_t = cellboundary_x - x(t)
            tmax_x = t_x * (cellboundary_x - x(t))
        if not no_dy:
            # dy_t = cellboundary_y - y(t)
            tmax_y = t_y * (cellboundary_y - y(t))

        # Find which dimension requires smallest step along t
        if tmax_x == tmax_y:
            ix += x_increment
            iy += y_increment
            tstep = tmax_x
        elif tmax_x < tmax_y:
            ix += x_increment
            tstep = tmax_x
        else:
            iy += y_increment
            tstep = tmax_y

        if (t + tstep) < t_end:
            dxs.append(x(t + tstep) - x(t))
            dys.append(y(t + tstep) - y(t))
            t += tstep
            # Store
            ixs.append(ix)
            iys.append(iy)
            segment_length.append(tstep)
        else:
            tstep = t_end - t
            # Store final step
            dxs.append(x(t + tstep) - x(t))
            dys.append(y(t + tstep) - y(t))
            segment_length.append(tstep)
            break

    if skipped_end > 0.0:
        segment_length.append(skipped_end)
        ixs.append(-1)
        iys.append(-1)
        dxs.append(x(length) - x(t_end))
        dys.append(y(length) - y(t_end))

    # Because of numerical precision, extremely small segments might be
    # included. Those are filtered out here.
    ixs_a = np.array(ixs)
    iys_a = np.array(iys)
    dxs_a = np.array(dxs)
    dys_a = np.array(dys)
    segment_length_a = np.array(segment_length)
    use = np.abs(segment_length_a) > 1.0e-6

    return ixs_a[use], iys_a[use], segment_length_a[use], dxs_a[use], dys_a[use]


def _bounding_box(xmin, xmax, ymin, ymax):
    a = (xmin, ymin)
    b = (xmax, ymin)
    c = (xmax, ymax)
    d = (xmin, ymax)
    return sg.Polygon([a, b, c, d])


def _cross_section(data, linecoords):
    dx, xmin, xmax, dy, ymin, ymax = imod.util.spatial.spatial_reference(data)
    if isinstance(dx, float):
        dx_a = np.full(data.x.size, dx)
    if isinstance(dy, float):
        dy_a = np.full(data.y.size, dy)
    x_decreasing = data.indexes["x"].is_monotonic_decreasing
    y_decreasing = data.indexes["y"].is_monotonic_decreasing

    # Create vertex edges
    nrow = data.y.size
    ncol = data.x.size
    ys = np.full(nrow + 1, ymin)
    xs = np.full(ncol + 1, xmin)
    # Always increasing
    if x_decreasing:
        xs[1:] += np.abs(dx_a[::-1]).cumsum()
    else:
        xs[1:] += np.abs(dx_a).cumsum()
    if y_decreasing:
        ys[1:] += np.abs(dy_a[::-1]).cumsum()
    else:
        ys[1:] += np.abs(dy_a).cumsum()

    ixs = []
    iys = []
    sdxs = []
    sdys = []
    segments_list = []

    bounding_box = _bounding_box(xmin, xmax, ymin, ymax)
    for start, end in zip(linecoords[:-1], linecoords[1:]):
        linestring = sg.LineString([start, end])
        if not linestring.length:
            continue
        if linestring.intersects(bounding_box):
            x0, y0 = start
            x1, y1 = end
            i, j, segment_length, sdx, sdy = _draw_line(
                xs, ys, x0, x1, y0, y1, xmin, xmax, ymin, ymax
            )
        else:  # append the linestring in full as nodata section
            i = np.array([-1])
            j = np.array([-1])
            sdx = np.array([-1])
            sdy = np.array([-1])
            segment_length = np.array([linestring.length])

        ixs.append(i)
        iys.append(j)
        sdxs.append(sdx)
        sdys.append(sdy)
        segments_list.append(segment_length)

    if len(ixs) == 0:
        raise ValueError("Linestring does not intersect data")

    # Concatenate into a single array
    ixs = np.concatenate(ixs)
    iys = np.concatenate(iys)
    sdxs = np.concatenate(sdxs)
    sdys = np.concatenate(sdys)
    segments = np.concatenate(segments_list)

    # Flip around indexes
    if x_decreasing:
        ixs = ncol - 1 - ixs
        ixs[ixs >= ncol] = -1
    if y_decreasing:
        iys = nrow - 1 - iys
        iys[iys >= nrow] = -1

    # Select data
    # use .where to get rid of out of nodata parts
    ind_x = xr.DataArray(ixs, dims=["s"])
    ind_y = xr.DataArray(iys, dims=["s"])
    section = data.isel(x=ind_x, y=ind_y).where(ind_x >= 0)
    # Set dimension values
    section.coords["s"] = segments.cumsum() - 0.5 * segments
    section = section.assign_coords(ds=("s", segments))
    section = section.assign_coords(dx=("s", sdxs))
    section = section.assign_coords(dy=("s", sdys))
    # Without this sort, the is_increasing_monotonic property of the "s" index
    # in the DataArray returns False, and plotting the DataArray as a quadmesh
    # appears to fail. TODO: investigate, seems like an xarray issue.
    section = section.sortby("s")

    return section


def cross_section_line(data, start, end):
    r"""
    Obtain an interpolated cross-sectional slice through gridded data.
    Utilizing the interpolation functionality in ``xarray``, this function
    takes a vertical cross-sectional slice along a line through the given
    data on a regular (possibly non-equidistant) grid, which is given as an
    `xarray.DataArray` so that we can utilize its coordinate data.

    Adapted from Metpy:
    https://github.com/Unidata/MetPy/blob/main/src/metpy/interpolate/slices.py

    Parameters
    ----------
    data: `xarray.DataArray` or `xarray.Dataset`
        Three- (or higher) dimensional field(s) to interpolate. The DataArray
        (or each DataArray in the Dataset) must have been parsed by MetPy and
        include both an x and y coordinate dimension and the added ``crs``

        coordinate.
    start: (2, ) array_like
        A latitude-longitude pair designating the start point of the cross
        section.
    end: (2, ) array_like
        A latitude-longitude pair designating the end point of the cross
        section.

    Returns
    -------
    `xarray.DataArray` or `xarray.Dataset`
        The interpolated cross section, with new dimension "s" along the
        cross-section. The cellsizes along "s" are given in the "ds" coordinate.
    """
    # Check for intersection
    _, xmin, xmax, _, ymin, ymax = imod.util.spatial.spatial_reference(data)
    bounding_box = _bounding_box(xmin, xmax, ymin, ymax)
    if not sg.LineString([start, end]).intersects(bounding_box):
        raise ValueError("Line does not intersect data")

    linecoords = [start, end]
    return _cross_section(data, linecoords)


def cross_section_linestring(data, linestring):
    r"""
    Obtain an interpolated cross-sectional slice through gridded data.
    Utilizing the interpolation functionality in ``xarray``, this function
    takes a vertical cross-sectional slice along a linestring through the given
    data on a regular grid, which is given as an `xarray.DataArray` so that
    we can utilize its coordinate data.

    Adapted from Metpy:
    https://github.com/Unidata/MetPy/blob/main/src/metpy/interpolate/slices.py

    Parameters
    ----------
    data: `xarray.DataArray` or `xarray.Dataset`
        Three- (or higher) dimensional field(s) to interpolate. The DataArray
        (or each DataArray in the Dataset) must have been parsed by MetPy and
        include both an x and y coordinate dimension and the added ``crs``

        coordinate.
    linestring : shapely.geometry.LineString
        Shapely geometry designating the linestring along which to sample the
        cross section.

        Note that a LineString can easily be taken from a geopandas.GeoDataFrame
        using the .geometry attribute. Please refer to the examples.

    Returns
    -------
    `xarray.DataArray` or `xarray.Dataset`
        The interpolated cross section, with new index dimension along the
        cross-section.

    Examples
    --------
    Load a shapefile (that you might have drawn before using a GIS program),
    take a linestring from it, and use it to extract the data for a cross
    section.

    >>> geodataframe = gpd.read_file("cross_section.shp")
    >>> linestring = geodataframe.geometry[0]
    >>> section = cross_section_linestring(data, linestring)

    Or, construct the linestring directly in Python:

    >>> import shapely.geometry as sg
    >>> linestring = sg.LineString([(0.0, 1.0), (5.0, 5.0), (7.5, 5.0)])
    >>> section = cross_section_linestring(data, linestring)

    If you have drawn multiple cross sections within a shapefile, simply loop
    over the linestrings:

    >>> sections = [cross_section_linestring(data, ls) for ls in geodataframe.geometry]

    """
    linecoords = np.array(linestring.coords)
    return _cross_section(data, linecoords)
