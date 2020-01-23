"""
This module creates unstructured grids from DataArrays.
Directly creating pyvista.UnstructuredGrids is advantageous for several reasons.

1. Structured (modflow) grids are rectilinear, so the pyvista.RectilinearGrid might
seem obvious. However, nodata values are also present, and have to removed using
a .treshold(). This means constructing every cell, followed by throwing most away.
This also returns an unstructured grid, and is slower.

2. Cells are rectangular, and they jump in the z dimension from one cell to the
other:
      __
  __ |2 |
 |1 ||__|
 |__|

In case of a rectilinear grid, the z value on the edge between cell 1 and 2 is
linearly interpolated, rather than making a jump. To create jumps, double the number
of cells is required, with dummy cells in between with width 0. This is clearly
wasteful.

The grid is constructed with:
offset, cells, cell_type, points, values

As presented here:
https://github.com/pyvista/pyvista/blob/master/examples/00-load/create-unstructured-surface.py

* offset: start of each cell in the cells array.
* cells: number of points in the cell, then point number; for every cell.
* cell_type: integers, informing vtk of the geometry type.
* points are the coordinates (x, y, z) for every cell corner.
* values: are the values from the data array. Can generally by used for coloring.

The methods below construct pyvista.UnstructuredGrids for voxel models (z1d),
"layer models" (z3d), and two dimensional data (e.g. a DEM).
"""
import numba
import numpy as np
import xarray as xr

from imod import util

try:
    import pyvista as pv
    import vtk
except ImportError:
    pass


@numba.njit
def _create_hexahedra_z1d(data, x, y, z):
    """
    Parameters
    ----------
    data : np.array of size (nz, ny, nx)
    x : np.array of size nx + 1
    y: np.array of size ny + 1
    z : np.array of size (nz + 1)

    Returns
    -------
    offset : np.array of int
    cells : np.array of int
    cell_type : np.array of vkt enum
    points : np.array of float
    values : np.array of float
    """
    nz, ny, nx = data.shape
    # First pass: count valid values
    n = 0
    for i in range(nz):
        for j in range(ny):
            for k in range(nx):
                if ~np.isnan(data[i, j, k]):
                    n += 1

    # Allocate
    # VTK_HEXAHEDRON is just an enum
    offset = np.arange(0, 9 * (n + 1), 9)
    cells = np.empty(n * 9)
    cell_type = np.full(n, vtk.VTK_HEXAHEDRON)
    # A hexahedron has 8 corners
    points = np.empty((n * 8, 3))
    values = np.empty(n)

    ii = 0
    jj = 0
    kk = 0
    for i in range(nz):
        for j in range(ny):
            for k in range(nx):
                if ~np.isnan(data[i, j, k]):
                    # Set coordinates of points
                    points[ii] = (x[k], y[j], z[i])
                    points[ii + 1] = (x[k + 1], y[j], z[i])
                    points[ii + 2] = (x[k + 1], y[j + 1], z[i])
                    points[ii + 3] = (x[k], y[j + 1], z[i])
                    points[ii + 4] = (x[k], y[j], z[i + 1])
                    points[ii + 5] = (x[k + 1], y[j], z[i + 1])
                    points[ii + 6] = (x[k + 1], y[j + 1], z[i + 1])
                    points[ii + 7] = (x[k], y[j + 1], z[i + 1])
                    # Set number of cells, and point number
                    cells[jj] = 8
                    cells[jj + 1] = ii
                    cells[jj + 2] = ii + 1
                    cells[jj + 3] = ii + 2
                    cells[jj + 4] = ii + 3
                    cells[jj + 5] = ii + 4
                    cells[jj + 6] = ii + 5
                    cells[jj + 7] = ii + 6
                    cells[jj + 8] = ii + 7
                    ii += 8
                    jj += 9
                    # Set values
                    values[kk] = data[i, j, k]
                    kk += 1

    return offset, cells, cell_type, points, values


@numba.njit
def _create_hexahedra_z3d(data, x, y, z3d):
    """
    Parameters
    ----------
    data : np.array of size (nz, ny, nx)
    x : np.array of size nx + 1
    y: np.array of size ny + 1
    z : np.array of size (nz + 1, ny, nx)

    Returns
    -------
    offset : np.array of int
    cells : np.array of int
    cell_type : np.array of vkt enum
    points : np.array of float
    values : np.array of float
    """
    nz, ny, nx = data.shape
    # First pass: count valid values
    n = 0
    for i in range(nz):
        for j in range(ny):
            for k in range(nx):
                if ~np.isnan(data[i, j, k]):
                    n += 1

    # Allocate
    # VTK_HEXAHEDRON is just an enum
    offset = np.arange(0, 9 * (n + 1), 9)
    cells = np.empty(n * 9)
    cell_type = np.full(n, vtk.VTK_HEXAHEDRON)
    # A hexahedron has 8 corners
    points = np.empty((n * 8, 3))
    values = np.empty(n)

    ii = 0
    jj = 0
    kk = 0
    for i in range(nz):
        for j in range(ny):
            for k in range(nx):
                v = data[i, j, k]
                if ~np.isnan(v):
                    # Set coordinates of points
                    points[ii] = (x[k], y[j], z3d[i, j, k])
                    points[ii + 1] = (x[k + 1], y[j], z3d[i, j, k])
                    points[ii + 2] = (x[k + 1], y[j + 1], z3d[i, j, k])
                    points[ii + 3] = (x[k], y[j + 1], z3d[i, j, k])
                    points[ii + 4] = (x[k], y[j], z3d[i + 1, j, k])
                    points[ii + 5] = (x[k + 1], y[j], z3d[i + 1, j, k])
                    points[ii + 6] = (x[k + 1], y[j + 1], z3d[i + 1, j, k])
                    points[ii + 7] = (x[k], y[j + 1], z3d[i + 1, j, k])
                    # Set number of cells, and point number
                    cells[jj] = 8
                    cells[jj + 1] = ii
                    cells[jj + 2] = ii + 1
                    cells[jj + 3] = ii + 2
                    cells[jj + 4] = ii + 3
                    cells[jj + 5] = ii + 4
                    cells[jj + 6] = ii + 5
                    cells[jj + 7] = ii + 6
                    cells[jj + 8] = ii + 7
                    ii += 8
                    jj += 9
                    # Set values
                    values[kk] = v
                    kk += 1

    return offset, cells, cell_type, points, values


@numba.njit
def _create_plane_surface(data, x, y, z):
    """
    Parameters
    ----------
    data : np.array of size (ny, nx)
    x : np.array of size nx + 1
    y: np.array of size ny + 1

    Returns
    -------
    offset : np.array of int
    cells : np.array of int
    cell_type : np.array of vkt enum
    points : np.array of float
    values : np.array of float
    """
    ny, nx = data.shape
    # First pass: count valid values
    n = 0
    for i in range(ny):
        for j in range(nx):
            if ~np.isnan(data[i, j]):
                n += 1

    # Allocate
    # VTK_HEXAHEDRON is just an enum
    offset = np.arange(0, 5 * (n + 1), 5)
    cells = np.empty(n * 5)
    cell_type = np.full(n, vtk.VTK_QUAD)
    # A hexahedron has r corners
    points = np.empty((n * 4, 3))
    values = np.empty(n)

    ii = 0
    jj = 0
    kk = 0
    for i in range(ny):
        for j in range(nx):
            v = data[i, j]
            if ~np.isnan(v):
                # Set coordinates of points
                points[ii] = (x[j], y[i], v)
                points[ii + 1] = (x[j + 1], y[i], v)
                points[ii + 2] = (x[j + 1], y[i + 1], v)
                points[ii + 3] = (x[j], y[i + 1], v)
                # Set number of cells, and point number
                cells[jj] = 4
                cells[jj + 1] = ii
                cells[jj + 2] = ii + 1
                cells[jj + 3] = ii + 2
                cells[jj + 4] = ii + 3
                ii += 4
                jj += 5
                # Set values
                values[kk] = v
                kk += 1

    return offset, cells, cell_type, points, values


def grid_3d(da):
    """
    Constructs a 3D PyVista representation of a DataArray.
    DataArrays should be two-dimensional or three-dimensional:

    * 2D: dimensions should be ``{"y", "x"}``. E.g. a DEM.
    * 3D: dimensions should be ``{"z", "y", "x"}``, for a voxel model.
    * 3D: dimensions should be ``{"layer", "y", "x"}``, with coordinates
        ``"top"({"layer", "y", "x"})`` and ``"bottom"({"layer", "y", "x"})``.

    Parameters
    ----------
    da : xr.DataArray

    Returns
    -------
    pyvista.UnstructuredGrid

    Examples
    --------

    >>> grid = imod.visualize.grid_3d(da)

    To plot the grid, call the ``.plot()`` method.

    >>> grid.plot()

    Use ``.assign_coords`` to assign tops and bottoms to layer models:

    >>> top = imod.idf.open("top*.idf")
    >>> bottom = imod.idf.open("bot*.idf")
    >>> kd = imod.idf.open("kd*.idf")
    >>> kd = kd.assign_coords(top=(("layer", "y", "x"), top))
    >>> kd = kd.assign_coords(bottom=(("layer", "y", "x"), bottom))
    >>> grid = imod.visualize.grid_3d(kd)
    >>> grid.plot()

    Refer to the PyVista documentation on how to customize plots:
    https://docs.pyvista.org/index.html
    """
    # x and y dimension
    dx, xmin, xmax, dy, ymin, ymax = util.spatial_reference(da)
    if isinstance(dx, float):
        dx = np.full(da.x.size, dx)
    if isinstance(dy, float):
        dy = np.full(da.y.size, dy)
    nx = da.coords["x"].size
    ny = da.coords["y"].size
    # TODO: Currently assuming dx positive, dy negative
    x = np.full(nx + 1, xmin)
    y = np.full(ny + 1, ymax)
    x[1:] += dx.cumsum()
    y[1:] += dy.cumsum()

    # z dimension
    if "top" in da.coords and "bottom" in da.coords:
        if not len(da.shape) == 3:
            raise ValueError(
                'Coordinates "top" and "bottom" are present, but data is not 3D.'
            )
        if not set(da.dims) == {"layer", "y", "x"}:
            raise ValueError(
                'Coordinates "top" and "bottom" are present, only dimensions allowed are: '
                '{"layer", "y", "x"}'
            )
        da = da.transpose("layer", "y", "x", transpose_coords=True)
        ztop = da.coords["top"]
        zbot = da.coords["bottom"]
        z3d = np.vstack([np.expand_dims(ztop.isel(layer=1).values, 0), zbot.values])
        offset, cells, cell_type, points, values = _create_hexahedra_z3d(
            da.values, x, y, z3d
        )
    elif "z" in da.coords:
        if not len(da.shape) == 3:
            raise ValueError('Coordinate "z" is present, but data is not 3D.')
        if not (set(da.dims) == {"layer", "y", "x"} or set(da.dims) == {"z", "y", "x"}):
            raise ValueError(
                'Coordinate "z" is present, only dimensions allowed are: '
                '{"layer", "y", "x"} or {"z", "y", "x"}'
            )
        if "z" not in da.dims:
            da = da.transpose("layer", "y", "x", transpose_coords=True)
        else:
            da = da.transpose("z", "y", "x", transpose_coords=True)

        dz, zmin, zmax = util.coord_reference(da["z"])
        nz = da.coords["z"].size
        z = np.full(nz + 1, zmin)
        if isinstance(dz, float):
            dz = np.full(nz, dz)
        z[1:] += dz.cumsum()
        offset, cells, cell_type, points, values = _create_hexahedra_z1d(
            da.values, x, y, z
        )
    elif set(da.dims) == {"y", "x"}:
        da = da.transpose("y", "x", transpose_coords=True)
        offset, cells, cell_type, points, values = _create_plane_surface(
            da.values, x, y
        )
    else:
        raise ValueError(
            'Incorrect coordinates and/or dimensions: Neither "z" nor "top" and '
            '"bottom" is present in the DataArray, but dimension are not {"y", "x"} '
            " either."
        )

    grid = pv.UnstructuredGrid(offset, cells, cell_type, points)
    grid.cell_arrays["values"] = values
    return grid


# For arrow plots: compute magnitude of vector
# Downsample, using max rule.
# Upsample again
# Select with .where?
# Create an array of (n, 3) shape
# Vectors are located somewhere
