import numba
import numpy as np
import pyvista as pv
import vtk
import xarray as xr

from imod import util


def to_grid(data):
    """
    Parameters
    ----------
    data : xr.DataArray

    Returns
    -------
    pyvista.RectilinearGrid or pyvista.StructuredGrid
    """
    # x and y dimension
    dx, xmin, xmax, dy, ymin, ymax = util.spatial_reference(data)
    if isinstance(dx, float):
        dx = np.full(data.x.size, dx)
    if isinstance(dy, float):
        dy = np.full(data.y.size, dy)
    nx = da.coords["x"].size
    ny = da.coords["y"].size
    # TODO: x and y increasing...
    x = np.full(nx + 1, xmin)
    y = np.full(ny + 1, ymin)
    x[1:] += dx.cumsum()
    y[1:] += dy.cumsum()
    # z dimension
    if "top" in da.coords and "bottom" in da.coords:
        # TODO
        raise NotImplementedError
    elif "z" in da.coords:
        dz, zmin, zmax = imod.util.coord_reference(da["z"])
        nz = da.coords["z"].size
        z = np.full(nz + 1, zmin)
        if isinstance(dz, float):
            dz = np.full(nz, dz)
        z[1:] += dz.cumsum()

    grid = pv.RectilinearGrid(x, y, z)
    grid.cell_arrays["values"] = da.values.ravel()
    # Test what is faster: Rectilinear -> filter
    # or: da.where -> unstructured
    # Unstructured might be required anyway.
    # Prisms of six corners, always. Pretty easy to generate.
    grid = grid.threshold([da.min(), da.max()])
    # This is sufficient for voxel models.
    # This is insufficient for varying tops and bottoms.
    # We need to specify all corners exactly of the prisms.
    # The question is whether building an unstructured grid isn't faster
    # Likely not, because rectilinear might still be able to make assumptions
    # about dimensions.
    # Cheaply: approximate, linear interpolation.
    return grid


@numba.njit
def _create_hexahedra_z1d(data, x, y, z):
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
def _create_plane_surface(data, x, y):
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
    ny, nx = data.shape
    # First pass: count valid values
    n = 0
    for i in range(ny):
        for j in range(nx):
            if ~np.isnan(data[i, j]):
                n += 1

    # Allocate
    # VTK_HEXAHEDRON is just an enum
    offset = np.arange(0, 4 * (n + 1), 4)
    cells = np.empty(n * 4)
    cell_type = np.full(n, vtk.VTK_PLANE_SURFACE)
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
                points[ii] = (x[k], y[j], v)
                points[ii + 1] = (x[k + 1], y[j], v)
                points[ii + 2] = (x[k + 1], y[j + 1], v)
                points[ii + 3] = (x[k], y[j + 1], v)
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

    return offset, cells, cell_type, points, value


def to_grid(data):
    """
    2.44 ms for filtering
    1.64 ms for this one
    Should be much more efficient with sparse data.

    Parameters
    ----------
    data : xr.DataArray

    Returns
    -------
    pyvista.RectilinearGrid or pyvista.StructuredGrid
    """
    # x and y dimension
    dx, xmin, xmax, dy, ymin, ymax = util.spatial_reference(data)
    if isinstance(dx, float):
        dx = np.full(data.x.size, dx)
    if isinstance(dy, float):
        dy = np.full(data.y.size, dy)
    nx = da.coords["x"].size
    ny = da.coords["y"].size
    # TODO: x and y increasing...
    x = np.full(nx + 1, xmin)
    y = np.full(ny + 1, ymin)
    x[1:] += dx.cumsum()
    y[1:] += dy.cumsum()
    # z dimension
    if "top" in da.coords and "bottom" in da.coords:
        ztop = da.coords["top"].transpose("layer", "y", "x")
        zbot = da.coords["bottom"].transpose("layer", "y", "x")
        z3d = np.stack([ztop.isel(layer=1).values, zbot.values])
        offset, cells, cell_type, points, values = _create_hexahedra_z3d(
            data.values, x, y, z3d
        )
    elif "z" in da.coords:
        dz, zmin, zmax = imod.util.coord_reference(da["z"])
        nz = da.coords["z"].size
        z = np.full(nz + 1, zmin)
        if isinstance(dz, float):
            dz = np.full(nz, dz)
        z[1:] += dz.cumsum()
        offset, cells, cell_type, points, values = _create_hexahedra_z1d(
            data.values, x, y, z
        )
    else:  # surface plot
        if not da.dims == ("y", "x"):
            raise ValueError()
        offset, cells, cell_type, points, values = _create_plane_surface(
            data.values, x, y
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
