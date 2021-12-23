"""
This module creates unstructured grids from DataArrays.
Directly creating pyvista.UnstructuredGrids is advantageous for several reasons.

1. Structured (modflow) grids are rectilinear, so the pyvista.RectilinearGrid might
seem obvious. However, nodata values are also present, and have to be removed using
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
https://github.com/pyvista/pyvista/blob/0.23.0/examples/00-load/create-unstructured-surface.py

* offset: start of each cell in the cells array.
* cells: number of points in the cell, then point number; for every cell.
* cell_type: integers, informing vtk of the geometry type.
* points are the coordinates (x, y, z) for every cell corner.
* values: are the values from the data array. Can generally by used for coloring.

The methods below construct pyvista.UnstructuredGrids for voxel models (z1d),
"layer models" (z3d), and two dimensional data (e.g. a DEM).
"""

from pathlib import Path
from typing import Optional, Tuple

import numba
import numpy as np
import pandas as pd
import scipy.ndimage.morphology
import tqdm
import xarray as xr

from imod import util
from imod.select import points_values

try:
    import pyvista as pv
    import vtk

    if vtk.vtkVersion().GetVTKMajorVersion() < 9:
        raise ImportError("VTK version of 9.0 or higher required")
except ImportError:
    pv = util.MissingOptionalModule("pyvista")
    vtk = util.MissingOptionalModule("vtk")


def exterior(da, n):
    has_data = da.notnull()
    eroded = da.copy(data=scipy.ndimage.binary_erosion(has_data.values, iterations=n))
    return has_data & ~eroded


@numba.njit
def _create_hexahedra_z1d(data, x, y, z):
    """
    This function creates the necessary arrays to create hexahedra, based on a
    one-dimensional z array: i.e. one that does not depend on x and y.
    These arrays are used to create a pyvista.UnstructuredGrid.

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
    cells = np.empty(n * 9, dtype=np.int32)
    indices = np.empty(n, dtype=np.int32)
    cell_type = np.full(n, vtk.VTK_HEXAHEDRON)
    # A hexahedron has 8 corners
    points = np.empty((n * 8, 3))
    values = np.empty(n, data.dtype)

    ii = 0
    jj = 0
    kk = 0
    linear_index = 0
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
                    # Set indices
                    indices[kk] = linear_index
                    kk += 1
                linear_index += 1

    return indices, offset, cells, cell_type, points, values


@numba.njit
def _create_hexahedra_z3d(data, x, y, z3d):
    """
    This function creates the necessary arrays to create hexahedra, based on a
    three-dimensional z array: i.e. one that depends on x and y.
    These arrays are used to create a pyvista.UnstructuredGrid.

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
    cells = np.empty(n * 9, dtype=np.int32)
    indices = np.empty(n, dtype=np.int32)
    cell_type = np.full(n, vtk.VTK_HEXAHEDRON)
    # A hexahedron has 8 corners
    points = np.empty((n * 8, 3))
    values = np.empty(n, data.dtype)

    ii = 0
    jj = 0
    kk = 0
    linear_index = 0
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
                    # Set index
                    indices[kk] = linear_index
                    kk += 1
                linear_index += 1

    return indices, offset, cells, cell_type, points, values


@numba.njit
def _create_plane_surface(data, x, y):
    """
    This function creates the necessary arrays to create quads, based on a
    two-dimensional data array. The data array is used for the z dimension.
    All the horizontal surfaces are connected by vertical faces, exactly
    representing the cell geometry, with a constant value per cell.
    The alternative is linear interpolation, which does not represent
    geometry exactly, or holes in the surface, with isolated quads floating
    in space.

    These arrays are used to create a pyvista.UnstructuredGrid.

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
                if j < nx - 1:
                    if ~np.isnan(data[i, j + 1]):
                        n += 1
                if i < ny - 1:
                    if ~np.isnan(data[i + 1, j]):
                        n += 1

    # Allocate
    # VTK_QUAD is just an enum
    offset = np.arange(0, 5 * (n + 1), 5)
    cells = np.empty(n * 5, dtype=np.int32)
    cell_type = np.full(n, vtk.VTK_QUAD)
    # A hexahedron has r corners
    points = np.empty((n * 4, 3))
    values = np.empty(n, dtype=data.dtype)

    ii = 0
    jj = 0
    kk = 0
    for i in range(ny):
        for j in range(nx):
            v = data[i, j]
            # Create horizontal quad
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
                # Set index
                kk += 1

                if j < nx - 1:
                    v01 = data[i, j + 1]
                    # Create vertical quads
                    if ~np.isnan(v01):
                        # Set coordinates of points
                        points[ii] = (x[j + 1], y[i], v)
                        points[ii + 1] = (x[j + 1], y[i + 1], v)
                        points[ii + 2] = (x[j + 1], y[i + 1], v01)
                        points[ii + 3] = (x[j + 1], y[i], v01)
                        # Set number of cells, and point number
                        cells[jj] = 4
                        cells[jj + 1] = ii
                        cells[jj + 2] = ii + 1
                        cells[jj + 3] = ii + 2
                        cells[jj + 4] = ii + 3
                        ii += 4
                        jj += 5
                        # Set values
                        values[kk] = v if v > v01 else v01
                        # Set index
                        kk += 1

                if i < ny - 1:
                    v10 = data[i + 1, j]
                    if ~np.isnan(v10):
                        # Set coordinates of points
                        points[ii] = (x[j], y[i + 1], v)
                        points[ii + 1] = (x[j + 1], y[i + 1], v)
                        points[ii + 2] = (x[j + 1], y[i + 1], v10)
                        points[ii + 3] = (x[j], y[i + 1], v10)
                        # Set number of cells, and point number
                        cells[jj] = 4
                        cells[jj + 1] = ii
                        cells[jj + 2] = ii + 1
                        cells[jj + 3] = ii + 2
                        cells[jj + 4] = ii + 3
                        ii += 4
                        jj += 5
                        # Set values
                        values[kk] = v if v > v10 else v10
                        # Set index
                        kk += 1

    return offset, cells, cell_type, points, values


def vertices_coords(dx, xmin, xmax, nx):
    """
    Return the coordinates of the vertices.
    (xarray stores midpoints)
    """
    if isinstance(dx, float):
        dx = np.full(nx, dx)
    if dx[0] > 0:
        x = np.full(nx + 1, xmin)
    else:
        x = np.full(nx + 1, xmax)
    x[1:] += dx.cumsum()
    return x


def grid_3d(
    da,
    vertical_exaggeration=30.0,
    exterior_only=True,
    exterior_depth=1,
    return_index=False,
):
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
    vertical_exaggeration : float, default 30.0
    exterior_only : bool, default True
        Whether or not to only draw the exterior. Greatly speeds up rendering,
        but it means that pyvista slices and filters produce "hollowed out"
        results.
    exterior_depth : int, default 1
        How many cells to consider as exterior. In case of large jumps, holes
        can occur. By settings this argument to a higher value, more of the
        inner cells will be rendered, reducing the chances of gaps occurring.
    return_index : bool, default False

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
    nx = da.coords["x"].size
    ny = da.coords["y"].size
    x = vertices_coords(dx, xmin, xmax, nx)
    y = vertices_coords(dy, ymin, ymax, ny)
    # Coordinates should always have dtype == np.float64 by now

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

        if exterior_only:
            da = da.where(exterior(da, exterior_depth))

        indices, offset, cells, cell_type, points, values = _create_hexahedra_z3d(
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
        z = vertices_coords(dz, zmin, zmax, nz)

        if exterior_only:
            da = da.where(exterior(da, exterior_depth))

        indices, offset, cells, cell_type, points, values = _create_hexahedra_z1d(
            da.values, x, y, z
        )
    elif set(da.dims) == {"y", "x"}:
        if return_index:
            raise ValueError("Cannot return indices for a 2D dataarray.")
        da = da.transpose("y", "x", transpose_coords=True)
        offset, cells, cell_type, points, values = _create_plane_surface(
            da.values.astype(np.float64), x, y
        )
    else:
        raise ValueError(
            'Incorrect coordinates and/or dimensions: Neither "z" nor "top" and '
            '"bottom" is present in the DataArray, but dimension are not {"y", "x"} '
            " either."
        )

    grid = pv.UnstructuredGrid(cells, cell_type, points)
    grid.points[:, -1] *= vertical_exaggeration
    grid.cell_arrays["values"] = values

    if return_index:
        return grid, indices
    else:
        return grid


# For arrow plots: compute magnitude of vector
# Downsample, using max rule.
# Upsample again
# Select with .where?
# Create an array of (n, 3) shape
# Vectors are located somewhere


def line_3d(polygon, z=0.0):
    """
    Returns the exterior line of a shapely polygon.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
    z : float or xr.DataArray
        z-coordinate to assign to line. If DataArray, assigns z-coordinate
        based on xy locations in DataArray.

    Returns
    -------
    pyvista.PolyData
    """
    x, y = map(np.array, polygon.exterior.coords.xy)
    if isinstance(z, xr.DataArray):
        z = points_values(z, x=x, y=y).values
    else:
        z = np.full_like(x, z)
    coords = np.vstack([x, y, z]).transpose()
    return pv.lines_from_points(coords)


class GridAnimation3D:
    """
    Class to easily setup 3D animations for transient data.
    Use the ``imod.visualize.StaticGridAnimation3D`` when the location of the
    displayed cells is constant over time: it will render much faster.

    You can iteratively add or change settings to the plotter, until you're
    satisfied. Call the ``.peek()`` method to take a look. When satisfied, call
    ``.output()`` to write to a file.


    Parameters
    ----------
    da : xr.DataArray
        The dataarray with transient data. Must contain a "time" dimension.
    vertical_exaggeration : float, defaults to 30.0
    mesh_kwargs : dict
        keyword arguments that are forwarded to the pyvista mesh representing
        "da". If "stitle" is given as one of the arguments, the special keyword "timestamp"
        can be used to render the plotted time as part of the title. See example.
    plotter_kwargs : dict
        keyword arguments that are forwarded to the pyvista plotter.

    Examples
    --------

    Initialize the animation:

    >>> animation = imod.visualize.GridAnimation3D(concentration, mesh_kwargs=dict(cmap="jet"))

    Check what it looks like (if a window pops up: press "q" instead of the X to return):

    >>> animation.peek()

    Change the camera position, add bounding box, and check the result:

    >>> animation.plotter.camera_position = (2, 1, 0.5)
    >>> animation.plotter.add_bounding_box()
    >>> animation.peek()

    When it looks good, write to a file:

    >>> animation.write("example.mp4")

    If you've made some changes that don't look good, call ``.reset()`` to start over:

    >>> animation.reset()

    Note that ``.reset()`` is automatically called when the animation has finished writing.

    You can use "stitle" in mesh_kwargs in conjunction with the "timestamp" keyword to print
    a formatted timestamp in the animation:

    >>> animation = imod.visualize.GridAnimation3D(concentration, mesh_kwargs=dict(stitle="Concentration on {timestamp:%Y-%m-%d}"))
    """

    def _initialize(self, da):
        self.mesh = grid_3d(
            da, vertical_exaggeration=self.vertical_exaggeration, return_index=False
        )
        mesh_kwargs = self.mesh_kwargs.copy()
        if "stitle" in mesh_kwargs and "{timestamp" in mesh_kwargs["stitle"]:
            mesh_kwargs["stitle"] = mesh_kwargs["stitle"].format(
                timestamp=pd.Timestamp(da.time.values)
            )

        self.mesh_actor = self.plotter.add_mesh(self.mesh, **mesh_kwargs)

    def _update(self, da):
        self.plotter.remove_actor(self.mesh_actor)
        self._initialize(da)

    def __init__(
        self, da, vertical_exaggeration=30.0, mesh_kwargs={}, plotter_kwargs={}
    ):
        # Store data
        self.da = da
        self.vertical_exaggeration = vertical_exaggeration
        self.mesh_kwargs = mesh_kwargs.copy()
        self.plotter_kwargs = plotter_kwargs
        self.plotter = pv.Plotter(**plotter_kwargs)
        # Initialize pyvista objects
        self._initialize(da.isel(time=0))

    def peek(self):
        """
        Display the current state of the animation plotter.
        """
        self.plotter.show(auto_close=False)

    def reset(self):
        """
        Reset the plotter to its base state.
        """
        self.plotter = pv.Plotter(**self.plotter_kwargs)
        self._update(self.da.isel(time=0))

    def write(self, filename, framerate=24):
        """
        Write the animation to a video or gif.

        Resets the plotter when finished animating.

        Parameters
        ----------
        filename : str, pathlib.Path
            Filename to write the video to. Should be an .mp4 or .gif.
        framerate : int, optional
            Frames per second. Not honoured for gif.
        """
        if Path(filename).suffix.lower() == ".gif":
            self.plotter.open_gif(
                Path(filename).with_suffix(".gif").as_posix()
            )  # only lowercase gif and no Path allowed
        else:
            self.plotter.open_movie(filename, framerate=framerate)
        self.plotter.show(auto_close=False, interactive=False)
        self.plotter.write_frame()

        for itime in tqdm.tqdm(range(1, self.da.coords["time"].size)):
            self._update(self.da.isel(time=itime))
            self.plotter.write_frame()

        # Close and reinitialize
        self.plotter.close()
        self.reset()


class StaticGridAnimation3D(GridAnimation3D):
    """
    Class to easily setup 3D animations for transient data;
    Should only be used when the location of the displayed cells is constant
    over time. It will render much faster than ``imod.visualize.GridAnimation3D``.

    Refer to examples of ``imod.visualize.GridAnimation3D``.
    """

    def _initialize(self, da):
        self.mesh, self.indices = grid_3d(
            da, vertical_exaggeration=self.vertical_exaggeration, return_index=True
        )

    def _update(self, da):
        self.mesh.cell_arrays["values"] = da.values.ravel()[self.indices]


def velocity_field(
    vx: xr.DataArray,
    vy: xr.DataArray,
    vz: xr.DataArray,
    z: Optional[xr.DataArray] = None,
    vertical_exaggeration: Optional[float] = 30.0,
    scale_by_magnitude: Optional[bool] = True,
    factor: Optional[float] = 1.0,
    tolerance: Optional[float] = 0.0,
    absolute: Optional[bool] = False,
    clamping: Optional[bool] = False,
    rng: Optional[Tuple[float, float]] = None,
):
    if z is None:
        z = vx["z"].values

    if not (vx.shape == vy.shape == vz.shape):
        raise ValueError("Shapes of velocity components vx, vy, vz do not match")
    if not (vx.dims == ("layer", "y", "x") or vx.dims == ("z", "y", "x")):
        raise ValueError(
            'Velocity components must have dimensions ("layer", "y", "x") '
            'or ("z", "y", "x") exactly.\n'
            f"Received {vx.dims} instead."
        )

    # Start by generating the location of the velocity arrows
    # Ensure the points follow the memory layout of the v DataArrays (z, y, x)
    # otherwise, the result of np.ravel() doesn't match up
    if z.dim == 1:
        zz, yy, xx = np.meshgrid(z, vx["y"].values, vx["x"].values, indexing="ij")
    elif z.dim == 3:
        if not z.shape == vx.shape:
            raise ValueError("Shape of `z` does not match velocity components.")
        _, yy, xx = np.meshgrid(
            np.arange(z.shape[0]), vx["x"].values, vy["y"].avlues, indexing="ij"
        )
        zz = z
    else:
        raise ValueError("z should be one or three dimensional.")

    zz *= vertical_exaggeration
    cellcenters = pv.PolyData(np.stack([np.ravel(x) for x in (xx, yy, zz)], axis=1))
    # Add the velocity components in x, y, z order
    cellcenters["velocity"] = np.stack([x.data.ravel() for x in (vx, vy, vz)], axis=1)
    scale = "velocity" if scale_by_magnitude else None

    return cellcenters.glyph(
        scale=scale,
        orient="velocity",
        factor=factor,
        tolerance=tolerance,
        absolute=absolute,
        clamping=clamping,
        rng=rng,
    )
