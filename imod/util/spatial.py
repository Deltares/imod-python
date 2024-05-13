"""
Utility functions for dealing with the spatial
location of rasters: :func:`imod.util.spatial.coord_reference`,
:func:`imod.util.spatial_reference` and :func:`imod.util.transform`. These are
used internally, but are not private since they may be useful to users as well.
"""

import collections
import re
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Tuple, Union

import affine
import numpy as np
import pandas as pd
import xarray as xr
import xugrid as xu

from imod.typing import FloatArray, GridDataset, IntArray
from imod.util.imports import MissingOptionalModule

# since rasterio, shapely, rioxarray, and geopandas are a big dependencies that are
# sometimes hard to install and not always required, we made this an optional
# dependency
try:
    import rasterio
except ImportError:
    rasterio = MissingOptionalModule("rasterio")

try:
    import shapely
except ImportError:
    shapely = MissingOptionalModule("shapely")

if TYPE_CHECKING:
    import geopandas as gpd
else:
    try:
        import geopandas as gpd
    except ImportError:
        gpd = MissingOptionalModule("geopandas")

try:
    import rioxarray
except ImportError:
    rasterio = MissingOptionalModule("rioxarray")


def _xycoords(bounds, cellsizes) -> Dict[str, Any]:
    """Based on bounds and cellsizes, construct coords with spatial information"""
    # unpack tuples
    xmin, xmax, ymin, ymax = bounds
    dx, dy = cellsizes
    coords: collections.OrderedDict[str, Any] = collections.OrderedDict()
    # from cell size to x and y coordinates
    if isinstance(dx, (int, float, np.int_)):  # equidistant
        coords["x"] = np.arange(xmin + dx / 2.0, xmax, dx)
        coords["y"] = np.arange(ymax + dy / 2.0, ymin, dy)
        coords["dx"] = np.array(float(dx))
        coords["dy"] = np.array(float(dy))
    else:  # nonequidistant
        # even though IDF may store them as float32, we always convert them to float64
        dx = dx.astype(np.float64)
        dy = dy.astype(np.float64)
        coords["x"] = xmin + np.cumsum(dx) - 0.5 * dx
        coords["y"] = ymax + np.cumsum(dy) - 0.5 * dy
        if np.allclose(dx, dx[0]) and np.allclose(dy, dy[0]):
            coords["dx"] = np.array(float(dx[0]))
            coords["dy"] = np.array(float(dy[0]))
        else:
            coords["dx"] = ("x", dx)
            coords["dy"] = ("y", dy)
    return coords


def coord_reference(da_coord) -> Tuple[float, float, float]:
    """
    Extracts dx, xmin, xmax for a coordinate DataArray, where x is any coordinate.

    If the DataArray coordinates are nonequidistant, dx will be returned as
    1D ndarray instead of float.

    Parameters
    ----------
    a : xarray.DataArray of a coordinate

    Returns
    --------------
    tuple
        (dx, xmin, xmax) for a coordinate x
    """
    x = da_coord.values

    # Possibly non-equidistant
    dx_string = f"d{da_coord.name}"
    if dx_string in da_coord.coords:
        dx = da_coord.coords[dx_string]
        if (dx.shape == x.shape) and (dx.size != 1):
            # choose correctly for decreasing coordinate
            if dx[0] < 0.0:
                end = 0
                start = -1
            else:
                start = 0
                end = -1
            dx = dx.values.astype(np.float64)
            xmin = float(x.min()) - 0.5 * abs(dx[start])
            xmax = float(x.max()) + 0.5 * abs(dx[end])
            # As a single value if equidistant
            if np.allclose(dx, dx[0]):
                dx = dx[0]
        else:
            dx = float(dx)
            xmin = float(x.min()) - 0.5 * abs(dx)
            xmax = float(x.max()) + 0.5 * abs(dx)
    elif x.size == 1:
        raise ValueError(
            f"DataArray has size 1 along {da_coord.name}, so cellsize must be provided"
            f" as a coordinate named d{da_coord.name}."
        )
    else:  # Equidistant
        # TODO: decide on decent criterium for what equidistant means
        # make use of floating point epsilon? E.g:
        # https://github.com/ioam/holoviews/issues/1869#issuecomment-353115449
        dxs = np.diff(x.astype(np.float64))
        dx = dxs[0]
        atolx = abs(1.0e-4 * dx)
        if not np.allclose(dxs, dx, atolx):
            raise ValueError(
                f"DataArray has to be equidistant along {da_coord.name}, or cellsizes"
                f" must be provided as a coordinate named d{da_coord.name}."
            )

        # as xarray uses midpoint coordinates
        xmin = float(x.min()) - 0.5 * abs(dx)
        xmax = float(x.max()) + 0.5 * abs(dx)

    return dx, xmin, xmax


def spatial_reference(
    a: xr.DataArray,
) -> Tuple[float, float, float, float, float, float]:
    """
    Extracts spatial reference from DataArray.

    If the DataArray coordinates are nonequidistant, dx and dy will be returned
    as 1D ndarray instead of float.

    Parameters
    ----------
    a : xarray.DataArray

    Returns
    --------------
    tuple
        (dx, xmin, xmax, dy, ymin, ymax)

    """
    dx, xmin, xmax = coord_reference(a["x"])
    dy, ymin, ymax = coord_reference(a["y"])
    return dx, xmin, xmax, dy, ymin, ymax


def transform(a: xr.DataArray) -> affine.Affine:
    """
    Extract the spatial reference information from the DataArray coordinates,
    into an affine.Affine object for writing to e.g. rasterio supported formats.

    Parameters
    ----------
    a : xarray.DataArray

    Returns
    -------
    affine.Affine

    """
    dx, xmin, _, dy, _, ymax = spatial_reference(a)

    def equidistant(dx, name):
        if isinstance(dx, np.ndarray):
            if np.unique(dx).size == 1:
                return dx[0]
            else:
                raise ValueError(f"DataArray is not equidistant along {name}")
        else:
            return dx

    dx = equidistant(dx, "x")
    dy = equidistant(dy, "y")

    if dx < 0.0:
        raise ValueError("dx must be positive")
    if dy > 0.0:
        raise ValueError("dy must be negative")
    return affine.Affine(dx, 0.0, xmin, 0.0, dy, ymax)


def ugrid2d_data(da: xr.DataArray, face_dim: str) -> xr.DataArray:
    """
    Reshape a structured (x, y) DataArray into unstructured (face) form.
    Extra dimensions are maintained:
    e.g. (time, layer, x, y) becomes (time, layer, face).

    Parameters
    ----------
    da: xr.DataArray
        Structured DataArray with last two dimensions ("y", "x").

    Returns
    -------
    Unstructured DataArray with dimensions ("y", "x") replaced by ("face",).
    """
    if da.dims[-2:] != ("y", "x"):
        raise ValueError('Last two dimensions of da must be ("y", "x")')
    dims = da.dims[:-2]
    coords = {k: da.coords[k] for k in dims}
    return xr.DataArray(
        da.data.reshape(*da.shape[:-2], -1),
        coords=coords,
        dims=[*dims, face_dim],
        name=da.name,
    )


def unstack_dim_into_variable(dataset: GridDataset, dim: str) -> GridDataset:
    """
    Unstack each variable containing ``dim`` into separate variables.
    """
    unstacked = dataset.copy()

    variables_containing_dim = [
        variable for variable in dataset.data_vars if dim in dataset[variable].dims
    ]

    for variable in variables_containing_dim:
        stacked = unstacked[variable]
        unstacked = unstacked.drop_vars(variable)  # type: ignore
        for index in stacked[dim].values:
            unstacked[f"{variable}_{dim}_{index}"] = stacked.sel(
                indexers={dim: index}, drop=True
            )
    if dim in unstacked.coords:
        unstacked = unstacked.drop_vars(dim)
    return unstacked


def mdal_compliant_ugrid2d(dataset: xr.Dataset) -> xr.Dataset:
    """
    Ensures the xarray Dataset will be written to a UGRID netCDF that will be
    accepted by MDAL.

    * Unstacks variables with a layer dimension into separate variables.
    * Removes absent entries from the mesh topology attributes.
    * Sets encoding to float for datetime variables.

    Parameters
    ----------
    dataset: xarray.Dataset

    Returns
    -------
    unstacked: xr.Dataset

    """
    ds = unstack_dim_into_variable(dataset, "layer")

    # Find topology variables
    for variable in ds.data_vars:
        attrs = ds[variable].attrs
        if attrs.get("cf_role") == "mesh_topology":
            # Possible attributes:
            #
            # "cf_role"
            # "long_name"
            # "topology_dimension"
            # "node_dimension": required
            # "node_coordinates": required
            # "edge_dimension": optional
            # "edge_node_connectivity": optional
            # "face_dimension": required
            # "face_node_connectivity": required
            # "max_face_nodes_dimension": required
            # "face_coordinates": optional

            node_dim = attrs.get("node_dimension")
            edge_dim = attrs.get("edge_dimension")
            face_dim = attrs.get("face_dimension")

            # Drop the coordinates on the UGRID dimensions
            to_drop = []
            for dim in (node_dim, edge_dim, face_dim):
                if dim is not None and dim in ds.coords:
                    to_drop.append(dim)
            ds = ds.drop_vars(to_drop)

            if edge_dim and edge_dim not in ds.dims:
                attrs.pop("edge_dimension")

            face_coords = attrs.get("face_coordinates")
            if face_coords and face_coords not in ds.coords:
                attrs.pop("face_coordinates")

            edge_nodes = attrs.get("edge_node_connectivity")
            if edge_nodes and edge_nodes not in ds:
                attrs.pop("edge_node_connectivity")

    # Make sure time is encoded as a float for MDAL
    # TODO: MDAL requires all data variables to be float (this excludes the UGRID topology data)
    for var in ds.coords:
        if np.issubdtype(ds[var].dtype, np.datetime64):
            ds[var].encoding["dtype"] = np.float64

    return ds


def from_mdal_compliant_ugrid2d(dataset: xu.UgridDataset) -> xu.UgridDataset:
    """
    Undo some of the changes of ``mdal_compliant_ugrid2d``: re-stack the
    layers.

    Parameters
    ----------
    dataset: xugrid.UgridDataset

    Returns
    -------
    restacked: xugrid.UgridDataset

    """
    ds = dataset.ugrid.obj
    pattern = re.compile(r"(\w+)_layer_(\d+)")
    matches = [(variable, pattern.search(variable)) for variable in ds.data_vars]
    matches = [(variable, match) for (variable, match) in matches if match is not None]
    if not matches:
        return dataset

    # First deal with the variables that may remain untouched.
    other_vars = set(ds.data_vars).difference([variable for (variable, _) in matches])
    restacked = ds[list(other_vars)]

    # Next group by name, which will be the output dataset variable name.
    grouped = collections.defaultdict(list)
    for variable, match in matches:
        name, layer = match.groups()  # type: ignore
        da = ds[variable]
        grouped[name].append(da.assign_coords(layer=int(layer)))

    # Concatenate, and make sure the dimension order is natural.
    ugrid_dims = {dim for grid in dataset.ugrid.grids for dim in grid.dimensions}
    for variable, das in grouped.items():
        da = xr.concat(sorted(das, key=lambda da: da["layer"]), dim="layer")
        newdims = list(da.dims)
        newdims.remove("layer")
        # If it's a spatial dataset, the layer should be second last.
        if ugrid_dims.intersection(newdims):
            newdims.insert(-1, "layer")
        # If not, the layer should be last.
        else:
            newdims.append("layer")
        if tuple(newdims) != da.dims:
            da = da.transpose(*newdims)

        restacked[variable] = da

    return xu.UgridDataset(restacked, grids=dataset.ugrid.grids)


def to_ugrid2d(data: Union[xr.DataArray, xr.Dataset]) -> xr.Dataset:
    """
    Convert a structured DataArray or Dataset into its UGRID-2D quadrilateral
    equivalent.

    See:
    https://ugrid-conventions.github.io/ugrid-conventions/#2d-flexible-mesh-mixed-triangles-quadrilaterals-etc-topology

    Parameters
    ----------
    data: Union[xr.DataArray, xr.Dataset]
        Dataset or DataArray with last two dimensions ("y", "x").
        In case of a Dataset, the 2D topology is defined once and variables are
        added one by one.
        In case of a DataArray, a name is required; a name can be set with:
        ``da.name = "..."``'

    Returns
    -------
    ugrid2d_dataset: xr.Dataset
        The equivalent data, in UGRID-2D quadrilateral form.
    """
    if not isinstance(data, (xr.DataArray, xr.Dataset)):
        raise TypeError("data must be xarray.DataArray or xr.Dataset")

    grid = xu.Ugrid2d.from_structured(data)
    ds = grid.to_dataset()

    if isinstance(data, xr.Dataset):
        for variable in data.data_vars:
            ds[variable] = ugrid2d_data(data[variable], grid.face_dimension)
    if isinstance(data, xr.DataArray):
        if data.name is None:
            raise ValueError(
                'A name is required for the DataArray. It can be set with ``da.name = "..."`'
            )
        ds[data.name] = ugrid2d_data(data, grid.face_dimension)
    return mdal_compliant_ugrid2d(ds)


def gdal_compliant_grid(
    data: Union[xr.DataArray, xr.Dataset],
    crs: Optional[Any] = None,
) -> Union[xr.DataArray, xr.Dataset]:
    """
    Assign attributes to x,y coordinates to make data accepted by GDAL.

    Parameters
    ----------
    data: xr.DataArray | xr.Dataset
        Structured data with a x and y coordinate.
    crs: Any, Optional
        Anything accepted by rasterio.crs.CRS.from_user_input
        Requires ``rioxarray`` installed.

    Returns
    -------
    data with attributes to be accepted by GDAL.
    """
    x_attrs = {
        "axis": "X",
        "long_name": "x coordinate of projection",
        "standard_name": "projection_x_coordinate",
    }
    y_attrs = {
        "axis": "Y",
        "long_name": "y coordinate of projection",
        "standard_name": "projection_y_coordinate",
    }

    dims = set(data.dims)
    missing_dims = {"x", "y"} - dims

    if len(missing_dims) > 0:
        raise ValueError(f"Missing dimensions: {missing_dims}")

    x_coord_attrs = data.coords["x"].assign_attrs(x_attrs)
    y_coord_attrs = data.coords["y"].assign_attrs(y_attrs)

    data_gdal = data.assign_coords(x=x_coord_attrs, y=y_coord_attrs)

    if crs is not None:
        if isinstance(rioxarray, MissingOptionalModule):
            raise ModuleNotFoundError("rioxarray is required for this functionality")
        data_gdal.rio.write_crs(crs, inplace=True)

    return data_gdal

def empty_2d(
    dx: Union[float, FloatArray],
    xmin: float,
    xmax: float,
    dy: Union[float, FloatArray],
    ymin: float,
    ymax: float,
) -> xr.DataArray:
    """
    Create an empty 2D (x, y) DataArray.

    ``dx`` and ``dy`` may be provided as:

        * scalar: for equidistant spacing
        * array: for non-equidistant spacing

    Note that xarray (and netCDF4) uses midpoint coordinates. ``xmin`` and
    ``xmax`` are used to generate the appropriate midpoints.

    Parameters
    ----------
    dx: float, 1d array of floats
        cell size along x
    xmin: float
    xmax: float
    dy: float, 1d array of floats
        cell size along y
    ymin: float
    ymax: float

    Returns
    -------
    empty: xr.DataArray
        Filled with NaN.
    """
    bounds = (xmin, xmax, ymin, ymax)
    cellsizes = (np.abs(dx), -np.abs(dy))
    coords = _xycoords(bounds, cellsizes)
    nrow = coords["y"].size
    ncol = coords["x"].size
    return xr.DataArray(
        data=np.full((nrow, ncol), np.nan), coords=coords, dims=["y", "x"]
    )


def empty_3d(
    dx: Union[float, FloatArray],
    xmin: float,
    xmax: float,
    dy: Union[float, FloatArray],
    ymin: float,
    ymax: float,
    layer: Union[int, Sequence[int], IntArray],
) -> xr.DataArray:
    """
    Create an empty 2D (x, y) DataArray.

    ``dx`` and ``dy`` may be provided as:

        * scalar: for equidistant spacing
        * array: for non-equidistant spacing

    Note that xarray (and netCDF4) uses midpoint coordinates. ``xmin`` and
    ``xmax`` are used to generate the appropriate midpoints.

    Parameters
    ----------
    dx: float, 1d array of floats
        cell size along x
    xmin: float
    xmax: float
    dy: float, 1d array of floats
        cell size along y
    ymin: float
    ymax: float
    layer: int, sequence of integers, 1d array of integers

    Returns
    -------
    empty: xr.DataArray
        Filled with NaN.
    """
    bounds = (xmin, xmax, ymin, ymax)
    cellsizes = (np.abs(dx), -np.abs(dy))
    coords = _xycoords(bounds, cellsizes)
    nrow = coords["y"].size
    ncol = coords["x"].size
    layer = _layer(layer)
    coords["layer"] = layer

    return xr.DataArray(
        data=np.full((layer.size, nrow, ncol), np.nan),
        coords=coords,
        dims=["layer", "y", "x"],
    )


def empty_2d_transient(
    dx: Union[float, FloatArray],
    xmin: float,
    xmax: float,
    dy: Union[float, FloatArray],
    ymin: float,
    ymax: float,
    time: Any,
) -> xr.DataArray:
    """
    Create an empty transient 2D (time, x, y) DataArray.

    ``dx`` and ``dy`` may be provided as:

        * scalar: for equidistant spacing
        * array: for non-equidistant spacing

    Note that xarray (and netCDF4) uses midpoint coordinates. ``xmin`` and
    ``xmax`` are used to generate the appropriate midpoints.

    Parameters
    ----------
    dx: float, 1d array of floats
        cell size along x
    xmin: float
    xmax: float
    dy: float, 1d array of floats
        cell size along y
    ymin: float
    ymax: float
    time: Any
        One or more of: str, numpy datetime64, pandas Timestamp

    Returns
    -------
    empty: xr.DataArray
        Filled with NaN.
    """
    bounds = (xmin, xmax, ymin, ymax)
    cellsizes = (np.abs(dx), -np.abs(dy))
    coords = _xycoords(bounds, cellsizes)
    nrow = coords["y"].size
    ncol = coords["x"].size
    time = _time(time)
    coords["time"] = time
    return xr.DataArray(
        data=np.full((time.size, nrow, ncol), np.nan),
        coords=coords,
        dims=["time", "y", "x"],
    )


def empty_3d_transient(
    dx: Union[float, FloatArray],
    xmin: float,
    xmax: float,
    dy: Union[float, FloatArray],
    ymin: float,
    ymax: float,
    layer: Union[int, Sequence[int], IntArray],
    time: Any,
) -> xr.DataArray:
    """
    Create an empty transient 3D (time, layer, x, y) DataArray.

    ``dx`` and ``dy`` may be provided as:

        * scalar: for equidistant spacing
        * array: for non-equidistant spacing

    Note that xarray (and netCDF4) uses midpoint coordinates. ``xmin`` and
    ``xmax`` are used to generate the appropriate midpoints.

    Parameters
    ----------
    dx: float, 1d array of floats
        cell size along x
    xmin: float
    xmax: float
    dy: float, 1d array of floats
        cell size along y
    ymin: float
    ymax: float
    layer: int, sequence of integers, 1d array of integers
    time: Any
        One or more of: str, numpy datetime64, pandas Timestamp

    Returns
    -------
    empty: xr.DataArray
        Filled with NaN.
    """
    bounds = (xmin, xmax, ymin, ymax)
    cellsizes = (np.abs(dx), -np.abs(dy))
    coords = _xycoords(bounds, cellsizes)
    nrow = coords["y"].size
    ncol = coords["x"].size
    layer = _layer(layer)
    coords["layer"] = layer
    time = _time(time)
    coords["time"] = time
    return xr.DataArray(
        data=np.full((time.size, layer.size, nrow, ncol), np.nan),
        coords=coords,
        dims=["time", "layer", "y", "x"],
    )


def _layer(layer: Union[int, Sequence[int], IntArray]) -> IntArray:
    layer = np.atleast_1d(layer)
    if layer.ndim > 1:
        raise ValueError("layer must be 1d")
    return layer


def _time(time: Any) -> Any:
    time = np.atleast_1d(time)
    if time.ndim > 1:
        raise ValueError("time must be 1d")
    return pd.to_datetime(time)


def is_divisor(numerator: Union[float, FloatArray], denominator: float) -> bool:
    """
    Parameters
    ----------
    numerator: np.array of floats or float
    denominator: float

    Returns
    -------
    is_divisor: bool
    """
    denominator = np.abs(denominator)
    remainder = np.abs(numerator) % denominator
    return bool(np.all(np.isclose(remainder, 0.0) | np.isclose(remainder, denominator)))


def _polygonize(da: xr.DataArray) -> "gpd.GeoDataFrame":
    """
    Polygonize a 2D-DataArray into a GeoDataFrame of polygons.

    Private method located in util.spatial to work around circular imports.
    """

    if da.dims != ("y", "x"):
        raise ValueError('Dimensions must be ("y", "x")')

    values = da.values
    if values.dtype == np.float64:
        values = values.astype(np.float32)

    affine_transform = transform(da)
    shapes = rasterio.features.shapes(values, transform=affine_transform)

    geometries = []
    colvalues = []
    for geom, colval in shapes:
        geometries.append(shapely.geometry.Polygon(geom["coordinates"][0]))
        colvalues.append(colval)

    gdf = gpd.GeoDataFrame({"value": colvalues, "geometry": geometries})
    gdf.crs = da.attrs.get("crs")
    return gdf
