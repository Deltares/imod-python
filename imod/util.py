"""
Miscellaneous Utilities.

Conventional IDF filenames can be understood and constructed using
:func:`imod.util.decompose` and :func:`imod.util.compose`. These are used
automatically in :func:`imod.idf`.

Furthermore there are some utility functions for dealing with the spatial
location of rasters: :func:`imod.util.coord_reference`,
:func:`imod.util.spatial_reference` and :func:`imod.util.transform`. These are
used internally, but are not private since they may be useful to users as well.
"""

import collections
import contextlib
import datetime
import functools
import os
import pathlib
import re
import tempfile
import warnings
from typing import Any, Dict, List, Sequence, Tuple, Union

import affine
import cftime
import dateutil
import numpy as np
import pandas as pd
import xarray as xr
import xugrid

try:
    Pattern = re._pattern_type
except AttributeError:
    Pattern = re.Pattern  # Python 3.7+


FloatArray = np.ndarray
IntArray = np.ndarray


DATETIME_FORMATS = {
    14: "%Y%m%d%H%M%S",
    12: "%Y%m%d%H%M",
    10: "%Y%m%d%H",
    8: "%Y%m%d",
    4: "%Y",
}


def to_datetime(s):
    try:
        time = datetime.datetime.strptime(s, DATETIME_FORMATS[len(s)])
    except (ValueError, KeyError):  # Try fullblown dateutil date parser
        time = dateutil.parser.parse(s)
    return time


def _groupdict(stem: str, pattern: str) -> Dict:
    if pattern is not None:
        if isinstance(pattern, Pattern):
            d = pattern.match(stem).groupdict()
        else:
            pattern = pattern.lower()
            # Get the variables between curly braces
            in_curly = re.compile(r"{(.*?)}").findall(pattern)
            regex_parts = {key: f"(?P<{key}>[\\w.-]+)" for key in in_curly}
            # Format the regex string, by filling in the variables
            simple_regex = pattern.format(**regex_parts)
            re_pattern = re.compile(simple_regex)
            # Use it to get the required variables
            d = re_pattern.match(stem).groupdict()
    else:  # Default to "iMOD conventions": {name}_c{species}_{time}_l{layer}
        has_layer = bool(re.search(r"_l\d+$", stem))
        has_species = bool(
            re.search(r"conc_c\d{1,3}_\d{8,14}", stem)
        )  # We are strict in recognizing species
        try:  # try for time
            base_pattern = r"(?P<name>[\w-]+)"
            if has_species:
                base_pattern += r"_c(?P<species>[0-9]+)"
            base_pattern += r"_(?P<time>[0-9-]{6,})"
            if has_layer:
                base_pattern += r"_l(?P<layer>[0-9]+)"
            re_pattern = re.compile(base_pattern)
            d = re_pattern.match(stem).groupdict()
        except AttributeError:  # probably no time
            base_pattern = r"(?P<name>[\w-]+)"
            if has_species:
                base_pattern += r"_c(?P<species>[0-9]+)"
            if has_layer:
                base_pattern += r"_l(?P<layer>[0-9]+)"
            re_pattern = re.compile(base_pattern)
            d = re_pattern.match(stem).groupdict()
    return d


def decompose(path, pattern: str = None) -> Dict[str, Any]:
    r"""
    Parse a path, returning a dict of the parts, following the iMOD conventions.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the file. Upper case is ignored.
    pattern : str, regex pattern, optional
        If the path is not made up of standard paths, and the default decompose
        does not produce the right result, specify the used pattern here. See
        the examples below.

    Returns
    -------
    d : dict
        Dictionary with name of variable and dimensions

    Examples
    --------
    Decompose a path, relying on default conventions:

    >>> decompose("head_20010101_l1.idf")

    Do the same, by specifying a format string pattern, excluding extension:

    >>> decompose("head_20010101_l1.idf", pattern="{name}_{time}_l{layer}")

    This supports an arbitrary number of variables:

    >>> decompose("head_slr_20010101_l1.idf", pattern="{name}_{scenario}_{time}_l{layer}")

    The format string pattern will only work on tidy paths, where variables are
    separated by underscores. You can also pass a compiled regex pattern.
    Make sure to include the ``re.IGNORECASE`` flag since all paths are lowered.

    >>> import re
    >>> pattern = re.compile(r"(?P<name>[\w]+)L(?P<layer>[\d+]*)")
    >>> decompose("headL11", pattern=pattern)

    However, this requires constructing regular expressions, which is generally
    a fiddly process. The website https://regex101.com is a nice help.
    Alternatively, the most pragmatic solution may be to just rename your files.
    """
    path = pathlib.Path(path)
    # We'll ignore upper case
    stem = path.stem.lower()

    d = _groupdict(stem, pattern)
    dims = list(d.keys())
    # If name is not provided, generate one from other fields
    if "name" not in d.keys():
        d["name"] = "_".join(d.values())
    else:
        dims.remove("name")

    # TODO: figure out what to with user specified variables
    # basically type inferencing via regex?
    # if purely numerical \d* -> int or float
    #    if \d*\.\d* -> float
    # else: keep as string

    # String -> type conversion
    if "layer" in d.keys():
        d["layer"] = int(d["layer"])
    if "species" in d.keys():
        d["species"] = int(d["species"])
    if "time" in d.keys():
        d["time"] = to_datetime(d["time"])
    if "steady-state" in d["name"]:
        # steady-state as time identifier isn't picked up by <time>[0-9] regex
        d["name"] = d["name"].replace("_steady-state", "")
        d["time"] = "steady-state"
        dims.append("time")

    d["extension"] = path.suffix
    d["directory"] = path.parent
    d["dims"] = dims
    return d


def _convert_datetimes(times, use_cftime):
    """
    Return times as np.datetime64[ns] or cftime.DatetimeProlepticGregorian
    depending on whether the dates fall within the inclusive bounds of
    np.datetime64[ns]: [1678-01-01 AD, 2261-12-31 AD].

    Alternatively, always returns as cftime.DatetimeProlepticGregorian if
    ``use_cf_time`` is True.
    """
    if all(time == "steady-state" for time in times):
        return times, False

    out_of_bounds = False
    if use_cftime:
        converted = [
            cftime.DatetimeProlepticGregorian(*time.timetuple()[:6]) for time in times
        ]
    else:
        for time in times:
            year = time.year
            if year < 1678 or year > 2261:
                out_of_bounds = True
                break

        if out_of_bounds:
            use_cftime = True
            msg = "Dates are outside of np.datetime64[ns] timespan. Converting to cftime.DatetimeProlepticGregorian."
            warnings.warn(msg)
            converted = [
                cftime.DatetimeProlepticGregorian(*time.timetuple()[:6])
                for time in times
            ]
        else:
            converted = [np.datetime64(time, "ns") for time in times]

    return converted, use_cftime


def _compose_timestring(time, time_format="%Y%m%d%H%M%S") -> str:
    """
    Compose timestring from time. Function takes care of different
    types of available time objects.
    """
    if time == "steady-state":
        return time
    else:
        if isinstance(time, np.datetime64):
            # The following line is because numpy.datetime64[ns] does not
            # support converting to datetime, but returns an integer instead.
            # This solution is 20 times faster than using pd.to_datetime()
            return time.astype("datetime64[us]").item().strftime(time_format)
        else:
            return time.strftime(time_format)


def compose(d, pattern=None) -> pathlib.Path:
    """
    From a dict of parts, construct a filename, following the iMOD
    conventions.
    """
    haslayer = "layer" in d
    hastime = "time" in d
    hasspecies = "species" in d

    if pattern is None:
        if hastime:
            time = d["time"]
            d["timestr"] = "_{}".format(_compose_timestring(time))
        else:
            d["timestr"] = ""

        if haslayer:
            d["layerstr"] = "_l{}".format(int(d["layer"]))
        else:
            d["layerstr"] = ""

        if hasspecies:
            d["speciesstr"] = "_c{}".format(int(d["species"]))
        else:
            d["speciesstr"] = ""

        s = "{name}{speciesstr}{timestr}{layerstr}{extension}".format(**d)
    else:
        if hastime:
            time = d["time"]
            if time != "steady-state":
                # Change time to datetime.datetime
                if isinstance(time, np.datetime64):
                    d["time"] = time.astype("datetime64[us]").item()
                elif isinstance(time, cftime.datetime):
                    # Take first six elements of timetuple and convert to datetime
                    d["time"] = datetime.datetime(*time.timetuple()[:6])
        s = pattern.format(**d)

    if "directory" in d:
        return pathlib.Path(d["directory"]) / s
    else:
        return pathlib.Path(s)


def _xycoords(bounds, cellsizes) -> Dict[str, Any]:
    """Based on bounds and cellsizes, construct coords with spatial information"""
    # unpack tuples
    xmin, xmax, ymin, ymax = bounds
    dx, dy = cellsizes
    coords = collections.OrderedDict()
    # from cell size to x and y coordinates
    if isinstance(dx, (int, float)):  # equidistant
        coords["x"] = np.arange(xmin + dx / 2.0, xmax, dx)
        coords["y"] = np.arange(ymax + dy / 2.0, ymin, dy)
        coords["dx"] = float(dx)
        coords["dy"] = float(dy)
    else:  # nonequidistant
        # even though IDF may store them as float32, we always convert them to float64
        dx = dx.astype(np.float64)
        dy = dy.astype(np.float64)
        coords["x"] = xmin + np.cumsum(dx) - 0.5 * dx
        coords["y"] = ymax + np.cumsum(dy) - 0.5 * dy
        if np.allclose(dx, dx[0]) and np.allclose(dy, dy[0]):
            coords["dx"] = float(dx[0])
            coords["dy"] = float(dy[0])
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


@contextlib.contextmanager
def cd(path: Union[str, pathlib.Path]):
    """
    Change directory, and change it back after the with block.

    Examples
    --------
    >>> with imod.util.cd("docs"):
            do_something_in_docs()

    """
    curdir = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(curdir)


def temporary_directory() -> pathlib.Path:
    tempdir = tempfile.TemporaryDirectory()
    return pathlib.Path(tempdir.name)


@contextlib.contextmanager
def ignore_warnings():
    """
    Contextmanager to ignore RuntimeWarnings as they are frequently
    raised by the Dask delayed scheduler.

    Examples
    --------
    >>> with imod.util.ignore_warnings():
            function_that_throws_warnings()

    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        yield


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


def mdal_compliant_ugrid2d(ds: xr.Dataset) -> xr.Dataset:
    """
    Ensures the xarray Dataset will be written to a UGRID netCDF that will be
    accepted by MDAL.

    * Breaks down variables with a layer dimension into separate variables.
    * Removes absent entries from the mesh topology attributes.
    * Sets encoding to float for datetime variables.
    * Convert face_node_connectivity to float and set _FillValue to NaN
      (xarray).

    """
    for variable in ds.data_vars:
        if "layer" in ds[variable].dims:
            stacked = ds[variable]
            ds = ds.drop_vars(variable)
            for layer in stacked["layer"].values:
                ds[f"{variable}_layer_{layer}"] = stacked.sel(layer=layer, drop=True)
    if "layer" in ds.coords:
        ds = ds.drop("layer")

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

            edge_dim = attrs.get("edge_dimension")
            if edge_dim and edge_dim not in ds.dims:
                attrs.pop("edge_dimension")

            face_coords = attrs.get("face_coordinates")
            if face_coords and face_coords not in ds.coords:
                attrs.pop("face_coordinates")

            edge_nodes = attrs.get("edge_node_connectivity")
            if edge_nodes and edge_nodes not in ds:
                attrs.pop("edge_node_connectivity")

    # Make sure time is encoded as a float for MDAL
    for var in ds.coords:
        if np.issubdtype(ds[var].dtype, np.datetime64):
            ds[var].encoding["dtype"] = np.float64

    return ds


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

    grid = xugrid.Ugrid2d.from_structured(data)
    ds = grid.dataset

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


def is_divisor(numerator: FloatArray, denominator: float) -> bool:
    """
    Parameters
    ----------
    numerator: np.array of floats
    denominator: float

    Returns
    -------
    is_divisor: bool
    """
    denominator = abs(denominator)
    remainder = np.abs(numerator) % denominator
    return (np.isclose(remainder, 0.0) | np.isclose(remainder, denominator)).all()


def initialize_nested_dict(depth: int) -> collections.defaultdict:
    """
    Initialize a nested dict with a fixed depth

    Parameters
    ----------
    depth : int
        depth of returned nested dict

    Returns
    -------
    nested defaultdicts of n depth

    """
    # In explicit form, say we have ndims=5
    # Then, writing it out, we get:
    # a = partial(defaultdict, {})
    # b = partial(defaultdict, a)
    # c = partial(defaultdict, b)
    # d = defaultdict(c)
    # This can obviously be done iteratively.
    if depth == 0:
        return {}
    elif depth == 1:
        return collections.defaultdict(dict)
    else:
        d = functools.partial(collections.defaultdict, dict)
        for _ in range(depth - 2):
            d = functools.partial(collections.defaultdict, d)
        return collections.defaultdict(d)


def set_nested(d: collections.defaultdict, keys: List[str], value: Any) -> None:
    """
    Set in the deepest dict of a set of nested dictionaries, as created by the
    initialize_nested_dict function above.

    Mutates d.

    Parameters
    ----------
    d : (Nested dict of) dict
    keys : list of keys
        Each key is a level of nesting
    value : dask array, typically

    Returns
    -------
    None
    """
    if len(keys) == 1:
        d[keys[0]] = value
    else:
        set_nested(d[keys[0]], keys[1:], value)


def append_nested_dict(dict1: Dict, dict2: Dict) -> None:
    """
    Recursively walk through two dicts to append dict2 to dict1.

    Mutates dict1

    Modified from:
    https://stackoverflow.com/a/58742155

    Parameters
    ----------
    dict1 : nested dict
        Nested dict to be appended to
    dict2 : nested dict
        Nested dict to append

    """
    for key, val in dict1.items():
        if isinstance(val, dict):
            if key in dict2 and isinstance(dict2[key], dict):
                append_nested_dict(dict1[key], dict2[key])
        else:
            if key in dict2:
                dict1[key] = dict2[key]

    for key, val in dict2.items():
        if key not in dict1:
            dict1[key] = val


def sorted_nested_dict(d: Dict) -> Dict:
    """
    Sorts a variably nested dict (of dicts) by keys.

    Each dictionary will be sorted by its keys.

    Parameters
    ----------
    d : (Nested dict of) dict

    Returns
    -------
    sorted_lists : list (of lists)
        Values sorted by keys, matches the nesting of d.
    """
    firstkey = next(iter(d.keys()))
    if not isinstance(d[firstkey], dict):  # Base case
        return [v for (_, v) in sorted(d.items(), key=lambda t: t[0])]
    else:  # Recursive case
        return [
            sorted_nested_dict(v) for (_, v) in sorted(d.items(), key=lambda t: t[0])
        ]


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
    cellsizes = (abs(dx), -abs(dy))
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
    cellsizes = (abs(dx), -abs(dy))
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
    cellsizes = (abs(dx), -abs(dy))
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
    cellsizes = (abs(dx), -abs(dy))
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


def where(condition, if_true, if_false, keep_nan: bool = True) -> xr.DataArray:
    """
    Wrapped version of xarray's ``.where``.

    This wrapped version does two differently:

    Firstly, it prioritizes the dimensions as: ``if_true > if_false > condition``.
    ``xarray.where(cond, a, b)`` will choose the dimension over ``a`` or ``b``.
    This may result in unwanted dimension orders such as ``("y", "x", "layer)``
    rather than ``("layer", "y', "x")``.

    Secondly, it preserves the NaN values of ``if_true`` by default.  If we
    wish to replace all values over 5 by 5, yet keep the NoData parts, this
    requires two operations with with xarray's ``where``.

    Parameters
    ----------
    condition: DataArray, Dataset
        Locations at which to preserve this object's values. dtype must be `bool`.
    if_true : scalar, DataArray or Dataset, optional
        Value to use for locations where ``cond`` is True.
    if_false : scalar, DataArray or Dataset, optional
        Value to use for locations where ``cond`` is False.
    keep_nan: bool, default: True
        Whether to keep the NaN values in place of ``if_true``.
    """
    xr_obj = (xr.DataArray, xr.Dataset)
    da_true = isinstance(if_true, xr_obj)
    da_false = isinstance(if_false, xr_obj)
    da_cond = isinstance(condition, xr_obj)

    # Give priority to where_true or where_false for broadcasting.
    if da_true:
        new = if_true.copy()
    elif da_false:
        new = xr.full_like(if_false, if_true)
    elif da_cond:
        new = xr.full_like(condition, if_true, dtype=type(if_true))
    else:
        raise ValueError(
            "at least one of {condition, if_true, if_false} should be a "
            "DataArray or Dataset"
        )

    new = new.where(condition, other=if_false)
    if keep_nan and da_true:
        new = new.where(if_true.notnull())

    return new


def replace(da: xr.DataArray, to_replace: Any, value: Any) -> xr.DataArray:
    """
    Replace values given in `to_replace` by `value`.

    Parameters
    ----------
    da: xr.DataArray
    to_replace: scalar or 1D array like
        Which values to replace. If to_replace and value are both array like,
        they must be the same length.
    value: scalar or 1D array like
        Value to replace any values matching `to_replace` with.

    Returns
    -------
    xr.DataArray
        DataArray after replacement.

    Examples
    --------

    Replace values of 1.0 by 10.0, and 2.0 by 20.0:

    >>> da = xr.DataArray([0.0, 1.0, 1.0, 2.0, 2.0])
    >>> replaced = imod.util.replace(da, to_replace=[1.0, 2.0], value=[10.0, 20.0])

    """
    from xarray.core.utils import is_scalar

    def _replace(
        a: np.ndarray, to_replace: np.ndarray, value: np.ndarray
    ) -> np.ndarray:
        # Use np.unique to create an inverse index
        flat = a.ravel()
        uniques, index = np.unique(flat, return_inverse=True)
        replaceable = np.isin(flat, to_replace)

        # Create a replacement array in which there is a 1:1 relation between
        # uniques and the replacement values, so that we can use the inverse index
        # to select replacement values.
        valid = np.isin(to_replace, uniques, assume_unique=True)
        # Remove to_replace values that are not present in da. If no overlap
        # exists between to_replace and the values in da, just return a copy.
        if not valid.any():
            return a.copy()
        to_replace = to_replace[valid]
        value = value[valid]

        replacement = np.full_like(uniques, np.nan)
        replacement[np.searchsorted(uniques, to_replace)] = value

        out = flat.copy()
        out[replaceable] = replacement[index[replaceable]]
        return out.reshape(a.shape)

    if is_scalar(to_replace):
        if not is_scalar(value):
            raise TypeError("if to_replace is scalar, then value must be a scalar")
        if np.isnan(to_replace):
            return da.fillna(value)
        else:
            return da.where(da != to_replace, other=value)
    else:
        to_replace = np.asarray(to_replace)
        if to_replace.ndim != 1:
            raise ValueError("to_replace must be 1D or scalar")
        if is_scalar(value):
            value = np.full_like(to_replace, value)
        else:
            value = np.asarray(value)
            if to_replace.shape != value.shape:
                raise ValueError(
                    f"Replacement arrays must match in shape. "
                    f"Expecting {to_replace.shape} got {value.shape} "
                )

    _, counts = np.unique(to_replace, return_counts=True)
    if (counts > 1).any():
        raise ValueError("to_replace contains duplicates")

    # Replace NaN values separately, as they will show up as separate values
    # from numpy.unique.
    isnan = np.isnan(to_replace)
    if isnan.any():
        i = np.nonzero(isnan)[0]
        da = da.fillna(value[i])

    return xr.apply_ufunc(
        _replace,
        da,
        kwargs={"to_replace": to_replace, "value": value},
        dask="parallelized",
        output_dtypes=[da.dtype],
    )


class MissingOptionalModule:
    """
    Presents a clear error for optional modules.
    """

    def __init__(self, name):
        self.name = name

    def __getattr__(self, name):
        raise ImportError(f"{self.name} is required for this functionality")
