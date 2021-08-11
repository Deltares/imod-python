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
import os
import pathlib
import re
from typing import Tuple, Union
import warnings

import affine
import cftime
import dateutil
import numpy as np
import xarray as xr
import functools

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


def _groupdict(stem, pattern):
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


def decompose(path, pattern=None):
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


def _compose_timestring(time, time_format="%Y%m%d%H%M%S"):
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


def compose(d, pattern=None):
    """
    From a dict of parts, construct a filename, following the iMOD
    conventions. Returns a pathlib.Path.
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


def _xycoords(bounds, cellsizes):
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


def coord_reference(da_coord):
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


def spatial_reference(a):
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


def transform(a):
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
def cd(path):
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


def _ugrid2d_dataset(
    node_x: FloatArray,
    node_y: FloatArray,
    face_x: FloatArray,
    face_y: FloatArray,
    face_nodes: IntArray,
) -> xr.Dataset:
    ds = xr.Dataset()
    ds["mesh2d"] = xr.DataArray(
        data=0,
        attrs={
            "cf_role": "mesh_topology",
            "long_name": "Topology data of 2D mesh",
            "topology_dimension": 2,
            "node_coordinates": "node_x node_y",
            "face_node_connectivity": "face_nodes",
            "edge_node_connectivity": "edge_nodes",
        },
    )
    ds = ds.assign_coords(
        node_x=xr.DataArray(
            data=node_x,
            dims=["node"],
        )
    )
    ds = ds.assign_coords(
        node_y=xr.DataArray(
            data=node_y,
            dims=["node"],
        )
    )
    ds["face_nodes"] = xr.DataArray(
        data=face_nodes,
        coords={
            "face_x": ("face", face_x),
            "face_y": ("face", face_y),
        },
        dims=["face", "nmax_face"],
        attrs={
            "cf_role": "face_node_connectivity",
            "long_name": "Vertex nodes of mesh faces (counterclockwise)",
            "start_index": 0,
            "_FillValue": -1,
        },
    )
    ds.attrs = {"Conventions": "CF-1.8 UGRID-1.0"}
    return ds


def ugrid2d_topology(data: Union[xr.DataArray, xr.Dataset]) -> xr.Dataset:
    """
    Derive the 2D-UGRID quadrilateral mesh topology from a structured DataArray
    or Dataset, with (2D-dimensions) "y" and "x".

    Parameters
    ----------
    data: Union[xr.DataArray, xr.Dataset]
        Structured data from which the "x" and "y" coordinate will be used to
        define the UGRID-2D topology.

    Returns
    -------
    ugrid_topology: xr.Dataset
        Dataset with the required arrays describing 2D unstructured topology:
        node_x, node_y, face_x, face_y, face_nodes (connectivity).
    """
    from imod.prepare import common

    # Transform midpoints into vertices
    # These are always returned monotonically increasing
    x = data["x"].values
    xcoord = common._coord(data, "x")
    if not data.indexes["x"].is_monotonic_increasing:
        xcoord = xcoord[::-1]
    y = data["y"].values
    ycoord = common._coord(data, "y")
    if not data.indexes["y"].is_monotonic_increasing:
        ycoord = ycoord[::-1]
    # Compute all vertices, these are the ugrid nodes
    node_y, node_x = (a.ravel() for a in np.meshgrid(ycoord, xcoord, indexing="ij"))
    face_y, face_x = (a.ravel() for a in np.meshgrid(y, x, indexing="ij"))
    linear_index = np.arange(node_x.size, dtype=np.int).reshape(
        ycoord.size, xcoord.size
    )
    # Allocate face_node_connectivity
    nfaces = (ycoord.size - 1) * (xcoord.size - 1)
    face_nodes = np.empty((nfaces, 4))
    # Set connectivity in counterclockwise manner
    face_nodes[:, 0] = linear_index[:-1, 1:].ravel()  # upper right
    face_nodes[:, 1] = linear_index[:-1, :-1].ravel()  # upper left
    face_nodes[:, 2] = linear_index[1:, :-1].ravel()  # lower left
    face_nodes[:, 3] = linear_index[1:, 1:].ravel()  # lower right
    # Tie it together
    ds = _ugrid2d_dataset(node_x, node_y, face_x, face_y, face_nodes)
    return ds


def ugrid2d_data(da: xr.DataArray) -> xr.DataArray:
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
    if da.dims[:-2] == ("y", "x"):
        raise ValueError('Last two dimensions must be ("y", "x").')
    extra_dims = list(set(da.dims) - set(["y", "x"]))
    shape = da.data.shape
    new_shape = shape[:-2] + (np.product(shape[-2:]),)
    return xr.DataArray(
        data=da.data.reshape(new_shape),
        coords={k: da[k] for k in da.coords if k not in ("y", "x", "dy", "dx")},
        dims=extra_dims + ["face"],
    )


def _unstack_layers(ds: xr.Dataset) -> xr.Dataset:
    """
    Unstack the layer dimensions, as MDAL does not have support for
    UGRID-2D-layered datasets yet. Layers are stored as separate variables
    instead for now.
    """
    for variable in ds.data_vars:
        if "layer" in ds[variable].dims:
            stacked = ds[variable]
            ds = ds.drop_vars(variable)
            for layer in stacked["layer"].values:
                ds[f"{variable}_layer_{layer}"] = stacked.sel(layer=layer)
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
    ds = ugrid2d_topology(data)
    if isinstance(data, xr.DataArray):
        if data.name is None:
            raise ValueError(
                'A name is required for the DataArray. It can be set with ``da.name = "..."`'
            )
        ds[data.name] = ugrid2d_data(data)
    elif isinstance(data, xr.Dataset):
        for variable in data.data_vars:
            ds[variable] = ugrid2d_data(data[variable])
    else:
        raise TypeError("data must be xarray.DataArray or xr.Dataset")
    return _unstack_layers(ds)


def is_divisor(numerator, denominator) -> bool:
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


def initialize_nested_dict(depth):
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
        for _ in range(depth - 1):
            d = functools.partial(collections.defaultdict, d)
        return collections.defaultdict(d)


def set_nested(d, keys, value):
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


def sorted_nested_dict(d):
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
