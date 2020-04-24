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
import warnings

import affine
import cftime
import dateutil
import numpy as np

try:
    Pattern = re._pattern_type
except AttributeError:
    Pattern = re.Pattern  # Python 3.7+


def to_datetime(s):
    try:
        try:
            time = datetime.datetime.strptime(s, "%Y%m%d%H%M%S")
        except ValueError:
            time = datetime.datetime.strptime(s, "%Y%m%d")
    except ValueError:  # Try fullblown dateutil date parser
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
    else:  # Default to "iMOD conventions": {name}_{time}_l{layer}
        has_layer = bool(re.search(r"_l\d+$", stem))
        has_species = bool(
            re.search(r"conc_\d{8,14}_c\d{1,3}", stem)
        )  # We are strict in recognizing species
        try:  # try for time
            base_pattern = r"(?P<name>[\w-]+)_(?P<time>[0-9-]{6,})"
            if has_species:
                base_pattern += r"_c(?P<species>[0-9]+)"
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
        # iMOD supports two datetime formats
        # try fast options first
        try:
            try:
                d["time"] = datetime.datetime.strptime(d["time"], "%Y%m%d%H%M%S")
            except ValueError:
                d["time"] = datetime.datetime.strptime(d["time"], "%Y%m%d")
        except ValueError:  # Try fullblown dateutil date parser
            d["time"] = dateutil.parser.parse(d["time"])
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


def compose(d, pattern=None):
    """
    From a dict of parts, construct a filename, following the iMOD
    conventions. Returns a pathlib.Path.
    """
    haslayer = "layer" in d
    hastime = "time" in d

    if pattern is None:
        if hastime:
            time = d["time"]
            if time == "steady-state":
                d["timestr"] = time
            else:
                if isinstance(time, np.datetime64):
                    # The following line is because numpy.datetime64[ns] does not
                    # support converting to datetime, but returns an integer instead.
                    # This solution is 20 times faster than using pd.to_datetime()
                    d["timestr"] = (
                        time.astype("datetime64[us]").item().strftime("%Y%m%d%H%M%S")
                    )
                else:
                    d["timestr"] = time.strftime("%Y%m%d%H%M%S")

        if haslayer:
            d["layer"] = int(d["layer"])
            if hastime:
                s = "{name}_{timestr}_l{layer}{extension}".format(**d)
            else:
                s = "{name}_l{layer}{extension}".format(**d)
        else:
            if hastime:
                s = "{name}_{timestr}{extension}".format(**d)
            else:
                s = "{name}{extension}".format(**d)
    else:
        if hastime:
            time = d["time"]
            if time == "steady-state":
                d["timestr"] = time
            else:
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
        atolx = abs(1.0e-6 * dx)
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
    x = a.x.values
    y = a.y.values
    ncol = x.size
    nrow = y.size
    if ncol > 1 and nrow > 1:
        dx, xmin, xmax = coord_reference(a["x"])
        dy, ymin, ymax = coord_reference(a["y"])
    elif ncol == 1:
        dy, ymin, ymax = coord_reference(a["y"])
        dx = 1.0
        xmin = float(x.min()) - 0.5 * abs(dy)
        xmax = float(x.max()) + 0.5 * abs(dy)
    elif nrow == 1:
        dx, xmin, xmax = coord_reference(a["x"])
        dy = -1.0
        ymin = float(y.min()) - 0.5 * abs(dy)
        ymax = float(y.max()) + 0.5 * abs(dy)
    else:  # ncol == 1 and nrow == 1:
        raise NotImplementedError("Not implemented for single cell DataArrays")
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
