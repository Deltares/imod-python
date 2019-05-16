import collections
import datetime
import pathlib
import re
import warnings

import affine
import cftime
import numpy as np

try:
    Pattern = re._pattern_type
except AttributeError:
    Pattern = re.Pattern  # Python 3.7+


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
    Make sure to include the `re.IGNORECASE` flag since all paths are lowered.

    >>> import re
    >>> pattern = re.compile(r"(?P<name>[\w]+)L(?P<layer>[\d+]*)")
    >>> decompose("headL11", pattern=pattern)

    However, this requires constructing regular expressions, which is generally
    a fiddly process. The website https://regex101.com is a nice help.
    Alternatively, the most pragmatic solution may be to just rename your files.
    """
    if isinstance(path, str):
        path = pathlib.Path(path)
    # We'll ignore upper case
    stem = path.stem.lower()

    if pattern is not None:
        if isinstance(pattern, Pattern):
            d = pattern.match(stem).groupdict()
        else:
            # Get the variables between curly braces
            in_curly = re.compile(r"{(.*?)}").findall(pattern)
            regex_parts = {key: f"(?P<{key}>[\\w-]+)" for key in in_curly}
            # Format the regex string, by filling in the variables
            simple_regex = pattern.format(**regex_parts)
            re_pattern = re.compile(simple_regex)
            # Use it to get the required variables
            d = re_pattern.match(stem).groupdict()
    else:  # Default to "iMOD conventions": {name}_{time}_l{layer}
        has_layer = bool(re.search(r"_l\d+$", stem))
        try:  # try for time
            base_pattern = r"(?P<name>[\w-]+)_(?P<time>[\w-]+)"
            if has_layer:
                base_pattern += r"_l(?P<layer>[\w]+)"
            re_pattern = re.compile(base_pattern)
            d = re_pattern.match(stem).groupdict()
        except AttributeError:  # probably no time
            base_pattern = r"(?P<name>[\w]+)"
            if has_layer:
                base_pattern += r"_l(?P<layer>[\w]+)"
            re_pattern = re.compile(base_pattern)
            d = re_pattern.match(stem).groupdict()

    # TODO: figure out what to with user specified variables
    # basically type inferencing via regex?
    # if purely numericdcal \d* -> int or float
    #    if \d*\.\d* -> float
    # else: keep as string

    # If name is not provided, generate one from other fields
    if "name" not in d.keys():
        d["name"] = "_".join(d.values())

    # String -> type conversion
    if "layer" in d.keys():
        d["layer"] = int(d["layer"])
    if "time" in d.keys():
        # iMOD supports two datetime formats
        if d["time"] == "steady-state":
            d.pop("time")
        else:
            try:
                d["time"] = datetime.datetime.strptime(d["time"], "%Y%m%d%H%M%S")
            except ValueError:
                d["time"] = datetime.datetime.strptime(d["time"], "%Y%m%d")

    d["extension"] = path.suffix
    d["directory"] = path.parent
    return d


def _convert_datetimes(times, use_cftime):
    """
    Return times as np.datetime64[ns] or cftime.DatetimeProlepticGregorian
    depending on whether the dates fall within the inclusive bounds of
    np.datetime64[ns]: [1678-01-01 AD, 2261-12-31 AD].

    Alternatively, always returns as cftime.DatetimeProlepticGregorian if
    `use_cf_time` is True.
    """
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


def compose(d):
    """
    From a dict of parts, construct a filename, following the iMOD
    conventions
    """
    haslayer = "layer" in d
    hastime = "time" in d
    if hastime:
        time = d["time"]
        if isinstance(time, np.datetime64):
            # The following line is because numpy.datetime64[ns] does not
            # support converting to datetime, but returns an integer instead.
            # This solution is 20 times faster than using pd.to_datetime()
            d["timestr"] = time.astype("datetime64[us]").item().strftime("%Y%m%d%H%M%S")
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
    if "directory" in d:
        return d["directory"].joinpath(s)
    else:
        return s


def _delta(x, coordname):
    dxs = np.diff(x.astype(np.float64))
    dx = dxs[0]
    atolx = abs(1.0e-6 * dx)
    if not np.allclose(dxs, dx, atolx):
        raise ValueError(f"DataArray has to be equidistant along {coordname}.")
    return dx


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

    # Possibly non-equidistant
    if ("dx" in a.coords) and ("dy" in a.coords):
        dx = a.coords["dx"]
        dy = a.coords["dy"]
        if (dx.shape == x.shape) and (dx.size != 1):
            dx = dx.values.astype(np.float64)
            xmin = float(x.min()) - 0.5 * abs(dx[0])
            xmax = float(x.max()) + 0.5 * abs(dx[-1])
        else:
            dx = float(dx)
            xmin = float(x.min()) - 0.5 * abs(dx)
            xmax = float(x.max()) + 0.5 * abs(dx)
        if (dy.shape == y.shape) and (dy.size != 1):
            dy = dy.values.astype(np.float64)
            ymin = float(y.min()) - 0.5 * abs(dy[-1])
            ymax = float(y.max()) + 0.5 * abs(dy[0])
        else:
            dy = float(dy)
            ymin = float(y.min()) - 0.5 * abs(dy)
            ymax = float(y.max()) + 0.5 * abs(dy)
    else:  # Equidistant
        # TODO: decide on decent criterium for what equidistant means
        # make use of floating point epsilon? E.g:
        # https://github.com/ioam/holoviews/issues/1869#issuecomment-353115449
        if ncol == 1:
            dy = _delta(y, "y")
            dx = -dy
        elif nrow == 1:
            dx = _delta(x, "x")
            dy = -dx
        else:
            dx = _delta(x, "x")
            dy = _delta(y, "y")

        # as xarray used midpoint coordinates
        xmin = float(x.min()) - 0.5 * abs(dx)
        xmax = float(x.max()) + 0.5 * abs(dx)
        ymin = float(y.min()) - 0.5 * abs(dy)
        ymax = float(y.max()) + 0.5 * abs(dy)

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
    if dx < 0.0:
        raise ValueError("dx must be positive")
    if dy > 0.0:
        raise ValueError("dy must be negative")
    return affine.Affine(dx, 0.0, xmin, 0.0, dy, ymax)
