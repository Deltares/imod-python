import re
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import cftime
import numpy as np
from affine import Affine


def decompose(path):
    """Parse a path, returning a dict of the parts,
    following the iMOD conventions"""
    if isinstance(path, str):
        path = Path(path)

    parts = path.stem.split("_")
    name = parts[0]
    assert name != "", ValueError("Name cannot be empty")
    d = OrderedDict()
    d["extension"] = path.suffix
    d["directory"] = path.parent
    d["name"] = name

    # Try to get time from idf name, iMODFLOW can output two datetime formats
    for s in parts:
        try:
            dt = datetime.strptime(s, "%Y%m%d%H%M%S")
            d["time"] = cftime.DatetimeProlepticGregorian(*dt.timetuple()[:6])
            break
        except ValueError:
            try:
                dt = datetime.strptime(s, "%Y%m%d")
                d["time"] = cftime.DatetimeProlepticGregorian(*dt.timetuple()[:6])
                break
            except ValueError:
                pass  # no time in dict

    # layer is always last
    p = re.compile(r"^l\d+$", re.IGNORECASE)
    if p.match(parts[-1]):
        d["layer"] = int(parts[-1][1:])
    return d


def compose(d):
    """From a dict of parts, construct a filename,
    following the iMOD conventions"""
    haslayer = "layer" in d
    hastime = "time" in d
    if hastime:
        d["timestr"] = d["time"].strftime("%Y%m%d%H%M%S")
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
    dxs = np.diff(x)
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
            dx = dx.values
            xmin = float(x.min()) - 0.5 * abs(dx[0])
            xmax = float(x.max()) + 0.5 * abs(dx[-1])
        else:
            dx = float(dx)
            xmin = float(x.min()) - 0.5 * abs(dx)
            xmax = float(x.max()) + 0.5 * abs(dx)
        if (dy.shape == y.shape) and (dy.size != 1):
            dy = dy.values
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
        # TODO: this is basically a work-around for iMODFLOW allowing only
        # square gridcells, ideally 1D IDF have a width of 1.0 (?)
        if ncol == 1:
            dx = dy = _delta(y, "y")
        elif nrow == 1:
            dy = dx = _delta(x, "x")
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
    return Affine(dx, 0.0, xmin, 0.0, dy, ymax)
