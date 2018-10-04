import re
from datetime import datetime
import numpy as np
from collections import OrderedDict
from pathlib import Path
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
    if len(parts) == 1:
        return d

    # Try to get time from idf name, iMODFLOW can output two datetime formats
    try:
        d["time"] = np.datetime64(datetime.strptime(parts[1], "%Y%m%d%H%M%S"))
    except ValueError:
        try:
            d["time"] = np.datetime64(datetime.strptime(parts[1], "%Y%m%d"))
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


def spatial_reference(a):
    """
    Extracts spatial reference from DataArray.

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

    # TODO: decide on decent criterium for what equidistant means
    # make use of floating point epsilon? E.g:
    # https://github.com/ioam/holoviews/issues/1869#issuecomment-353115449
    if ncol == 1:
        dx = 1.0
    else:
        dxs = np.diff(x)
        dx = dxs[0]
        atolx = abs(1.0e-6 * dx)
        assert np.allclose(dxs, dx, atolx), "DataArray has to be equidistant along x."

    if nrow == 1:
        dy = 1.0
    else:
        dys = np.diff(y)
        dy = dys[0]
        atoly = abs(1.0e-6 * dy)
        assert np.allclose(dys, dy, atoly), "DataArray has to be equidistant along y."
        
    xmin = x.min() - 0.5 * abs(dx) # as xarray used midpoint coordinates
    ymax = y.max() + 0.5 * abs(dy)
    xmax = xmin + ncol * abs(dx)
    ymin = ymax - nrow * abs(dy)
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
    return Affine(dx, 0.0, xmin, 0.0, dy, ymax)
