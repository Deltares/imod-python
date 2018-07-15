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
    try:
        # TODO try pandas parse date?
        d["time"] = np.datetime64(datetime.strptime(parts[1], "%Y%m%d%H%M%S"))
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
    dxs = np.diff(x)
    dys = np.diff(y)
    dx = dxs[0]
    dy = dys[0]
    assert np.allclose(dxs, dx, atol=1.e-8), "DataArray has to be equidistant along x." 
    assert np.allclose(dys, dy, atol=1.e-8), "DataArray has to be equidistant along y." 
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