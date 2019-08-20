"""
Functions that make use of `rasterio
<https://rasterio.readthedocs.io/en/stable/>`_ for input and output to other
raster formats.

Currently only :func:`imod.rasterio.write` is implemented.
"""

import collections
import functools
import glob
import pathlib
import re

import numpy as np
import xarray as xr

# since rasterio is a big dependency that is sometimes hard to install
# and not always required, we made this an optional dependency
try:
    import rasterio
except ImportError:
    pass

from imod import idf, util


def write(path, da, driver=None, nodata=np.nan):
    """Write ``xarray.DataArray`` to GDAL supported geospatial rasters using ``rasterio``.
    
    Parameters
    ----------
    path: str or Path
        path to the dstput raste
    da: xarray DataArray
        The DataArray to be written. Should have only x and y dimensions.
    driver: str; optional
        Which GDAL format driver to use. The complete list is at
        https://www.gdal.org/formats_list.html.
        By default tries to guess from the file extension.
    nodata: float
        Nodata value to use. Should be convertible to the DataArray and GDAL dtype.
        Default value is np.nan
        
    Examples
    --------
    Save ``xarray.DataArray`` in ASCII format:

    >>> imod.rasterio.write("example.asc", da)
    
    Save ``xarray.DataArray`` in ASCII format, with 6 significant digits:
    
    >>> da.attrs['SIGNIFICANT_DIGITS'] = 6
    >>> imod.rasterio.write("example.asc", da)
    """
    # Not directly related to iMOD, but provides a missing link, together
    # with xarray.open_rasterio.
    # Note that this function can quickly become dstdated as
    # the xarray rasterio connection matures, see for instance:
    # https://github.com/pydata/xarray/issues/1736
    # https://github.com/pydata/xarray/pull/1712
    if isinstance(path, str):
        path = pathlib.Path(path)
    profile = da.attrs.copy()
    if driver is None:
        ext = path.suffix.lower()
        if ext in (".tif", ".tiff"):
            driver = "GTiff"
        elif ext == ".asc":
            driver = "AAIGrid"
        elif ext == ".map":
            driver = "PCRaster"
        else:
            raise ValueError(f"Unknown extension {ext}, specifiy driver")
    # prevent rasterio warnings
    if driver == "AAIGrid":
        profile.pop("res", None)
        profile.pop("is_tiled", None)
    extradims = idf._extra_dims(da)
    # TODO only equidistant IDFs are compatible with GDAL / rasterio
    # TODO try squeezing extradims here, such that 1 layer, 1 time, etc. is acccepted
    if extradims:
        raise ValueError(f"Only x and y dimensions supported, found {da.dims}")
    # transform will be affine object in next xarray
    profile["transform"] = util.transform(da)
    profile["driver"] = driver
    profile["height"] = da.y.size
    profile["width"] = da.x.size
    profile["count"] = 1
    profile["dtype"] = da.dtype
    profile["nodata"] = nodata
    if (nodata is None) or np.isnan(nodata):
        # NaN is the default missing value in xarray
        # None is different in that the raster won't have a nodata value
        dafilled = da
    else:
        dafilled = da.fillna(nodata)
    with rasterio.Env():
        with rasterio.open(path, "w", **profile) as ds:
            ds.write(dafilled.values, 1)


def ndconcat(das, dims):
    """
    Parameters
    ----------
    das : dict (of dicts) of lists, n levels deep. Bottoms out at a list.
        E.g. {2000-01-01: [da1, da2], 2001-01-01: [da3, da4]}
        for n = 2.
    dims : tuple
        Tuple of dimensions over which to concatenate. Has to be n elements long.
        E.g. ("time", "layer") for n = 2.

    Returns
    -------
    concatenated : xr.DataArray
        Input concatenated over n dimensions.
    """
    if len(dims) == 1:  # base case
        return xr.concat(das, dim=dims[0])
    else:
        dims_in = dims[1:]  # recursive case
        out = [ndconcat(das_in, dims_in) for das_in in das.values()]
        return xr.concat(out, dims[0])


def initialize_groupby(ndims):
    # In explicit form, say we have ndims=4
    # Then, writing it out, we get:
    # a = partial(defaultdict, list)
    # b = partial(defaultdict, a)
    # c = partial(defaultdict, b)
    # d = defaultdict(c)
    # This can obviously be written as a for loop, as below.
    d = collections.defaultdict(list)
    if ndims == 1:
        return d
    else:
        for _ in range(ndims - 1):
            d = functools.partial(collections.defaultdict, d)
        return d


def read(path, use_cftime=False, pattern=None):
    if isinstance(path, list):
        return _read(path, use_cftime, pattern)
    elif isinstance(path, pathlib.Path):
        path = str(path)

    paths = [pathlib.Path(p) for p in glob.glob(path)]
    n = len(paths)
    if n == 0:
        raise FileNotFoundError(f"Could not find any files matching {path}")
    return _read(paths, use_cftime, pattern)
