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
        https://gdal.org/drivers/raster/index.html.
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
    elif driver == "PCRaster":
        if da.dtype == "float64":
            da = da.astype("float32")
        elif da.dtype == "int64":
            da = da.astype("int32")
        elif da.dtype == "bool":
            da = da.astype("uint8")
        if "PCRASTER_VALUESCALE" not in profile:
            if da.dtype == "int32":
                profile["PCRASTER_VALUESCALE"] = "VS_NOMINAL"
            elif da.dtype == "uint8":
                profile["PCRASTER_VALUESCALE"] = "VS_BOOLEAN"
            else:
                profile["PCRASTER_VALUESCALE"] = "VS_SCALAR"
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
        das.sort(key=lambda da: da.coords[dims[0]])
        return xr.concat(das, dim=dims[0])
    else:
        dims_in = dims[1:]  # recursive case
        out = [ndconcat(das_in, dims_in) for das_in in das.values()]
        return xr.concat(out, dims[0])


def set_nested(d, keys, value):
    if len(keys) == 1:
        d.append(value)
    else:
        set_nested(d[keys[0]], keys[1:], value)


def _read(paths, use_cftime, pattern):
    if len(paths) == 1:
        return xr.open_rasterio(paths[0]).squeeze("band", drop=True)

    dicts = []
    firstlen = len(util.decompose(paths[0], pattern=pattern))
    for path in paths:
        d = util.decompose(path, pattern=pattern)
        if not len(d) == firstlen:
            raise ValueError("Number of dimensions on grids do not match.")
        d["path"] = path
        dicts.append(d)

    dict_dims = [
        key for key in dicts[0] if key not in ("name", "extension", "directory", "path")
    ]
    ndims = len(dict_dims)

    dims = dict_dims
    groupby = initialize_groupby(ndims)
    for d in dicts:
        # Read array
        da = xr.open_rasterio(d["path"]).squeeze("band", drop=True)
        # Assign coordinates
        groupbykeys = []
        for dim in dict_dims:
            value = d[dim]
            da = da.assign_coords(**{dim: value})
            groupbykeys.append(value)
        # Group in the right dimension
        set_nested(groupby, groupbykeys, da)

    nd = ndconcat(groupby, dims)
    nd.coords["dx"] = abs(nd.res[0])
    nd.coords["dy"] = -abs(nd.res[1])
    return nd


def initialize_groupby(ndims):
    # In explicit form, say we have ndims=5
    # Then, writing it out, we get:
    # a = partial(defaultdict, list)
    # b = partial(defaultdict, a)
    # c = partial(defaultdict, b)
    # d = defaultdict(c)
    # This can obviously be done iteratively.
    if ndims == 1:
        return list()
    elif ndims == 2:
        return collections.defaultdict(list)
    else:
        d = functools.partial(collections.defaultdict, list)
        for _ in range(ndims - 2):
            d = functools.partial(collections.defaultdict, d)
        return collections.defaultdict(d)


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
