"""
Functions that make use of `rasterio
<https://rasterio.readthedocs.io/en/stable/>`_ for input and output to other
raster formats.

Currently only :func:`imod.rasterio.write` is implemented.
"""

import glob
import pathlib

import numpy as np
import xarray as xr

# since rasterio is a big dependency that is sometimes hard to install
# and not always required, we made this an optional dependency
try:
    import rasterio
except ImportError:
    pass

import imod
from imod import idf, util
from . import array_IO


def _limitations(riods, path):
    if riods.count != 1:
        raise NotImplementedError(
            f"Cannot open multi-band grid: {path}. Try xarray.open_rasterio() instead."
        )
    if not riods.transform.is_rectilinear:
        raise NotImplementedError(
            f"Cannot open non-rectilinear grid: {path}. Try xarray.open_rasterio() instead."
        )


def header(path, pattern):
    attrs = util.decompose(path, pattern)
    
    # TODO:
    # Check bands, datatypes, rotation, etc.
    # Raise NotImplementedErrors and point to xr.open_rasterio
    with rasterio.open(path, "r") as riods:
        _limitations(riods, path)
        nrow = riods.height
        ncol = riods.width
        dx, xmin, _, _, dy, ymax = tuple(riods.transform[:6])
        xmax = xmin + ncol * dx
        ymin = ymax + nrow * dy
        attrs["nodata"] = riods.nodata

    attrs["nrow"] = nrow
    attrs["ncol"] = ncol
    attrs["dx"] = dx
    attrs["dy"] = dy
    attrs["xmin"] = xmin
    attrs["xmax"] = xmax
    attrs["ymin"] = ymin
    attrs["ymax"] = ymax
    attrs["headersize"] = None
    return attrs


def _read(path, *args):
    with rasterio.open(path, "r") as dataset:
        a = dataset.read(1)
    return a


# Open IDFs for multiple times and/or layers into one DataArray
def open(path, use_cftime=False, pattern=None):
    r"""
    Open one or more IDF files as an xarray.DataArray.

    In accordance with xarray's design, ``open`` loads the data of IDF files
    lazily. This means the data of the IDFs are not loaded into memory until the
    data is needed. This allows for easier handling of large datasets, and
    more efficient computations.

    Parameters
    ----------
    path : str, Path or list
        This can be a single file, 'head_l1.idf', a glob pattern expansion,
        'head_l*.idf', or a list of files, ['head_l1.idf', 'head_l2.idf'].
        Note that each file needs to be of the same name (part before the
        first underscore) but have a different layer and/or timestamp,
        such that they can be combined in a single xarray.DataArray.
    use_cftime : bool, optional
        Use ``cftime.DatetimeProlepticGregorian`` instead of `np.datetime64[ns]`
        for the time axis.

        Dates are normally encoded as ``np.datetime64[ns]``; however, if dates
        fall before 1678 or after 2261, they are automatically encoded as
        ``cftime.DatetimeProlepticGregorian`` objects rather than
        ``np.datetime64[ns]``.
    pattern : str, regex pattern, optional
        If the filenames do match default naming conventions of
        {name}_{time}_l{layer}, a custom pattern can be defined here either
        as a string, or as a compiled regular expression pattern. See the
        examples below.

    Returns
    -------
    xarray.DataArray
        A float32 xarray.DataArray of the values in the IDF file(s).
        All metadata needed for writing the file to IDF or other formats
        using imod.rasterio are included in the xarray.DataArray.attrs.

    Examples
    --------
    Open an IDF file:

    >>> da = imod.idf.open("example.idf")

    Open an IDF file, relying on default naming conventions to identify
    layer:

    >>> da = imod.idf.open("example_l1.idf")

    Open an IDF file, relying on default naming conventions to identify layer
    and time:

    >>> head = imod.idf.open("head_20010101_l1.idf")

    Open multiple IDF files, in this case files for the year 2001 for all
    layers, again relying on default conventions for naming:

    >>> head = imod.idf.open("head_2001*_l*.idf")

    The same, this time explicitly specifying ``name``, ``time``, and ``layer``:

    >>> head = imod.idf.open("head_2001*_l*.idf", pattern="{name}_{time}_l{layer}")

    The format string pattern will only work on tidy paths, where variables are
    separated by underscores. You can also pass a compiled regex pattern.
    Make sure to include the ``re.IGNORECASE`` flag since all paths are lowered.

    >>> import re
    >>> pattern = re.compile(r"(?P<name>[\w]+)L(?P<layer>[\d+]*)", re.IGNORECASE)
    >>> head = imod.idf.open("headL11", pattern=pattern)

    However, this requires constructing regular expressions, which is
    generally a fiddly process. Regex notation is also impossible to
    remember. The website https://regex101.com is a nice help. Alternatively,
    the most pragmatic solution may be to just rename your files.
    """

    if isinstance(path, list):
        return array_IO.reading._load(path, use_cftime, pattern, _read, header)
    elif isinstance(path, pathlib.Path):
        path = str(path)

    paths = [pathlib.Path(p) for p in glob.glob(path)]
    n = len(paths)
    if n == 0:
        raise FileNotFoundError(f"Could not find any files matching {path}")
    return array_IO.reading._load(paths, use_cftime, pattern, _read, header)


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
    extradims = list(filter(lambda dim: dim not in ("y", "x"), da.dims))
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


def save(path, a, nodata=1.0e20, pattern=None):
    """
    Write a xarray.DataArray to one or more IDF files

    If the DataArray only has ``y`` and ``x`` dimensions, a single IDF file is
    written, like the ``imod.idf.write`` function. This function is more general
    and also supports ``time`` and ``layer`` dimensions. It will split these up,
    give them their own filename according to the conventions in
    ``imod.util.compose``, and write them each.

    Parameters
    ----------
    path : str or Path
        Path to the IDF file to be written. This function decides on the
        actual filename(s) using conventions, so it only takes the directory and
        name from this parameter.
    a : xarray.DataArray
        DataArray to be written. It needs to have dimensions ('y', 'x'), and
        optionally ``layer`` and ``time``.
    nodata : float, optional
        Nodata value in the saved IDF files. Xarray uses nan values to represent
        nodata, but these tend to work unreliably in iMOD(FLOW).
        Defaults to a value of 1.0e20.
    pattern : str
        Format string which defines how to create the filenames. See examples.

    Example
    -------
    Consider a DataArray ``da`` that has dimensions 'layer', 'y' and 'x', with the
    'layer' dimension consisting of layer 1 and 2::

        save('path/to/head', da)

    This writes the following two IDF files: 'path/to/head_l1.idf' and
    'path/to/head_l2.idf'.


    It is possible to generate custom filenames using a format string. The
    default filenames would be generated by the following format string:

        save("example", pattern="{name}_l{layer}{extension}")

    If you desire zero-padded numbers that show up neatly sorted in a
    file manager, you may specify:

        save("example", pattern="{name}_l{layer:02d}{extension}")

    In this case, a 0 will be padded for single digit numbers ('1' will become
    '01').

    To get a date with dashes, use the following pattern:

        "{name}_{time:%Y-%m-%d}_l{layer}{extension}"

    """
    array_IO.writing._save(path, a, nodata, pattern, write)
