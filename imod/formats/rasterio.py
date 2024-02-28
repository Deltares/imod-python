"""
Functions that make use of `rasterio
<https://rasterio.readthedocs.io/en/stable/>`_ for input and output to other
raster formats.

Currently only :func:`imod.rasterio.write` is implemented.
"""

import pathlib
import warnings
from typing import Dict

import numpy as np
import pandas as pd

from imod.util.imports import MissingOptionalModule
from imod.formats import array_io

# since rasterio is a big dependency that is sometimes hard to install
# and not always required, we made this an optional dependency
try:
    import rasterio
except ImportError:
    rasterio = MissingOptionalModule("rasterio")

f_open = open


# Based on this comment
# https://github.com/mapbox/rasterio/issues/265#issuecomment-367044836
def _create_ext_driver_code_map():
    from osgeo import gdal

    if hasattr(gdal, "DCAP_RASTER"):

        def _check_driver(drv):
            return drv.GetMetadataItem(gdal.DCAP_RASTER)

    else:

        def _check_driver(drv):
            return True

    output = {}
    for i in range(gdal.GetDriverCount()):
        drv = gdal.GetDriver(i)
        if _check_driver(drv):
            if drv.GetMetadataItem(gdal.DCAP_CREATE) or drv.GetMetadataItem(
                gdal.DCAP_CREATECOPY
            ):
                ext = drv.GetMetadataItem(gdal.DMD_EXTENSION)
                if ext is not None and len(ext) > 0:
                    output[drv.GetMetadataItem(gdal.DMD_EXTENSION)] = drv.ShortName
    sortedkeys = sorted(output.keys())
    output = {k: output[k] for k in sortedkeys}
    return output


# tiff and jpeg keys have been added manually.
EXTENSION_GDAL_DRIVER_CODE_MAP = {
    "asc": "AAIGrid",
    "bag": "BAG",
    "bil": "EHdr",
    "blx": "BLX",
    "bmp": "BMP",
    "bt": "BT",
    "dat": "ZMap",
    "dem": "USGSDEM",
    "ers": "ERS",
    "gen": "ADRG",
    "gif": "GIF",
    "gpkg": "GPKG",
    "grd": "NWT_GRD",
    "gsb": "NTv2",
    "gtx": "GTX",
    "hdr": "MFF",
    "hf2": "HF2",
    "hgt": "SRTMHGT",
    "img": "HFA",
    "jp2": "JP2OpenJPEG",
    "jpg": "JPEG",
    "jpeg": "JPEG",
    "kea": "KEA",
    "kro": "KRO",
    "lcp": "LCP",
    "map": "PCRaster",
    "mbtiles": "MBTiles",
    "mrf": "MRF",
    "nc": "netCDF",
    "ntf": "NITF",
    "pdf": "PDF",
    "pix": "PCIDSK",
    "png": "PNG",
    "rda": "R",
    "rgb": "SGI",
    "rst": "RST",
    "rsw": "RMF",
    "sigdem": "SIGDEM",
    "sqlite": "Rasterlite",
    "ter": "Terragen",
    "tif": "GTiff",
    "tiff": "GTiff",
    "vrt": "VRT",
    "xml": "PDS4",
    "xpm": "XPM",
    "xyz": "XYZ",
}


def _get_driver(path):
    ext = path.suffix.lower()[1:]  # skip the period
    try:
        return EXTENSION_GDAL_DRIVER_CODE_MAP[ext]
    except KeyError:
        raise ValueError(
            f'Unknown extension "{ext}", available extensions: '
            f'{", ".join(EXTENSION_GDAL_DRIVER_CODE_MAP.keys())}'
        )


def _limitations(riods, path):
    if riods.count != 1:
        raise NotImplementedError(
            f"Cannot open multi-band grid: {path}. Try rioxarray instead."
        )
    if not riods.transform.is_rectilinear:
        raise NotImplementedError(
            f"Cannot open non-rectilinear grid: {path}. Try rioxarray instead."
        )


def header(path, pattern):
    attrs = imod.util.path.decompose(path, pattern)

    # TODO:
    # Check bands, rotation, etc.
    # Raise NotImplementedErrors and point to rioxarray.open_rasterio
    with rasterio.open(path, "r") as riods:
        _limitations(riods, path)
        attrs["nrow"] = riods.height
        attrs["ncol"] = riods.width
        xmin, ymin, xmax, ymax = riods.bounds
        attrs["dx"] = riods.transform[0]
        attrs["dy"] = riods.transform[4]
        attrs["nodata"] = riods.nodata
        attrs["dtype"] = riods.dtypes[0]
        crs = riods.crs
        if crs is not None:
            attrs["crs"] = crs

    attrs["xmin"] = xmin
    attrs["xmax"] = xmax
    attrs["ymin"] = ymin
    attrs["ymax"] = ymax
    attrs["headersize"] = None
    return attrs


def _read(path, headersize, nrow, ncol, nodata, dtype):
    with rasterio.open(path, "r") as dataset:
        a = dataset.read(1)

    # None signifies no replacement; skip if nodata already is nan
    if (nodata is None) or np.isnan(nodata):
        return a
    # Only set nodata to nan if the dtype supports it
    if (a.dtype == np.float64) or (a.dtype == np.float32):
        return array_io.reading._to_nan(a, nodata)
    else:
        return a


def read(path, use_cftime=False, pattern=None):
    warnings.warn(
        "The imod.rasterio.read() function is deprecated. Instead use imod.rasterio.open().",
        FutureWarning,
    )
    return open(path, use_cftime=False, pattern=None).load()


def open(path, use_cftime=False, pattern=None):
    r"""
    Open one or more GDAL supported raster files as an xarray.DataArray.

    In accordance with xarray's design, ``open`` loads the data of the files
    lazily. This means the data of the rasters are not loaded into memory until the
    data is needed. This allows for easier handling of large datasets, and
    more efficient computations.

    Parameters
    ----------
    path : str, Path or list
        This can be a single file, 'head_l1.tif', a glob pattern expansion,
        'head_l*.tif', or a list of files, ['head_l1.tif', 'head_l2.tif'].
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
        A float32 xarray.DataArray of the values in the raster file(s).
        All metadata needed for writing the file to raster or other formats
        using imod.rasterio are included in the xarray.DataArray.attrs.

    Examples
    --------
    Open a raster file:

    >>> da = imod.rasterio.open("example.tif")

    Open a raster file, relying on default naming conventions to identify
    layer:

    >>> da = imod.rasterio.open("example_l1.tif")

    Open an IDF file, relying on default naming conventions to identify layer
    and time:

    >>> head = imod.rasterio.open("head_20010101_l1.tif")

    Open multiple files, in this case files for the year 2001 for all
    layers, again relying on default conventions for naming:

    >>> head = imod.rasterio.open("head_2001*_l*.tif")

    The same, this time explicitly specifying ``name``, ``time``, and ``layer``:

    >>> head = imod.rasterio.open("head_2001*_l*.tif", pattern="{name}_{time}_l{layer}")

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
    return array_io.reading._open(path, use_cftime, pattern, header, _read)


def write_aaigrid(path: pathlib.Path, a: np.ndarray, profile: Dict) -> None:
    """
    Fall-back function to write ESRII ASCII grids even if rasterio is not
    installed.

    This function takes care to mimick the idiosyncracies of the GDAL driver.

    Parameters
    ----------
    path: str or Path
        path to the output raster
    a: np.ndarray
        The raster data.
    profile: dict
        The rasterio profile metadata.
    """
    dtype = profile["dtype"]
    nodata = profile["nodata"]
    df = pd.DataFrame(a.astype(dtype))
    # GDAL writes white space before every line. pandas doesn't support this,
    # but it will write nodata values -- encoded as NaN -- as white space.
    df.index = np.full(len(df), np.nan)
    if np.issubdtype(dtype, np.integer):
        is_float = False
        space = ""
        fmt = "%d"
        str_nodata = f"{int(nodata)}"
    elif np.issubdtype(dtype, np.floating):
        # For some reason, GDAL inserts a space before the nodata value if
        # dtype is float.
        is_float = True
        space = " "
        precision = profile.get("decimal_precision")
        digits = profile.get("significant_digits")
        # See: https://docs.python.org/3/library/string.html#formatspec
        if precision is not None:
            fmt = f"%.{precision}f"
            str_nodata = f"{nodata:.{precision}f}"
        elif digits is not None:
            fmt = f"%.{digits}g"
            str_nodata = f"{nodata:.{digits}g}"
        else:
            fmt = "%.20g"
            str_nodata = f"{nodata:.20g}"
    else:
        raise TypeError(f"invalid dtype: {dtype}")

    dx, _, xmin, _, dy, ymax = profile["transform"][:6]
    ymin = ymax + profile["height"] * dy

    header = (
        f'ncols        {profile["width"]}\n'
        f'nrows        {profile["height"]}\n'
        f"xllcorner    {xmin:.12f}\n"
        f"yllcorner    {ymin:.12f}\n"
        f"cellsize     {dx:.12f}\n"
        f"NODATA_value {space}{str_nodata}\n"
    )

    # GDAL writes only a linefeed, not a carriage return. By default, Python
    # adds a carriage return as well, on Windows. This is disabled by
    # explicitly setting the newline argument.
    with f_open(path, "w", newline="") as f:
        f.write(header)

        first = df.iloc[0, 0]
        # is_float is the result of a typecheck above.
        # is_integer() checks whether a float has a decimal fraction:
        # (1.0).is_integer() -> True
        # (1.1).is_integer() -> False
        if is_float and first.is_integer():
            # GDAL uses the "general" float (g) format by default. However, if
            # the first value is a float without decimals, it will write a
            # single trailing 0 for the first value, presumably to aid type
            # inference when reading values back in. All subsequent values are
            # written without decimals, however.
            precision = profile.get("decimal_precision", 1)
            # Write the first value, with trailing zero if needed.
            f.write(f" {first:.{precision}f} ")
            # Write remainder of first row. Since we've already written the
            # first value with a space in front of it, we skip writing the NaN
            # index value here.
            df.iloc[[0], 1:].to_csv(
                f, index=False, header=False, sep=" ", float_format=fmt
            )
            # Write all other rows.
            df.iloc[1:].to_csv(f, index=True, header=False, sep=" ", float_format=fmt)
        else:
            df.to_csv(f, index=True, header=False, sep=" ", float_format=fmt)

    return


def write(path, da, driver=None, nodata=np.nan, dtype=None):
    """Write ``xarray.DataArray`` to GDAL supported geospatial rasters using ``rasterio``.

    Parameters
    ----------
    path: str or Path
        path to the output raster
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
    # with xioxarray.
    # Note that this function can quickly become dstdated as
    # the xarray rasterio connection matures, see for instance:
    # https://github.com/pydata/xarray/issues/1736
    # https://github.com/pydata/xarray/pull/1712
    path = pathlib.Path(path)
    profile = da.attrs.copy()
    if driver is None:
        driver = _get_driver(path)

    flip = slice(None, None, -1)
    if not da.indexes["x"].is_monotonic_increasing:
        da = da.isel(x=flip)
    if not da.indexes["y"].is_monotonic_decreasing:
        da = da.isel(y=flip)

    # Only try to fill data that can contains nan's
    # Do this before casting to another type!
    ignore_nodata = (nodata is None) or np.isnan(nodata)
    if not ignore_nodata:
        if (da.dtype == np.float32) or (da.dtype == np.float64):
            # NaN is the default missing value in xarray
            # None is different in that the raster won't have a nodata value
            da = da.fillna(nodata)

    # Cast to dtype if dtype is given
    if dtype is not None:
        if da.dtype != dtype:
            da = da.astype(dtype)

    # Cast to supported pcraster dtypes
    if driver == "PCRaster":
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
    profile["transform"] = util.spatial.transform(da)
    profile["driver"] = driver
    profile["height"] = da.y.size
    profile["width"] = da.x.size
    profile["count"] = 1
    profile["dtype"] = da.dtype
    profile["nodata"] = nodata

    # Allow writing ASCII grids even if rasterio isn't installed. This is
    # useful for e.g. MetaSWAP input.
    if isinstance(rasterio, MissingOptionalModule) and driver == "AAIGrid":
        write_aaigrid(path, da.values, profile)
    else:
        with rasterio.Env():
            with rasterio.open(path, "w", **profile) as ds:
                ds.write(da.values, 1)
    return


def save(path, a, driver=None, nodata=np.nan, pattern=None, dtype=None):
    """
    Write a xarray.DataArray to one or more rasterio supported files

    If the DataArray only has ``y`` and ``x`` dimensions, a single raster file is
    written, like the ``imod.rasterio.write`` function. This function is more general
    and also supports ``time`` and ``layer`` and other dimensions. It will split these up,
    give them their own filename according to the conventions in
    ``imod.imod.util.path.compose``, and write them each.

    Parameters
    ----------
    path : str or Path
        Path to the raster file to be written. This function decides on the
        actual filename(s) using conventions, so it only takes the directory and
        name from this parameter.
    a : xarray.DataArray
        DataArray to be written. It needs to have dimensions ('y', 'x'), and
        optionally ``layer`` and ``time``.
    driver: str, optional
        Which GDAL format driver to use. The complete list is at
        https://gdal.org/drivers/raster/index.html.
        By default tries to guess from the file extension.
    nodata : float, optional
        Nodata value in the saved raster files. Defaults to a value of nan.
    pattern : str
        Format string which defines how to create the filenames. See examples.

    Example
    -------
    Consider a DataArray ``da`` that has dimensions 'layer', 'y' and 'x', with the
    'layer' dimension consisting of layer 1 and 2::

        save('path/to/head', da)

    This writes the following two tif files: 'path/to/head_l1.tif' and
    'path/to/head_l2.tif'.


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
    path = pathlib.Path(path)
    # defaults to geotiff
    if driver is None:
        if path.suffix == "":
            path = path.with_suffix(".tif")
            driver = "GTiff"
        else:
            driver = _get_driver(path)

    # Use a closure to skip the driver argument
    # so it takes the same arguments as the idf write
    def _write(path, a, nodata, dtype):
        return write(path, a, driver, nodata, dtype)

    array_io.writing._save(path, a, nodata, pattern, dtype, _write)
