import numpy as np
import xarray as xr
import rasterio
import affine
from rasterio.warp import Resampling
from pathlib import Path
from glob import glob
from imod import idf, util


def write(path, da, driver=None, nodata=np.nan):
    """Write DataArray to GDAL supported geospatial rasters using rasterio
    
    Parameters
    ----------
    path: str or Path
        path to the dstput raste
    da: xarray DataArray
        The DataArray to be written. Should have only x and y dimensions.
    driver: str; optional
        Which GDAL format driver to use. The complete list is at
        http://www.gdal.org/formats_list.html.
        By default tries to guess from the file extension.
    nodata: float
        Nodata value to use. Should be convertible to the DataArray and GDAL dtype.
        Default value is np.nan
    """
    # Not directly related to iMOD, but provides a missing link, together
    # with xarray.open_rasterio.
    # Note that this function can quickly become dstdated as
    # the xarray rasterio connection matures, see for instance:
    # https://github.com/pydata/xarray/issues/1736
    # https://github.com/pydata/xarray/pull/1712
    if isinstance(path, str):
        path = Path(path)
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


def _reproject_dst(source, src_crs, dst_crs, src_transform):
    """
    Prepares destination transform Affine and DataArray for projection.
    """
    src_height, src_width = source.y.size, source.x.size
    bounds = rasterio.transform.array_bounds(src_height, src_width, src_transform)
    dst_transform, dst_width, dst_height = rasterio.warp.calculate_default_transform(
        src_crs, dst_crs, src_width, src_height, *bounds
    )
    # from: http://xarray.pydata.org/en/stable/generated/xarray.open_rasterio.html
    x, y = (
        np.meshgrid(np.arange(dst_width) + 0.5, np.arange(dst_height) + 0.5)
        * dst_transform
    )
    dst = xr.DataArray(
        data=np.zeros((dst_height, dst_width), source.dtype),
        coords={"y": y[:, 0], "x": x[0, :]},
        dims=("y", "x"),
    )
    return dst_transform, dst


def resample(
    source,
    like=None,
    src_crs=None,
    dst_crs=None,
    method="nearest",
    use_src_attrs=False,
    src_nodata=np.nan,
    **reproject_kwargs,
):
    """
    Reprojects and/or resamples a 2D xarray DataArray to a 
    different cellsize or coordinate system.

    * To resample to a new cellsize in the same projection: provide only `like`.
    * To only reproject: provide only `src_crs` and `src_crs`.
    * To reproject and resample to a specific domain: provide `src_crs`, `src_crs`, and `like`.
    
    Note: when only `like` is provided, Cartesian (projected) coordinates are a
    ssumed for resampling. In case of non-Cartesian coordinates, specify 
    `src_crs` and `dst_crs` for correct resampling.

    Parameters
    ----------
    source: xarray DataArray
        The DataArray to be resampled and/or reprojected. Must contain dimensions
        `y` and `x`.
    like: xarray DataArray
        Example DataArray that shows what the resampled result should look like 
        in terms of coordinates. Must contain dimensions `y` and `x`.
    src_crs: string, dict, rasterio.crs.CRS
        Coordinate system of `source`. Options:

        * string: e.g. `"+init=EPSG:4326"`
        * dict: e.g. `{"init":"EPSG:4326"}`
        * rasterio.crs.CRS
    dst_crs: string, dict, rasterio.crs.CRS
        Coordinate system of result. Options:

        * string: e.g. `"+init=EPSG:4326"`
        * dict: e.g. `{"init":"EPSG:4326"}`
        * rasterio.crs.CRS
    use_src_attrs: boolean
        If True: Use metadata in `source.attrs`, as generated by `xarray.open_rasterio()`, to do 
        reprojection.
    method: string
        The method to use for resampling/reprojection.
        Defaults to "nearest". GDAL methods are available:

        * nearest
        * bilinear
        * cubic
        * cubic_spline
        * lanczos
        * average
        * mode
        * gauss
        * max
        * min
        * med (50th percentile)
        * q1 (25th percentile)
        * q3 (75th percentile)
    reproject_kwargs: dict, optional
        keyword arguments for `rasterio.warp.reproject()`.

    Returns
    ------- 
    xarray.DataArray
        Resampled/reprojected DataArray.

    Examples
    --------
    Resample a DataArray `a` to a new cellsize, using an existing DataArray `b`:
    
    >>> c = imod.rasterio.resample(source=a, like=b)
    
    Resample a DataArray to a new cellsize of 100.0, by creating a `like` DataArray first:
    (Note that dy must be negative, as is usual for geospatial grids.)
    
    >>> dims = ("y", "x")
    >>> coords = {"y": np.arange(200_000.0, 100_000.0, -100.0), "x": np.arange(0.0, 100_000.0, 100.0)}
    >>> b = xr.DataArray(data=np.empty((200, 100)), coords=coords, dims=dims)
    >>> c = imod.rasterio.resample(source=a, like=b)

    Reproject a DataArray from one coordinate system (WGS84, EPSG:4326) to another (UTM30N, EPSG:32630):

    >>> c = imod.rasterio.resample(source=a, src_crs="+init=EPSG:4326", dst_crs="+init=EPSG:32630")

    Get the reprojected DataArray in the desired shape and coordinates by providing `like`:

    >>> c = imod.rasterio.resample(source=a, like=b, src_crs="+init=EPSG:4326", dst_crs="+init=EPSG:32630")

    Open a single band raster, and reproject to RD new coordinate system (EPSG:28992), without explicitly specifying `src_crs`.
    `src_crs` is taken from `a.attrs`, so the raster file has to include coordinate system metadata for this to work.

    >>> a = xr.open_rasterio("example.tif").squeeze("band")
    >>> c = imod.rasterio.resample(source=a, use_src_attrs=True, dst_crs="+init=EPSG:28992")

    In case of a rotated `source`, provide `src_transform` directly or `use_src_attrs=True` to rely on generated attributes:

    >>> rotated = xr.open_rasterio("rotated_example.tif").squeeze("band")
    >>> c = imod.rasterio.resample(source=rotated, dst_crs="+init=EPSG:28992", reproject_kwargs={"src_transform":affine.Affine(...)})
    >>> c = imod.rasterio.resample(source=rotated, dst_crs="+init=EPSG:28992", use_src_attrs=True)
    """
    assert source.dims == (
        "y",
        "x",
    ), "resample does not support dimensions other than `x` and `y` for `source`."
    if like is not None:
        assert like.dims == (
            "y",
            "x",
        ), "resample does not support dimensions other than `x` and `y` for `like`."
    if use_src_attrs:  # only provided when reproject is necessary
        src_crs = rasterio.crs.CRS.from_string(source.attrs["crs"])
        src_nodata = source.attrs["nodatavals"][0]

    resampling_methods = {e.name: e for e in Resampling}

    if isinstance(method, str):
        try:
            resampling_method = resampling_methods[method]
        except KeyError as e:
            raise ValueError(
                "Invalid resampling method. Available methods are: {}".format(
                    resampling_methods.keys()
                )
            ) from e
    elif isinstance(method, Resampling):
        resampling_method = method
    else:
        raise TypeError("method must be a string or rasterio.warp.Resampling")

    # Givens: source, like, method. No reprojection necessary.
    if src_crs is None and dst_crs is None:
        if like is None:
            raise ValueError(
                "If crs information is not provided, `like` must be provided."
            )
        if resampling_method == Resampling.nearest:
            # this can be handled with xarray
            # xarray 0.10.9 needs .compute()
            # see https://github.com/pydata/xarray/issues/2454
            return source.compute().reindex_like(like, method="nearest")
        else:
            # if no crs is defined, assume it should remain the same
            # in this case use UTM30, ESPG:32630, as a dummy value for GDAL
            # (Any projected coordinate system should suffice, Cartesian plane == Cartesian plane)
            dst = like.copy()
            src_crs = dst_crs = rasterio.crs.CRS.from_epsg(32630)
        src_transform = util.transform(source)
        dst_transform = util.transform(like)

    elif src_crs and dst_crs:
        if use_src_attrs:
            # TODO: modify if/when xarray uses affine by default for transform
            src_transform = affine.Affine(*source.attrs["transform"][:6])
        elif "src_transform" in reproject_kwargs.keys():
            src_transform = reproject_kwargs.pop("src_transform")
        else:
            src_transform = util.transform(source)

        # If no like is provided, just resample to different coordinate system
        if like is None:
            dst_transform, dst = _reproject_dst(source, src_crs, dst_crs, src_transform)
        else:
            dst_transform = util.transform(like)
            dst = like.copy()

    else:
        raise ValueError(
            "At least `like`, or crs information for source and destination must be provided."
        )

    assert src_transform[0] > 0, "dx of 'source' must be positive"
    assert src_transform[4] < 0, "dy of 'source' must be negative"
    assert dst_transform[0] > 0, "dx of 'like' must be positive"
    assert dst_transform[4] < 0, "dy of 'like' must be negative"

    rasterio.warp.reproject(
        source.values,
        dst.values,
        src_transform=src_transform,
        dst_transform=dst_transform,
        src_crs=src_crs,
        dst_crs=dst_crs,
        resampling=resampling_method,
        src_nodata=src_nodata,
        **reproject_kwargs,
    )

    dst.attrs = source.attrs
    dst.attrs["transform"] = dst_transform
    dst.attrs["res"] = (abs(dst_transform[0]), abs(dst_transform[4]))
    dst.attrs["crs"] = dst_crs
    # TODO: what should be the type of "crs" field in attrs?
    # Doesn't actually matter for functionality
    # rasterio accepts string, dict, and CRS object
    return dst
