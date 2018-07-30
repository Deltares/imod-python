import numpy as np
import xarray as xr
import rasterio
from rasterio import crs
from rasterio.warp import Resampling
from rasterio.warp import calculate_default_transform
from rasterio.warp import reproject
from pathlib import Path
from glob import glob
from imod import util


def write(path, da, driver=None, nodata=np.nan):
    """Write DataArray to GDAL supported geospatial rasters using rasterio
    
    Parameters
    ----------
    path: str or Path
        path to the output raster
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
    # Note that this function can quickly become outdated as
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
            raise ValueError("Unknown extension {}, specifiy driver".format(ext))
    # prevent rasterio warnings
    if driver == "AAIGrid":
        profile.pop("res", None)
        profile.pop("is_tiled", None)
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


def resample(source, from_epsg=None, to_epsg=None, template=None, method="nearest"):
    """
    Reprojects and/or resamples a 2D xarray DataArray to a different cellsize
    or coordinate system.

    * To resample to a new cellsize in the same projection: provide only template.
    * To only reproject: provide only from_epsg and to_epsg.
    * To reproject and resample to a specific domain: provide from_epsg, to_epsg,
    and template.

    Parameters
    ----------
    source: xarray DataArray
        The DataArray to be resampled and/or reprojected. Must contain dimensions "y" and "x".
    from_epsg: int, string
        ESPG code of the source DataArray
    to_espg: 
        EPSG code of the coordinate system to reproject to
    template: xarray DataArray
        The template DataArray. Must contain dimensions "y" and "x".
    method: string
        The method to use for resampling or reprojection.
        Defaults to "nearest". GDAL method are available:
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
            * med: 50th percentile
            * q1: 25th percentile
            * q3: 75th percentile

    Returns
    ------- 
    xarray DataArray

    """
    assert source.dims == (
        "y",
        "x",
    ), "resample does not support dimensions other than 'x' and 'y' for source."
    if template is not None:
        assert template.dims == (
            "y",
            "x",
        ), "resample does not support dimensions other than 'x' and 'y' for template."

    # Givens: source, template, method. No reprojection necessary.
    if from_epsg is None and to_epsg is None:
        if template is None:
            raise ValueError("If EPSG's are not provided, template must be provided.")

        if method == "nearest":
            # this can be handled with xarray
            return source.reindex_like(template, method="nearest")
        else:
            # if no crs is defined, assume it should remain the same
            # in this case use UTM30, ESPG:32630, as a dummy value for GDAL
            # (Any projected coordinate system should suffice, Cartesian plane == Cartesian plane)
            out = template.copy()
            src_crs = dst_crs = crs.CRS.from_epsg(32630)

    elif from_epsg and to_epsg:
        src_crs = crs.CRS.from_epsg(from_epsg)
        dst_crs = crs.CRS.from_epsg(to_epsg)

        # If no template is provided, just resample to different coordinate system
        if template is None:
            src_height, src_width = source.y.size, source.x.size
            src_transform = util.transform(source)
            bounds = rasterio.transform.array_bounds(
                src_height, src_width, src_transform
            )
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src_crs, dst_crs, src_width, src_height, *bounds
            )

            # from: http://xarray.pydata.org/en/stable/generated/xarray.open_rasterio.html
            x, y = (
                np.meshgrid(np.arange(dst_width) + 0.5, np.arange(dst_height) + 0.5)
                * dst_transform
            )

            out = xr.DataArray(
                data=np.empty((dst_height, dst_width), source.dtype),
                coords={"y": y[:, 0], "x": x[0, :]},
                dims=("y", "x"),
            )
        # When both template and epsg's are provided
        else:
            src_transform = util.transform(source)
            dst_transform = util.transform(template)
            out = source.reindex_like(template)
    else:
        raise ValueError(
            "At least template, or both from_epsg and to_epsg, must be provided."
        )

    reproject(
        source.values,
        out.values,
        src_transform=src_transform,
        dst_transform=dst_transform,
        src_crs=src_crs,
        dst_crs=dst_crs,
        resampling=eval("Resampling." + method),
    )

    out.attrs = source.attrs
    out.attrs["transform"] = dst_transform
    out.attrs["crs"] = dst_crs.to_string()
    out.attrs["res"] = (abs(dst_transform[0]), abs(dst_transform[4]))

    return out
