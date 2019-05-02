import numpy as np
import scipy.ndimage
import xarray as xr
import rasterio
import rasterio.features
import imod


def round_extent(extent, cellsize):
    """Increases the extent until all sides lie on a coordinate
    divisible by cellsize."""
    xmin, ymin, xmax, ymax = extent
    xmin = np.floor(xmin / cellsize) * cellsize
    ymin = np.floor(ymin / cellsize) * cellsize
    xmax = np.ceil(xmax / cellsize) * cellsize
    ymax = np.ceil(ymax / cellsize) * cellsize
    return xmin, ymin, xmax, ymax


def round_z(z_extent, dz):
    """Increases the extent until all sides lie on a coordinate
    divisible by dz."""
    zmin, zmax = z_extent
    zmin = np.floor(zmin / dz) * dz
    zmax = np.ceil(zmax / dz) * dz
    return zmin, zmax


def _fill_np(data, invalid):
    """Basic nearest neighbour interpolation"""
    # see: https://stackoverflow.com/questions/5551286/filling-gaps-in-a-numpy-array
    ind = scipy.ndimage.distance_transform_edt(
        invalid, return_distances=False, return_indices=True
    )
    return data[tuple(ind)]


def fill(da, invalid=None, by=None):
    """
    Replace the value of invalid `da` cells (indicated by `invalid`)
    using basic nearest neighbour interpolation.

    Parameters
    ----------
    da: xr.DataArray with gaps
        array containing missing value
        if one of the dimensions is layer, it will interpolate one layer at a
        a time (2D interpolation over x and y in case of dims == (by, "y", "x")).

    invalid: xr.DataArray
        a binary array of same shape as `da`.
        data value are replaced where invalid is True
        If None (default), uses: `invalid = np.isnan(data)`

    Returns
    -------
    xarray.DataArray
        with the same coordinates as the input.
    """

    out = xr.full_like(da, np.nan)
    if invalid is None:
        invalid = np.isnan(da)
    if by:
        for layer in da[by]:
            out.sel(by=layer)[...] = _fill_np(
                da.sel(by=layer).values, invalid.sel(by=layer).values
            )
    else:
        out.values = _fill_np(da.values, invalid.values)

    return out


def rasterize(geodataframe, like, column=None, fill=np.nan, **kwargs):
    """
    Rasterize a geopandas GeoDataFrame onto the given
    xarray coordinates.

    Parameters
    ----------
    geodataframe : geopandas.GeoDataFrame
    column : str, int, float
        column name of geodataframe to burn into raster
    like : xarray.DataArray
        Example DataArray. The rasterized result will match the shape and
        coordinates of this DataArray.
    fill : float, int
        Fill value for nodata areas. Optional, default value is np.nan.
    kwargs : additional keyword arguments for rasterio.features.rasterize.
        See: https://rasterio.readthedocs.io/en/stable/api/rasterio.features.html#rasterio.features.rasterize

    Returns
    -------
    rasterized : xarray.DataArray
        Vector data rasterized. Matches shape and coordinates of `like`.
    """

    if column is not None:
        shapes = [
            (geom, value)
            for geom, value in zip(geodataframe.geometry, geodataframe[column])
        ]
    else:
        shapes = [geom for geom in geodataframe.geometry]

    # shapes must be an iterable
    try:
        iter(shapes)
    except TypeError:
        shapes = (shapes,)

    raster = rasterio.features.rasterize(
        shapes,
        out_shape=like.shape,
        fill=fill,
        transform=imod.util.transform(like),
        **kwargs,
    )

    return xr.DataArray(raster, like.coords, like.dims)
