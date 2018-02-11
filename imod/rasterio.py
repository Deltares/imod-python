import os
import rasterio


def write(path, da, driver=None, nodata=-9999):
    """Write DataArray to GDAL supported geospatial rasters using rasterio"""
    # Not directly related to iMOD, but provides a missing link, together
    # with xarray.open_rasterio.
    # Note that this function can quickly become outdated as
    # the xarray rasterio connection matures, see for instance:
    # https://github.com/pydata/xarray/issues/1736
    # https://github.com/pydata/xarray/pull/1712
    profile = da.attrs.copy()
    if driver is None:
        ext = os.path.splitext(path)[1].lower()
        if ext in ('.tif', '.tiff'):
            driver = 'GTiff'
        elif ext == '.asc':
            driver = 'AAIGrid'
        else:
            raise ValueError(
                'Unknown extension {}, specifiy driver'.format(ext))
    # prevent rasterio warnings
    if driver == 'AAIGrid':
        profile.pop('res', None)
        profile.pop('is_tiled', None)
    # transform will be affine object in next xarray
    profile['transform'] = profile['transform'][:6]
    profile['driver'] = driver
    profile['height'] = da.y.size
    profile['width'] = da.x.size
    profile['count'] = 1
    profile['dtype'] = da.dtype
    profile['nodata'] = nodata
    dafilled = da.fillna(nodata)
    with rasterio.Env():
        with rasterio.open(path, 'w', **profile) as ds:
            ds.write(dafilled.values, 1)
