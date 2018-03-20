import rasterio
import numpy as np
from pathlib import Path


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
        if ext in ('.tif', '.tiff'):
            driver = 'GTiff'
        elif ext == '.asc':
            driver = 'AAIGrid'
        elif ext == '.map':
            driver = 'PCRaster'
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
    if (nodata is None) or np.isnan(nodata):
        # NaN is the default missing value in xarray
        # None is different in that the raster won't have a nodata value
        dafilled = da
    else:
        dafilled = da.fillna(nodata)
    with rasterio.Env():
        with rasterio.open(path, 'w', **profile) as ds:
            ds.write(dafilled.values, 1)
