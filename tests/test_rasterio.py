import numpy as np
import pytest
import rasterio
import xarray as xr

import imod


@pytest.fixture(scope="module")
def test_da():
    nrow, ncol = 3, 4
    dx, dy = 1.0, -1.0
    xmin, xmax = 0.0, 4.0
    ymin, ymax = 0.0, 3.0
    coords = imod.util._xycoords((xmin, xmax, ymin, ymax), (dx, dy))
    kwargs = {"name": "test", "coords": coords, "dims": ("y", "x")}
    data = np.ones((nrow, ncol), dtype=np.float32)
    da = xr.DataArray(data, **kwargs)
    da.attrs["crs"] = rasterio.crs.CRS(init="EPSG:28992")
    return da


def test_saveopen(test_da, tmp_path):
    da = test_da
    imod.rasterio.save(tmp_path / "withcrs.tif", da)
    daback = imod.rasterio.open(tmp_path / "withcrs.tif").load()
    daback.name = "test"
    assert daback.identical(test_da)

    da.attrs["crs"] = None
    imod.rasterio.save(tmp_path / "no_crs.asc", da)
    daback = imod.rasterio.open(tmp_path / "no_crs.asc").load()
    daback.name = "test"
    assert daback.identical(da)
