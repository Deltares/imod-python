import numpy as np
import xarray as xr

from imod.msw.idf_mapping import IdfMapping
from imod.tests.fixtures.msw_regrid_fixture import get_3x3_area, get_5x5_new_grid
from imod.util.regrid import RegridderWeightsCache


def test_idf_mapping_regrid():
    area = get_3x3_area()

    idf_mapping = IdfMapping(area, np.nan)

    new_grid = get_5x5_new_grid()
    regrid_context = RegridderWeightsCache()

    idf_mapping_regridded = idf_mapping.regrid_like(new_grid, regrid_context)

    assert len(idf_mapping_regridded.dataset["rows"]) == 5
    assert len(idf_mapping_regridded.dataset["columns"]) == 5
    assert np.isnan(idf_mapping_regridded.dataset["nodata"].values[()])


def test_idf_mapping_clip():
    area = get_3x3_area()

    idf_mapping = IdfMapping(area, np.nan)

    idf_mapping_selected = idf_mapping.clip_box(
        x_min=1.0, x_max=2.5, y_min=1.0, y_max=2.5
    )

    expected_area = area.sel(x=slice(1.0, 2.5), y=slice(2.5, 1.0))
    xr.testing.assert_allclose(idf_mapping_selected.dataset["area"], expected_area)
