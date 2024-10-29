import numpy as np

from imod.mf6.utilities.regrid import RegridderWeightsCache
from imod.msw.idf_mapping import IdfMapping
from imod.tests.fixtures.msw_regrid_fixture import get_3x3_area, get_5x5_new_grid


def test_idf_mapping_regrid():
    area = get_3x3_area()

    idf_mapping = IdfMapping(area, np.nan)

    new_grid = get_5x5_new_grid()
    regrid_context = RegridderWeightsCache()

    idf_mapping_regridded = idf_mapping.regrid_like(new_grid, regrid_context)

    assert len(idf_mapping_regridded.dataset["rows"]) == 5
    assert len(idf_mapping_regridded.dataset["columns"]) == 5
    assert np.isnan(idf_mapping_regridded.dataset["nodata"].values[()])
