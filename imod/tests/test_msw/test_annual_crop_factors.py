import numpy as np
import xarray as xr

from imod.msw import AnnualCropFactors
from imod.util.regrid import (
    RegridderWeightsCache,
)


def setup_cropfactors():
    day_of_year = np.arange(1, 367)
    vegetation_index = np.arange(1, 4)

    coords = {"day_of_year": day_of_year, "vegetation_index": vegetation_index}
    soil_cover = xr.DataArray(
        data=np.zeros(day_of_year.shape + vegetation_index.shape),
        coords=coords,
        dims=("day_of_year", "vegetation_index"),
    )
    soil_cover[132:254, :] = 1.0
    leaf_area_index = soil_cover * 3

    vegetation_factor = xr.zeros_like(soil_cover)
    vegetation_factor[132:142, :] = 0.7
    vegetation_factor[142:152, :] = 0.9
    vegetation_factor[152:162, :] = 1.0
    vegetation_factor[162:192, :] = 1.2
    vegetation_factor[192:244, :] = 1.1
    vegetation_factor[244:254, :] = 0.7

    cropfactors = AnnualCropFactors(
        soil_cover=soil_cover,
        leaf_area_index=leaf_area_index,
        interception_capacity=xr.zeros_like(soil_cover),
        vegetation_factor=vegetation_factor,
        interception_factor=xr.ones_like(soil_cover),
        bare_soil_factor=xr.ones_like(soil_cover),
        ponding_factor=xr.ones_like(soil_cover),
    )
    return cropfactors


def test_cropfactor_regrid(simple_2d_grid_with_subunits):
    crop_factors = setup_cropfactors()
    new_grid = simple_2d_grid_with_subunits

    regrid_context = RegridderWeightsCache()
    regridded = crop_factors.regrid_like(new_grid, regrid_context)

    assert regridded.dataset.equals(crop_factors.dataset)
