import tempfile
from pathlib import Path

import numpy as np
import xarray as xr
from numpy.testing import assert_almost_equal, assert_equal

from imod.mf6.utilities.regrid import (
    RegridderWeightsCache,
)
from imod.msw import LanduseOptions


def create_landuse_dict():
    landuse_index = np.arange(1, 4)
    names = ["grassland", "maize", "potatoes"]

    coords = {"landuse_index": landuse_index}

    landuse_names = xr.DataArray(data=names, coords=coords, dims=("landuse_index",))
    vegetation_index_da = xr.DataArray(
        data=np.arange(1, 4), coords=coords, dims=("landuse_index",)
    )
    lu = xr.ones_like(vegetation_index_da, dtype=float)

    options = dict(
        landuse_name=landuse_names,
        vegetation_index=vegetation_index_da,
        jarvis_o2_stress=xr.ones_like(lu),
        jarvis_drought_stress=xr.ones_like(lu),
        feddes_p1=xr.full_like(lu, 99.0),
        feddes_p2=xr.full_like(lu, 99.0),
        feddes_p3h=lu * [-2.0, -4.0, -3.0],
        feddes_p3l=lu * [-8.0, -5.0, -5.0],
        feddes_p4=lu * [-80.0, -100.0, -100.0],
        feddes_t3h=xr.full_like(lu, 5.0),
        feddes_t3l=xr.full_like(lu, 1.0),
        threshold_sprinkling=lu * [-8.0, -5.0, -5.0],
        fraction_evaporated_sprinkling=xr.full_like(lu, 0.05),
        gift=xr.full_like(lu, 20.0),
        gift_duration=xr.full_like(lu, 0.25),
        rotational_period=lu * [10, 7, 7],
        start_sprinkling_season=lu * [120, 180, 150],
        end_sprinkling_season=lu * [230, 230, 240],
        interception_option=xr.ones_like(lu, dtype=int),
        interception_capacity_per_LAI=xr.zeros_like(lu),
        interception_intercept=xr.ones_like(lu),
    )
    return options


def get_new_grid():
    x = [1.0, 1.5, 2.0, 2.5, 3.0]
    y = [3.0, 2.5, 2.0, 1.5, 1.0]
    subunit = [0, 1]
    dx = 0.5
    dy = 0.5
    # fmt: off
    new_grid = xr.DataArray(
        dims=("subunit", "y", "x"),
        coords={"subunit": subunit, "y": y, "x": x, "dx": dx, "dy": dy}
    )
    new_grid.values[:,:,:] = 1
    return new_grid


def test_landuse_options(fixed_format_parser):
    options = create_landuse_dict()
    lu_options = LanduseOptions(**options)

    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        lu_options.write(output_dir, None, None)

        results = fixed_format_parser(
            output_dir / LanduseOptions._file_name, LanduseOptions._metadata_dict
        )

    for option, value in options.items():
        if option == "landuse_name":
            # Strip spaces of names first
            assert_equal(np.char.rstrip(results[option]), value.values)
        elif option == "interception_capacity_per_LAI":
            option1 = option + "_Rutter"
            option2 = option + "_VonHoyningen"
            assert_almost_equal(results[option1], value.values)
            assert_almost_equal(results[option2], value.values)
        elif np.issubdtype(value.dtype, np.floating):
            assert_almost_equal(results[option], value.values)
        else:
            assert_equal(results[option], value.values)


def test_landuse_options_regrid():
    new_grid = get_new_grid()
    options = create_landuse_dict()
    lu_options = LanduseOptions(**options)

    regrid_context = RegridderWeightsCache(new_grid, new_grid)
    regridded_land_use = lu_options.regrid_like(new_grid, regrid_context)

    assert len(regridded_land_use.dataset.coords.keys()) == 1
    assert np.all(
        regridded_land_use.dataset.coords["landuse_index"].values == np.array([1, 2, 3])
    )
