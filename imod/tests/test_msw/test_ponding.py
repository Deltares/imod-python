import tempfile
from pathlib import Path

import numpy as np
import xarray as xr
from numpy import nan
from numpy.testing import assert_almost_equal, assert_equal

from imod.mf6.utilities.regrid import (
    RegridderWeightsCache,
)
from imod.msw import Ponding


def setup_ponding():
    x = [1.0, 2.0, 3.0]
    y = [1.0, 2.0, 3.0]
    subunit = [0, 1]
    dx = 1.0
    dy = 1.0

    # fmt: off

    ponding_depth = xr.DataArray(
        np.array(
            [
                [[0.5, 0.5, 0.5],
                 [nan, nan, nan],
                 [1.0, 1.0, 1.0]],

                [[0.5, 0.5, 0.5],
                 [1.0, 1.0, 1.0],
                 [nan, nan, nan]],
            ]
        ),
        dims=("subunit", "y", "x"),
        coords={"subunit": subunit, "y": y, "x": x, "dx": dx, "dy": dy}
    )

    runoff_resistance = xr.DataArray(
        np.array(
            [
                [[0.5, 0.5, 0.5],
                 [nan, nan, nan],
                 [1.0, 1.0, 1.0]],
                [[0.5, 0.5, 0.5],
                 [1.0, 1.0, 1.0],
                 [nan, nan, nan]],
            ]
        ),
        dims=("subunit", "y", "x"),
        coords={"subunit": subunit, "y": y, "x": x, "dx": dx, "dy": dy}
    )
    svat = xr.DataArray(
        np.array(
            [
                [[0, 1, 0],
                 [0, 0, 0],
                 [0, 2, 0]],

                [[0, 3, 0],
                 [0, 4, 0],
                 [0, 0, 0]],
            ]
        ),
        dims=("subunit", "y", "x"),
        coords={"subunit": subunit, "y": y, "x": x, "dx": dx, "dy": dy}
    )
    # fmt: on
    index = (svat != 0).values.ravel()

    ponding = Ponding(
        ponding_depth=ponding_depth,
        runoff_resistance=runoff_resistance,
        runon_resistance=runoff_resistance,
    )
    return ponding, index, svat


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


def test_simple_model(fixed_format_parser):
    ponding, index, svat = setup_ponding()
    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        ponding.write(output_dir, index, svat)

        results = fixed_format_parser(
            output_dir / Ponding._file_name, Ponding._metadata_dict
        )

    assert_equal(results["svat"], np.array([1, 2, 3, 4]))
    assert_equal(results["swnr"], np.array([0, 0, 0, 0]))
    assert_almost_equal(results["ponding_depth"], np.array([0.5, 1.0, 0.5, 1.0]))
    assert_almost_equal(results["runoff_resistance"], np.array([0.5, 1.0, 0.5, 1.0]))
    assert_almost_equal(results["runoff_resistance"], np.array([0.5, 1.0, 0.5, 1.0]))


def test_regrid_ponding():
    ponding, index, svat = setup_ponding()
    new_grid = get_new_grid()

    old_grid = ponding.dataset["ponding_depth"].isel(subunit=0)
    regrid_context = RegridderWeightsCache(old_grid, new_grid)

    regridded_ponding = ponding.regrid_like(new_grid, regrid_context)
    pass
