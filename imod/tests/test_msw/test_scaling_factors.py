import tempfile
from pathlib import Path

import numpy as np
import xarray as xr
from numpy import nan
from numpy.testing import assert_almost_equal, assert_equal

from imod.mf6.utilities.regrid import (
    RegridderWeightsCache,
)
from imod.msw import ScalingFactors


def setup_scaling_factor():
    x = [1.0, 2.0, 3.0]
    y = [1.0, 2.0, 3.0]
    subunit = [0, 1]
    dx = 1.0
    dy = 1.0

    # fmt: off
    scale = xr.DataArray(
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

    depth_perched_water_table = xr.DataArray(
        np.array(
            [[0.5, 0.5, 0.5],
             [0.7, 0.7, 0.7],
             [1.0, 1.0, 1.0]]
        ),
        dims=("y", "x"),
        coords={"y": y, "x": x, "dx": dx, "dy": dy}
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

    scaling_factors = ScalingFactors(
        scale_soil_moisture=scale,
        scale_hydraulic_conductivity=scale,
        scale_pressure_head=scale,
        depth_perched_water_table=depth_perched_water_table,
    )

    return scaling_factors, index, svat


def test_simple_model(fixed_format_parser):
    scaling_factors, index, svat = setup_scaling_factor()

    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        scaling_factors.write(output_dir, index, svat)

        results = fixed_format_parser(
            output_dir / ScalingFactors._file_name, ScalingFactors._metadata_dict
        )

    assert_equal(results["svat"], np.array([1, 2, 3, 4]))
    assert_almost_equal(results["scale_soil_moisture"], np.array([0.5, 1.0, 0.5, 1.0]))
    assert_almost_equal(
        results["scale_hydraulic_conductivity"], np.array([0.5, 1.0, 0.5, 1.0])
    )
    assert_almost_equal(results["scale_pressure_head"], np.array([0.5, 1.0, 0.5, 1.0]))
    assert_almost_equal(
        results["depth_perched_water_table"], np.array([0.5, 1.0, 0.5, 0.7])
    )


def test_regrid_scaling_factor(fixed_format_parser, simple_2d_grid_with_subunits):
    scaling_factors, _, _ = setup_scaling_factor()
    new_grid = simple_2d_grid_with_subunits

    regrid_context = RegridderWeightsCache()

    regridded_scaling_factor = scaling_factors.regrid_like(new_grid, regrid_context)

    assert np.all(regridded_scaling_factor.dataset["x"].values == new_grid["x"].values)
    assert np.all(regridded_scaling_factor.dataset["y"].values == new_grid["y"].values)
