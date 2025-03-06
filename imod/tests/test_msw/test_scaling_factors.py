import tempfile
from pathlib import Path

import numpy as np
import xarray as xr
from numpy import nan
from numpy.testing import assert_almost_equal, assert_equal

from imod.msw import ScalingFactors
from imod.typing.grid import ones_like
from imod.util.regrid import (
    RegridderWeightsCache,
)


def setup_scaling_factor_grids():
    x = [1.0, 2.0, 3.0]
    y = [3.0, 2.0, 1.0]
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

    return scale, depth_perched_water_table, index, svat


def test_simple_model(fixed_format_parser):
    scale, depth_perched_water_table, index, svat = setup_scaling_factor_grids()

    scaling_factors = ScalingFactors(
        scale_soil_moisture=scale,
        scale_hydraulic_conductivity=scale,
        scale_pressure_head=scale,
        depth_perched_water_table=depth_perched_water_table,
    )

    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        scaling_factors.write(output_dir, index, svat, None, None)

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


def test_regrid_scaling_factor(simple_2d_grid_with_subunits):
    scale, depth_perched_water_table, _, _ = setup_scaling_factor_grids()

    scaling_factors = ScalingFactors(
        scale_soil_moisture=scale,
        scale_hydraulic_conductivity=scale,
        scale_pressure_head=scale,
        depth_perched_water_table=depth_perched_water_table,
    )

    new_grid = simple_2d_grid_with_subunits

    regrid_context = RegridderWeightsCache()

    regridded_scaling_factor = scaling_factors.regrid_like(new_grid, regrid_context)

    assert np.all(regridded_scaling_factor.dataset["x"].values == new_grid["x"].values)
    assert np.all(regridded_scaling_factor.dataset["y"].values == new_grid["y"].values)


def test_clip_box():
    scale, depth_perched_water_table, _, _ = setup_scaling_factor_grids()

    scaling_factors = ScalingFactors(
        scale_soil_moisture=scale,
        scale_hydraulic_conductivity=scale,
        scale_pressure_head=scale,
        depth_perched_water_table=depth_perched_water_table,
    )
    clipped = scaling_factors.clip_box(x_min=1.0, x_max=2.5, y_min=1.0, y_max=2.5)
    expected = scale.sel(x=slice(1.0, 2.5), y=slice(2.5, 1.0))
    xr.testing.assert_allclose(clipped.dataset["scale_soil_moisture"], expected)


def test_from_imod5_data(fixed_format_parser):
    scale, depth_perched_water_table, index, svat = setup_scaling_factor_grids()

    imod5_data = {"cap": {}}
    scale_rural = scale.sel(subunit=0, drop=True)
    imod5_data["cap"]["boundary"] = ones_like(scale_rural)
    imod5_data["cap"]["soil_moisture_fraction"] = scale_rural
    imod5_data["cap"]["conductivitiy_factor"] = scale_rural
    imod5_data["cap"]["perched_water_table_level"] = depth_perched_water_table

    scaling_factors = ScalingFactors.from_imod5_data(imod5_data)

    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        scaling_factors.write(output_dir, index, svat, None, None)

        results = fixed_format_parser(
            output_dir / ScalingFactors._file_name, ScalingFactors._metadata_dict
        )

    assert_equal(results["svat"], np.array([1, 2, 3, 4]))
    assert_almost_equal(results["scale_soil_moisture"], np.array([0.5, 1.0, 1.0, 1.0]))
    assert_almost_equal(
        results["scale_hydraulic_conductivity"], np.array([0.5, 1.0, 1.0, 1.0])
    )
    assert_almost_equal(results["scale_pressure_head"], np.array([1.0, 1.0, 1.0, 1.0]))
    assert_almost_equal(
        results["depth_perched_water_table"], np.array([0.5, 1.0, 0.5, 0.7])
    )
