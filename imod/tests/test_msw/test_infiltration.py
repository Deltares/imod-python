import tempfile
from pathlib import Path

import numpy as np
import xarray as xr
from hypothesis import given, settings
from hypothesis.strategies import floats
from numpy import nan
from numpy.testing import assert_almost_equal, assert_equal

from imod.mf6.utilities.regrid import (
    RegridderWeightsCache,
)
from imod.msw import Infiltration
from imod.msw.fixed_format import format_fixed_width


def setup_infiltration_package(subunit, y, x, dy, dx):
    infiltration_capacity = xr.DataArray(
        np.array(
            [
                [[0.5, 0.5, 0.5], [nan, nan, nan], [1.0, 1.0, 1.0]],
                [[0.5, 0.5, 0.5], [1.0, 1.0, 1.0], [nan, nan, nan]],
            ]
        ),
        dims=("subunit", "y", "x"),
        coords={"subunit": subunit, "y": y, "x": x, "dx": dx, "dy": dy},
    )

    downward_resistance = xr.DataArray(
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
        dims=("y", "x"),
        coords={"y": y, "x": x, "dx": dx, "dy": dy},
    )

    upward_resistance = xr.DataArray(
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
        dims=("y", "x"),
        coords={"y": y, "x": x, "dx": dx, "dy": dy},
    )

    bottom_resistance = xr.DataArray(
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
        dims=("y", "x"),
        coords={"y": y, "x": x, "dx": dx, "dy": dy},
    )

    extra_storage_coefficient = xr.DataArray(
        np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]),
        dims=("y", "x"),
        coords={"y": y, "x": x, "dx": dx, "dy": dy},
    )

    svat = xr.DataArray(
        np.array(
            [
                [[0, 1, 0], [0, 0, 0], [0, 2, 0]],
                [[0, 3, 0], [0, 4, 0], [0, 0, 0]],
            ]
        ),
        dims=("subunit", "y", "x"),
        coords={"subunit": subunit, "y": y, "x": x, "dx": dx, "dy": dy},
    )
    # fmt: on
    index = (svat != 0).values.ravel()

    infiltration = Infiltration(
        infiltration_capacity,
        downward_resistance,
        upward_resistance,
        bottom_resistance,
        extra_storage_coefficient,
    )

    return infiltration, svat, index


@given(
    floats(
        Infiltration._metadata_dict["infiltration_capacity"].min_value,
        Infiltration._metadata_dict["infiltration_capacity"].max_value,
    ),
    floats(
        Infiltration._metadata_dict["downward_resistance"].min_value,
        Infiltration._metadata_dict["downward_resistance"].max_value,
    ),
    floats(
        Infiltration._metadata_dict["upward_resistance"].min_value,
        Infiltration._metadata_dict["upward_resistance"].max_value,
    ),
    floats(
        Infiltration._metadata_dict["bottom_resistance"].min_value,
        Infiltration._metadata_dict["bottom_resistance"].max_value,
    ),
    floats(
        Infiltration._metadata_dict["extra_storage_coefficient"].min_value,
        Infiltration._metadata_dict["extra_storage_coefficient"].max_value,
    ),
)
@settings(deadline=None)
def test_write(
    fixed_format_parser,
    infiltration_capacity,
    downward_resistance,
    upward_resistance,
    bottom_resistance,
    extra_storage_coefficient,
):
    infiltration = Infiltration(
        xr.DataArray(infiltration_capacity).expand_dims(subunit=[0]),
        xr.DataArray(downward_resistance),
        xr.DataArray(upward_resistance),
        xr.DataArray(bottom_resistance),
        xr.DataArray(extra_storage_coefficient),
    )

    index = np.array([True])
    svat = xr.DataArray(1).expand_dims(subunit=[0])

    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        infiltration.write(output_dir, index, svat)

        results = fixed_format_parser(
            output_dir / Infiltration._file_name, Infiltration._metadata_dict
        )

    assert_almost_equal(
        results["infiltration_capacity"],
        float(
            format_fixed_width(
                infiltration_capacity,
                Infiltration._metadata_dict["infiltration_capacity"],
            )
        ),
    )
    assert_almost_equal(
        results["downward_resistance"],
        float(
            format_fixed_width(
                downward_resistance,
                Infiltration._metadata_dict["downward_resistance"],
            )
        ),
    )

    assert_almost_equal(
        results["upward_resistance"],
        float(
            format_fixed_width(
                upward_resistance,
                Infiltration._metadata_dict["upward_resistance"],
            )
        ),
    )

    assert_almost_equal(
        results["bottom_resistance"],
        float(
            format_fixed_width(
                bottom_resistance,
                Infiltration._metadata_dict["bottom_resistance"],
            )
        ),
    )

    assert_almost_equal(
        results["extra_storage_coefficient"],
        float(
            format_fixed_width(
                extra_storage_coefficient,
                Infiltration._metadata_dict["extra_storage_coefficient"],
            )
        ),
    )


def test_simple_model(fixed_format_parser):
    x = [1.0, 2.0, 3.0]
    y = [1.0, 2.0, 3.0]
    subunit = [0, 1]
    dx = 1.0
    dy = 1.0
    # fmt: off
    infiltration, svat, index = setup_infiltration_package(subunit, y, x, dy, dx)

    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        infiltration.write(output_dir, index, svat)

        results = fixed_format_parser(
            output_dir / Infiltration._file_name, Infiltration._metadata_dict
        )

    assert_equal(results["svat"], np.array([1, 2, 3, 4]))
    assert_almost_equal(
        results["infiltration_capacity"], np.array([0.5, 1.0, 0.5, 1.0])
    )
    assert_almost_equal(results["downward_resistance"], np.array([2.0, 8.0, 2.0, 5.0]))
    assert_almost_equal(results["upward_resistance"], np.array([2.0, 8.0, 2.0, 5.0]))
    assert_almost_equal(results["bottom_resistance"], np.array([2.0, 8.0, 2.0, 5.0]))
    assert_almost_equal(
        results["extra_storage_coefficient"], np.array([0.2, 0.8, 0.2, 0.5])
    )


def test_regrid():
    x = [1.0, 2.0, 3.0]
    y = [3.0, 2.0, 1.0]
    subunit = [0, 1]
    dx = 1.0
    dy = 1.0

    infiltration, _, _ = setup_infiltration_package(subunit, y, x, dy, dx)

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

    regrid_context = RegridderWeightsCache(infiltration.dataset["infiltration_capacity"], new_grid)
    regridded = infiltration.regrid_like(new_grid, regrid_context )
    assert_almost_equal(regridded.dataset.coords["x"].values, x)
    assert_almost_equal(regridded.dataset.coords["y"].values, y)
