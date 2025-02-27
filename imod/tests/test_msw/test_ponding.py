import tempfile
from pathlib import Path

import numpy as np
import xarray as xr
from numpy import nan
from numpy.testing import assert_almost_equal, assert_equal

from imod.msw import Ponding
from imod.typing import GridDataArray, GridDataDict
from imod.util.regrid import (
    RegridderWeightsCache,
)


def setup_ponding() -> tuple[GridDataDict, np.ndarray, GridDataArray]:
    x = [1.0, 2.0, 3.0]
    y = [3.0, 2.0, 1.0]
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

    data_ponding = {
        "ponding_depth": ponding_depth,
        "runoff_resistance": runoff_resistance,
        "runon_resistance": runoff_resistance,
    }
    return data_ponding, index, svat


def test_simple_model(fixed_format_parser):
    data_ponding, index, svat = setup_ponding()
    ponding = Ponding(**data_ponding)
    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        ponding.write(output_dir, index, svat, None, None)

        results = fixed_format_parser(
            output_dir / Ponding._file_name, Ponding._metadata_dict
        )

    assert_equal(results["svat"], np.array([1, 2, 3, 4]))
    assert_equal(results["swnr"], np.array([0, 0, 0, 0]))
    assert_almost_equal(results["ponding_depth"], np.array([0.5, 1.0, 0.5, 1.0]))
    assert_almost_equal(results["runoff_resistance"], np.array([0.5, 1.0, 0.5, 1.0]))
    assert_almost_equal(results["runoff_resistance"], np.array([0.5, 1.0, 0.5, 1.0]))


def test_regrid_ponding(simple_2d_grid_with_subunits):
    data_ponding, _, _ = setup_ponding()
    ponding = Ponding(**data_ponding)
    new_grid = simple_2d_grid_with_subunits

    regrid_context = RegridderWeightsCache()

    regridded_ponding = ponding.regrid_like(new_grid, regrid_context)

    assert np.all(regridded_ponding.dataset["x"].values == new_grid["x"].values)
    assert np.all(regridded_ponding.dataset["y"].values == new_grid["y"].values)


def test_from_imod5_data():
    data_ponding, _, _ = setup_ponding()
    expected_ponding = Ponding(**data_ponding)

    # Create cap data
    cap_data = {}
    mapping_ls = [
        ("rural_runoff_resistance", "runoff_resistance", 0),
        ("urban_runoff_resistance", "runoff_resistance", 1),
        ("rural_runon_resistance", "runon_resistance", 0),
        ("urban_runon_resistance", "runon_resistance", 1),
        ("rural_ponding_depth", "ponding_depth", 0),
        ("urban_ponding_depth", "ponding_depth", 1),
    ]
    for cap_key, pkg_key, subunit_nr in mapping_ls:
        cap_data[cap_key] = data_ponding[pkg_key].sel(subunit=subunit_nr, drop=True)

    imod5_data = {"cap": cap_data}

    actual_ponding = Ponding.from_imod5_data(imod5_data)

    xr.testing.assert_equal(expected_ponding.dataset, actual_ponding.dataset)
