import tempfile
from pathlib import Path

import numpy as np
import xarray as xr
from numpy import nan
from numpy.testing import assert_almost_equal, assert_equal

from imod import msw
from imod.mf6.mf6_wel_adapter import Mf6Wel
from imod.mf6.wel import derive_cellid_from_points


def test_simple_model(fixed_format_parser):
    x = [1.0, 2.0, 3.0]
    y = [1.0, 2.0, 3.0]
    subunit = [0, 1]
    dx = 1.0
    dy = 1.0
    # fmt: off
    max_abstraction_groundwater = xr.DataArray(
        np.array(
            [[nan, 100.0, nan],
             [nan, 200.0, nan],
             [nan, 300.0, nan]]),
        dims=("y", "x"),
        coords={"y": y, "x": x, "dx": dx, "dy": dy}
    )

    max_abstraction_surfacewater = xr.DataArray(
        np.array(
            [[nan, 100.0, nan],
             [nan, 200.0, nan],
             [nan, 300.0, nan]]),
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

    # Well
    well_layer = [3, 2, 1]
    well_y = y
    well_x = [2.0, 2.0, 2.0]
    well_rate = [-5.0] * 3
    cellids = derive_cellid_from_points(svat, well_x, well_y, well_layer)
    well = Mf6Wel(cellids, well_rate)

    sprinkling = msw.Sprinkling(
        max_abstraction_groundwater,
        max_abstraction_surfacewater,
    )

    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        sprinkling.write(output_dir, index, svat, None, well)

        results = fixed_format_parser(
            output_dir / msw.Sprinkling._file_name,
            msw.Sprinkling._metadata_dict,
        )

    assert_equal(results["svat"], np.array([1, 2, 3, 4]))
    assert_almost_equal(
        results["max_abstraction_groundwater"],
        np.array([100.0, 300.0, 100.0, 200.0]),
    )
    assert_almost_equal(
        results["max_abstraction_surfacewater"],
        np.array([100.0, 300.0, 100.0, 200.0]),
    )
    assert_equal(results["layer"], np.array([3, 1, 3, 2]))


def test_simple_model_1_subunit(fixed_format_parser):
    x = [1.0, 2.0, 3.0]
    y = [1.0, 2.0, 3.0]
    subunit = [0]
    dx = 1.0
    dy = 1.0
    # fmt: off
    max_abstraction_groundwater = xr.DataArray(
        np.array(
            [[nan, 100.0, nan],
             [nan, 200.0, nan],
             [nan, 300.0, nan]]),
        dims=("y", "x"),
        coords={"y": y, "x": x, "dx": dx, "dy": dy}
    )

    max_abstraction_surfacewater = xr.DataArray(
        np.array(
            [[nan, 100.0, nan],
             [nan, 200.0, nan],
             [nan, 300.0, nan]]),
        dims=("y", "x"),
        coords={"y": y, "x": x, "dx": dx, "dy": dy}
    )

    svat = xr.DataArray(
        np.array(
            [
                [[0, 1, 0],
                 [0, 0, 0],
                 [0, 2, 0]],
            ]
        ),
        dims=("subunit", "y", "x"),
        coords={"subunit": subunit, "y": y, "x": x, "dx": dx, "dy": dy}
    )
    # fmt: on
    index = (svat != 0).values.ravel()

    # Well
    well_layer = [3, 2]
    well_y = [1.0, 3.0]
    well_x = [2.0, 2.0]
    well_rate = [-5.0] * 2
    cellids = derive_cellid_from_points(svat, well_x, well_y, well_layer)
    well = Mf6Wel(cellids, well_rate)

    sprinkling = msw.Sprinkling(
        max_abstraction_groundwater,
        max_abstraction_surfacewater,
    )

    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        sprinkling.write(output_dir, index, svat, None, well)

        results = fixed_format_parser(
            output_dir / msw.Sprinkling._file_name,
            msw.Sprinkling._metadata_dict,
        )

    assert_equal(results["svat"], np.array([1, 2]))
    assert_almost_equal(
        results["max_abstraction_groundwater"],
        np.array([100.0, 300.0]),
    )
    assert_almost_equal(
        results["max_abstraction_surfacewater"],
        np.array([100.0, 300.0]),
    )
    assert_equal(results["layer"], np.array([3, 2]))
