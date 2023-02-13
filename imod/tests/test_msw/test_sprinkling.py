import tempfile
from pathlib import Path

import numpy as np
import xarray as xr
from numpy import nan
from numpy.testing import assert_almost_equal, assert_equal

from imod import mf6, msw


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
    well_row = [1, 2, 3]
    well_column = [2, 2, 2]
    well_rate = [-5.0] * 3
    well = mf6.WellDisStructured(
        layer=well_layer,
        row=well_row,
        column=well_column,
        rate=well_rate,
    )

    coupler_mapping = msw.Sprinkling(
        max_abstraction_groundwater,
        max_abstraction_surfacewater,
        well,
    )

    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        coupler_mapping.write(output_dir, index, svat)

        results = fixed_format_parser(
            output_dir / msw.Sprinkling._file_name,
            msw.Sprinkling._metadata_dict,
        )

    assert_equal(results["svat"], np.array([1, 2, 3, 4]))
    assert_almost_equal(
        results["max_abstraction_groundwater_m3_d"],
        np.array([100.0, 300.0, 100.0, 200.0]),
    )
    assert_almost_equal(
        results["max_abstraction_surfacewater_m3_d"],
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
    well_row = [1, 3]
    well_column = [2, 2]
    well_rate = [-5.0] * 2
    well = mf6.WellDisStructured(
        layer=well_layer,
        row=well_row,
        column=well_column,
        rate=well_rate,
    )

    coupler_mapping = msw.Sprinkling(
        max_abstraction_groundwater,
        max_abstraction_surfacewater,
        well,
    )

    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        coupler_mapping.write(output_dir, index, svat)

        results = fixed_format_parser(
            output_dir / msw.Sprinkling._file_name,
            msw.Sprinkling._metadata_dict,
        )

    assert_equal(results["svat"], np.array([1, 2]))
    assert_almost_equal(
        results["max_abstraction_groundwater_m3_d"],
        np.array([100.0, 300.0]),
    )
    assert_almost_equal(
        results["max_abstraction_surfacewater_m3_d"],
        np.array([100.0, 300.0]),
    )
    assert_equal(results["layer"], np.array([3, 2]))
