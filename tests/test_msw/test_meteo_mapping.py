import tempfile
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from numpy import nan
from numpy.testing import assert_equal

from imod import msw


def test_precipitation_mapping_simple(fixed_format_parser):

    x_svat = [1.0, 2.0, 3.0]
    y_svat = [1.0, 2.0, 3.0]
    subunit_svat = [0, 1]
    dx_svat = 1.0
    dy_svat = 1.0
    # fmt: off
    area = xr.DataArray(
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
        coords={"subunit": subunit_svat, "y": y_svat, "x": x_svat, "dx": dx_svat, "dy": dy_svat}
    )

    active = xr.DataArray(
        np.array(
            [[False, True, False],
             [False, True, False],
             [False, True, False]]),
        dims=("y", "x"),
        coords={"y": y_svat, "x": x_svat, "dx": dx_svat, "dy": dy_svat}
    )
    # fmt: on

    x_meteo = [-1.0, 1.0, 3.0]
    y_meteo = [1.0, 3.0, 5.0]
    subunit_meteo = [0, 1]
    dx_meteo = 2.0
    dy_meteo = 2.0
    # fmt: off
    precipitation = xr.DataArray(
        np.array(
            [
                [[1.0, 1.0, 1.0],
                 [1.0, 1.0, 1.0],
                 [1.0, 1.0, 1.0]],

                [[1.0, 1.0, 1.0],
                 [1.0, 1.0, 1.0],
                 [1.0, 1.0, 1.0]],
            ]
        ),
        dims=("subunit", "y", "x"),
        coords={"subunit": subunit_meteo, "y": y_meteo, "x": x_meteo, "dx": dx_meteo, "dy": dy_meteo}
    )
    # fmt: on

    precipitation_mapping = msw.PrecipitationMapping(area, active, precipitation)

    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        precipitation_mapping.write(output_dir)

        results = fixed_format_parser(
            output_dir / msw.PrecipitationMapping._file_name,
            msw.PrecipitationMapping._metadata_dict,
        )

    assert_equal(results["svat"], np.array([1, 2, 3, 4]))
    assert_equal(results["row"], np.array([1, 2, 1, 1]))
    assert_equal(results["column"], np.array([2, 2, 2, 2]))


def test_precipitation_mapping_negative_dy(fixed_format_parser):

    x_svat = [1.0, 2.0, 3.0]
    y_svat = [1.0, 2.0, 3.0]
    subunit_svat = [0, 1]
    dx_svat = 1.0
    dy_svat = 1.0
    # fmt: off
    area = xr.DataArray(
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
        coords={"subunit": subunit_svat, "y": y_svat, "x": x_svat, "dx": dx_svat, "dy": dy_svat}
    )

    active = xr.DataArray(
        np.array(
            [[False, True, False],
             [False, True, False],
             [False, True, False]]),
        dims=("y", "x"),
        coords={"y": y_svat, "x": x_svat, "dx": dx_svat, "dy": dy_svat}
    )
    # fmt: on

    x_meteo = [-1.0, 1.0, 3.0]
    y_meteo = [5.0, 3.0, 1.0]
    subunit_meteo = [0, 1]
    dx_meteo = 2.0
    dy_meteo = -2.0
    # fmt: off
    precipitation = xr.DataArray(
        np.array(
            [
                [[1.0, 1.0, 1.0],
                 [1.0, 1.0, 1.0],
                 [1.0, 1.0, 1.0]],

                [[1.0, 1.0, 1.0],
                 [1.0, 1.0, 1.0],
                 [1.0, 1.0, 1.0]],
            ]
        ),
        dims=("subunit", "y", "x"),
        coords={"subunit": subunit_meteo, "y": y_meteo, "x": x_meteo, "dx": dx_meteo, "dy": dy_meteo}
    )
    # fmt: on

    precipitation_mapping = msw.PrecipitationMapping(area, active, precipitation)

    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        precipitation_mapping.write(output_dir)

        results = fixed_format_parser(
            output_dir / msw.PrecipitationMapping._file_name,
            msw.PrecipitationMapping._metadata_dict,
        )

    assert_equal(results["svat"], np.array([1, 2, 3, 4]))
    assert_equal(results["row"], np.array([3, 2, 3, 3]))
    assert_equal(results["column"], np.array([2, 2, 2, 2]))


def test_precipitation_mapping_out_of_bound():

    x_svat = [1.0, 2.0, 3.0]
    y_svat = [1.0, 2.0, 3.0]
    subunit_svat = [0, 1]
    dx_svat = 1.0
    dy_svat = 1.0
    # fmt: off
    area = xr.DataArray(
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
        coords={"subunit": subunit_svat, "y": y_svat, "x": x_svat, "dx": dx_svat, "dy": dy_svat}
    )

    active = xr.DataArray(
        np.array(
            [[False, True, False],
             [False, True, False],
             [False, True, False]]),
        dims=("y", "x"),
        coords={"y": y_svat, "x": x_svat, "dx": dx_svat, "dy": dy_svat}
    )
    # fmt: on

    x_meteo = [-1.0, 1.0, 3.0]
    y_meteo = [3.0, 5.0, 7.0]
    subunit_meteo = [0, 1]
    dx_meteo = 2.0
    dy_meteo = 2.0
    # fmt: off
    precipitation = xr.DataArray(
        np.array(
            [
                [[1.0, 1.0, 1.0],
                 [1.0, 1.0, 1.0],
                 [1.0, 1.0, 1.0]],

                [[1.0, 1.0, 1.0],
                 [1.0, 1.0, 1.0],
                 [1.0, 1.0, 1.0]],
            ]
        ),
        dims=("subunit", "y", "x"),
        coords={"subunit": subunit_meteo, "y": y_meteo, "x": x_meteo, "dx": dx_meteo, "dy": dy_meteo}
    )
    # fmt: on

    precipitation_mapping = msw.PrecipitationMapping(area, active, precipitation)

    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        # The grid is out of bounds, which is why we expect a ValueError to be raisen
        with pytest.raises(ValueError):
            precipitation_mapping.write(output_dir)
