import tempfile
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_equal

from imod import msw


def test_evapotranspiration_mapping_simple(fixed_format_parser):
    x_meteo = [-0.5, 1.5, 3.5]
    y_meteo = [0.5, 2.5, 4.5]
    subunit_meteo = [0, 1]
    dx_meteo = 2.0
    dy_meteo = 2.0
    # fmt: off
    evapotranspiration = xr.DataArray(
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
    x_svat = [1.0, 2.0, 3.0]
    y_svat = [1.0, 2.0, 3.0]
    subunit_svat = [0, 1]
    dx_svat = 1.0
    dy_svat = 1.0

    # fmt: off
    svat = xr.DataArray(
        np.array(
            [
                [[0, 1, 0],
                 [0, 0, 0],
                 [0, 2, 0]],

                [[0, 3, 0],
                 [4, 5, 6],
                 [0, 0, 0]],
            ]
        ),
        dims=("subunit", "y", "x"),
        coords={"subunit": subunit_svat, "y": y_svat, "x": x_svat, "dx": dx_svat, "dy": dy_svat}
    )
    # fmt: on
    index = (svat != 0).values.ravel()

    evapotranspiration_mapping = msw.EvapotranspirationMapping(evapotranspiration)

    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        evapotranspiration_mapping.write(output_dir, index, svat, None, None)

        results = fixed_format_parser(
            output_dir / msw.EvapotranspirationMapping._file_name,
            msw.EvapotranspirationMapping._metadata_dict,
        )

    assert_equal(results["svat"], np.array([1, 2, 3, 4, 5, 6]))
    assert_equal(results["row"], np.array([1, 2, 1, 2, 2, 2]))
    assert_equal(results["column"], np.array([2, 2, 2, 2, 2, 3]))


def test_evapotranspiration_mapping_negative_dx(fixed_format_parser):
    x_meteo = [3.5, 1.5, -0.5]
    y_meteo = [0.5, 2.5, 4.5]
    subunit_meteo = [0, 1]
    dx_meteo = -2.0
    dy_meteo = 2.0
    # fmt: off
    evapotranspiration = xr.DataArray(
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

    x_svat = [1.0, 2.0, 3.0]
    y_svat = [1.0, 2.0, 3.0]
    subunit_svat = [0, 1]
    dx_svat = 1.0
    dy_svat = 1.0
    # fmt: off
    svat = xr.DataArray(
        np.array(
            [
                [[0, 1, 0],
                 [0, 0, 0],
                 [0, 2, 0]],

                [[0, 3, 0],
                 [4, 5, 6],
                 [0, 0, 0]],
            ]
        ),
        dims=("subunit", "y", "x"),
        coords={"subunit": subunit_svat, "y": y_svat, "x": x_svat, "dx": dx_svat, "dy": dy_svat}
    )
    # fmt: on
    index = (svat != 0).values.ravel()

    evapotranspiration_mapping = msw.EvapotranspirationMapping(evapotranspiration)

    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        evapotranspiration_mapping.write(output_dir, index, svat, None, None)

        results = fixed_format_parser(
            output_dir / msw.EvapotranspirationMapping._file_name,
            msw.EvapotranspirationMapping._metadata_dict,
        )

    assert_equal(results["svat"], np.array([1, 2, 3, 4, 5, 6]))
    assert_equal(results["row"], np.array([1, 2, 1, 2, 2, 2]))
    assert_equal(results["column"], np.array([2, 2, 2, 2, 2, 1]))


def test_evapotranspiration_mapping_out_of_bound():
    x_meteo = [-2.5, -0.5, 1.5]
    y_meteo = [0.5, 2.5, 4.5]
    subunit_meteo = [0, 1]
    dx_meteo = 2.0
    dy_meteo = 2.0

    # fmt: off
    evapotranspiration = xr.DataArray(
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

    x_svat = [1.0, 2.0, 3.0]
    y_svat = [1.0, 2.0, 3.0]
    subunit_svat = [0, 1]
    dx_svat = 1.0
    dy_svat = 1.0
    # fmt: off
    svat = xr.DataArray(
        np.array(
            [
                [[0, 1, 0],
                 [0, 0, 0],
                 [0, 2, 0]],

                [[0, 3, 0],
                 [4, 5, 6],
                 [0, 0, 0]],
            ]
        ),
        dims=("subunit", "y", "x"),
        coords={"subunit": subunit_svat, "y": y_svat, "x": x_svat, "dx": dx_svat, "dy": dy_svat}
    )
    # fmt: on
    index = (svat != 0).values.ravel()

    evapotranspiration_mapping = msw.EvapotranspirationMapping(evapotranspiration)

    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        # The grid is out of bounds, which is why we expect a ValueError to be raisen
        with pytest.raises(ValueError):
            evapotranspiration_mapping.write(output_dir, index, svat, None, None)
