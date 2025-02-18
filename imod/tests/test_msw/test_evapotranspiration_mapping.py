import re
import tempfile
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_equal
from pytest_cases import parametrize_with_cases

from imod import msw


@pytest.fixture(scope="function")
def svat_index() -> xr.DataArray:
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
    return svat, index


def create_meteo_grid(x, y, subunit, dx, dy):
    # fmt: off
    return xr.DataArray(
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
        coords={"subunit": subunit, "y": y, "x": x, "dx": dx, "dy": dy}
    )
    # fmt: on


def test_evapotranspiration_mapping_simple(fixed_format_parser, svat_index):
    svat, index = svat_index

    x = [-0.5, 1.5, 3.5]
    y = [0.5, 2.5, 4.5]
    subunit = [0, 1]
    dx = 2.0
    dy = 2.0
    evapotranspiration = create_meteo_grid(x, y, subunit, dx, dy)
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


def test_evapotranspiration_mapping_negative_dx(fixed_format_parser, svat_index):
    svat, index = svat_index

    x = [3.5, 1.5, -0.5]
    y = [0.5, 2.5, 4.5]
    subunit = [0, 1]
    dx = -2.0
    dy = 2.0

    evapotranspiration = create_meteo_grid(x, y, subunit, dx, dy)
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


def test_evapotranspiration_mapping_out_of_bound(svat_index):
    svat, index = svat_index

    x = [-2.5, -0.5, 1.5]
    y = [0.5, 2.5, 4.5]
    subunit = [0, 1]
    dx = 2.0
    dy = 2.0

    evapotranspiration = create_meteo_grid(x, y, subunit, dx, dy)
    evapotranspiration_mapping = msw.EvapotranspirationMapping(evapotranspiration)

    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        # The grid is out of bounds, which is why we expect a ValueError to be raisen
        with pytest.raises(ValueError):
            evapotranspiration_mapping.write(output_dir, index, svat, None, None)


def setup_meteo_grid(datadir):
    """Setup precipitation grid and write mete_grid.inp"""
    # Arrange
    x = [-0.5, 1.5, 3.5]
    y = [4.5, 2.5, 0.5]
    subunit = [0, 1]
    dx = 2.0
    dy = -2.0

    time = [np.datetime64(t) for t in ["2001-01-01", "2001-01-02", "2001-01-03"]]
    time_da = xr.DataArray([1.0, 1.0, 1.0], coords={"time": time})

    precipitation = create_meteo_grid(x, y, subunit, dx, dy).isel(subunit=0, drop=True)
    precipitation_times = time_da * precipitation
    mete_grid = msw.MeteoGrid(precipitation_times, precipitation_times)
    mete_grid.write(datadir)
    return precipitation


class MeteGridCases:
    """
    Cases return strings to replace in mete_grid.inp, to set paths to floats.
    """

    def case_all_paths(self):
        return r"nothing_to_replace"

    def case_some_paths(self):
        return r'"meteo_grids\\(\w+)_20010101000000.asc"'

    def case_no_paths(self):
        return r'"meteo_grids\\(\w+)_([0-9]+).asc"'


@parametrize_with_cases("replace_string", cases=MeteGridCases)
def test_evapotranspiration_from_imod5(tmpdir_factory, replace_string, request):
    datadir = Path(tmpdir_factory.mktemp("evapotranspiration_mapping"))
    evapotranspiration = setup_meteo_grid(datadir)

    paths = [["foo"], [datadir / "mete_grid.inp"], ["bar"]]
    imod5_data = {"extra": {"paths": paths}}

    # Replace text in existing mete_grid.inp
    with open(datadir / "mete_grid.inp", "r") as f:
        text = f.read()
    text_replaced = re.sub(replace_string, '"0.0"', text)
    with open(datadir / "mete_grid.inp", "w") as f:
        f.write(text_replaced)

    # Act
    if request.node.callspec.id == "no_paths":
        with pytest.raises(ValueError):
            msw.EvapotranspirationMapping.from_imod5_data(imod5_data)

    else:
        evapotranspiration_mapping = msw.EvapotranspirationMapping.from_imod5_data(
            imod5_data
        )
        actual = evapotranspiration_mapping.meteo

        # Assert
        assert len(actual.coords["time"]) == 1
        xr.testing.assert_equal(evapotranspiration, actual.isel(time=0, drop=True))
