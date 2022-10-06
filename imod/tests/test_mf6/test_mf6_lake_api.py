import textwrap

import numpy as np
import pytest
import xarray as xr

import imod.mf6.lake_package.lake_api as lp


def create_lake_data(is_lake, starting_stage, name):
    HORIZONTAL = 0
    connection_type = xr.full_like(is_lake, HORIZONTAL, dtype=np.float64).where(is_lake)
    bed_leak = xr.full_like(is_lake, 0.2, dtype=np.float64).where(is_lake)
    top_elevation = xr.full_like(is_lake, 0.3, dtype=np.float64).where(is_lake)
    bot_elevation = xr.full_like(is_lake, 0.4, dtype=np.float64).where(is_lake)
    connection_length = xr.full_like(is_lake, 0.5, dtype=np.float64).where(is_lake)
    connection_width = xr.full_like(is_lake, 0.6, dtype=np.float64).where(is_lake)
    return lp.LakeData(
        starting_stage=starting_stage,
        boundname=name,
        connection_type=connection_type,
        bed_leak=bed_leak,
        top_elevation=top_elevation,
        bot_elevation=bot_elevation,
        connection_length=connection_length,
        connection_width=connection_width,
    )


@pytest.fixture(scope="function")
def naardermeer(basic_dis):
    idomain, _, _ = basic_dis
    is_lake = xr.full_like(idomain, False, dtype=bool)
    is_lake[0, 1, 1] = True
    is_lake[0, 1, 2] = True
    is_lake[0, 2, 2] = True
    return create_lake_data(is_lake, starting_stage=11.0, name="Naardermeer")


@pytest.fixture(scope="function")
def ijsselmeer(basic_dis):
    idomain, _, _ = basic_dis
    is_lake = xr.full_like(idomain, False, dtype=bool)
    is_lake[0, 4, 4] = True
    is_lake[0, 4, 5] = True
    is_lake[0, 5, 5] = True
    return create_lake_data(is_lake, starting_stage=15.0, name="IJsselmeer")


def test_lake_api(naardermeer, ijsselmeer):
    outlet1 = lp.OutletManning(1, "Naardermeer", "IJsselmeer", 23.0, 24.0, 25.0, 26.0)
    outlet2 = lp.OutletManning(2, "IJsselmeer", "Naardermeer", 27.0, 28.0, 29.0, 30.0)

    lake_package = lp.from_lakes_and_outlets(
        [naardermeer, ijsselmeer], [outlet1, outlet2]
    )
    actual = lake_package.render(None, None, None, False)
    expected = textwrap.dedent(
        """\
        begin options
        end options

        begin dimensions
          nlakes 2
          noutlets 2
          ntables 0
        end dimensions

        begin packagedata
          1  11.0  3  Naardermeer
          2  15.0  3  IJsselmeer
        end packagedata

        begin connectiondata
          1 1 1 2 2 horizontal  0.2 0.4  0.3  0.6 0.5
          1 2 1 2 3 horizontal  0.2 0.4  0.3  0.6 0.5
          1 3 1 3 3 horizontal  0.2 0.4  0.3  0.6 0.5
          2 1 1 5 5 horizontal  0.2 0.4  0.3  0.6 0.5
          2 2 1 5 6 horizontal  0.2 0.4  0.3  0.6 0.5
          2 3 1 6 6 horizontal  0.2 0.4  0.3  0.6 0.5
        end connectiondata

        begin outlets
          1 2 manning 23.0 25.0 24.0 26.0
          2 1 manning 27.0 29.0 28.0 30.0
        end outlets
        """
    )

    assert actual == expected


def test_helper_function_get_1d_array(naardermeer):
    row, col, layer, values = naardermeer.get_1d_array(naardermeer.bottom_elevation)
    assert np.array_equal(row, np.array([2, 3, 3]))
    assert np.array_equal(col, np.array([2, 2, 3]))
    assert np.array_equal(layer, np.array([1, 1, 1]))
    assert np.array_equal(values, np.array([0.4, 0.4, 0.4]))


def test_helper_function_nparray_to_xarray_1d():
    result = lp.nparray_to_xarray_1d([23, 24, 25, 26], "velocity")
    assert result.dims == ("velocity",)
    assert result.values[0] == 23
    assert result.values[1] == 24
    assert result.values[2] == 25
    assert result.values[3] == 26


def test_helper_function_outlet_list_prop_to_xarray_1d():
    outlet1 = lp.OutletManning(1, "Naardermeer", "IJsselmeer", 23.0, 24.0, 25.0, 26.0)
    outlet2 = lp.OutletManning(2, "IJsselmeer", "Naardermeer", 27.0, 28.0, 29.0, 30.0)
    result = lp.outlet_list_prop_to_xarray_1d([outlet1, outlet2], "lake_in", "lakeIn")
    assert result.values[0] == "Naardermeer"
    assert result.values[1] == "IJsselmeer"


def test_helper_function_lake_list_connection_prop_to_xarray_1d(
    naardermeer, ijsselmeer
):
    result = lp.lake_list_connection_prop_to_xarray_1d(
        [naardermeer, ijsselmeer], "bottom_elevation"
    )
    assert result.dims == ("connection_nr",)
    assert result.values.min() == result.values.max() == 0.4


def test_helper_function_lake_list_lake_prop_to_xarray_1d(naardermeer, ijsselmeer):
    result = lp.lake_list_lake_prop_to_xarray_1d(
        [naardermeer, ijsselmeer], "starting_stage"
    )
    assert result.values[0] == 11.0
    assert result.values[1] == 15.0
    assert result.dims == ("lake_nr",)
