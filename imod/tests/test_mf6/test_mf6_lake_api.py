import textwrap

import numpy as np
import xarray as xr

from imod.mf6.lake_package import lak, lake_api


def create_gridcovering_array(idomain, lake_cells, fillvalue, dtype):
    """
    creates an array similar in dimensions/coords to idomain, but with value NaN (orr the missing value for integers)
    everywhere, except in the cells contained in list "lake_cells". In those cells, the output array has value fillvalue.
    """
    result = xr.full_like(idomain, fill_value=np.nan, dtype=dtype)
    for cell in lake_cells:
        result[{"layer":cell[0] -1, "y":cell[1]-1,"x":cell[2]-1}]= fillvalue
    return result


def create_lakelake(idomain, starting_stage, boundname, lake_cells):
    connection_type = create_gridcovering_array(
        idomain, lake_cells, lak.connection_types["horizontal"], np.float64
    )
    bed_leak = create_gridcovering_array(idomain, lake_cells, 0.2, np.float64)
    top_elevation = create_gridcovering_array(idomain, lake_cells, 0.3, np.float64)
    bot_elevation = create_gridcovering_array(idomain, lake_cells, 0.4, np.float64)
    connection_length = create_gridcovering_array(idomain, lake_cells, 0.5, np.float64)
    connection_width = create_gridcovering_array(idomain, lake_cells, 0.6, np.float64)
    result = lake_api.LakeLake(
        starting_stage,
        boundname,
        connection_type,
        bed_leak,
        top_elevation,
        bot_elevation,
        connection_length,
        connection_width,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    return result


def test_lake_api(basic_dis):
    idomain, _, _ = basic_dis

    outlet1 = lake_api.OutletManning(
        1, "Naardermeer", "IJsselmeer", 23.0, 24.0, 25.0, 26.0
    )
    outlet2 = lake_api.OutletManning(
        2, "IJsselmeer", "Naardermeer", 27.0, 28.0, 29.0, 30.0
    )

    lake1 = create_lakelake(
        idomain, 11.0, "Naardermeer", [(1, 2, 2), (1, 2, 3), (1, 3, 3)]
    )
    lake2 = create_lakelake(
        idomain, 15.0, "IJsselmeer", [(1, 5, 5), (1, 5, 6), (1, 6, 6)]
    )
    lake_package = lake_api.from_lakes_and_outlets([lake1, lake2], [outlet1, outlet2])
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


def test_helper_function_get_1d_array(basic_dis):
    idomain, _, _ = basic_dis
    lake1 = create_lakelake(
        idomain, 11.0, "Naardermeer", [(1, 2, 2), (1, 2, 3), (1, 3, 3)]
    )
    row, col, layer, values = lake1.get_1d_array(lake1.bottom_elevation)
    assert np.array_equal(row, np.array([2, 3, 3]))
    assert np.array_equal(col, np.array([2, 2, 3]))
    assert np.array_equal(layer, np.array([1, 1, 1]))
    assert np.array_equal(values, np.array([0.4, 0.4, 0.4]))


def test_helper_function_nparray_to_xarray_1d():
    result = lake_api.nparray_to_xarray_1d([23, 24, 25, 26], "velocity")
    assert result.dims == ("velocity",)
    assert result.values[0] == 23
    assert result.values[1] == 24
    assert result.values[2] == 25
    assert result.values[3] == 26


def test_helper_function_outlet_list_prop_to_xarray_1d():
    outlet1 = lake_api.OutletManning(
        1, "Naardermeer", "IJsselmeer", 23.0, 24.0, 25.0, 26.0
    )
    outlet2 = lake_api.OutletManning(
        2, "IJsselmeer", "Naardermeer", 27.0, 28.0, 29.0, 30.0
    )
    result = lake_api.outlet_list_prop_to_xarray_1d(
        [outlet1, outlet2], "lake_in", "lakeIn"
    )
    assert result.values[0] == "Naardermeer"
    assert result.values[1] == "IJsselmeer"


def test_helper_function_lake_list_connection_prop_to_xarray_1d(basic_dis):
    idomain, _, _ = basic_dis
    lake1 = create_lakelake(
        idomain, 11.0, "Naardermeer", [(1, 2, 2), (1, 2, 3), (1, 3, 3)]
    )
    lake2 = create_lakelake(
        idomain, 15.0, "IJsselmeer", [(1, 5, 5), (1, 5, 6), (1, 6, 6)]
    )
    result = lake_api.lake_list_connection_prop_to_xarray_1d(
        [lake1, lake2], "bottom_elevation"
    )
    assert result.dims == ("connection_nr",)
    assert result.values.min() == result.values.max() == 0.4


def test_helper_function_lake_list_lake_prop_to_xarray_1d(basic_dis):
    idomain, _, _ = basic_dis
    lake1 = create_lakelake(
        idomain, 11.0, "Naardermeer", [(1, 2, 2), (1, 2, 3), (1, 3, 3)]
    )
    lake2 = create_lakelake(
        idomain, 15.0, "IJsselmeer", [(1, 5, 5), (1, 5, 6), (1, 6, 6)]
    )
    result = lake_api.lake_list_lake_prop_to_xarray_1d([lake1, lake2], "starting_stage")
    assert result.values[0] == 11.0
    assert result.values[1] == 15.0
    assert result.dims == ("lake_nr",)
