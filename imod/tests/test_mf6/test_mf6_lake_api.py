import textwrap

import numpy as np
import xarray as xr

from imod.mf6.lake_package import lak, lake_api


def create_gridcovering_array(idomain, lake_cells, fillvalue, dtype):
    """
    creates an array similar in dimensions/coords to idomain, but with value NaN (orr the missing value for integers)
    everywhere, except in the cells contained in list "lake_cells". In those cells, the output array has value fillvalue.
    """
    result = xr.full_like(
        idomain, fill_value=lake_api.missing_values[np.dtype(dtype).name], dtype=dtype
    )
    for cell in lake_cells:
        result.values[cell[0], cell[1], cell[2]] = fillvalue
    return result


def create_lakelake(
    idomain, starting_stage, boundname, lake_cells, rainfall=None, inflow=None
):
    """
    creates a LakeLake object with a specified starting stage, location, and optionally time dependent rainfall and inflow
    """
    connectionType = create_gridcovering_array(
        idomain, lake_cells, lak.connection_types["HORIZONTAL"], np.int32
    )
    bed_leak = create_gridcovering_array(idomain, lake_cells, 0.2, np.float32)
    top_elevation = create_gridcovering_array(idomain, lake_cells, 0.3, np.float32)
    bot_elevation = create_gridcovering_array(idomain, lake_cells, 0.4, np.float32)
    connection_length = create_gridcovering_array(idomain, lake_cells, 0.5, np.float32)
    connection_width = create_gridcovering_array(idomain, lake_cells, 0.6, np.float32)
    result = lake_api.LakeLake(
        starting_stage,
        boundname,
        connectionType,
        bed_leak,
        top_elevation,
        bot_elevation,
        connection_length,
        connection_width,
        None,
        None,
        rainfall,
        None,
        None,
        inflow,
        None,
        None,
    )
    return result


def create_lakelake_transient(
    idomain, starting_stage, boundname, lake_cells, rainfall, inflow
):
    """
    helper function that creates a lake package with time-dependent data
    """
    connectionType = create_gridcovering_array(
        idomain, lake_cells, lak.connection_types["HORIZONTAL"], np.int32
    )
    bed_leak = create_gridcovering_array(idomain, lake_cells, 0.2, np.float32)
    top_elevation = create_gridcovering_array(idomain, lake_cells, 0.3, np.float32)
    bot_elevation = create_gridcovering_array(idomain, lake_cells, 0.4, np.float32)
    connection_length = create_gridcovering_array(idomain, lake_cells, 0.5, np.float32)
    connection_width = create_gridcovering_array(idomain, lake_cells, 0.6, np.float32)
    result = lake_api.LakeLake(
        starting_stage,
        boundname,
        connectionType,
        bed_leak,
        top_elevation,
        bot_elevation,
        connection_length,
        connection_width,
        None,
        None,
        rainfall,
        None,
        None,
        inflow,
        None,
        None,
    )
    return result


def test_lake_api(basic_dis):
    idomain, _, _ = basic_dis

    outlet1 = lake_api.OutletManning(
        1, "Naardermeer", "Ijsselmeer", 23.0, 24.0, 25.0, 26.0
    )
    outlet2 = lake_api.OutletManning(
        2, "Ijsselmeer", "Naardermeer", 27.0, 28.0, 29.0, 30.0
    )

    lake1 = create_lakelake(
        idomain, 11.0, "Naardermeer", [(1, 2, 2), (1, 2, 3), (1, 3, 3)]
    )
    lake2 = create_lakelake(
        idomain, 15.0, "Ijsselmeer", [(1, 5, 5), (1, 5, 6), (1, 6, 6)]
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
          2  15.0  3  Ijsselmeer
        end packagedata

        begin connectiondata
          1 1 1 2 2 HORIZONTAL  0.2 0.4  0.3  0.6 0.5
          1 2 1 2 3 HORIZONTAL  0.2 0.4  0.3  0.6 0.5
          1 3 1 3 3 HORIZONTAL  0.2 0.4  0.3  0.6 0.5
          2 1 1 5 5 HORIZONTAL  0.2 0.4  0.3  0.6 0.5
          2 2 1 5 6 HORIZONTAL  0.2 0.4  0.3  0.6 0.5
          2 3 1 6 6 HORIZONTAL  0.2 0.4  0.3  0.6 0.5
        end connectiondata

        begin outlets
          1 2 MANNING 23.0 25.0 24.0 26.0
          2 1 MANNING 27.0 29.0 28.0 30.0
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
    assert row == [2, 3, 3]
    assert col == [2, 2, 3]
    assert layer == [1, 1, 1]
    assert values[0] == values[1] == values[2] == np.float32(0.4)


def test_helper_function_list_1d_to_xarray_1d():
    result = lake_api.list_1d_to_xarray_1d([23, 24, 25, 26], "velocity")
    assert result.dims == ("velocity",)
    assert result.values[0] == 23
    assert result.values[1] == 24
    assert result.values[2] == 25
    assert result.values[3] == 26


def test_helper_function_outlet_list_prop_to_xarray_1d():
    outlet1 = lake_api.OutletManning(
        1, "Naardermeer", "Ijsselmeer", 23.0, 24.0, 25.0, 26.0
    )
    outlet2 = lake_api.OutletManning(
        2, "Ijsselmeer", "Naardermeer", 27.0, 28.0, 29.0, 30.0
    )
    result = lake_api.outlet_list_prop_to_xarray_1d(
        [outlet1, outlet2], "lake_in", "lakeIn"
    )
    assert result.values[0] == "Naardermeer"
    assert result.values[1] == "Ijsselmeer"


def test_helper_function_lake_list_connection_prop_to_xarray_1d(basic_dis):
    idomain, _, _ = basic_dis
    lake1 = create_lakelake(
        idomain, 11.0, "Naardermeer", [(1, 2, 2), (1, 2, 3), (1, 3, 3)]
    )
    lake2 = create_lakelake(
        idomain, 15.0, "Ijsselmeer", [(1, 5, 5), (1, 5, 6), (1, 6, 6)]
    )
    result = lake_api.lake_list_connection_prop_to_xarray_1d(
        [lake1, lake2], "bottom_elevation"
    )
    assert result.dims == ("connection_nr",)
    assert result.values.min() == result.values.max() == np.float32(0.4)


def test_helper_function_lake_list_lake_prop_to_xarray_1d(basic_dis):
    idomain, _, _ = basic_dis
    lake1 = create_lakelake(
        idomain, 11.0, "Naardermeer", [(1, 2, 2), (1, 2, 3), (1, 3, 3)]
    )
    lake2 = create_lakelake(
        idomain, 15.0, "Ijsselmeer", [(1, 5, 5), (1, 5, 6), (1, 6, 6)]
    )
    result = lake_api.lake_list_lake_prop_to_xarray_1d([lake1, lake2], "starting_stage")
    assert result.values[0] == 11.0
    assert result.values[1] == 15.0
    assert result.dims == ("lake_nr",)


def test_lake_rendering_transient(basic_dis):
    idomain, _, _ = basic_dis
    times_rainfall = [
        np.datetime64("2000-01-01"),
        np.datetime64("2000-03-01"),
        np.datetime64("2000-05-01"),
    ]
    rainfall = xr.DataArray(
        np.full((len(times_rainfall)), 5.0),
        coords={"time": times_rainfall},
        dims=["time"],
    )
    times_inflow = [np.datetime64("2000-02-01"), np.datetime64("2000-04-01")]
    inflow = xr.DataArray(
        np.full((len(times_inflow)), 4.0), coords={"time": times_inflow}, dims=["time"]
    )

    lake1 = create_lakelake(
        idomain,
        11.0,
        "Naardermeer",
        [(1, 2, 2), (1, 2, 3), (1, 3, 3)],
        rainfall,
        inflow,
    )
    lake2 = create_lakelake(idomain, 11.0, "Ijsselmeer", [(1, 4, 4)], rainfall, inflow)
    times_invert = [
        np.datetime64("2000-01-01"),
        np.datetime64("2000-03-01"),
        np.datetime64("2000-05-01"),
    ]
    invert = xr.DataArray(
        np.full((len(times_invert)), 3.0),
        coords={"time": times_rainfall},
        dims=["time"],
    )
    outlet1 = lake_api.OutletManning(
        1, "Naardermeer", "Ijsselmeer", invert, 24.0, 25.0, 26.0
    )
    lake_package = lake_api.from_lakes_and_outlets([lake1, lake2], [outlet1])
    global_times = np.array(
        [
            np.datetime64("1999-01-01"),
            np.datetime64("2000-01-01"),
            np.datetime64("2000-02-01"),
            np.datetime64("2000-03-01"),
            np.datetime64("2000-04-01"),
            np.datetime64("2000-05-01"),
        ]
    )
    actual = lake_package.render(None, None, global_times, False)
    expected = textwrap.dedent(
        """\
        begin options
        end options

        begin dimensions
          nlakes 2
          noutlets 1
          ntables 0
        end dimensions

        begin packagedata
          1  11.0  3  Naardermeer
          2  11.0  1  Ijsselmeer
        end packagedata

        begin connectiondata
          1 1 1 2 2 HORIZONTAL  0.2 0.4  0.3  0.6 0.5
          1 2 1 2 3 HORIZONTAL  0.2 0.4  0.3  0.6 0.5
          1 3 1 3 3 HORIZONTAL  0.2 0.4  0.3  0.6 0.5
          2 1 1 4 4 HORIZONTAL  0.2 0.4  0.3  0.6 0.5
        end connectiondata

        begin outlets
          1 2 MANNING 0.0 25.0 24.0 26.0
        end outlets

        begin period 2
          1  rainfall 5.0
          2  rainfall 5.0
          1  invert 3.0
        end period

        begin period 3
          1  inflow 4.0
          2  inflow 4.0
        end period

        begin period 4
          1  rainfall 5.0
          2  rainfall 5.0
          1  invert 3.0
        end period

        begin period 5
          1  inflow 4.0
          2  inflow 4.0
        end period

        begin period 6
          1  rainfall 5.0
          2  rainfall 5.0
          1  invert 3.0
        end period
        """
    )
    assert actual == expected


def test_lakelake_get_times(basic_dis):
    idomain, _, _ = basic_dis

    times_rainfall = [
        np.datetime64("2000-01-01"),
        np.datetime64("2000-03-01"),
        np.datetime64("2000-05-01"),
    ]
    rainfall = xr.DataArray(
        np.full((len(times_rainfall)), 5.0),
        coords={"time": times_rainfall},
        dims=["time"],
    )

    lake1 = create_lakelake(
        idomain, 11.0, "Naardermeer", [(1, 2, 2), (1, 2, 3), (1, 3, 3)], rainfall
    )
    returned_times = lake1.get_times("rainfall")

    assert returned_times == times_rainfall


def test_lakeapi_get_timeseries(basic_dis):
    idomain, _, _ = basic_dis

    times_rainfall = [
        np.datetime64("2000-01-01"),
        np.datetime64("2000-03-01"),
        np.datetime64("2000-05-01"),
    ]
    rainfall = xr.DataArray(
        np.full((len(times_rainfall)), 5.0),
        coords={"time": times_rainfall},
        dims=["time"],
    )
    times_inflow = [np.datetime64("2000-02-01"), np.datetime64("2000-04-01")]
    inflow = xr.DataArray(
        np.full((len(times_inflow)), 4.0), coords={"time": times_inflow}, dims=["time"]
    )

    lake1 = create_lakelake(
        idomain,
        11.0,
        "Naardermeer",
        [(1, 2, 2), (1, 2, 3), (1, 3, 3)],
        rainfall,
        inflow,
    )
    times = lake_api.collect_all_times([lake1], [])
    rainfall_ts = lake_api.create_timeseries([lake1], times, "rainfall")
    assert len(rainfall_ts.coords["time"].values) == 5
    assert len(rainfall_ts.coords["index"].values) == 1
    assert rainfall_ts.values[0] == 5
    assert np.isnan(rainfall_ts.values[1])
    assert rainfall_ts.values[2] == 5
    assert np.isnan(rainfall_ts.values[3])
    assert rainfall_ts.values[4] == 5
