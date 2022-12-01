import textwrap

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from imod.mf6.lak import Lake, LakeData, OutletManning  # , OutletSpecified, OutletWeir


def create_lake_data(is_lake, starting_stage, name, rainfall=None, inflow=None):
    HORIZONTAL = 0
    connection_type = xr.full_like(is_lake, HORIZONTAL, dtype=np.float64).where(is_lake)
    bed_leak = xr.full_like(is_lake, 0.2, dtype=np.float64).where(is_lake)
    top_elevation = xr.full_like(is_lake, 0.3, dtype=np.float64).where(is_lake)
    bot_elevation = xr.full_like(is_lake, 0.4, dtype=np.float64).where(is_lake)
    connection_length = xr.full_like(is_lake, 0.5, dtype=np.float64).where(is_lake)
    connection_width = xr.full_like(is_lake, 0.6, dtype=np.float64).where(is_lake)
    return LakeData(
        starting_stage=starting_stage,
        boundname=name,
        connection_type=connection_type,
        bed_leak=bed_leak,
        top_elevation=top_elevation,
        bot_elevation=bot_elevation,
        connection_length=connection_length,
        connection_width=connection_width,
        rainfall=rainfall,
        inflow=inflow,
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
    return create_lake_data(
        is_lake,
        starting_stage=15.0,
        name="IJsselmeer",
    )


@pytest.fixture(scope="function")
def lake_package(naardermeer, ijsselmeer):
    outlet1 = OutletManning(1, "Naardermeer", "IJsselmeer", 23.0, 24.0, 25.0, 26.0)
    outlet2 = OutletManning(2, "IJsselmeer", "Naardermeer", 27.0, 28.0, 29.0, 30.0)
    return Lake.from_lakes_and_outlets([naardermeer, ijsselmeer], [outlet1, outlet2])


def test_alternative_constructor(naardermeer, ijsselmeer):
    outlet1 = OutletManning(1, "Naardermeer", "IJsselmeer", 23.0, 24.0, 25.0, 26.0)
    outlet2 = OutletManning(2, "IJsselmeer", "Naardermeer", 27.0, 28.0, 29.0, 30.0)
    actual = Lake.from_lakes_and_outlets([naardermeer, ijsselmeer], [outlet1, outlet2])
    _ = actual.render(None, None, None, False)
    assert isinstance(actual, Lake)


def write_and_read(package, path, filename, globaltimes=None) -> str:
    package.write(path, filename, globaltimes, False)
    with open(path / f"{filename}.lak") as f:
        actual = f.read()
    return actual


def test_lake_render(lake_package):
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
          1 11.0 3 Naardermeer
          2 15.0 3 IJsselmeer
        end packagedata
        """
    )
    assert actual == expected


def test_lake_connection_dataframe(lake_package):
    df = lake_package._connection_dataframe()
    assert isinstance(df, pd.DataFrame)
    actual = list(df.columns)
    expected = [
        "connection_lake_number",
        "iconn",
        "layer",
        "y",
        "x",
        "connection_type",
        "connection_bed_leak",
        "connection_bottom_elevation",
        "connection_top_elevation",
        "connection_width",
        "connection_length",
    ]
    assert actual == expected


def test_lake_outlet_dataframe(lake_package):
    df = lake_package._outlet_dataframe()
    assert isinstance(df, pd.DataFrame)
    actual = list(df.columns)
    expected = [
        "outlet_lakein",
        "outlet_lakeout",
        "outlet_couttype",
        "outlet_invert",
        "outlet_roughness",
        "outlet_width",
        "outlet_slope",
    ]
    assert actual == expected


def test_lake_write(tmp_path, naardermeer, ijsselmeer):
    outlet1 = OutletManning(1, "Naardermeer", "IJsselmeer", 23.0, 24.0, 25.0, 26.0)
    outlet2 = OutletManning(2, "IJsselmeer", "Naardermeer", 27.0, 28.0, 29.0, 30.0)

    lake_package = Lake.from_lakes_and_outlets(
        [naardermeer, ijsselmeer], [outlet1, outlet2]
    )
    actual = write_and_read(lake_package, tmp_path, "lake-test")
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
          1 11.0 3 Naardermeer
          2 15.0 3 IJsselmeer
        end packagedata

        begin connectiondata
        1 1 0 1 1 horizontal 0.2 0.4 0.3 0.6 0.5
        1 2 0 1 2 horizontal 0.2 0.4 0.3 0.6 0.5
        1 3 0 2 2 horizontal 0.2 0.4 0.3 0.6 0.5
        2 1 0 4 4 horizontal 0.2 0.4 0.3 0.6 0.5
        2 2 0 4 5 horizontal 0.2 0.4 0.3 0.6 0.5
        2 3 0 5 5 horizontal 0.2 0.4 0.3 0.6 0.5
        end connectiondata

        begin outlets
        1 2 manning 23.0 25.0 24.0 26.0
        2 1 manning 27.0 29.0 28.0 30.0
        end outlets
        """
    )
    assert actual == expected


def test_lake_write_disv_three_lakes(basic_dis, tmp_path):
    """
    Create 6 connections.

    We assume an unstructured grid, so there are two indexes per cell (index
    and layer) instead of the usual three (row, col, layer).

    connection_number  lake_nr cell_id connection_type bed_leak bottom_elevation top_elevation connection_width connection_length
    1                        1      3,1     vertical        0.2        -1              0             0.1             0.2
    2                        1      4,1     vertical        0.3        -2              0.1           0.2             0.3
    3                        1      3,2     vertical        0.4        -3              -0.1          0.3             0.4
    4                        2     17,1     horizontal     None        -4              0.2           0.5             0.6
    5                        2     18,1     horizontal     None        -5              -0.2          0.6             0.7
    6                        3     23,1     embeddedv      None        -6              0             0.7             0.8
    """

    ibound, _, _ = basic_dis

    lake_like = xr.DataArray(np.ones(3, dtype=np.float64), dims=("lake"))
    boundnames = lake_like.copy(
        data=["IJsselmeer", "Vinkeveense_plas", "Reeuwijkse_plas"]
    )
    starting_stages = lake_like.copy(data=[1.0, 2.0, 3.0])
    lake_numbers = lake_like.copy(data=[1, 2, 3])

    connection_like = xr.DataArray(
        data=np.ones(6, dtype=np.float64), dims=("connection",)
    )
    connection_lake_number = connection_like.copy(data=[1, 1, 1, 2, 2, 3])
    connection_type = connection_like.copy(
        data=[
            "vertical",
            "vertical",
            "vertical",
            "horizontal",
            "horizontal",
            "embeddedv",
        ]
    )
    connection_bed_leak = connection_like.copy(data=[0.2, 0.3, 0.4, -1, -1, -1])
    connection_cell_id = xr.DataArray(
        data=[
            [1, 3],
            [1, 4],
            [2, 3],
            [1, 17],
            [1, 18],
            [1, 23],
        ],
        coords={"celldim": ["layer", "cell2d"]},
        dims=("connection", "celldim"),
    )
    connection_bottom_elevation = connection_like.copy(
        data=[-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]
    )
    connection_top_elevation = connection_like.copy(
        data=[0.0, 0.1, -0.1, 0.2, -0.2, 0.0]
    )
    connection_width = connection_like.copy(data=[-1.0, -2.0, -3.0, -4.0, -5.0, -6.0])
    connection_length = connection_like.copy(data=[-1.0, -2.0, -3.0, -4.0, -5.0, -6.0])

    lake = Lake(
        # lake data
        lake_numbers,
        starting_stages,
        boundnames,
        # connection data
        connection_lake_number,
        connection_cell_id,
        connection_type,
        connection_bed_leak,
        connection_bottom_elevation,
        connection_top_elevation,
        connection_width,
        connection_length,
    )

    actual = write_and_read(lake, tmp_path, "lake-test")
    expected = textwrap.dedent(
        """\
        begin options
        end options

        begin dimensions
          nlakes 3
          noutlets 0
          ntables 0
        end dimensions

        begin packagedata
          1 1.0 3 IJsselmeer
          2 2.0 2 Vinkeveense_plas
          3 3.0 1 Reeuwijkse_plas
        end packagedata

        begin connectiondata
        1 1 1 3 vertical 0.2 -1.0 0.0 -1.0 -1.0
        1 2 1 4 vertical 0.3 -2.0 0.1 -2.0 -2.0
        1 3 2 3 vertical 0.4 -3.0 -0.1 -3.0 -3.0
        2 1 1 17 horizontal -1.0 -4.0 0.2 -4.0 -4.0
        2 2 1 18 horizontal -1.0 -5.0 -0.2 -5.0 -5.0
        3 1 1 23 embeddedv -1.0 -6.0 0.0 -6.0 -6.0
        end connectiondata
        """
    )
    assert actual == expected


def test_lake_rendering_transient(basic_dis, tmp_path):
    idomain, _, _ = basic_dis

    is_lake1 = xr.full_like(idomain, False, dtype=bool)
    is_lake1[1, 2, 2] = True
    is_lake1[1, 2, 3] = True
    is_lake1[1, 3, 3] = True
    is_lake2 = xr.full_like(idomain, False, dtype=bool)
    is_lake2[1, 4, 4] = True
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

    lake1 = create_lake_data(is_lake1, 11.0, "Naardermeer", rainfall, inflow)
    lake2 = create_lake_data(is_lake2, 11.0, "Ijsselmeer", rainfall, inflow)
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
    outlet1 = OutletManning(1, "Naardermeer", "Ijsselmeer", invert, 24.0, 25.0, 26.0)
    lake_package = Lake.from_lakes_and_outlets([lake1, lake2], [outlet1])
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
    actual = write_and_read(lake_package, tmp_path, "lake-test", global_times)
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
          1 11.0 3 Naardermeer
          2 11.0 1 Ijsselmeer
        end packagedata

        begin connectiondata
        1 1 1 2 2 horizontal 0.2 0.4 0.3 0.6 0.5
        1 2 1 2 3 horizontal 0.2 0.4 0.3 0.6 0.5
        1 3 1 3 3 horizontal 0.2 0.4 0.3 0.6 0.5
        2 1 1 4 4 horizontal 0.2 0.4 0.3 0.6 0.5
        end connectiondata

        begin outlets
        1 2 manning 0.0 25.0 24.0 26.0
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
