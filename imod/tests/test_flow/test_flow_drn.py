import os
import pathlib
import textwrap

import numpy as np
import pytest
import xarray as xr

from imod.flow import Drain


@pytest.fixture(scope="module")
def drain_transient(basic_dis, three_days):
    ibound, _, _ = basic_dis
    x = ibound.x.values

    times = three_days

    # Boundary_conditions
    # Create rising trend
    trend = np.cumsum(np.ones(times.shape))
    trend = xr.DataArray(trend, coords={"time": times}, dims=["time"])

    sides = ibound.where(ibound.x.isin([x[0], x[-1]]))
    elevation = trend * sides

    return Drain(conductance=10.0, elevation=elevation)


@pytest.fixture(scope="module")
def drain_transient_scalar(three_days):
    times = three_days

    layers = np.arange(2) + 1

    # Boundary_conditions
    # Create rising trend
    data = (np.arange(6) + 1.0).reshape((3, 2))
    elevation = xr.DataArray(
        data, coords={"time": times, "layer": layers}, dims=["time", "layer"]
    )

    conductance = elevation + 100.0

    return Drain(conductance=conductance, elevation=elevation)


@pytest.fixture(scope="module")
def drain_transient_scalar_paths(three_days):
    times = three_days

    layers = np.arange(2) + 1

    # Boundary_conditions
    # Create rising trend
    data_elevation = [
        [f"/path/to/drn/elevation_day1_L{layer}.idf" for layer in layers],
        [f"/path/to/drn/elevation_day2_L{layer}.idf" for layer in layers],
        [f"/path/to/drn/elevation_day3_L{layer}.idf" for layer in layers],
    ]
    elevation = xr.DataArray(
        data_elevation, coords={"time": times, "layer": layers}, dims=["time", "layer"]
    )

    data_conductance = [
        [f"/path/to/drn/conductance_day1_L{layer}.idf" for layer in layers],
        [f"/path/to/drn/conductance_day2_L{layer}.idf" for layer in layers],
        [f"/path/to/drn/conductance_day3_L{layer}.idf" for layer in layers],
    ]
    conductance = xr.DataArray(
        data_conductance,
        coords={"time": times, "layer": layers},
        dims=["time", "layer"],
    )

    return Drain(conductance=conductance, elevation=elevation)


def test_drain_transient(get_render_dict, drain_transient, three_days):
    # Resolve in advance, so that comparisons have the same directory
    # See e.g. https://github.com/omarkohl/pytest-datafiles/issues/6
    directory = str(pathlib.Path(".").resolve())

    nlayer = len(drain_transient["layer"])
    times = three_days

    time_composed = {
        "1": "2018-01-01 00:00:00",
        "2": "2018-01-02 00:00:00",
        "3": "2018-01-03 00:00:00",
    }

    to_render = get_render_dict(drain_transient, directory, times, nlayer)
    to_render["n_entry"] = nlayer
    to_render["times"] = time_composed

    compare = textwrap.dedent(
        f"""\
        0003, (drn), 1, Drain, ['conductance', 'elevation']
        2018-01-01 00:00:00
        002, 003
        1, 1, 001, 1.000, 0.000, 10.0, ""
        1, 1, 002, 1.000, 0.000, 10.0, ""
        1, 1, 003, 1.000, 0.000, 10.0, ""
        1, 2, 001, 1.000, 0.000, -9999., {directory}{os.sep}elevation_20180101000000_l1.idf
        1, 2, 002, 1.000, 0.000, -9999., {directory}{os.sep}elevation_20180101000000_l2.idf
        1, 2, 003, 1.000, 0.000, -9999., {directory}{os.sep}elevation_20180101000000_l3.idf
        2018-01-02 00:00:00
        002, 003
        1, 1, 001, 1.000, 0.000, 10.0, ""
        1, 1, 002, 1.000, 0.000, 10.0, ""
        1, 1, 003, 1.000, 0.000, 10.0, ""
        1, 2, 001, 1.000, 0.000, -9999., {directory}{os.sep}elevation_20180102000000_l1.idf
        1, 2, 002, 1.000, 0.000, -9999., {directory}{os.sep}elevation_20180102000000_l2.idf
        1, 2, 003, 1.000, 0.000, -9999., {directory}{os.sep}elevation_20180102000000_l3.idf
        2018-01-03 00:00:00
        002, 003
        1, 1, 001, 1.000, 0.000, 10.0, ""
        1, 1, 002, 1.000, 0.000, 10.0, ""
        1, 1, 003, 1.000, 0.000, 10.0, ""
        1, 2, 001, 1.000, 0.000, -9999., {directory}{os.sep}elevation_20180103000000_l1.idf
        1, 2, 002, 1.000, 0.000, -9999., {directory}{os.sep}elevation_20180103000000_l2.idf
        1, 2, 003, 1.000, 0.000, -9999., {directory}{os.sep}elevation_20180103000000_l3.idf"""
    )
    rendered = drain_transient._render_projectfile(**to_render)

    assert compare == rendered


def test_drain_transient_scalar(get_render_dict, drain_transient_scalar, three_days):
    # Resolve in advance, so that comparisons have the same directory
    # See e.g. https://github.com/omarkohl/pytest-datafiles/issues/6
    directory = str(pathlib.Path(".").resolve())

    nlayer = len(drain_transient_scalar["layer"])
    times = three_days

    time_composed = {
        "1": "2018-01-01 00:00:00",
        "2": "2018-01-02 00:00:00",
        "3": "2018-01-03 00:00:00",
    }

    to_render = get_render_dict(drain_transient_scalar, directory, times, nlayer)
    to_render["n_entry"] = nlayer
    to_render["times"] = time_composed

    compare = textwrap.dedent(
        '''\
        0003, (drn), 1, Drain, ['conductance', 'elevation']
        2018-01-01 00:00:00
        002, 002
        1, 1, 001, 1.000, 0.000, 101.0, ""
        1, 1, 002, 1.000, 0.000, 102.0, ""
        1, 1, 001, 1.000, 0.000, 1.0, ""
        1, 1, 002, 1.000, 0.000, 2.0, ""
        2018-01-02 00:00:00
        002, 002
        1, 1, 001, 1.000, 0.000, 103.0, ""
        1, 1, 002, 1.000, 0.000, 104.0, ""
        1, 1, 001, 1.000, 0.000, 3.0, ""
        1, 1, 002, 1.000, 0.000, 4.0, ""
        2018-01-03 00:00:00
        002, 002
        1, 1, 001, 1.000, 0.000, 105.0, ""
        1, 1, 002, 1.000, 0.000, 106.0, ""
        1, 1, 001, 1.000, 0.000, 5.0, ""
        1, 1, 002, 1.000, 0.000, 6.0, ""'''
    )
    rendered = drain_transient_scalar._render_projectfile(**to_render)

    assert compare == rendered


def test_drain_transient_scalar_paths(
    get_render_dict, drain_transient_scalar_paths, three_days
):
    # Resolve in advance, so that comparisons have the same directory
    # See e.g. https://github.com/omarkohl/pytest-datafiles/issues/6
    directory = str(pathlib.Path(".").resolve())

    nlayer = len(drain_transient_scalar_paths["layer"])
    times = three_days

    time_composed = {
        "1": "2018-01-01 00:00:00",
        "2": "2018-01-02 00:00:00",
        "3": "2018-01-03 00:00:00",
    }

    to_render = get_render_dict(drain_transient_scalar_paths, directory, times, nlayer)
    to_render["n_entry"] = nlayer
    to_render["times"] = time_composed

    compare = textwrap.dedent(
        """\
        0003, (drn), 1, Drain, ['conductance', 'elevation']
        2018-01-01 00:00:00
        002, 002
        1, 2, 001, 1.000, 0.000, -9999., /path/to/drn/conductance_day1_L1.idf
        1, 2, 002, 1.000, 0.000, -9999., /path/to/drn/conductance_day1_L2.idf
        1, 2, 001, 1.000, 0.000, -9999., /path/to/drn/elevation_day1_L1.idf
        1, 2, 002, 1.000, 0.000, -9999., /path/to/drn/elevation_day1_L2.idf
        2018-01-02 00:00:00
        002, 002
        1, 2, 001, 1.000, 0.000, -9999., /path/to/drn/conductance_day2_L1.idf
        1, 2, 002, 1.000, 0.000, -9999., /path/to/drn/conductance_day2_L2.idf
        1, 2, 001, 1.000, 0.000, -9999., /path/to/drn/elevation_day2_L1.idf
        1, 2, 002, 1.000, 0.000, -9999., /path/to/drn/elevation_day2_L2.idf
        2018-01-03 00:00:00
        002, 002
        1, 2, 001, 1.000, 0.000, -9999., /path/to/drn/conductance_day3_L1.idf
        1, 2, 002, 1.000, 0.000, -9999., /path/to/drn/conductance_day3_L2.idf
        1, 2, 001, 1.000, 0.000, -9999., /path/to/drn/elevation_day3_L1.idf
        1, 2, 002, 1.000, 0.000, -9999., /path/to/drn/elevation_day3_L2.idf"""
    )

    rendered = drain_transient_scalar_paths._render_projectfile(**to_render)

    assert compare == rendered
