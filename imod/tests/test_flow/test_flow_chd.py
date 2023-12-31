import os
import pathlib
import textwrap
from copy import deepcopy

import numpy as np
import pytest
import xarray as xr

from imod.flow import ConstantHead


@pytest.fixture(scope="module")
def constant_head(basic_dis, three_days):
    ibound, _, _ = basic_dis
    x = ibound.x.values

    times = three_days

    # Boundary_conditions
    # Create rising trend
    trend = np.cumsum(np.ones(times.shape))
    trend = xr.DataArray(trend, coords={"time": times}, dims=["time"])

    sides = ibound.where(ibound.x.isin([x[0], x[-1]]))
    head = trend * sides

    return ConstantHead(head=head)


@pytest.fixture(scope="module")
def periodic_constant_head(constant_head, three_days):
    mapping = {three_days[0]: "day1", three_days[1]: "day2", three_days[2]: "day3"}

    chd = deepcopy(constant_head)
    chd.periodic_stress(mapping)

    return chd


@pytest.fixture(scope="module")
def steady_state_constant_head(basic_dis):
    ibound, _, _ = basic_dis
    x = ibound.x.values

    head = ibound.where(ibound.x.isin([x[0], x[-1]]))

    return ConstantHead(head=head)


def test_constant_head(constant_head, get_render_dict, three_days):
    # Resolve in advance, so that comparisons have the same directory
    # See e.g. https://github.com/omarkohl/pytest-datafiles/issues/6
    directory = str(pathlib.Path(".").resolve())

    nlayer = len(constant_head["layer"])
    times = three_days

    time_composed = {
        "1": "2018-01-01 00:00:00",
        "2": "2018-01-02 00:00:00",
        "3": "2018-01-03 00:00:00",
    }

    to_render = get_render_dict(constant_head, directory, times, nlayer)
    to_render["n_entry"] = nlayer
    to_render["times"] = time_composed

    compare = textwrap.dedent(
        f"""\
        0003, (chd), 1, ConstantHead, ['head']
        2018-01-01 00:00:00
        001, 003
        1, 2, 001, 1.000, 0.000, -9999., {directory}{os.sep}head_20180101000000_l1.idf
        1, 2, 002, 1.000, 0.000, -9999., {directory}{os.sep}head_20180101000000_l2.idf
        1, 2, 003, 1.000, 0.000, -9999., {directory}{os.sep}head_20180101000000_l3.idf
        2018-01-02 00:00:00
        001, 003
        1, 2, 001, 1.000, 0.000, -9999., {directory}{os.sep}head_20180102000000_l1.idf
        1, 2, 002, 1.000, 0.000, -9999., {directory}{os.sep}head_20180102000000_l2.idf
        1, 2, 003, 1.000, 0.000, -9999., {directory}{os.sep}head_20180102000000_l3.idf
        2018-01-03 00:00:00
        001, 003
        1, 2, 001, 1.000, 0.000, -9999., {directory}{os.sep}head_20180103000000_l1.idf
        1, 2, 002, 1.000, 0.000, -9999., {directory}{os.sep}head_20180103000000_l2.idf
        1, 2, 003, 1.000, 0.000, -9999., {directory}{os.sep}head_20180103000000_l3.idf"""
    )
    rendered = constant_head._render_projectfile(**to_render)

    assert compare == rendered


def test_periodic_constant_head(periodic_constant_head, get_render_dict, three_days):
    # Resolve in advance, so that comparisons have the same directory
    # See e.g. https://github.com/omarkohl/pytest-datafiles/issues/6

    assert periodic_constant_head._is_periodic()

    directory = str(pathlib.Path(".").resolve())

    nlayer = len(periodic_constant_head["layer"])
    times = three_days

    time_composed = {
        "day1": "day1",
        "day2": "day2",
        "day3": "day3",
    }

    to_render = get_render_dict(periodic_constant_head, directory, times, nlayer)
    to_render["n_entry"] = nlayer
    to_render["times"] = time_composed

    compare = textwrap.dedent(
        f"""\
        0003, (chd), 1, ConstantHead, ['head']
        day1
        001, 003
        1, 2, 001, 1.000, 0.000, -9999., {directory}{os.sep}head_20180101000000_l1.idf
        1, 2, 002, 1.000, 0.000, -9999., {directory}{os.sep}head_20180101000000_l2.idf
        1, 2, 003, 1.000, 0.000, -9999., {directory}{os.sep}head_20180101000000_l3.idf
        day2
        001, 003
        1, 2, 001, 1.000, 0.000, -9999., {directory}{os.sep}head_20180102000000_l1.idf
        1, 2, 002, 1.000, 0.000, -9999., {directory}{os.sep}head_20180102000000_l2.idf
        1, 2, 003, 1.000, 0.000, -9999., {directory}{os.sep}head_20180102000000_l3.idf
        day3
        001, 003
        1, 2, 001, 1.000, 0.000, -9999., {directory}{os.sep}head_20180103000000_l1.idf
        1, 2, 002, 1.000, 0.000, -9999., {directory}{os.sep}head_20180103000000_l2.idf
        1, 2, 003, 1.000, 0.000, -9999., {directory}{os.sep}head_20180103000000_l3.idf"""
    )
    rendered = periodic_constant_head._render_projectfile(**to_render)

    assert compare == rendered


def test_steady_state_constant_head(steady_state_constant_head, get_render_dict):
    # Resolve in advance, so that comparisons have the same directory
    # See e.g. https://github.com/omarkohl/pytest-datafiles/issues/6

    directory = str(pathlib.Path(".").resolve())

    nlayer = len(steady_state_constant_head["layer"])

    time_composed = {"steady-state": "steady-state"}

    to_render = get_render_dict(steady_state_constant_head, directory, [], nlayer)
    to_render["n_entry"] = nlayer
    to_render["times"] = time_composed

    compare = textwrap.dedent(
        f"""\
        0001, (chd), 1, ConstantHead, ['head']
        steady-state
        001, 003
        1, 2, 001, 1.000, 0.000, -9999., {directory}{os.sep}head_l1.idf
        1, 2, 002, 1.000, 0.000, -9999., {directory}{os.sep}head_l2.idf
        1, 2, 003, 1.000, 0.000, -9999., {directory}{os.sep}head_l3.idf"""
    )
    rendered = steady_state_constant_head._render_projectfile(**to_render)

    assert compare == rendered
