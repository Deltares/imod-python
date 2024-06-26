import os
import pathlib
import textwrap
from copy import deepcopy

import numpy as np
import pytest
import xarray as xr

from imod.flow import ConstantHead, Well
from imod.flow.pkggroup import ConstantHeadGroup, WellGroup


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
def constant_head_no_time(basic_dis):
    ibound, _, _ = basic_dis
    x = ibound.x.values

    sides = ibound.where(ibound.x.isin([x[0], x[-1]]))

    return ConstantHead(head=sides)


def test_group_rendered_no_time(constant_head_no_time, three_days):
    chd1 = constant_head_no_time
    chd2 = deepcopy(constant_head_no_time)

    d = {"primary": chd1, "secondary": chd2}

    chd_group = ConstantHeadGroup(**d)
    nlayer = len(chd1["layer"])
    times = three_days
    directory = pathlib.Path(".").resolve()

    time_composed = {
        "1": "2018-01-01 00:00:00",
        "2": "2018-01-02 00:00:00",
        "3": "2018-01-03 00:00:00",
        "steady-state": "steady-state",
    }

    group_composition = chd_group.compose(directory, times, nlayer)

    pkg_id = chd1._pkg_id

    to_render = {
        "pkg_id": pkg_id,
        "name": "ConstantHead",
        "variable_order": chd1._variable_order,
        "package_data": group_composition[pkg_id],
        "times": time_composed,
        "n_entry": 6,
    }

    rendered = chd1._render_projectfile(**to_render)

    compare = textwrap.dedent(
        f"""\
        0001, (chd), 1, ConstantHead, ['head']
        steady-state
        001, 006
        1, 2, 001, 1.000, 0.000, -9999., {directory}{os.sep}primary{os.sep}head_l1.idf
        1, 2, 002, 1.000, 0.000, -9999., {directory}{os.sep}primary{os.sep}head_l2.idf
        1, 2, 003, 1.000, 0.000, -9999., {directory}{os.sep}primary{os.sep}head_l3.idf
        1, 2, 001, 1.000, 0.000, -9999., {directory}{os.sep}secondary{os.sep}head_l1.idf
        1, 2, 002, 1.000, 0.000, -9999., {directory}{os.sep}secondary{os.sep}head_l2.idf
        1, 2, 003, 1.000, 0.000, -9999., {directory}{os.sep}secondary{os.sep}head_l3.idf"""
    )

    assert compare == rendered


def test_group_rendered_mixed_time_no_time(
    constant_head, constant_head_no_time, three_days
):
    chd1 = constant_head
    chd2 = constant_head_no_time
    d = {"primary": chd1, "secondary": chd2}

    chd_group = ConstantHeadGroup(**d)

    nlayer = len(constant_head["layer"])
    times = three_days
    directory = pathlib.Path(".").resolve()

    time_composed = {
        "1": "2018-01-01 00:00:00",
        "2": "2018-01-02 00:00:00",
        "3": "2018-01-03 00:00:00",
    }

    group_composition = chd_group.compose(directory, times, nlayer)

    pkg_id = chd1._pkg_id

    to_render = {
        "pkg_id": pkg_id,
        "name": "ConstantHead",
        "variable_order": chd1._variable_order,
        "package_data": group_composition[pkg_id],
        "times": time_composed,
        "n_entry": 6,
    }

    rendered = chd1._render_projectfile(**to_render)

    compare = textwrap.dedent(
        f"""\
        0003, (chd), 1, ConstantHead, ['head']
        2018-01-01 00:00:00
        001, 006
        1, 2, 001, 1.000, 0.000, -9999., {directory}{os.sep}primary{os.sep}head_20180101000000_l1.idf
        1, 2, 002, 1.000, 0.000, -9999., {directory}{os.sep}primary{os.sep}head_20180101000000_l2.idf
        1, 2, 003, 1.000, 0.000, -9999., {directory}{os.sep}primary{os.sep}head_20180101000000_l3.idf
        1, 2, 001, 1.000, 0.000, -9999., {directory}{os.sep}secondary{os.sep}head_l1.idf
        1, 2, 002, 1.000, 0.000, -9999., {directory}{os.sep}secondary{os.sep}head_l2.idf
        1, 2, 003, 1.000, 0.000, -9999., {directory}{os.sep}secondary{os.sep}head_l3.idf
        2018-01-02 00:00:00
        001, 006
        1, 2, 001, 1.000, 0.000, -9999., {directory}{os.sep}primary{os.sep}head_20180102000000_l1.idf
        1, 2, 002, 1.000, 0.000, -9999., {directory}{os.sep}primary{os.sep}head_20180102000000_l2.idf
        1, 2, 003, 1.000, 0.000, -9999., {directory}{os.sep}primary{os.sep}head_20180102000000_l3.idf
        1, 2, 001, 1.000, 0.000, -9999., {directory}{os.sep}secondary{os.sep}head_l1.idf
        1, 2, 002, 1.000, 0.000, -9999., {directory}{os.sep}secondary{os.sep}head_l2.idf
        1, 2, 003, 1.000, 0.000, -9999., {directory}{os.sep}secondary{os.sep}head_l3.idf
        2018-01-03 00:00:00
        001, 006
        1, 2, 001, 1.000, 0.000, -9999., {directory}{os.sep}primary{os.sep}head_20180103000000_l1.idf
        1, 2, 002, 1.000, 0.000, -9999., {directory}{os.sep}primary{os.sep}head_20180103000000_l2.idf
        1, 2, 003, 1.000, 0.000, -9999., {directory}{os.sep}primary{os.sep}head_20180103000000_l3.idf
        1, 2, 001, 1.000, 0.000, -9999., {directory}{os.sep}secondary{os.sep}head_l1.idf
        1, 2, 002, 1.000, 0.000, -9999., {directory}{os.sep}secondary{os.sep}head_l2.idf
        1, 2, 003, 1.000, 0.000, -9999., {directory}{os.sep}secondary{os.sep}head_l3.idf"""
    )

    assert compare == rendered


def test_group_rendered(constant_head, three_days):
    chd1 = constant_head
    chd2 = deepcopy(constant_head)
    d = {"primary": chd1, "secondary": chd2}

    nlayer = len(constant_head["layer"])
    times = three_days
    directory = pathlib.Path(".").resolve()

    time_composed = {
        "1": "2018-01-01 00:00:00",
        "2": "2018-01-02 00:00:00",
        "3": "2018-01-03 00:00:00",
    }

    chd_group = ConstantHeadGroup(**d)

    group_composition = chd_group.compose(directory, times, nlayer)

    pkg_id = chd1._pkg_id

    to_render = {
        "pkg_id": pkg_id,
        "name": "ConstantHead",
        "variable_order": chd1._variable_order,
        "package_data": group_composition[pkg_id],
        "times": time_composed,
        "n_entry": 6,
    }

    rendered = chd1._render_projectfile(**to_render)

    compare = textwrap.dedent(
        f"""\
        0003, (chd), 1, ConstantHead, ['head']
        2018-01-01 00:00:00
        001, 006
        1, 2, 001, 1.000, 0.000, -9999., {directory}{os.sep}primary{os.sep}head_20180101000000_l1.idf
        1, 2, 002, 1.000, 0.000, -9999., {directory}{os.sep}primary{os.sep}head_20180101000000_l2.idf
        1, 2, 003, 1.000, 0.000, -9999., {directory}{os.sep}primary{os.sep}head_20180101000000_l3.idf
        1, 2, 001, 1.000, 0.000, -9999., {directory}{os.sep}secondary{os.sep}head_20180101000000_l1.idf
        1, 2, 002, 1.000, 0.000, -9999., {directory}{os.sep}secondary{os.sep}head_20180101000000_l2.idf
        1, 2, 003, 1.000, 0.000, -9999., {directory}{os.sep}secondary{os.sep}head_20180101000000_l3.idf
        2018-01-02 00:00:00
        001, 006
        1, 2, 001, 1.000, 0.000, -9999., {directory}{os.sep}primary{os.sep}head_20180102000000_l1.idf
        1, 2, 002, 1.000, 0.000, -9999., {directory}{os.sep}primary{os.sep}head_20180102000000_l2.idf
        1, 2, 003, 1.000, 0.000, -9999., {directory}{os.sep}primary{os.sep}head_20180102000000_l3.idf
        1, 2, 001, 1.000, 0.000, -9999., {directory}{os.sep}secondary{os.sep}head_20180102000000_l1.idf
        1, 2, 002, 1.000, 0.000, -9999., {directory}{os.sep}secondary{os.sep}head_20180102000000_l2.idf
        1, 2, 003, 1.000, 0.000, -9999., {directory}{os.sep}secondary{os.sep}head_20180102000000_l3.idf
        2018-01-03 00:00:00
        001, 006
        1, 2, 001, 1.000, 0.000, -9999., {directory}{os.sep}primary{os.sep}head_20180103000000_l1.idf
        1, 2, 002, 1.000, 0.000, -9999., {directory}{os.sep}primary{os.sep}head_20180103000000_l2.idf
        1, 2, 003, 1.000, 0.000, -9999., {directory}{os.sep}primary{os.sep}head_20180103000000_l3.idf
        1, 2, 001, 1.000, 0.000, -9999., {directory}{os.sep}secondary{os.sep}head_20180103000000_l1.idf
        1, 2, 002, 1.000, 0.000, -9999., {directory}{os.sep}secondary{os.sep}head_20180103000000_l2.idf
        1, 2, 003, 1.000, 0.000, -9999., {directory}{os.sep}secondary{os.sep}head_20180103000000_l3.idf"""
    )

    assert compare == rendered


def test_group_synchronize_times_rendered(constant_head, three_days):
    chd1 = constant_head
    chd2 = ConstantHead(head=2.0)
    d = {"primary": chd1, "secondary": chd2}

    nlayer = len(constant_head["layer"])
    times = three_days
    directory = pathlib.Path(".").resolve()

    time_composed = {
        "1": "2018-01-01 00:00:00",
        "2": "2018-01-02 00:00:00",
        "3": "2018-01-03 00:00:00",
    }

    chd_group = ConstantHeadGroup(**d)

    group_composition = chd_group.compose(directory, times, nlayer)

    pkg_id = chd1._pkg_id

    to_render = {
        "pkg_id": pkg_id,
        "name": "ConstantHead",
        "variable_order": chd1._variable_order,
        "package_data": group_composition[pkg_id],
        "times": time_composed,
        "n_entry": 6,
    }

    rendered = chd1._render_projectfile(**to_render)

    compare = textwrap.dedent(
        f'''\
        0003, (chd), 1, ConstantHead, ['head']
        2018-01-01 00:00:00
        001, 006
        1, 2, 001, 1.000, 0.000, -9999., {directory}{os.sep}primary{os.sep}head_20180101000000_l1.idf
        1, 2, 002, 1.000, 0.000, -9999., {directory}{os.sep}primary{os.sep}head_20180101000000_l2.idf
        1, 2, 003, 1.000, 0.000, -9999., {directory}{os.sep}primary{os.sep}head_20180101000000_l3.idf
        1, 1, 001, 1.000, 0.000, 2.0, ""
        1, 1, 002, 1.000, 0.000, 2.0, ""
        1, 1, 003, 1.000, 0.000, 2.0, ""
        2018-01-02 00:00:00
        001, 006
        1, 2, 001, 1.000, 0.000, -9999., {directory}{os.sep}primary{os.sep}head_20180102000000_l1.idf
        1, 2, 002, 1.000, 0.000, -9999., {directory}{os.sep}primary{os.sep}head_20180102000000_l2.idf
        1, 2, 003, 1.000, 0.000, -9999., {directory}{os.sep}primary{os.sep}head_20180102000000_l3.idf
        1, 1, 001, 1.000, 0.000, 2.0, ""
        1, 1, 002, 1.000, 0.000, 2.0, ""
        1, 1, 003, 1.000, 0.000, 2.0, ""
        2018-01-03 00:00:00
        001, 006
        1, 2, 001, 1.000, 0.000, -9999., {directory}{os.sep}primary{os.sep}head_20180103000000_l1.idf
        1, 2, 002, 1.000, 0.000, -9999., {directory}{os.sep}primary{os.sep}head_20180103000000_l2.idf
        1, 2, 003, 1.000, 0.000, -9999., {directory}{os.sep}primary{os.sep}head_20180103000000_l3.idf
        1, 1, 001, 1.000, 0.000, 2.0, ""
        1, 1, 002, 1.000, 0.000, 2.0, ""
        1, 1, 003, 1.000, 0.000, 2.0, ""'''
    )

    assert compare == rendered


def test_two_wels(well_df, three_days, get_render_dict):
    directory = pathlib.Path(".").resolve()
    nlayer = 3
    times = three_days

    well = Well(**well_df)
    well2 = Well(**well_df)
    d = {"well_1": well, "well_2": well2}

    well_group = WellGroup(**d)

    group_composition = well_group.compose(directory, times, nlayer)

    pkg_id = well._pkg_id

    time_composed = {
        "1": "2018-01-01 00:00:00",
        "2": "2018-01-02 00:00:00",
        "3": "2018-01-03 00:00:00",
    }

    to_render = {
        "pkg_id": pkg_id,
        "name": "Well",
        "variable_order": well._variable_order,
        "package_data": group_composition[pkg_id],
        "times": time_composed,
        "n_entry": 2,
    }

    compare = textwrap.dedent(
        f"""\
        0003, (wel), 1, Well, ['rate']
        2018-01-01 00:00:00
        001, 002
        1, 2, 002, 1.000, 0.000, -9999., {directory}{os.sep}well_1{os.sep}well_1_20180101000000_l2.ipf
        1, 2, 002, 1.000, 0.000, -9999., {directory}{os.sep}well_2{os.sep}well_2_20180101000000_l2.ipf
        2018-01-02 00:00:00
        001, 002
        1, 2, 002, 1.000, 0.000, -9999., {directory}{os.sep}well_1{os.sep}well_1_20180102000000_l2.ipf
        1, 2, 002, 1.000, 0.000, -9999., {directory}{os.sep}well_2{os.sep}well_2_20180102000000_l2.ipf
        2018-01-03 00:00:00
        001, 002
        1, 2, 002, 1.000, 0.000, -9999., {directory}{os.sep}well_1{os.sep}well_1_20180103000000_l2.ipf
        1, 2, 002, 1.000, 0.000, -9999., {directory}{os.sep}well_2{os.sep}well_2_20180103000000_l2.ipf"""
    )
    rendered = well._render_projectfile(**to_render)

    assert compare == rendered
