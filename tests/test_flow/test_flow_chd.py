from imod.flow import ConstantHead
import pathlib
import os
import numpy as np
import xarray as xr
import pytest

@pytest.fixture(scope="module")
def constant_head(basic_dis, three_days):
    ibound, _, _ = basic_dis
    x = ibound.x.values

    times = three_days 

    #Boundary_conditions
    trend = np.ones(times.shape)
    trend = np.cumsum(trend)

    head = ibound.where(ibound.x.isin([x[0], x[-1]]))
    head = xr.DataArray(trend, 
                coords = {"time" : times}, 
                dims = ["time"]
                ) * head

    return ConstantHead(head=head)


def test_constant_head(constant_head, get_render_dict, three_days):
    # Resolve in advance, so that comparisons have the same directory
    # See e.g. https://github.com/omarkohl/pytest-datafiles/issues/6
    directory = str(pathlib.Path(".").resolve())

    nlayer = len(constant_head["layer"])
    times = three_days 

    time_composed = {
        "1" : "2018-01-01 00:00:00",
        "2" : "2018-01-02 00:00:00",
        "3" : "2018-01-03 00:00:00"
        }

    to_render = get_render_dict(constant_head, directory, times, nlayer)
    to_render["n_entry"] = nlayer
    to_render["times"] = time_composed

    compare = (
    "0003, (chd), 1, ConstantHead, ['head']\n"
    "2018-01-01 00:00:00\n"
    "001, 003\n"
    f"1, 2, 001, 1.000, 0.000, -9999., {directory}{os.sep}head_20180101000000_l1.idf\n"
    f"1, 2, 002, 1.000, 0.000, -9999., {directory}{os.sep}head_20180101000000_l2.idf\n"
    f"1, 2, 003, 1.000, 0.000, -9999., {directory}{os.sep}head_20180101000000_l3.idf\n"
    "2018-01-02 00:00:00\n"
    "001, 003\n"
    f"1, 2, 001, 1.000, 0.000, -9999., {directory}{os.sep}head_20180102000000_l1.idf\n"
    f"1, 2, 002, 1.000, 0.000, -9999., {directory}{os.sep}head_20180102000000_l2.idf\n"
    f"1, 2, 003, 1.000, 0.000, -9999., {directory}{os.sep}head_20180102000000_l3.idf\n"
    "2018-01-03 00:00:00\n"
    "001, 003\n"
    f"1, 2, 001, 1.000, 0.000, -9999., {directory}{os.sep}head_20180103000000_l1.idf\n"
    f"1, 2, 002, 1.000, 0.000, -9999., {directory}{os.sep}head_20180103000000_l2.idf\n"
    f"1, 2, 003, 1.000, 0.000, -9999., {directory}{os.sep}head_20180103000000_l3.idf"
    )
    rendered = constant_head._render_projectfile(**to_render)

    assert compare == rendered

def test_chd_2_systems(constant_head, get_render_dict, three_days):
    # Resolve in advance, so that comparisons have the same directory
    # See e.g. https://github.com/omarkohl/pytest-datafiles/issues/6
    directory = str(pathlib.Path(".").resolve())

    nlayer = len(constant_head["layer"])
    times = three_days 

    time_composed = {
        "1" : "2018-01-01 00:00:00",
        "2" : "2018-01-02 00:00:00",
        "3" : "2018-01-03 00:00:00"
        }

    to_render = get_render_dict(
        constant_head, directory, times, nlayer
        )

    # Manually attach second system in package data
    # For some reason supplying a system_index and a composition
    # to get_render_dict did not work to get the same result in pytest
    for time_step in to_render["package_data"].keys():
        to_render["package_data"][time_step]["head"][2] = to_render["package_data"][time_step]["head"][1]

    to_render["n_entry"] = nlayer * 2
    to_render["times"] = time_composed

    compare = (
    "0003, (chd), 1, ConstantHead, ['head']\n"
    "2018-01-01 00:00:00\n"
    "001, 006\n"
    f"1, 2, 001, 1.000, 0.000, -9999., {directory}{os.sep}head_20180101000000_l1.idf\n"
    f"1, 2, 002, 1.000, 0.000, -9999., {directory}{os.sep}head_20180101000000_l2.idf\n"
    f"1, 2, 003, 1.000, 0.000, -9999., {directory}{os.sep}head_20180101000000_l3.idf\n"
    f"1, 2, 001, 1.000, 0.000, -9999., {directory}{os.sep}head_20180101000000_l1.idf\n"
    f"1, 2, 002, 1.000, 0.000, -9999., {directory}{os.sep}head_20180101000000_l2.idf\n"
    f"1, 2, 003, 1.000, 0.000, -9999., {directory}{os.sep}head_20180101000000_l3.idf\n"
    "2018-01-02 00:00:00\n"
    "001, 006\n"
    f"1, 2, 001, 1.000, 0.000, -9999., {directory}{os.sep}head_20180102000000_l1.idf\n"
    f"1, 2, 002, 1.000, 0.000, -9999., {directory}{os.sep}head_20180102000000_l2.idf\n"
    f"1, 2, 003, 1.000, 0.000, -9999., {directory}{os.sep}head_20180102000000_l3.idf\n"
    f"1, 2, 001, 1.000, 0.000, -9999., {directory}{os.sep}head_20180102000000_l1.idf\n"
    f"1, 2, 002, 1.000, 0.000, -9999., {directory}{os.sep}head_20180102000000_l2.idf\n"
    f"1, 2, 003, 1.000, 0.000, -9999., {directory}{os.sep}head_20180102000000_l3.idf\n"
    "2018-01-03 00:00:00\n"
    "001, 006\n"
    f"1, 2, 001, 1.000, 0.000, -9999., {directory}{os.sep}head_20180103000000_l1.idf\n"
    f"1, 2, 002, 1.000, 0.000, -9999., {directory}{os.sep}head_20180103000000_l2.idf\n"
    f"1, 2, 003, 1.000, 0.000, -9999., {directory}{os.sep}head_20180103000000_l3.idf\n"
    f"1, 2, 001, 1.000, 0.000, -9999., {directory}{os.sep}head_20180103000000_l1.idf\n"
    f"1, 2, 002, 1.000, 0.000, -9999., {directory}{os.sep}head_20180103000000_l2.idf\n"
    f"1, 2, 003, 1.000, 0.000, -9999., {directory}{os.sep}head_20180103000000_l3.idf"
    )
    rendered = constant_head._render_projectfile(**to_render)

    assert compare == rendered