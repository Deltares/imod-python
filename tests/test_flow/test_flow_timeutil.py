from imod.flow import ConstantHead, Well
from imod.flow import timeutil
import numpy as np
import xarray as xr
import pytest
import pathlib
import os
from copy import deepcopy


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


def test_insert_unique_package_times(constant_head, three_days):
    outer_days = [three_days[0], three_days[-1]]
    inner_day = three_days[1]

    head = constant_head["head"]

    chd1 = ConstantHead(head=head.sel(time=outer_days))
    chd2 = ConstantHead(head=head.sel(time=inner_day))
    d = {"primary": chd1, "secondary": chd2}

    times = []

    times = timeutil.insert_unique_package_times(d.items(), times)

    compare = [
        np.datetime64("2018-01-01T00:00:00.000000000"),
        np.datetime64("2018-01-02T00:00:00.000000000"),
        np.datetime64("2018-01-03T00:00:00.000000000"),
    ]

    assert times == compare
