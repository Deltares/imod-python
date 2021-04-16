from imod.flow import ConstantHead
from imod.flow import timeutil
import numpy as np
import xarray as xr
import pytest
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


@pytest.fixture()
def correct_datetime():
    return np.array(
        [
            np.datetime64("2018-01-01T00:00:00.000000000"),
            np.datetime64("2018-01-02T00:00:00.000000000"),
            np.datetime64("2018-01-03T00:00:00.000000000"),
        ]
    )


def test_insert_unique_package_times(constant_head, three_days, correct_datetime):
    outer_days = [three_days[0], three_days[-1]]
    inner_day = three_days[1]

    head = constant_head["head"]

    chd1 = ConstantHead(head=head.sel(time=outer_days))
    chd2 = ConstantHead(head=head.sel(time=inner_day))
    d = {"primary": chd1, "secondary": chd2}

    times, _ = timeutil.insert_unique_package_times(d.items())

    assert np.all(times == correct_datetime)


def test_insert_unique_package_times_overlap(
    constant_head, three_days, correct_datetime
):
    outer_days = [three_days[0], three_days[-1]]
    first_days = three_days[:-1]

    head = constant_head["head"]

    chd1 = ConstantHead(head=head.sel(time=outer_days))
    chd2 = ConstantHead(head=head.sel(time=first_days))
    d = {"primary": chd1, "secondary": chd2}

    times, _ = timeutil.insert_unique_package_times(d.items())

    assert np.all(times == correct_datetime)


def test_insert_unique_package_times_manual_insert(
    constant_head, three_days, correct_datetime
):
    first_day = three_days[0]
    second_day = np.datetime64(three_days[1], "ns")
    third_day = three_days[-1]

    head = constant_head["head"]

    chd1 = ConstantHead(head=head.sel(time=first_day))
    chd2 = ConstantHead(head=head.sel(time=third_day))
    d = {"primary": chd1, "secondary": chd2}

    times, _ = timeutil.insert_unique_package_times(d.items(), manual_insert=second_day)

    assert np.all(times == correct_datetime)