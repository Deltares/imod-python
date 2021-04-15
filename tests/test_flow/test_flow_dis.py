from imod.flow import TimeDiscretization
from imod.wq import timeutil
import xarray as xr
import numpy as np
import pytest
import cftime


@pytest.fixture(scope="module")
def time_discretization(three_days):
    times = three_days
    duration = timeutil.timestep_duration(times, False)

    timestep_duration = xr.DataArray(
        duration, coords={"time": np.array(times)[:-1]}, dims=("time",)
    )

    return TimeDiscretization(timestep_duration=timestep_duration)


@pytest.fixture(scope="module")
def time_discretization_cftime(three_days):
    times = three_days
    use_cftime = True
    times = [timeutil.to_datetime(time, use_cftime) for time in times]
    duration = timeutil.timestep_duration(times, use_cftime)

    timestep_duration = xr.DataArray(
        duration, coords={"time": np.array(times)[:-1]}, dims=("time",)
    )

    return TimeDiscretization(timestep_duration=timestep_duration)


def test_time_discretization(time_discretization):
    rendered = time_discretization._render()

    compare = "20180101000000,1,1,1.0\n" "20180102000000,1,1,1.0\n"

    assert rendered == compare


def test_time_discretization_cftime(time_discretization_cftime):
    first_time = time_discretization_cftime.dataset["time"].values[0]

    # Double check if we really are going to test cftimes next
    assert isinstance(first_time, cftime.datetime)

    rendered = time_discretization_cftime._render()

    compare = "20180101000000,1,1,1.0\n" "20180102000000,1,1,1.0\n"

    assert rendered == compare
