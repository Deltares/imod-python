import datetime

import cftime
import numpy as np
import pytest

import imod


def test_datetime_conversion__withinbounds():
    times = [datetime.datetime(y, 1, 1) for y in range(2000, 2010)]
    converted, use_cftime = imod.util.time._convert_datetimes(times, use_cftime=False)
    assert use_cftime is False
    assert all(t.dtype == "<M8[ns]" for t in converted)
    assert converted[0] == np.datetime64("2000-01-01", "ns")
    assert converted[-1] == np.datetime64("2009-01-01", "ns")


def test_datetime_conversion__outofbounds():
    times = [datetime.datetime(y, 1, 1) for y in range(1670, 1680)]
    with pytest.warns(UserWarning):
        converted, use_cftime = imod.util.time._convert_datetimes(times, use_cftime=False)
    assert use_cftime is True
    assert all(type(t) is cftime.DatetimeProlepticGregorian for t in converted)
    assert converted[0] == cftime.DatetimeProlepticGregorian(1670, 1, 1)
    assert converted[-1] == cftime.DatetimeProlepticGregorian(1679, 1, 1)


def test_datetime_conversion__withinbounds_cftime():
    times = [datetime.datetime(y, 1, 1) for y in range(2000, 2010)]
    converted, use_cftime = imod.util.time._convert_datetimes(times, use_cftime=True)
    assert use_cftime is True
    assert all(type(t) is cftime.DatetimeProlepticGregorian for t in converted)
    assert converted[0] == cftime.DatetimeProlepticGregorian(2000, 1, 1)
    assert converted[-1] == cftime.DatetimeProlepticGregorian(2009, 1, 1)
