import datetime

import cftime
import numpy as np
import pandas as pd
import pytest

import imod
from imod.util.time import _check_year, forcing_starts_ends, to_datetime_internal


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


def test__check_year():
    year_pass = 1678
    _check_year(year_pass)

    year_fail = 1677
    with pytest.raises(ValueError):
        _check_year(year_fail)


def test_to_datetime__string_to_datetime64():
    present_day = "2000-01-01"
    datetime = to_datetime_internal(present_day, False)
    assert datetime == np.datetime64("2000-01-01", "ns")

    the_past = "1000-01-01"
    with pytest.raises(ValueError):
        to_datetime_internal(the_past, False)


def test_to_datetime__string_to_cftime():
    present_day = "2000-01-01"
    datetime = to_datetime_internal(present_day, True)
    assert datetime == cftime.DatetimeProlepticGregorian(2000, 1, 1, 0, 0, 0, 0)

    the_past = "1000-01-01"
    datetime_past = to_datetime_internal(the_past, True)
    assert datetime_past == cftime.DatetimeProlepticGregorian(1000, 1, 1, 0, 0, 0, 0)


def test_to_datetime__cftime_to_cftime():
    present_day = cftime.DatetimeProlepticGregorian(2000, 1, 1, 0, 0, 0, 0)
    datetime = to_datetime_internal(present_day, False)
    assert datetime == cftime.DatetimeProlepticGregorian(2000, 1, 1, 0, 0, 0, 0)

    the_past = cftime.DatetimeProlepticGregorian(1000, 1, 1, 0, 0, 0, 0)
    datetime_past = to_datetime_internal(the_past, False)
    assert datetime_past == cftime.DatetimeProlepticGregorian(1000, 1, 1, 0, 0, 0, 0)


def test_to_datetime__datetime64_to_datetime64():
    # test if days converted to nanoseconds
    present_day = np.datetime64("2000-01-01", "D")
    datetime = to_datetime_internal(present_day, False)
    assert datetime == np.datetime64("2000-01-01", "ns")

    the_past = np.datetime64("1000-01-01", "D")
    with pytest.raises(ValueError):
        to_datetime_internal(the_past, False)


def test_forcing__month_day():
    package_times = pd.date_range("2000-01-01", "2000-03-31", freq="MS")
    globaltimes = pd.date_range("2000-01-01", "2000-03-31")

    starts_ends = forcing_starts_ends(
        package_times=package_times, globaltimes=globaltimes
    )
    assert starts_ends == ["1:31", "32:60", "61:91"]


def test_forcing_partial__month_day():
    package_times = pd.date_range("2000-02-01", "2000-03-31", freq="MS")
    globaltimes = pd.date_range("2000-01-01", "2000-03-31")

    starts_ends = forcing_starts_ends(
        package_times=package_times, globaltimes=globaltimes
    )
    assert starts_ends == ["32:60", "61:91"]


def test_forcing__day_day():
    package_times = pd.date_range("2000-01-01", "2000-03-31")
    globaltimes = pd.date_range("2000-01-01", "2000-03-31")

    starts_ends = forcing_starts_ends(
        package_times=package_times, globaltimes=globaltimes
    )
    assert starts_ends == list(map(str, range(1, 92)))


def test_forcing_partial__day_day():
    package_times = pd.date_range("2000-02-01", "2000-03-31")
    globaltimes = pd.date_range("2000-01-01", "2000-03-31")

    starts_ends = forcing_starts_ends(
        package_times=package_times, globaltimes=globaltimes
    )
    assert starts_ends == list(map(str, range(32, 92)))


def test_forcing__irregular_day():
    datestr = ["2000-01-01", "2000-01-02", "2000-01-05", "2000-01-06"]
    package_times = pd.to_datetime(datestr, format="%Y-%m-%d")
    globaltimes = pd.date_range("2000-01-01", "2000-01-06")

    starts_ends = forcing_starts_ends(
        package_times=package_times, globaltimes=globaltimes
    )
    assert starts_ends == ["1", "2:4", "5", "6"]
