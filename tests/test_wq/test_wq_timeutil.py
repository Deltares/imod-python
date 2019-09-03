import pandas as pd
import numpy as np
from imod.wq import timeutil


def test_forcing__month_day():
    package_times = pd.date_range("2000-01-01", "2000-03-31", freq="MS")
    globaltimes = pd.date_range("2000-01-01", "2000-03-31")

    starts_ends = timeutil.forcing_starts_ends(
        package_times=package_times, globaltimes=globaltimes
    )
    assert starts_ends == ["1:31", "32:60", "61:91"]


def test_forcing_partial__month_day():
    package_times = pd.date_range("2000-02-01", "2000-03-31", freq="MS")
    globaltimes = pd.date_range("2000-01-01", "2000-03-31")

    starts_ends = timeutil.forcing_starts_ends(
        package_times=package_times, globaltimes=globaltimes
    )
    assert starts_ends == ["32:60", "61:91"]


def test_forcing__day_day():
    package_times = pd.date_range("2000-01-01", "2000-03-31")
    globaltimes = pd.date_range("2000-01-01", "2000-03-31")

    starts_ends = timeutil.forcing_starts_ends(
        package_times=package_times, globaltimes=globaltimes
    )
    assert starts_ends == list(map(str, range(1, 92)))


def test_forcing_partial__day_day():
    package_times = pd.date_range("2000-02-01", "2000-03-31")
    globaltimes = pd.date_range("2000-01-01", "2000-03-31")

    starts_ends = timeutil.forcing_starts_ends(
        package_times=package_times, globaltimes=globaltimes
    )
    assert starts_ends == list(map(str, range(32, 92)))


def test_forcing__irregular_day():
    datestr = ["2000-01-01", "2000-01-02", "2000-01-05", "2000-01-06"]
    package_times = pd.to_datetime(datestr, format="%Y-%m-%d")
    globaltimes = pd.date_range("2000-01-01", "2000-01-06")

    starts_ends = timeutil.forcing_starts_ends(
        package_times=package_times, globaltimes=globaltimes
    )
    assert starts_ends == ["1", "2:4", "5", "6"]
