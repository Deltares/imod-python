import numpy as np
import pandas as pd
import xarray as xr

import imod
from imod.mf6.pkgbase import Package


def test_slice_repeat_stress__all_repeats():
    dataset = xr.Dataset()
    dataset["multiplier"] = xr.DataArray(
        data=np.arange(1, 13),
        coords={"time": pd.date_range("2000-01-01", "2000-12-01", freq="MS")},
        dims=("time",),
    )
    keys = pd.date_range("2001-01-01", "2009-12-01", freq="MS")
    values = np.tile(dataset["time"], reps=9)

    dataset["repeat_stress"] = xr.DataArray(
        data=np.column_stack((keys, values)),
        dims=("repeat", "repeat_items"),
    )

    time_start = imod.wq.timeutil.to_datetime("2005-01-01", False)
    time_end = imod.wq.timeutil.to_datetime("2008-12-01", False)
    selection = dataset.sel(time=slice(time_start, time_end))

    actual = Package._slice_repeat_stress(
        dataset=dataset,
        selection=selection,
        time_start=time_start,
        time_end=time_end,
    )
    keys = actual["repeat_stress"].loc[:, 0]
    values = actual["repeat_stress"].loc[:, 1]
    assert actual["time"].size == 12
    assert np.array_equal(keys.dt.year, np.repeat([2006, 2007, 2008], repeats=12))
    assert np.array_equal(keys.dt.month, values.dt.month)
    assert np.isin(values, actual["time"]).all()


def test_slice_repeat_stress__some_repeats():
    dataset = xr.Dataset()
    time = np.concatenate(
        [
            pd.date_range("2000-01-01", "2000-12-01", freq="MS").values,
            np.array(["2005-01-15", "2005-02-15", "2005-03-15"], dtype=np.datetime64),
        ]
    )

    dataset["multiplier"] = xr.DataArray(
        data=np.arange(1, 16),
        coords={"time": time},
        dims=("time",),
    )
    keys = pd.date_range("2001-01-01", "2009-12-01", freq="MS")
    values = np.tile(dataset["time"][:12], reps=9)

    dataset["repeat_stress"] = xr.DataArray(
        data=np.column_stack((keys, values)),
        dims=("repeat", "repeat_items"),
    )

    time_start = imod.wq.timeutil.to_datetime("2005-01-01", False)
    time_end = imod.wq.timeutil.to_datetime("2008-12-01", False)
    selection = dataset.sel(time=slice(time_start, time_end))

    actual = Package._slice_repeat_stress(
        dataset=dataset,
        selection=selection,
        time_start=time_start,
        time_end=time_end,
    )
    keys = actual["repeat_stress"].loc[:, 0]
    values = actual["repeat_stress"].loc[:, 1]
    assert actual["time"].size == 15
    assert np.array_equal(
        actual["multiplier"], [1, 13, 2, 14, 3, 15, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    )
    assert np.array_equal(keys.dt.year, np.repeat([2006, 2007, 2008], repeats=12))
    assert np.array_equal(keys.dt.month, values.dt.month)
    assert np.isin(values, actual["time"]).all()
