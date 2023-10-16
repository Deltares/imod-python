import numpy as np
import pandas as pd
import pytest
import xarray as xr

import imod
from imod.mf6.package import Package
from imod.tests.fixtures.mf6_small_models_fixture import (
    grid_data_structured,
    grid_data_unstructured,
)


@pytest.fixture(scope="function")
def dataset():
    dataset = xr.Dataset()
    dataset["multiplier"] = xr.DataArray(
        data=np.arange(1, 13),
        coords={"time": pd.date_range("2000-01-01", "2000-12-01", freq="MS")},
        dims=("time",),
    )
    return dataset


@pytest.fixture(scope="function")
def dataset2():
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
    return dataset


def create_expected(index, time):
    return xr.DataArray(
        data=index,
        coords={"time": time},
        dims=("time"),
    )


def test_clip_time_indexer__full(dataset):
    time = dataset["time"].values
    time_start = imod.wq.timeutil.to_datetime("2000-01-01", False)
    time_end = imod.wq.timeutil.to_datetime("2000-12-01", False)
    indexer = Package._clip_time_indexer(
        time=time,
        time_start=time_start,
        time_end=time_end,
    )
    expected = create_expected(np.arange(12), time)
    assert indexer.equals(expected)


def test_clip_time_indexer__after(dataset):
    time = dataset["time"].values
    time_start = imod.wq.timeutil.to_datetime("2001-01-01", False)
    time_end = imod.wq.timeutil.to_datetime("2001-12-01", False)
    indexer = Package._clip_time_indexer(
        time=time,
        time_start=time_start,
        time_end=time_end,
    )
    expected = create_expected([11], [time_start])
    assert indexer.equals(expected)


def test_clip_time_indexer__before(dataset):
    time = dataset["time"].values
    time_start = imod.wq.timeutil.to_datetime("1999-01-01", False)
    time_end = imod.wq.timeutil.to_datetime("1999-12-01", False)
    indexer = Package._clip_time_indexer(
        time=time,
        time_start=time_start,
        time_end=time_end,
    )
    expected = create_expected([0], [time_start])
    assert indexer.equals(expected)


def test_clip_time_indexer__within(dataset):
    time = dataset["time"].values
    time_start = imod.wq.timeutil.to_datetime("2000-01-15", False)
    time_end = imod.wq.timeutil.to_datetime("2000-03-01", False)
    indexer = Package._clip_time_indexer(
        time=time,
        time_start=time_start,
        time_end=time_end,
    )
    expected = create_expected([0, 1, 2], [time_start, time[1], time[2]])
    assert indexer.equals(expected)


def test_clip_repeat_stress__all_repeats(dataset):
    dataset = dataset.copy()
    keys = pd.date_range("2001-01-01", "2009-12-01", freq="MS")
    values = np.tile(dataset["time"], reps=9)
    dataset["repeat_stress"] = xr.DataArray(
        data=np.column_stack((keys, values)),
        dims=("repeat", "repeat_items"),
    )

    time = dataset["time"].values
    time_start = imod.wq.timeutil.to_datetime("2005-01-01", False)
    time_end = imod.wq.timeutil.to_datetime("2008-12-01", False)

    indexer = Package._clip_time_indexer(
        time=dataset["time"].values,
        time_start=time_start,
        time_end=time_end,
    )
    repeat_indexer, repeat_stress = Package._clip_repeat_stress(
        repeat_stress=dataset["repeat_stress"],
        time=time,
        time_start=time_start,
        time_end=time_end,
    )
    indexer = repeat_indexer.combine_first(indexer).astype(int)

    keys = repeat_stress.loc[:, 0]
    values = repeat_stress.loc[:, 1]
    actual = dataset.drop_vars("time").isel(time=indexer)
    assert actual["time"].size == 12
    assert np.array_equal(keys.dt.year, np.repeat([2006, 2007, 2008], repeats=12))
    assert np.array_equal(keys.dt.month, values.dt.month)
    assert np.isin(values, actual["time"]).all()


def test_clip_repeat_stress__some_repeats(dataset2):
    dataset = dataset2.copy()
    keys = pd.date_range("2001-01-01", "2009-12-01", freq="MS")
    values = np.tile(dataset["time"][:12], reps=9)
    dataset["repeat_stress"] = xr.DataArray(
        data=np.column_stack((keys, values)),
        dims=("repeat", "repeat_items"),
    )

    time = dataset["time"].values
    time_start = imod.wq.timeutil.to_datetime("2005-01-01", False)
    time_end = imod.wq.timeutil.to_datetime("2008-12-01", False)

    indexer = Package._clip_time_indexer(
        time=dataset["time"].values,
        time_start=time_start,
        time_end=time_end,
    )
    repeat_indexer, repeat_stress = Package._clip_repeat_stress(
        repeat_stress=dataset["repeat_stress"],
        time=time,
        time_start=time_start,
        time_end=time_end,
    )
    indexer = repeat_indexer.combine_first(indexer).astype(int)

    keys = repeat_stress.loc[:, 0]
    values = repeat_stress.loc[:, 1]
    actual = dataset.drop_vars("time").isel(time=indexer)
    assert np.array_equal(
        actual["multiplier"], [1, 13, 2, 14, 3, 15, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    )
    assert np.array_equal(keys.dt.year, np.repeat([2006, 2007, 2008], repeats=12))
    assert np.array_equal(keys.dt.month, values.dt.month)
    assert np.isin(values, actual["time"]).all()


def test_mask_structured():
    head = grid_data_structured(np.float64, 2.1, 2.0)
    pkg = imod.mf6.ConstantHead(head=head)
    mask = grid_data_structured(np.int32, 1, 2.0)
    inactive_cell_location = {"layer": 1, "x": 2.0, "y": 4.0}
    mask.loc[inactive_cell_location] = 0

    masked_package = pkg.mask(mask)

    masked_head = masked_package.dataset["head"]
    assert type(masked_head) is type(head)
    assert masked_head.dtype == head.dtype
    assert np.isnan(masked_head.sel(inactive_cell_location).values[()])
    masked_head.loc[inactive_cell_location] = 2.1
    assert (masked_head == head).all().values[()]


def test_mask_unstructured():
    head = grid_data_unstructured(np.float64, 2.1, 2.0)
    pkg = imod.mf6.ConstantHead(head=head)
    mask = grid_data_unstructured(np.int32, 1, 2.0)
    inactive_cell_location = {"layer": 1, "mesh2d_nFaces": 23}
    mask.loc[inactive_cell_location] = 0

    masked_package = pkg.mask(mask)

    masked_head = masked_package.dataset["head"]
    assert type(masked_head) is type(head)
    assert masked_head.dtype == head.dtype
    assert np.isnan(masked_head.sel(inactive_cell_location).values[()])
    masked_head.loc[inactive_cell_location] = 2.1
    assert (masked_head == head).all().values[()]


def test_mask_scalar_input():
    # Create a storage package with scalar input
    storage_pack = imod.mf6.SpecificStorage(
        specific_storage=1.0e-5,
        specific_yield=0.15,
        transient=False,
        convertible=0,
    )
    mask = grid_data_unstructured(np.int32, 1, 2.0)
    inactive_cell_location = {"layer": 1, "mesh2d_nFaces": 23}
    mask.loc[inactive_cell_location] = 0

    masked_package = storage_pack.mask(mask)
    ss = masked_package["specific_storage"]
    assert np.isscalar(ss.values[()])


def test_mask_layered_input():
    # Create a npf package with scalar input
    model_layers = np.array([1, 2, 3])
    k = xr.DataArray([1.0e-3, 1.0e-4, 2.0e-4], {"layer": model_layers}, ("layer",))
    icelltype = xr.DataArray([1, 0, 0], {"layer": model_layers}, ("layer",))
    npf_pack = imod.mf6.NodePropertyFlow(
        icelltype=icelltype,
        k=k,
        variable_vertical_conductance=True,
        dewatered=True,
        perched=False,
        save_flows=True,
    )

    # Create a mask
    mask = grid_data_unstructured(np.int32, 1, 2.0)
    inactive_cell_location = {"layer": 1, "mesh2d_nFaces": 23}
    mask.loc[inactive_cell_location] = 0

    # Apply the mask
    masked_package = npf_pack.mask(mask)

    # Check layered array intact after masking
    assert (masked_package["k"] == k).all()
