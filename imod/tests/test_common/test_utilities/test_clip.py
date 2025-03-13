from unittest.mock import MagicMock

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
import xarray as xr
import xugrid as xu
from shapely.testing import assert_geometries_equal

import imod
from imod.common.interfaces.ipackagebase import IPackageBase
from imod.common.utilities.clip import (
    clip_by_grid,
    clip_repeat_stress,
    clip_time_indexer,
)
from imod.common.utilities.grid import broadcast_to_full_domain
from imod.mf6 import HorizontalFlowBarrierResistance


@pytest.fixture(scope="function")
def horizontal_flow_barrier():
    ztop = -5.0
    zbottom = -135.0

    barrier_y = [70.0, 40.0, 0.0]
    barrier_x = [120.0, 40.0, 0.0]

    geometry = gpd.GeoDataFrame(
        geometry=[shapely.linestrings(barrier_x, barrier_y)],
        data={
            "resistance": [1e3],
            "ztop": [ztop],
            "zbottom": [zbottom],
        },
    )
    return HorizontalFlowBarrierResistance(geometry)


def test_clip_by_grid_convex_grid(basic_dis):
    # Arrange
    x_min = 35.0
    y_min = 55.0

    idomain, top, bottom = basic_dis
    top, bottom = broadcast_to_full_domain(idomain, top, bottom)
    pkg = imod.mf6.StructuredDiscretization(top.sel(layer=1), bottom, idomain)

    active = idomain.sel(layer=1, drop=True)
    active = active.where((active.x > x_min) & (active.y > y_min), -1)

    # Act
    clipped_pkg = clip_by_grid(pkg, active)

    # Assert
    assert pkg.dataset.data_vars.keys() == clipped_pkg.dataset.data_vars.keys()
    assert clipped_pkg.dataset.x.min() > x_min
    assert clipped_pkg.dataset.y.min() > y_min

    expected_idomain_shape = active.where(active > 0, 0, drop=True).shape
    assert clipped_pkg.dataset["idomain"].sel(layer=1).shape == expected_idomain_shape


def test_clip_by_grid_concave_grid(basic_dis):
    # Arrange
    x_start_cut = 35.0
    y_start_cut = 55.0

    idomain, top, bottom = basic_dis
    top, bottom = broadcast_to_full_domain(idomain, top, bottom)
    pkg = imod.mf6.StructuredDiscretization(top.sel(layer=1), bottom, idomain)

    active = idomain.sel(layer=1, drop=True)
    active = active.where((active.x > x_start_cut) & (active.y > y_start_cut), -1)
    active = active * -1
    active = active.where(active > 0, 0)

    # Act
    clipped_pkg = clip_by_grid(pkg, active)

    # Assert
    assert pkg.dataset.data_vars.keys() == clipped_pkg.dataset.data_vars.keys()

    expected_idomain = active.where(active > 0, 0, drop=True)
    expected_idomain_shape = expected_idomain.shape
    assert clipped_pkg.dataset["idomain"].sel(layer=1).shape == expected_idomain_shape
    assert (
        clipped_pkg.dataset["idomain"].sel(layer=1, drop=True) == expected_idomain
    ).all()


def test_clip_by_grid_unstructured_grid(basic_unstructured_dis):
    # Arrange
    idomain, top, bottom = basic_unstructured_dis
    top, bottom = broadcast_to_full_domain(idomain, top, bottom)
    pkg = imod.mf6.VerticesDiscretization(top.sel(layer=1), bottom, idomain)

    active = idomain.sel(layer=1, drop=True)
    active = active.where(active.grid.face_x > 0, -1)

    # Act
    clipped_pkg = clip_by_grid(pkg, active)

    # Assert
    assert pkg.dataset.data_vars.keys() == clipped_pkg.dataset.data_vars.keys()

    clipped_active_cells = clipped_pkg.dataset["idomain"].sel(layer=1).count()
    expected_active_cells = active.where(active > 0).count()

    assert clipped_active_cells == expected_active_cells


def test_clip_by_grid_wrong_grid_type():
    # Arrange
    pkg = MagicMock(spec_set=IPackageBase)
    active = "wrong type"

    # Act/Assert
    with pytest.raises(TypeError):
        _ = clip_by_grid(pkg, active)


def test_clip_by_grid_with_line_data_package__structured(
    basic_dis, horizontal_flow_barrier
):
    # Arrange
    idomain, _, _ = basic_dis
    active = idomain.sel(layer=1, drop=True)

    # Act
    hfb_clipped = clip_by_grid(horizontal_flow_barrier, active)

    # Assert
    with pytest.raises(AssertionError):
        assert_geometries_equal(
            hfb_clipped["geometry"].values.item(),
            horizontal_flow_barrier["geometry"].values.item(),
        )

    x, y = hfb_clipped["geometry"].values.item().xy
    np.testing.assert_allclose(x, np.array([90.0, 40.0, 0.0]))
    np.testing.assert_allclose(y, np.array([58.75, 40.0, 0.0]))


def test_clip_by_grid_with_line_data_package__unstructured(
    basic_dis, horizontal_flow_barrier
):
    # Arrange
    idomain, _, _ = basic_dis
    active = idomain.sel(layer=1, drop=True)
    active_uda = xu.UgridDataArray.from_structured2d(active)

    # Act
    hfb_clipped = clip_by_grid(horizontal_flow_barrier, active_uda)

    # Assert
    with pytest.raises(AssertionError):
        assert_geometries_equal(
            hfb_clipped["geometry"].values.item(),
            horizontal_flow_barrier["geometry"].values.item(),
        )

    x, y = hfb_clipped["geometry"].values.item().xy
    np.testing.assert_allclose(x, np.array([90.0, 40.0, 0.0]))
    np.testing.assert_allclose(y, np.array([58.75, 40.0, 0.0]))


def test_clip_by_grid__structured_grid_full(
    basic_dis, well_high_lvl_test_data_stationary
):
    """All wells are included within the structured grid bounds"""
    # Arrange
    idomain, _, _ = basic_dis
    active = idomain.sel(layer=1, drop=True)
    wel = imod.mf6.Well(*well_high_lvl_test_data_stationary, print_flows=True)

    # Act
    wel_clipped = clip_by_grid(wel, active)

    # Assert
    assert isinstance(wel_clipped, imod.mf6.Well)
    assert wel_clipped.dataset["rate"].shape == wel.dataset["rate"].shape
    # Test if options are copied
    assert wel_clipped.dataset["print_flows"] == wel.dataset["print_flows"]


def test_clip_by_grid__structured_grid_clipped(
    basic_dis, well_high_lvl_test_data_stationary
):
    """Half of the wells are included within the structured grid bounds"""
    # Arrange
    idomain, _, _ = basic_dis
    active = idomain.sel(layer=1, drop=True)
    wel = imod.mf6.Well(*well_high_lvl_test_data_stationary, print_flows=True)
    # Clip grid so that xmax is set to 70.0 instead of 90.0
    active_selected = active.where(idomain.x < 70.0, -1)

    # Act
    wel_clipped = clip_by_grid(wel, active_selected)

    # Assert
    assert isinstance(wel_clipped, imod.mf6.Well)
    assert wel_clipped.dataset["rate"].shape == (4,)
    # Test if options are copied
    assert wel_clipped.dataset["print_flows"] == wel.dataset["print_flows"]


def test_clip_by_grid__unstructured_grid_full(
    basic_dis, well_high_lvl_test_data_stationary
):
    """All the wells are included within the unstructured grid bounds"""
    # Arrange
    idomain, _, _ = basic_dis
    active = idomain.sel(layer=1, drop=True)
    wel = imod.mf6.Well(*well_high_lvl_test_data_stationary, print_flows=True)
    active_ugrid = xu.UgridDataArray.from_structured2d(active)

    # Act
    wel_clipped = clip_by_grid(wel, active_ugrid)

    # Assert
    assert isinstance(wel_clipped, imod.mf6.Well)
    assert wel_clipped.dataset["rate"].shape == wel.dataset["rate"].shape
    # Test if options are copied
    assert wel_clipped.dataset["print_flows"] == wel.dataset["print_flows"]


def test_clip_by_grid__unstructured_grid_clipped(
    basic_dis, well_high_lvl_test_data_stationary
):
    """Half of the wells are included within the unstructured grid bounds"""
    # Arrange
    idomain, _, _ = basic_dis
    active = idomain.sel(layer=1, drop=True)
    wel = imod.mf6.Well(*well_high_lvl_test_data_stationary, print_flows=True)
    # Clip grid so that xmax is set to 70.0 instead of 90.0
    active_selected = active.sel(x=slice(None, 70.0))
    active_ugrid = xu.UgridDataArray.from_structured2d(active_selected)

    # Act
    wel_clipped = clip_by_grid(wel, active_ugrid)

    # Assert
    assert isinstance(wel_clipped, imod.mf6.Well)
    assert wel_clipped.dataset["rate"].shape == (4,)
    # Test if options are copied
    assert wel_clipped.dataset["print_flows"] == wel.dataset["print_flows"]


def test_clip_by_grid_contains_non_grid_data_variables(basic_dis):
    # Arrange
    x_min = 35.0
    y_min = 55.0

    idomain, _, _ = basic_dis
    k = xr.full_like(idomain, 1.0, dtype=float)

    pkg = imod.mf6.NodePropertyFlow(
        k=k,
        icelltype=0,
        variable_vertical_conductance=True,
        dewatered=True,
        perched=True,
        save_flows=True,
    )

    active = idomain.sel(layer=1, drop=True)
    active = active.where((active.x > x_min) & (active.y > y_min), -1)

    # Act
    clipped_pkg = clip_by_grid(pkg, active)

    # Assert
    assert pkg.dataset["icelltype"] == clipped_pkg.dataset["icelltype"]
    assert (
        pkg.dataset["variable_vertical_conductance"]
        == clipped_pkg.dataset["variable_vertical_conductance"]
    )
    assert pkg.dataset["dewatered"] == clipped_pkg.dataset["dewatered"]
    assert pkg.dataset["perched"] == clipped_pkg.dataset["perched"]
    assert pkg.dataset["save_flows"] == clipped_pkg.dataset["save_flows"]


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
    time_start = imod.util.time.to_datetime_internal("2000-01-01", False)
    time_end = imod.util.time.to_datetime_internal("2000-12-01", False)
    indexer = clip_time_indexer(
        time=time,
        time_start=time_start,
        time_end=time_end,
    )
    expected = create_expected(np.arange(12), time)
    assert indexer.equals(expected)


def test_clip_time_indexer__after(dataset):
    time = dataset["time"].values
    time_start = imod.util.time.to_datetime_internal("2001-01-01", False)
    time_end = imod.util.time.to_datetime_internal("2001-12-01", False)
    indexer = clip_time_indexer(
        time=time,
        time_start=time_start,
        time_end=time_end,
    )
    expected = create_expected([11], [time_start])
    assert indexer.equals(expected)


def test_clip_time_indexer__before(dataset):
    time = dataset["time"].values
    time_start = imod.util.time.to_datetime_internal("1999-01-01", False)
    time_end = imod.util.time.to_datetime_internal("1999-12-01", False)
    indexer = clip_time_indexer(
        time=time,
        time_start=time_start,
        time_end=time_end,
    )
    expected = create_expected([0], [time_start])
    assert indexer.equals(expected)


def test_clip_time_indexer__within(dataset):
    time = dataset["time"].values
    time_start = imod.util.time.to_datetime_internal("2000-01-15", False)
    time_end = imod.util.time.to_datetime_internal("2000-03-01", False)
    indexer = clip_time_indexer(
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
    time_start = imod.util.time.to_datetime_internal("2005-01-01", False)
    time_end = imod.util.time.to_datetime_internal("2008-12-01", False)

    indexer = clip_time_indexer(
        time=dataset["time"].values,
        time_start=time_start,
        time_end=time_end,
    )
    repeat_indexer, repeat_stress = clip_repeat_stress(
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
    # Assert
    np.testing.assert_array_equal(
        actual.coords["time"].dt.year, np.repeat([2005], repeats=12)
    )
    np.testing.assert_array_equal(actual.coords["time"].dt.month, np.arange(1, 13))
    np.testing.assert_array_equal(actual.coords["time"].dt.day, np.ones(12))
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
    time_start = imod.util.time.to_datetime_internal("2005-01-01", False)
    time_end = imod.util.time.to_datetime_internal("2008-12-01", False)

    indexer = clip_time_indexer(
        time=dataset["time"].values,
        time_start=time_start,
        time_end=time_end,
    )
    repeat_indexer, repeat_stress = clip_repeat_stress(
        repeat_stress=dataset["repeat_stress"],
        time=time,
        time_start=time_start,
        time_end=time_end,
    )
    indexer = repeat_indexer.combine_first(indexer).astype(int)

    keys = repeat_stress.loc[:, 0]
    values = repeat_stress.loc[:, 1]
    actual = dataset.drop_vars("time").isel(time=indexer)
    # Assert
    np.testing.assert_array_equal(
        actual.coords["time"].dt.year, np.repeat([2005], repeats=15)
    )
    np.testing.assert_array_equal(
        actual.coords["time"].dt.month,
        np.array([1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    )
    np.testing.assert_array_equal(
        actual.coords["time"].dt.day,
        np.array([1, 15, 1, 15, 1, 15, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
    )
    assert np.array_equal(
        actual["multiplier"], [1, 13, 2, 14, 3, 15, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    )
    assert np.array_equal(keys.dt.year, np.repeat([2006, 2007, 2008], repeats=12))
    assert np.array_equal(keys.dt.month, values.dt.month)
    assert np.isin(values, actual["time"]).all()


def test_clip_repeat_stress__no_end(dataset2):
    dataset = dataset2.copy()
    keys = pd.date_range("2001-01-01", "2009-12-01", freq="MS")
    values = np.tile(dataset["time"][:12], reps=9)
    dataset["repeat_stress"] = xr.DataArray(
        data=np.column_stack((keys, values)),
        dims=("repeat", "repeat_items"),
    )

    time = dataset["time"].values
    time_start = imod.util.time.to_datetime_internal("2005-01-01", False)
    time_end = None

    indexer = clip_time_indexer(
        time=dataset["time"].values,
        time_start=time_start,
        time_end=time_end,
    )
    repeat_indexer, repeat_stress = clip_repeat_stress(
        repeat_stress=dataset["repeat_stress"],
        time=time,
        time_start=time_start,
        time_end=time_end,
    )
    indexer = repeat_indexer.combine_first(indexer).astype(int)

    keys = repeat_stress.loc[:, 0]
    values = repeat_stress.loc[:, 1]
    actual = dataset.drop_vars("time").isel(time=indexer)
    np.testing.assert_array_equal(
        actual.coords["time"].dt.year, np.repeat([2005], repeats=15)
    )
    np.testing.assert_array_equal(
        actual.coords["time"].dt.month,
        np.array([1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    )
    np.testing.assert_array_equal(
        actual.coords["time"].dt.day,
        np.array([1, 15, 1, 15, 1, 15, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
    )
    assert np.array_equal(
        actual["multiplier"], [1, 13, 2, 14, 3, 15, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    )
    assert np.array_equal(keys.dt.year, np.repeat([2006, 2007, 2008, 2009], repeats=12))
    assert np.array_equal(keys.dt.month, values.dt.month)
    assert np.isin(values, actual["time"]).all()


def test_clip_repeat_stress__no_start_no_end(dataset2):
    dataset = dataset2.copy()
    keys = pd.date_range("2001-01-01", "2009-12-01", freq="MS")
    values = np.tile(dataset["time"][:12], reps=9)
    dataset["repeat_stress"] = xr.DataArray(
        data=np.column_stack((keys, values)),
        dims=("repeat", "repeat_items"),
    )

    time = dataset["time"].values
    time_start = None
    time_end = None

    indexer = clip_time_indexer(
        time=dataset["time"].values,
        time_start=time_start,
        time_end=time_end,
    )
    repeat_indexer, repeat_stress = clip_repeat_stress(
        repeat_stress=dataset["repeat_stress"],
        time=time,
        time_start=time_start,
        time_end=time_end,
    )
    indexer = repeat_indexer.combine_first(indexer).astype(int)

    keys = repeat_stress.loc[:, 0]
    values = repeat_stress.loc[:, 1]
    actual = dataset.drop_vars("time").isel(time=indexer)
    # Assert
    assert dataset.coords["time"].equals(actual.coords["time"])
    assert np.array_equal(actual["multiplier"], np.arange(1, 16))
    assert np.array_equal(keys.dt.year, np.repeat(np.arange(9) + 2001, repeats=12))
    assert np.array_equal(keys.dt.month, values.dt.month)
    assert np.isin(values, actual["time"]).all()
