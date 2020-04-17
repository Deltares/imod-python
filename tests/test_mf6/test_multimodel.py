import numpy as np
import pytest
import xarray as xr

import imod


def test_coord_union():
    x1 = np.array([0.0, 2.0, 3.0])
    x2 = np.array([0.5, 1.5])
    x = imod.mf6.multimodel.coord_union(x1, x2, decreasing=False)
    expected_x = np.array([0.0, 0.5, 1.5, 2.0, 3.0])
    assert np.array_equal(x, expected_x)

    y = imod.mf6.multimodel.coord_union(x1, x2, decreasing=True)
    expected_y = expected_x[::-1]
    assert np.array_equal(y, expected_y)


def test_is_touching():
    data = [0, 0]
    da1 = xr.DataArray(data, {"x": [0.5, 1.5]}, ["x"])
    da2 = xr.DataArray(data, {"x": [2.5, 3.5]}, ["x"])
    actual = imod.mf6.multimodel.is_touching(da1, da2, "x")
    assert actual == True

    da1 = xr.DataArray(data, {"x": [0.5, 1.5]}, ["x"])
    da2 = xr.DataArray(data, {"x": [1.5, 2.5]}, ["x"])
    actual = imod.mf6.multimodel.is_touching(da1, da2, "x")
    assert actual == False

    da1 = xr.DataArray(data, {"x": [0.5, 1.5]}, ["x"])
    da2 = xr.DataArray(data, {"x": [3.5, 4.5]}, ["x"])
    with pytest.raises(ValueError):
        actual = imod.mf6.multimodel.is_touching(da1, da2, "x")


def test_find_exchange():
    a = np.array([[-1, -1, 2]])
    b = np.array([[0, 1, -1]])
    expected1 = np.array([2])
    expected2 = np.array([1])
    actual1, actual2 = imod.mf6.multimodel.find_exchange(a, b)
    assert np.array_equal(actual1, expected1)
    assert np.array_equal(actual2, expected2)

    a = np.array([[0, 1, -1]])
    b = np.array([[-1, -1, 2]])
    expected1 = np.array([1])
    expected2 = np.array([2])
    actual1, actual2 = imod.mf6.multimodel.find_exchange(a, b)
    assert np.array_equal(actual1, expected1)
    assert np.array_equal(actual2, expected2)

    a = np.array([[0, -1, -1]])
    b = np.array([[-1, 1, 2]])
    expected1 = np.array([0])
    expected2 = np.array([1])
    actual1, actual2 = imod.mf6.multimodel.find_exchange(a, b)
    assert np.array_equal(actual1, expected1)
    assert np.array_equal(actual2, expected2)

    a = np.array([[0, 1, -1, -1, 3]])
    b = np.array([[-1, -1, 2, -1, -1]])
    expected1 = np.array([1])
    expected2 = np.array([2])
    actual1, actual2 = imod.mf6.multimodel.find_exchange(a, b)
    assert np.array_equal(actual1, expected1)
    assert np.array_equal(actual2, expected2)
