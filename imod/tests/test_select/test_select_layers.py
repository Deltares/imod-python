import numpy as np
import pytest
import xarray as xr

import imod

# seed random generator
np.random.seed(0)


def test_upper_active_layer_ibound():
    x = np.arange(10)
    y = np.arange(10)
    layer = np.arange(1, 6)
    values = np.ones((5, 10, 10))
    active_layer = np.random.randint(low=1, high=6, size=(10, 10))
    coords = {"x": x, "y": y, "layer": layer}
    dims = ["layer", "y", "x"]

    # set to 0 above active layer (in loop to make sure we know what we're doing)
    for i in range(10):
        for j in range(10):
            li = active_layer[i, j] - 1  # layer is 1-based
            values[:li, i, j] = 0

    da_test = xr.DataArray(data=values, coords=coords, dims=dims, name="test")

    active_layer_test = imod.select.upper_active_layer(da_test, is_ibound=True)

    # assert
    np.testing.assert_equal(active_layer_test, active_layer)


def test_upper_active_layer_ibound_some_inactive():
    x = np.arange(10)
    y = np.arange(10)
    layer = np.arange(1, 6)
    values = np.ones((5, 10, 10))
    active_layer = np.random.randint(low=1, high=6, size=(10, 10))
    coords = {"x": x, "y": y, "layer": layer}
    dims = ["layer", "y", "x"]

    # set to 0 above active layer (in loop to make sure we know what we're doing)
    for i in range(10):
        for j in range(10):
            li = active_layer[i, j] - 1  # layer is 1-based
            values[:li, i, j] = 0

    # all layers 0 in ibound, nan in active_layer
    active_layer = active_layer.astype(float)
    active_layer[3, 3] = np.nan
    values[:, 3, 3] = 0

    da_test = xr.DataArray(data=values, coords=coords, dims=dims, name="test")

    active_layer_test = imod.select.upper_active_layer(da_test, is_ibound=True)

    # assert
    np.testing.assert_equal(active_layer_test, active_layer)


def test_upper_active_layer_ibound_all_inactive():
    x = np.arange(10)
    y = np.arange(10)
    layer = np.arange(1, 6)
    values = np.zeros((5, 10, 10))
    active_layer = np.empty((10, 10))
    active_layer[...] = np.nan
    coords = {"x": x, "y": y, "layer": layer}
    dims = ["layer", "y", "x"]

    da_test = xr.DataArray(data=values, coords=coords, dims=dims, name="test")

    active_layer_test = imod.select.upper_active_layer(da_test, is_ibound=True)

    # assert
    np.testing.assert_equal(active_layer_test, active_layer)


def test_upper_active_layer_values():
    x = np.arange(10)
    y = np.arange(10)
    layer = np.arange(1, 6)
    values = np.random.random((5, 10, 10))
    values[:, 0, 0] = 0  # introduce some zeros
    active_layer = np.random.randint(low=1, high=6, size=(10, 10))
    coords = {"x": x, "y": y, "layer": layer}
    dims = ["layer", "y", "x"]

    # set to nan above active layer (in loop to make sure we know what we're doing)
    for i in range(10):
        for j in range(10):
            li = active_layer[i, j] - 1  # layer is 1-based
            values[:li, i, j] = np.nan

    da_test = xr.DataArray(data=values, coords=coords, dims=dims, name="test")

    active_layer_test = imod.select.upper_active_layer(da_test, is_ibound=False)

    # assert
    np.testing.assert_equal(active_layer_test, active_layer)


def test_upper_active_layer_not_an_ibound():
    x = np.arange(10)
    y = np.arange(10)
    layer = np.arange(1, 6)
    values = np.random.random((5, 10, 10))
    # TODO:? active_layer = np.random.randint(low=1, high=6, size=(10, 10))
    coords = {"x": x, "y": y, "layer": layer}
    dims = ["layer", "y", "x"]

    da_test = xr.DataArray(data=values, coords=coords, dims=dims, name="test")

    # assert
    with pytest.raises(ValueError):
        assert imod.select.upper_active_layer(da_test, is_ibound=True)
