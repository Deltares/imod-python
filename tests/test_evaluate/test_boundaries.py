import numpy as np
import xarray as xr

import imod


def test_interpolate_value_boundaries():
    data = np.array([[[0.5], [1.2]], [[1.0], [0.5]], [[1.2], [3.0]]])
    z = np.array([[[0.5], [0.5]], [[-1.0], [-1.0]], [[-5.0], [-5.0]]])
    result = np.array([[[-1.0], [0.5]], [[np.nan], [0.071428571]], [[np.nan], [-1.8]]])
    coords = {"layer": [1, 2, 3], "y": [0.5, 1.5], "x": [0.5]}
    dims = ("layer", "y", "x")
    coords2 = {"boundary": [1, 2, 3], "y": [0.5, 1.5], "x": [0.5]}
    dims2 = ("boundary", "y", "x")

    da = xr.DataArray(data, coords, dims)
    z = xr.DataArray(z, coords, dims)
    result = xr.DataArray(result, coords2, dims2)
    bnds = imod.evaluate.interpolate_value_boundaries(da, z, 1.0)
    assert result.round(5).equals(bnds.round(5))


def test_interpolate_value_boundaries2():
    # up- and downward boundary on same z-coord
    data = np.array([[[0.5], [1.2]], [[1.0], [0.5]], [[0.8], [3.0]]])
    z = np.array([[[0.5], [0.5]], [[-1.0], [-1.0]], [[-5.0], [-5.0]]])
    result = np.array([[[-1.0], [0.5]], [[-1.0], [0.071428571]], [[np.nan], [-1.8]]])
    coords = {"layer": [1, 2, 3], "y": [0.5, 1.5], "x": [0.5]}
    dims = ("layer", "y", "x")
    coords2 = {"boundary": [1, 2, 3], "y": [0.5, 1.5], "x": [0.5]}
    dims2 = ("boundary", "y", "x")

    da = xr.DataArray(data, coords, dims)
    z = xr.DataArray(z, coords, dims)
    result = xr.DataArray(result, coords2, dims2)
    bnds = imod.evaluate.interpolate_value_boundaries(da, z, 1.0)
    assert result.round(5).equals(bnds.round(5))


def test_interpolate_value_boundaries_nan():
    # up- and downward boundary on same z-coord
    data = np.array([[[0.5], [np.nan]], [[np.nan], [0.5]], [[0.8], [3.0]]])
    z = np.array([[[0.5], [0.5]], [[-1.0], [-1.0]], [[-5.0], [-5.0]]])
    result = np.array([[[-1.0], [0.5]], [[-1.0], [0.071428571]], [[np.nan], [-1.8]]])
    coords = {"layer": [1, 2, 3], "y": [0.5, 1.5], "x": [0.5]}
    dims = ("layer", "y", "x")
    coords2 = {"boundary": [1, 2, 3], "y": [0.5, 1.5], "x": [0.5]}
    dims2 = ("boundary", "y", "x")

    da = xr.DataArray(data, coords, dims)
    z = xr.DataArray(z, coords, dims)
    result = xr.DataArray(result, coords2, dims2)
    bnds = imod.evaluate.interpolate_value_boundaries(da, z, 1.0)
    assert result.round(5).equals(bnds.round(5))
