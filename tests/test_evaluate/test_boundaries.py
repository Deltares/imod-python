import numpy as np
import xarray as xr

import imod


def test_interpolate_value_boundaries_layers():
    data = np.array([[[0.5], [1.2]], [[1.0], [0.5]], [[1.2], [3.0]]])
    z = np.array([[[0.5], [0.5]], [[-1.0], [-1.0]], [[-5.0], [-5.0]]])
    dz = np.array([[[1.0], [1.0]], [[2.0], [2.0]], [[6.0], [6.0]]])
    coords = {"layer": [1, 2, 3], "y": [0.5, 1.5], "x": [0.5]}
    dims = ("layer", "y", "x")
    da = xr.DataArray(data, coords, dims)
    z = xr.DataArray(z, coords, dims)
    z = z.assign_coords({"dz": (("layer", "y", "x"), dz)})

    exc_ref = np.array([[[-1.0], [1.0]], [[np.nan], [-1.8]]])
    fallb_ref = np.array([[[np.nan], [0.071428571]]])
    coords2 = {"boundary": [0, 1], "y": [0.5, 1.5], "x": [0.5]}
    dims2 = ("boundary", "y", "x")
    exc_ref = xr.DataArray(exc_ref, coords2, dims2)
    coords2["boundary"] = [0]
    fallb_ref = xr.DataArray(fallb_ref, coords2, dims2)

    exc, fallb = imod.evaluate.interpolate_value_boundaries(da, z, 1.0)

    assert exc_ref.round(5).equals(exc.round(5))
    assert fallb_ref.round(5).equals(fallb.round(5))


def test_interpolate_value_boundaries_layers_dz1d():
    data = np.array([[[0.5], [1.2]], [[1.0], [0.5]], [[1.2], [3.0]]])
    z = np.array([[[0.5], [0.5]], [[-1.0], [-1.0]], [[-5.0], [-5.0]]])
    dz = np.array([1.0, 2.0, 6.0])
    coords = {"layer": [1, 2, 3], "y": [0.5, 1.5], "x": [0.5]}
    dims = ("layer", "y", "x")
    da = xr.DataArray(data, coords, dims)
    z = xr.DataArray(z, coords, dims)
    z = z.assign_coords({"dz": ("layer", dz)})

    exc_ref = np.array([[[-1.0], [1.0]], [[np.nan], [-1.8]]])
    fallb_ref = np.array([[[np.nan], [0.071428571]]])
    coords2 = {"boundary": [0, 1], "y": [0.5, 1.5], "x": [0.5]}
    dims2 = ("boundary", "y", "x")
    exc_ref = xr.DataArray(exc_ref, coords2, dims2)
    coords2["boundary"] = [0]
    fallb_ref = xr.DataArray(fallb_ref, coords2, dims2)

    exc, fallb = imod.evaluate.interpolate_value_boundaries(da, z, 1.0)

    assert exc_ref.round(5).equals(exc.round(5))
    assert fallb_ref.round(5).equals(fallb.round(5))


def test_interpolate_value_boundaries2():
    # up- and downward boundary on same z-coord
    data = np.array([[[0.5], [1.2]], [[1.0], [0.5]], [[0.8], [3.0]]])
    z = np.array([[[0.5], [0.5]], [[-1.0], [-1.0]], [[-5.0], [-5.0]]])
    dz = np.array([[[1.0], [1.0]], [[2.0], [2.0]], [[6.0], [6.0]]])
    coords = {"layer": [1, 2, 3], "y": [0.5, 1.5], "x": [0.5]}
    dims = ("layer", "y", "x")
    da = xr.DataArray(data, coords, dims)
    z = xr.DataArray(z, coords, dims)
    z = z.assign_coords({"dz": (("layer", "y", "x"), dz)})

    exc_ref = np.array([[[-1.0], [1.0]], [[np.nan], [-1.8]]])
    fallb_ref = np.array([[[-1.0], [0.071428571]]])
    coords2 = {"boundary": [0, 1], "y": [0.5, 1.5], "x": [0.5]}
    dims2 = ("boundary", "y", "x")
    exc_ref = xr.DataArray(exc_ref, coords2, dims2)
    coords2["boundary"] = [0]
    fallb_ref = xr.DataArray(fallb_ref, coords2, dims2)

    exc, fallb = imod.evaluate.interpolate_value_boundaries(da, z, 1.0)

    assert exc_ref.round(5).equals(exc.round(5))
    assert fallb_ref.round(5).equals(fallb.round(5))


def test_interpolate_value_boundaries_nan():
    # nan values in data
    data = np.array([[[0.5], [np.nan]], [[np.nan], [0.5]], [[0.8], [3.0]]])
    z = np.array([[[0.5], [0.5]], [[-1.0], [-1.0]], [[-5.0], [-5.0]]])
    dz = np.array([[[1.0], [1.0]], [[2.0], [2.0]], [[6.0], [6.0]]])
    coords = {"layer": [1, 2, 3], "y": [0.5, 1.5], "x": [0.5]}
    dims = ("layer", "y", "x")
    da = xr.DataArray(data, coords, dims)
    z = xr.DataArray(z, coords, dims)
    z = z.assign_coords({"dz": (("layer", "y", "x"), dz)})

    exc_ref = np.array([[[np.nan], [-1.8]]])
    fallb_ref = np.empty((0, 2, 1))
    coords2 = {"boundary": [0], "y": [0.5, 1.5], "x": [0.5]}
    dims2 = ("boundary", "y", "x")
    exc_ref = xr.DataArray(exc_ref, coords2, dims2)
    coords2["boundary"] = []
    fallb_ref = xr.DataArray(fallb_ref, coords2, dims2)

    exc, fallb = imod.evaluate.interpolate_value_boundaries(da, z, 1.0)

    assert exc_ref.round(5).equals(exc.round(5))
    assert fallb_ref.round(5).equals(fallb.round(5))


def test_interpolate_value_boundaries_voxels():
    data = np.array([[[0.5], [1.2]], [[1.0], [0.5]], [[1.2], [3.0]]])
    z = np.array([0.5, -1.0, -5.0])
    dz = np.array([1.0, 2.0, 6.0])
    coords = {"layer": [1, 2, 3], "y": [0.5, 1.5], "x": [0.5]}
    zcoords = {"layer": [1, 2, 3]}
    dims = ("layer", "y", "x")
    zdims = "layer"
    da = xr.DataArray(data, coords, dims)
    z = xr.DataArray(z, zcoords, zdims)
    z = z.assign_coords({"dz": ("layer", dz)})

    exc_ref = np.array([[[-1.0], [1.0]], [[np.nan], [-1.8]]])
    fallb_ref = np.array([[[np.nan], [0.071428571]]])
    coords2 = {"boundary": [0, 1], "y": [0.5, 1.5], "x": [0.5]}
    dims2 = ("boundary", "y", "x")
    exc_ref = xr.DataArray(exc_ref, coords2, dims2)
    coords2["boundary"] = [0]
    fallb_ref = xr.DataArray(fallb_ref, coords2, dims2)

    exc, fallb = imod.evaluate.interpolate_value_boundaries(da, z, 1.0)

    assert exc_ref.round(5).equals(exc.round(5))
    assert fallb_ref.round(5).equals(fallb.round(5))
