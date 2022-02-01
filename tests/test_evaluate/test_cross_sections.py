import numpy as np
import xarray as xr

import imod


def test_streamfunction():
    data = np.zeros((2, 3, 3))
    coords = {
        "layer": [1, 2],
        "y": [0.5, 1.5, 2.5],
        "x": [0.5, 1.5, 2.5],
        "dx": 1.0,
        "dy": 1.0,
    }
    dims = ("layer", "y", "x")
    da = xr.DataArray(data, coords, dims)
    front = xr.full_like(da, 1.0)
    right = xr.full_like(da, 2.0)
    # TODO: lower = xr.full_like(da, 3.0)

    # ref
    data = np.zeros((2, 2))
    coords = {"layer": [1, 2], "s": [0.5, 1.5]}
    dims = ("layer", "s")
    da = xr.DataArray(data, coords, dims)
    sf_y_ref = da.copy(data=[[2.0, 2.0], [1.0, 1.0]])
    sf_x_ref = da.copy(data=[[-4.0, -4.0], [-2.0, -2.0]])
    sf_angle_ref = np.array([-1.0 / 2.0**0.5] * 2)
    sf_angle_ref = da.copy(data=[2 * sf_angle_ref, sf_angle_ref])
    # TODO: v_ref = xr.full_like(da, -3.0)

    # test different planes:
    sf = imod.evaluate.streamfunction_line(right, front, (1, 0), (1, 2))
    np.testing.assert_allclose(sf, sf_y_ref)

    sf = imod.evaluate.streamfunction_line(right, front, (0, 0), (2, 2))
    np.testing.assert_allclose(sf, sf_angle_ref)

    sf = imod.evaluate.streamfunction_line(right, front, (0, 1), (2, 1))
    np.testing.assert_allclose(sf, sf_x_ref)


def test_quiver():
    data = np.zeros((2, 3, 3))
    coords = {
        "layer": [1, 2],
        "y": [0.5, 1.5, 2.5],
        "x": [0.5, 1.5, 2.5],
        "dx": 1.0,
        "dy": 1.0,
    }
    dims = ("layer", "y", "x")
    da = xr.DataArray(data, coords, dims)
    front = xr.full_like(da, 1.0)
    right = xr.full_like(da, 2.0)
    lower = xr.full_like(da, 3.0)

    # ref
    data = np.zeros((2, 1))
    coords = {"layer": [1, 2], "s": [0.5]}
    dims = ("layer", "s")
    da = xr.DataArray(data, coords, dims)
    u_y_ref = xr.full_like(da, 1.0)
    u_x_ref = xr.full_like(da, -2.0)
    u_angle_ref = xr.full_like(da, -1.0 / 2.0**0.5)
    v_ref = xr.full_like(da, 3.0)

    # test different planes:
    u, v = imod.evaluate.quiver_line(right, front, lower, (1, 1), (1, 2))
    np.testing.assert_allclose(u, u_y_ref)
    np.testing.assert_allclose(v, v_ref)

    u, v = imod.evaluate.quiver_line(right, front, lower, (1, 1), (2, 2))
    np.testing.assert_allclose(u, u_angle_ref)
    np.testing.assert_allclose(v, v_ref)

    u, v = imod.evaluate.quiver_line(right, front, lower, (1, 1), (2, 1))
    np.testing.assert_allclose(u, u_x_ref)
    np.testing.assert_allclose(v, v_ref)
