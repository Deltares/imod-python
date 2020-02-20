import numpy as np
import xarray as xr

import imod


def test_flowlower_up():
    data = np.zeros((3, 2, 1))
    coords = {"layer": [1, 2, 3], "y": [0.5, 1.5], "x": [0.5]}
    dims = ("layer", "y", "x")

    da = xr.DataArray(data, coords, dims)
    front = xr.full_like(da, 0.0)
    right = xr.full_like(da, 0.0)
    lower = xr.full_like(da, 0.0)
    budgetzone = xr.full_like(da, 0.0)

    lower[:2] = 1.0
    budgetzone[:2] = 1
    assert imod.evaluate.facebudget(budgetzone, front, lower, right).sum() == 2.0
    rf, rl, rr = imod.evaluate.facebudget(budgetzone, front, lower, right, False)
    assert rf.sum() == 0.0
    assert rl.sum() == 2.0
    assert rr.sum() == 0.0


def test_flowlower_down():
    data = np.zeros((2, 2, 1))
    coords = {"layer": [1, 2], "y": [0.5, 1.5], "x": [0.5]}
    dims = ("layer", "y", "x")

    da = xr.DataArray(data, coords, dims)
    front = xr.full_like(da, 0.0)
    right = xr.full_like(da, 0.0)
    lower = xr.full_like(da, 0.0)
    budgetzone = xr.full_like(da, 0.0)

    lower[0] = 1.0
    budgetzone[0] = 1
    assert imod.evaluate.facebudget(budgetzone, front, lower, right).sum() == 2.0

    lower[:1] = -1.0
    assert imod.evaluate.facebudget(budgetzone, front, lower, right).sum() == -2.0

    lower = xr.full_like(da, 0.0)
    budgetzone = xr.full_like(da, 0.0)
    lower[0] = 1.0
    budgetzone[1] = 1
    assert imod.evaluate.facebudget(budgetzone, front, lower, right).sum() == -2.0


def test_lower_netzero():
    data = np.zeros((3, 2, 1))
    coords = {"layer": [1, 2, 3], "y": [0.5, 1.5], "x": [0.5]}
    dims = ("layer", "y", "x")

    da = xr.DataArray(data, coords, dims)
    front = xr.full_like(da, 0.0)
    right = xr.full_like(da, 0.0)
    lower = xr.full_like(da, 0.0)
    budgetzone = xr.full_like(da, 0.0)

    lower[:2] = 1.0
    budgetzone[1] = 1
    assert imod.evaluate.facebudget(budgetzone, front, lower, right).sum() == 0.0


def test_lower_right():
    data = np.zeros((2, 2, 1))
    coords = {"layer": [1, 2], "y": [0.5, 1.5], "x": [0.5]}
    dims = ("layer", "y", "x")

    da = xr.DataArray(data, coords, dims)
    front = xr.full_like(da, 0.0)
    right = xr.full_like(da, 0.0)
    lower = xr.full_like(da, 0.0)
    budgetzone = xr.full_like(da, 0.0)

    right[:] = 1.0
    lower[:] = 1.0
    budgetzone[0] = 1
    budgetzone[1, 0] = 1
    assert imod.evaluate.facebudget(budgetzone, front, lower, right).sum() == 2.0


def test_flow_right_lower_netzero():
    data = np.zeros((1, 3, 3))
    coords = {"layer": [1], "y": [0.5, 1.5, 2.5], "x": [0.5, 1.5, 2.5]}
    dims = ("layer", "y", "x")

    da = xr.DataArray(data, coords, dims)
    front = xr.full_like(da, 1.0)
    right = xr.full_like(da, 1.0)
    lower = xr.full_like(da, 1.0)
    budgetzone = xr.full_like(da, 0.0)

    budgetzone[:, 0, :] = 1
    budgetzone[:, 1, 1:] = 1
    budgetzone[:, 2, 2:] = 1
    assert imod.evaluate.facebudget(budgetzone, front, lower, right).sum() == 0.0


def test_flow_right_lower():
    data = np.zeros((1, 3, 3))
    coords = {"layer": [1], "y": [0.5, 1.5, 2.5], "x": [0.5, 1.5, 2.5]}
    dims = ("layer", "y", "x")
    da = xr.DataArray(data, coords, dims)
    front = xr.full_like(da, 1.0)
    right = xr.full_like(da, 1.0)
    lower = xr.full_like(da, 1.0)
    budgetzone = xr.full_like(da, 0.0)

    budgetzone[:, 0, :] = 1
    budgetzone[:, 1, :1] = 1
    budgetzone[:, 2, :2] = 1
    assert imod.evaluate.facebudget(budgetzone, front, lower, right).sum() == 3.0

    # Inverted
    budgetzone = abs(budgetzone - 1)
    assert imod.evaluate.facebudget(budgetzone, front, lower, right).sum() == -3.0
