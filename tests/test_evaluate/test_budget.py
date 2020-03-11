import numpy as np
import pandas as pd
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
    budgetzone = xr.full_like(da, np.nan)

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
    budgetzone = xr.full_like(da, np.nan)

    lower[0] = 1.0
    budgetzone[0] = 1
    assert imod.evaluate.facebudget(budgetzone, front, lower, right).sum() == 2.0

    lower[:1] = -1.0
    assert imod.evaluate.facebudget(budgetzone, front, lower, right).sum() == -2.0

    lower = xr.full_like(da, 0.0)
    budgetzone = xr.full_like(da, np.nan)
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
    budgetzone = xr.full_like(da, np.nan)

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
    budgetzone = xr.full_like(da, np.nan)

    right[:] = 1.0
    lower[:] = 1.0
    budgetzone[0] = 1
    budgetzone[1, 0] = 1
    assert imod.evaluate.facebudget(budgetzone, front, lower, right).sum() == 1.0


def test_flow_right_lower_netzero():
    data = np.zeros((1, 3, 3))
    coords = {"layer": [1], "y": [0.5, 1.5, 2.5], "x": [0.5, 1.5, 2.5]}
    dims = ("layer", "y", "x")

    da = xr.DataArray(data, coords, dims)
    front = xr.full_like(da, 1.0)
    right = xr.full_like(da, 1.0)
    lower = xr.full_like(da, 1.0)
    budgetzone = xr.full_like(da, np.nan)

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
    right = xr.full_like(da, 5.0)
    lower = xr.full_like(da, 1.0)
    budgetzone = xr.full_like(da, np.nan)

    budgetzone[:, 0, :] = 1
    budgetzone[:, 1, :1] = 1
    budgetzone[:, 2, :2] = 1
    assert (
        float(imod.evaluate.facebudget(budgetzone, front, lower, right).sum()) == 11.0
    )

    # Inverted
    budgetzone = xr.full_like(budgetzone, 1.0).where(budgetzone.isnull())
    assert (
        float(imod.evaluate.facebudget(budgetzone, front, lower, right).sum()) == -11.0
    )

    # Test two zones
    budgetzone = xr.full_like(da, np.nan)
    budgetzone[:, 0, :] = 1
    budgetzone[:, 1, :1] = 1
    budgetzone[:, 2, :2] = 1
    budgetzone = budgetzone.fillna(2)
    netflow = imod.evaluate.facebudget(budgetzone, front, lower, right)
    assert float(netflow.sum()) == 0.0
    assert float(netflow.where(budgetzone == 1).sum()) == 11.0
    assert float(netflow.where(budgetzone == 2).sum()) == -11.0


def test_flow_right_lower__aniflow():
    data = np.zeros((2, 3, 3))
    coords = {"layer": [1, 2], "y": [0.5, 1.5, 2.5], "x": [0.5, 1.5, 2.5]}
    dims = ("layer", "y", "x")
    da = xr.DataArray(data, coords, dims)
    front = xr.full_like(da, 1.0)
    right = xr.full_like(da, 2.0)
    lower = xr.full_like(da, 3.0)
    budgetzone = xr.full_like(da, np.nan)

    budgetzone[0, 1, 1] = 1
    assert float(imod.evaluate.facebudget(budgetzone, front, lower, right).sum()) == 3.0

    # reset
    budgetzone[...] = np.nan
    budgetzone[:, :, -1] = 1
    assert float(imod.evaluate.facebudget(budgetzone, front, lower, right).sum()) == -(
        2 * 3 * 2.0
    )

    # reset
    budgetzone[...] = np.nan
    budgetzone[:, -1, :] = 1
    assert float(imod.evaluate.facebudget(budgetzone, front, lower, right).sum()) == -(
        2 * 3 * 1.0
    )


def test_transient_flow():
    data = np.zeros((4, 3, 2, 1))
    coords = {
        "time": pd.date_range("2000-01-01", "2000-01-04"),
        "layer": [1, 2, 3],
        "y": [0.5, 1.5],
        "x": [0.5],
    }
    dims = ("time", "layer", "y", "x")

    da = xr.DataArray(data, coords, dims)
    front = xr.full_like(da, 0.0)
    right = xr.full_like(da, 0.0)
    lower = xr.full_like(da, 0.0)
    budgetzone = xr.DataArray(
        data=np.zeros((3, 2, 1)),
        coords={"layer": [1, 2, 3], "y": [0.5, 1.5], "x": [0.5]},
        dims=("layer", "y", "x"),
    )
    netflow = imod.evaluate.facebudget(budgetzone, front, lower, right)
    assert netflow.shape == (4, 3, 2, 1)
    assert float(netflow.sum()) == 0.0
