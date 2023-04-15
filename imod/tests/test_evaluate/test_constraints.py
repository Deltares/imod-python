import numpy as np
import pandas as pd
import pytest
import xarray as xr

import imod
from imod.testing import assert_frame_equal


@pytest.fixture(scope="module")
def test_da():
    data = np.ones((3, 2, 2))
    coords = {"layer": [1, 2, 3], "y": [0.5, 1.5], "x": [0.5, 1.5], "dx": 1.0, "dy": -1}
    dims = ("layer", "y", "x")
    da = xr.DataArray(data, coords, dims)
    return da


@pytest.fixture(scope="module")
def test_da1():
    data = np.ones((1, 1, 1))
    coords = {"layer": [1], "y": [0.5], "x": [0.5], "dx": 1.0, "dy": -1}
    dims = ("layer", "y", "x")
    da = xr.DataArray(data, coords, dims)
    return da


def test_stability_constraint_wel(test_da):
    top = (test_da.T * np.array([0.0, -1.0, -3.0])).T
    bottom = (test_da.T * np.array([-1.0, -3.0, -4.0])).T
    top_bot = xr.Dataset({"top": top, "bot": bottom})
    wel = pd.DataFrame(
        [[0.6, 0.6, 1, 1.0], [0.6, 0.6, 2, 1.0]], columns=["x", "y", "layer", "Q"]
    )
    wel_ref = pd.DataFrame(
        [[0.6, 0.6, 1, 1.0, 1.0, 0.3], [0.6, 0.6, 2, 1.0, 0.5, 0.6]],
        columns=["x", "y", "layer", "Q", "qs", "dt"],
    )

    wel_dt = imod.evaluate.stability_constraint_wel(wel, top_bot, porosity=0.3, R=1.0)

    assert_frame_equal(wel_ref, wel_dt)


def test_stability_constraint_advection(test_da):
    front = test_da * 2.0
    lower = (test_da.T * np.array([1.0, 1.0, 0.0])).T
    right = test_da
    top = (test_da.T * np.array([0.0, -1.0, -3.0])).T
    bottom = (test_da.T * np.array([-1.0, -3.0, -4.0])).T
    top_bot = xr.Dataset({"top": top, "bot": bottom})

    dt, dt_xyz = imod.evaluate.stability_constraint_advection(
        front, lower, right, top_bot
    )

    dt_x = (test_da.T * np.array([0.3, 0.6, 0.3])).T.assign_coords({"direction": "x"})
    dt_y = (test_da.T * np.array([0.15, 0.3, 0.15])).T.assign_coords({"direction": "y"})
    dt_z = (test_da.T * np.array([0.3, 0.6, np.nan])).T.assign_coords(
        {"direction": "z"}
    )
    dtref = (1 / (1 / dt_x + 1 / dt_y + 1 / dt_z)).drop_vars("direction")

    assert dtref.round(5).equals(dt.round(5))
    assert dt_x.round(5).equals(dt_xyz.sel(direction="x").round(5))
    assert dt_y.round(5).equals(dt_xyz.sel(direction="y").round(5))
    assert dt_z.round(5).equals(dt_xyz.sel(direction="z").round(5))


def test_intra_cell_boundary_conditions(test_da1):
    top_bot = xr.Dataset({"top": test_da1 * 0.0, "bot": test_da1 * -1.0})
    riv1 = xr.Dataset(
        {
            "stage": test_da1 * 1.0,
            "conductance": test_da1 * 100.0,
            "bottom_elevation": test_da1 * 0.0,
        }
    )
    riv2 = xr.Dataset(
        {
            "stage": test_da1 * 1.0,
            "conductance": test_da1 * 100.0,
            "bottom_elevation": test_da1 * 1.0,
        }
    )
    ghb = xr.Dataset({"head": test_da1 * 1.0, "conductance": test_da1 * 200.0})
    drn = xr.Dataset({"elevation": test_da1 * 0.0, "conductance": test_da1 * 150.0})

    dt_min, dt_all = imod.evaluate.intra_cell_boundary_conditions(
        top_bot, porosity=0.3, riv=[riv1, riv2], ghb=ghb, drn=drn, drop_allnan=True
    )

    ghbdrn = test_da1 * (0.3 * 1.0) / min(200.0, 150) * (1.0 - 0.0)
    riv1drn = test_da1 * (0.3 * 1.0) / min(100.0, 150) * (1.0 - 0.0)
    dt_min_ref = np.minimum(ghbdrn, riv1drn)

    assert dt_min_ref.equals(dt_min)
    assert dt_all.equals(
        xr.concat(
            (ghbdrn, riv1drn), pd.Index(["ghb-drn", "riv_0-drn"], name="combination")
        )
    )


def test_intra_cell_boundary_conditions_thickness_zero(test_da1):
    top_bot = xr.Dataset({"top": test_da1 * -1.0, "bot": test_da1 * -1.0})
    riv1 = xr.Dataset(
        {
            "stage": test_da1 * 1.0,
            "conductance": test_da1 * 100.0,
            "bottom_elevation": test_da1 * 0.0,
        }
    )
    drn = xr.Dataset({"elevation": test_da1 * 0.0, "conductance": test_da1 * 150.0})

    with pytest.raises(ValueError):
        _ = imod.evaluate.intra_cell_boundary_conditions(
            top_bot, porosity=0.3, riv=[riv1], drn=drn, drop_allnan=True
        )
