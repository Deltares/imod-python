import numpy as np
import pandas as pd
import pytest
import xarray as xr

import imod
from imod import util
from imod.visualize.pyvista import velocity_field


@pytest.fixture(scope="module")
def test_da():
    nrow, ncol = 3, 4
    dx, dy = 1.0, -1.0
    xmin, xmax = 0.0, 4.0
    ymin, ymax = 0.0, 3.0
    coords = util.spatial._xycoords((xmin, xmax, ymin, ymax), (dx, dy))
    kwargs = {"name": "test", "coords": coords, "dims": ("y", "x")}
    data = np.ones((nrow, ncol))
    da = xr.DataArray(data, **kwargs)
    return da


@pytest.fixture(scope="module")
def test_zda():
    nlay, nrow, ncol = 5, 3, 4
    dx, dy = 1.0, -1.0
    xmin, xmax = 0.0, 4.0
    ymin, ymax = 0.0, 3.0
    coords = util.spatial._xycoords((xmin, xmax, ymin, ymax), (dx, dy))
    coords["z"] = np.arange(5.0) + 0.5
    kwargs = {"name": "z", "coords": coords, "dims": ("z", "y", "x")}
    data = np.ones((nlay, nrow, ncol), dtype=np.float32)
    da = xr.DataArray(data, **kwargs)
    return da


@pytest.fixture(scope="module")
def test_3dzda():
    nlay, nrow, ncol = 5, 3, 4
    dx, dy = 1.0, -1.0
    xmin, xmax = 0.0, 4.0
    ymin, ymax = 0.0, 3.0
    coords = util.spatial._xycoords((xmin, xmax, ymin, ymax), (dx, dy))
    coords["layer"] = np.arange(nlay) + 1

    ncell = nlay * nrow * ncol
    scaled_addition = np.arange(1.0, 1.0 + ncell, 1.0) / ncell
    dz = (1.0 + scaled_addition).reshape((nlay, nrow, ncol))
    top = np.full((nlay, nrow, ncol), 0.0)
    top -= dz.cumsum(axis=0)
    bottom = top - dz

    dims = ("layer", "y", "x")
    coords["top"] = (dims, top)
    coords["bottom"] = (dims, bottom)
    data = np.ones((nlay, nrow, ncol))
    da = xr.DataArray(data, coords, dims)
    return da


def test_grid3_plane(tmp_path, test_da):
    da = test_da
    g = imod.visualize.grid_3d(da)
    assert g.bounds
    # g.plot(screenshot=tmp_path / "plane.png", off_screen=True)


def test_grid3d_z(tmp_path, test_zda):
    da = test_zda
    g = imod.visualize.grid_3d(da)
    assert g.bounds
    # g.plot(screenshot=tmp_path / "z.png", off_screen=True)


def test_grid3d_z3d(tmp_path, test_3dzda):
    da = test_3dzda
    g = imod.visualize.grid_3d(da)
    assert g.bounds
    # g.plot(screenshot=tmp_path / "z3d.png", off_screen=True)


def _layered_da(top, bottom):
    shape = (len(top), 2, 3)
    dims = ("layer", "y", "x")
    coords = {
        "layer": np.arange(1, len(top) + 1),
        "y": np.array([102.5, 101.5]),
        "x": np.array([0.5, 1.5, 2.5]),
        "top": (dims, np.ones(shape) * np.reshape(top, (-1, 1, 1))),
        "bottom": (dims, np.ones(shape) * np.reshape(bottom, (-1, 1, 1))),
    }
    return xr.DataArray(np.ones(shape), coords, dims)


def test_grid3d_z3d_model_top():
    da = _layered_da(top=[10.0, 5.0], bottom=[5.0, 0.0])
    g = imod.visualize.grid_3d(da, vertical_exaggeration=1.0)
    assert g.bounds[4] == 0.0
    assert g.bounds[5] == 10.0


def test_grid3d_z3d_single_layer():
    da = _layered_da(top=[10.0], bottom=[0.0])
    g = imod.visualize.grid_3d(da, vertical_exaggeration=1.0)
    assert g.n_cells == 6
    assert g.bounds[4] == 0.0
    assert g.bounds[5] == 10.0


def test_velocity_field_z3d():
    # Non-square grid, with x and y in distinct ranges to detect a swap.
    v = _layered_da(top=[0.0, -1.0], bottom=[-1.0, -2.0])
    z = v.copy(data=np.stack([np.full((2, 3), -1.0), np.full((2, 3), -2.0)]))
    z_before = z.values.copy()
    glyphs = velocity_field(
        v,
        v,
        v,
        z=z,
        vertical_exaggeration=30.0,
        scale_by_magnitude=False,
        factor=1.0e-3,
    )
    # With tiny arrows, the glyph bounds are the bounds of the arrow bases.
    xmin, xmax, ymin, ymax, zmin, zmax = glyphs.bounds
    np.testing.assert_allclose([xmin, xmax], [0.5, 2.5], atol=0.01)
    np.testing.assert_allclose([ymin, ymax], [101.5, 102.5], atol=0.01)
    np.testing.assert_allclose([zmin, zmax], [-60.0, -30.0], atol=0.01)
    np.testing.assert_array_equal(z.values, z_before)


def _transient_layered_da():
    da = _layered_da(top=[0.0, -1.0], bottom=[-1.0, -2.0])
    time = pd.date_range("2000-01-01", periods=2)
    values = np.stack([np.full(da.shape, 1.0), np.full(da.shape, 2.0)])
    return xr.DataArray(
        values,
        coords={"time": time, **da.coords},
        dims=("time", "layer", "y", "x"),
    )


def _has_mesh_actor(animation):
    return any(
        getattr(getattr(actor, "mapper", None), "dataset", None) is animation.mesh
        for actor in animation.plotter.renderer.actors.values()
    )


def test_static_grid_animation_3d_mesh_added():
    da = _transient_layered_da()
    animation = imod.visualize.StaticGridAnimation3D(
        da, plotter_kwargs={"off_screen": True}
    )
    assert _has_mesh_actor(animation)
    np.testing.assert_array_equal(animation.mesh.cell_data["values"], 1.0)

    animation._update(da.isel(time=1))
    np.testing.assert_array_equal(animation.mesh.cell_data["values"], 2.0)

    animation.reset()
    assert _has_mesh_actor(animation)
    np.testing.assert_array_equal(animation.mesh.cell_data["values"], 1.0)
    animation.plotter.close()


def test_grid_animation_3d_reset():
    da = _transient_layered_da()
    animation = imod.visualize.GridAnimation3D(da, plotter_kwargs={"off_screen": True})
    assert _has_mesh_actor(animation)
    animation._update(da.isel(time=1))
    assert _has_mesh_actor(animation)
    np.testing.assert_array_equal(animation.mesh.cell_data["values"], 2.0)
    animation.reset()
    assert _has_mesh_actor(animation)
    np.testing.assert_array_equal(animation.mesh.cell_data["values"], 1.0)
    animation.plotter.close()


@pytest.mark.skip(reason="CI isn't cooperating")
def test_grid_animation_3d(tmp_path, test_zda):
    # Can't test show functions right now...
    # Due to OpenGL issues on CI
    da = test_zda
    das = [da.assign_coords(time=i) for i in range(3)]
    timeda = xr.concat(das, dim="time")
    animation = imod.visualize.GridAnimation3D(timeda)
    # Make a random change
    animation.plotter.camera_position = [0.0, 0.0, 0.0]
    animation.reset()


@pytest.mark.skip(reason="CI isn't cooperating")
def test_static_grid_animation_3d(tmp_path, test_zda):
    # Can't test show functions right now...
    # Due to OpenGL issues on CI
    da = test_zda
    das = [da.assign_coords(time=i) for i in range(3)]
    timeda = xr.concat(das, dim="time")
    animation = imod.visualize.StaticGridAnimation3D(timeda)
    # Make a random change
    animation.plotter.camera_position = [0.0, 0.0, 0.0]
    animation.reset()
