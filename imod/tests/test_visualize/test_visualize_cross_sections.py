import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

import imod
from imod.visualize.cross_sections import _meshcoords


def _voxel_section(x, dx=None):
    coords = {
        "layer": [1, 2],
        "x": x,
        "top": ("layer", [1.0, 0.5]),
        "bottom": ("layer", [0.5, 0.0]),
    }
    if dx is not None:
        coords["dx"] = ("x", dx) if np.ndim(dx) else dx
    return xr.DataArray(np.ones((2, len(x))), coords, ("layer", "x"))


@pytest.fixture(scope="function")
def testda():
    def _testda(equidistant):
        if equidistant:
            coords = {"layer": [1, 2], "x": [1.0, 2.0, 3.0]}
        else:
            dx = np.arange(1.0, 4.0)
            x = dx.cumsum() - 0.5 * dx
            coords = {"layer": [1, 2], "x": x, "dx": ("x", dx)}

        dims = ("layer", "x")
        da = xr.DataArray(np.random.randn(2, 3), coords, dims)
        top = xr.full_like(da, 1.0)
        bottom = xr.full_like(da, 0.0)
        da = da.assign_coords(top=top)
        da = da.assign_coords(bottom=bottom)
        return da

    return _testda


@pytest.mark.parametrize("equidistant", [True, False])
@pytest.mark.parametrize("transposed", [True, False])
@pytest.mark.parametrize("transpose_coords", [True, False])
def test_plot_cross_section(testda, equidistant, transposed, transpose_coords):
    da = testda(equidistant)
    if transposed:
        da = da.transpose("x", "layer", transpose_coords=transpose_coords)
    levels = [0.0, 0.25, 0.50, 0.75, 1.0]
    colors = ["#ffffcc", "#c7e9b4", "#7fcdbb", "#41b6c4", "#2c7fb8", "#253494"]
    fig, ax = imod.visualize.cross_section(
        da, layers=True, levels=levels, colors=colors
    )
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)


def test_meshcoords_nonequidistant_x():
    # Cell widths 1, 2, 3 starting at x=0: edges at 0, 1, 3, 6.
    da = _voxel_section(np.array([0.5, 2.0, 4.5]), np.array([1.0, 2.0, 3.0]))
    X, Y, C, _ = _meshcoords(da)
    expected = [0.0, 1.0, 1.0, 3.0, 3.0, 6.0]
    np.testing.assert_allclose(X[0], expected)
    np.testing.assert_allclose(X[-1], expected)
    assert X.shape == Y.shape == (3, 6)
    assert C.shape == (2, 5)


@pytest.mark.parametrize("with_dx_coord", [True, False])
@pytest.mark.parametrize("ncol", [3, 7, 10, 13])
@pytest.mark.parametrize("dx", [0.01, 0.1, 0.3, 0.7, 1.1])
def test_meshcoords_decimal_cellsize(dx, ncol, with_dx_coord):
    # np.arange with a decimal step could include xmax through float rounding.
    x = np.round(dx * (np.arange(ncol) + 0.5), 10)
    da = _voxel_section(x, dx if with_dx_coord else None)
    X, Y, C, _ = _meshcoords(da)
    edges = dx * np.arange(ncol + 1)
    expected = np.repeat(edges, 2)[1:-1]
    assert X.shape == Y.shape == (3, 2 * ncol)
    assert C.shape == (2, 2 * ncol - 1)
    np.testing.assert_allclose(X[0], expected, atol=1e-9)


def test_plot_cross_section_decimal_cellsize():
    x = np.round(np.arange(0.05, 0.66, 0.1), 10)
    da = _voxel_section(x, 0.1)
    levels = [0.0, 0.25, 0.50, 0.75, 1.0]
    colors = ["#ffffcc", "#c7e9b4", "#7fcdbb", "#41b6c4", "#2c7fb8", "#253494"]
    fig, ax = imod.visualize.cross_section(
        da, layers=True, levels=levels, colors=colors
    )
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)
