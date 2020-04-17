import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

import imod


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
