import numpy as np
import pytest
import xarray as xr

from imod.typing.grid import zeros_like
from imod.util.regrid import RegridderWeightsCache


def finer_grid(grid):
    xmin = grid.coords["x"].min().values[()]
    xmax = grid.coords["x"].max().values[()]
    ymin = grid.coords["y"].min().values[()]
    ymax = grid.coords["y"].max().values[()]
    steps_x = len(grid.coords["x"])
    steps_y = len(grid.coords["y"])
    layers = grid.coords["layer"]
    new_x = np.arange(xmin, xmax, (xmax - xmin) / (steps_x * 2))
    new_y = np.arange(ymax, ymin, -(ymax - ymin) / (steps_y * 2))
    return xr.DataArray(
        dims=["layer", "y", "x"], coords={"layer": layers, "y": new_y, "x": new_x}
    )


def test_mf6_simulation_partition_with_lakes(rectangle_with_lakes, tmp_path):
    simulation = rectangle_with_lakes

    label_array = zeros_like(simulation["GWF_1"].domain.sel(layer=1))
    label_array.values[10:, 10:] = 1

    with pytest.raises(ValueError, match="simulation(.+)lake(.+)GWF_1"):
        _ = simulation.split(label_array)


def test_mf6_simulation_regrid_with_lakes(rectangle_with_lakes, tmp_path):
    simulation = rectangle_with_lakes

    new_grid = finer_grid(simulation["GWF_1"].domain)

    with pytest.raises(ValueError, match="simulation(.+)lake(.+)GWF_1"):
        _ = simulation.regrid_like("regridded_simulation", new_grid, True)


def test_mf6_model_regrid_with_lakes(rectangle_with_lakes, tmp_path):
    simulation = rectangle_with_lakes
    model = simulation["GWF_1"]
    new_grid = finer_grid(simulation["GWF_1"].domain)

    with pytest.raises(ValueError, match="model(.+)lake"):
        _ = model.regrid_like(new_grid, True)


def test_mf6_package_regrid_with_lakes(rectangle_with_lakes, tmp_path):
    simulation = rectangle_with_lakes
    package = simulation["GWF_1"]["lake"]
    new_grid = finer_grid(simulation["GWF_1"].domain)
    regrid_cache = RegridderWeightsCache()
    with pytest.raises(ValueError, match="package(.+)not be regridded"):
        _ = package.regrid_like(new_grid, regrid_cache)


def test_mf6_simulation_clip_with_lakes(rectangle_with_lakes, tmp_path):
    simulation = rectangle_with_lakes

    with pytest.raises(ValueError, match="simulation(.+)clipped(.*)lake"):
        _ = simulation.clip_box(x_min=200, y_min=200, x_max=1000, y_max=1000)


def test_mf6_model_clip_with_lakes(rectangle_with_lakes, tmp_path):
    model = rectangle_with_lakes["GWF_1"]

    with pytest.raises(ValueError, match="model(.+)clipped(.*)lake"):
        _ = model.clip_box(x_min=200, y_min=200, x_max=1000, y_max=1000)


def test_mf6_package_clip_with_lakes(rectangle_with_lakes, tmp_path):
    simulation = rectangle_with_lakes
    package = simulation["GWF_1"]["lake"]

    with pytest.raises(ValueError, match="package(.+)clipping"):
        _ = package.clip_box(x_min=200, y_min=200, x_max=1000, y_max=1000)
