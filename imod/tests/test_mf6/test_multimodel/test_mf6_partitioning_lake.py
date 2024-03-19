import pytest
import xarray as xr
import numpy as np

from imod.typing.grid import zeros_like
def finer_grid(grid):
    xmin = grid.coords["x"].min().values[()]
    xmax = grid.coords["x"].max().values[()]
    ymin = grid.coords["y"].min().values[()]
    ymax = grid.coords["y"].max().values[()]
    steps_x = len(grid.coords["x"])
    steps_y = len(grid.coords["y"])   
    layers = grid.coords["layer"]
    new_x = np.arange(xmin, xmax, (xmax-xmin)/(steps_x*2))
    new_y =  np.arange(ymax, ymin, -(ymax-ymin)/(steps_y*2))
    return xr.DataArray(dims=["layer", "y","x"], coords ={"layer":layers, "y": new_y, "x": new_x})

@pytest.mark.usefixtures("rectangle_with_lakes")
def test_mf6_simulation_partition_with_lakes(rectangle_with_lakes, tmp_path):
    simulation = rectangle_with_lakes


    label_array = zeros_like(simulation["GWF_1"].domain.sel(layer=1))
    label_array.values[10:, 10:] = 1

    with pytest.raises(ValueError, match="simulation(.+)lake(.+)GWF_1"):
        _ = simulation.split(label_array)

@pytest.mark.usefixtures("rectangle_with_lakes")
def test_mf6_simulation_regrid_with_lakes(rectangle_with_lakes, tmp_path):
    simulation = rectangle_with_lakes

    new_grid = finer_grid(simulation["GWF_1"].domain)

    with pytest.raises(ValueError, match="simulation(.+)lake(.+)GWF_1"):
        _ = simulation.regrid_like("regridded_simulation", new_grid, True)

@pytest.mark.usefixtures("rectangle_with_lakes")
def test_mf6_model_regrid_with_lakes(rectangle_with_lakes, tmp_path):
    simulation = rectangle_with_lakes
    model = simulation["GWF_1"]
    new_grid = finer_grid(simulation["GWF_1"].domain)

    with pytest.raises(ValueError, match="model(.+)lake"):
        _ = model.regrid_like( new_grid, True)


@pytest.mark.usefixtures("rectangle_with_lakes")
def test_mf6_package_regrid_with_lakes(rectangle_with_lakes, tmp_path):
    simulation = rectangle_with_lakes
    package = simulation["GWF_1"]["lake"]
    new_grid = finer_grid(simulation["GWF_1"].domain)

    with pytest.raises(ValueError, match="package(.+)not be regridded"):
        _ = package.regrid_like( new_grid)