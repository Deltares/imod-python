import pathlib
import textwrap
import xarray as xr
import numpy as np
import pytest
import imod

@pytest.fixture(scope="function")
def grid_array():
    nlay = 3
    nrow = 15
    ncol = 15
    shape = (nlay, nrow, ncol)

    dx = 5000.0
    dy = -5000.0
    xmin = 0.0
    xmax = dx * ncol
    ymin = 0.0
    ymax = abs(dy) * nrow
    dims = ("layer", "y", "x")

    layer = np.array([1, 2, 3])
    y = np.arange(ymax, ymin, dy) + 0.5 * dy
    x = np.arange(xmin, xmax, dx) + 0.5 * dx
    coords = {"layer": layer, "y": y, "x": x}
    idomain = xr.DataArray(np.ones(shape, dtype=np.int8), coords=coords, dims=dims)

    return idomain

def test_ist_simple(grid_array):
    grid = grid_array
    immobile_porosity = xr.full_like(grid, dtype=np.float64, fill_value=0.1)
    mobile_immobile_mass_transfer_rate =  xr.full_like(grid, dtype=np.float64, fill_value=88.0)
    ist = imod.mf6.ImmobileStorage(immobile_porosity=immobile_porosity, mobile_immobile_mass_transfer_rate=mobile_immobile_mass_transfer_rate)
    directory = pathlib.Path("mymodel")
    globaltimes = [np.datetime64("2000-01-01")]
    actual=ist.render(directory,"ist", globaltimes, False)
    expected  = textwrap.dedent(
        """\
        begin options
          CIM FILEOUT cim.dat
          PRINT_FORMAT COLUMNS  WIDTH 10 DIGITS 7 EXPONENTIAL
        end options

        begin griddata
          thetaim
            open/close mymodel/ist/thetaim.dat
          zetaim
            open/close mymodel/ist/zetaim.dat
        end griddata"""
    )
    assert actual == expected
