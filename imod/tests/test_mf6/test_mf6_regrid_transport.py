from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
import xarray as xr

import imod


def create_regridding_idomain(domain: xr.DataArray , ncol: int, nrow: int):
    #Create an equidistant structured grid with the same horizontal and vertical
    #extent as the input grid, and the same number of layer, but a different
    #amount of rows and columns
    dx = domain.coords["dx"].values[()]
    dy = domain.coords["dy"].values[()]
    x_min = domain.coords["x"].values[0] - dx/2
    x_max = domain.coords["x"].values[-1] + dx/2
    y_min = domain.coords["y"].values[-1] - dy/2
    y_max = domain.coords["y"].values[0] + dy/2
    nlayer = domain.coords["layer"][-1]

    column_size = (x_max - x_min) / ncol
    row_size = (y_max -y_min) / nrow

    x = np.arange(x_min, x_max, column_size) + column_size/2
    y = np.arange(y_max, y_min, -row_size) - row_size/2

    new_idomain = xr.DataArray(dims=["layer", "y", "x"], coords={"layer": np.arange(nlayer)+1, "y": y, "x": x, "dx": column_size, "dy": row_size})
    new_idomain.values[:,:,:] =1

    return new_idomain

@pytest.mark.parametrize("col_row_dimension",[(101,4 ), (55,3 ), (155,4 )])
def test_regrid_transport(
    tmp_path: Path,
    flow_transport_simulation: imod.mf6.Modflow6Simulation,
    col_row_dimension: Tuple[int, int]
):
    # Run the original simulation
    flow_transport_simulation.write( tmp_path/"original", binary=False)
    flow_transport_simulation.run()

    #Set up the regridded domain. The original domain is 101 columns * 2 rows * 1 layer
    domain = flow_transport_simulation["flow"].domain
    other_idomain = create_regridding_idomain(domain, col_row_dimension[0], col_row_dimension[1])

    new_simulation = flow_transport_simulation.regrid_like(
        "regridded_simulation", other_idomain
    )

    # Test that the newly regridded simulation can run
    new_simulation.write( tmp_path/"regridded", binary=False)
    new_simulation.run()

    # simulation results
    conc = flow_transport_simulation.open_concentration(["species_a", "species_b", "species_c", "species_d"])
    regridded_conc = new_simulation.open_concentration(["species_a", "species_b", "species_c", "species_d"])

    dx = domain.coords["dx"].values[()]
    dy = domain.coords["dy"].values[()]
    cell_volume = dx * dy * 1

    new_dx = other_idomain.coords["dx"].values[()]
    new_dy = other_idomain.coords["dy"].values[()]
    regridded_cell_volume = new_dx * new_dy

    conc = conc.where(conc > -1e29, 0)
    regridded_conc = regridded_conc.where(regridded_conc > -1e29, 0)

    #Compute and compare the total mass of the original model and the regridded mode.
    #Constants like porosity are not included. 
    for species in ["species_a", "species_b", "species_c", "species_d"]: 
        original_mass = np.sum(conc.sel(time=2000, species =species).values * cell_volume)
        regridded_mass = np.sum(regridded_conc.sel(time=2000, species =species ).values * regridded_cell_volume)
        assert abs((original_mass  - regridded_mass))/original_mass < 3e-2






