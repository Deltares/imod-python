from pathlib import Path

import numpy as np
import xarray as xr

import imod
from imod.tests.fixtures.mf6_modelrun_fixture import assert_simulation_can_run


def test_regrid_transport(
    tmp_path: Path,
    flow_transport_simulation: imod.mf6.Modflow6Simulation,
):
    assert_simulation_can_run(flow_transport_simulation, "dis", tmp_path/"original")
    domain = flow_transport_simulation["flow"].domain
    dx = domain.coords["dx"].values[()]
    dy = domain.coords["dy"].values[()]
    x_min = domain.coords["x"].values[0] - dx/2
    x_max = domain.coords["x"].values[-1] + dx/2
    y_min = domain.coords["y"].values[-1] - dy/2
    y_max = domain.coords["y"].values[0] + dy/2
    nlayer = domain.coords["layer"][-1]

    cellsize_x = (x_max - x_min) / 200
    cellsize_y = (y_max -y_min) / 3

    x = np.arange(x_min, x_max, cellsize_x) + cellsize_x/2
    y = np.arange(y_max, y_min, -cellsize_y) - cellsize_y/2

    finer_idomain = xr.DataArray(dims=["layer", "y", "x"], coords={"layer": np.arange(nlayer)+1, "y": y, "x": x, "dx": cellsize_x, "dy": cellsize_y})
    finer_idomain.values[:,:,:] =1



    new_simulation = flow_transport_simulation.regrid_like(
        "regridded_simulation", finer_idomain
    )

    # Test that the newly regridded simulation can run
    assert_simulation_can_run(new_simulation, "dis", tmp_path/"regridded")

