from pathlib import Path

import numpy as np
import xarray as xr

import imod
from imod.tests.fixtures.mf6_modelrun_fixture import assert_simulation_can_run


def test_regrid_transport(
    tmp_path: Path,
    flow_transport_simulation: imod.mf6.Modflow6Simulation,
):
    domain = flow_transport_simulation["flow"].domain
    x_min = domain.coords["x"].values[0]
    x_max = domain.coords["x"].values[-1]
    y_min = domain.coords["y"].values[-1]
    y_max = domain.coords["y"].values[0]
    nlayer = domain.coords["layer"][-1]

    cellsize_x = (x_max - x_min) / 3
    cellsize_y = (y_max -y_min) / 200

    x = np.arange(x_min, x_max, cellsize_x)
    y = np.arange(y_max, y_min, -cellsize_y)

    finer_idomain = xr.DataArray(dims=["layer", "y", "x"], coords={"layer": np.arange(nlayer)+1, "y": y, "x": x, "dx": cellsize_x, "dy": cellsize_y})
    finer_idomain.values[:,:,:] =1



    new_simulation = flow_transport_simulation.regrid_like(
        "regridded_simulation", finer_idomain
    )

    # Test that the newly regridded simulation can run
    assert_simulation_can_run(new_simulation, "dis", tmp_path)