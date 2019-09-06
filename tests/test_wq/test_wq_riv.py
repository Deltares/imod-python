import pathlib
import textwrap

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from imod.wq import River


@pytest.fixture(scope="function")
def river():
    layer = np.arange(1, 4)
    y = np.arange(4.5, 0.0, -1.0)
    x = np.arange(0.5, 5.0, 1.0)
    stage = xr.DataArray(
        np.full((3, 5, 5), 1.0),
        coords={"layer": layer, "y": y, "x": x, "dx": 1.0, "dy": -1.0},
        dims=("layer", "y", "x"),
    )

    riv = River(
        stage=stage,
        conductance=stage.copy(),
        bottom_elevation=stage.copy(),
        concentration=stage.copy(),
        density=stage.copy(),
    )
    return riv


def test_render(river):
    riv = river
    directory = pathlib.Path(".")

    compare = """
    stage_p?_s1_l1 = stage_l1.idf
    stage_p?_s1_l2 = stage_l2.idf
    stage_p?_s1_l3 = stage_l3.idf
    cond_p?_s1_l1 = conductance_l1.idf
    cond_p?_s1_l2 = conductance_l2.idf
    cond_p?_s1_l3 = conductance_l3.idf
    rbot_p?_s1_l1 = bottom_elevation_l1.idf
    rbot_p?_s1_l2 = bottom_elevation_l2.idf
    rbot_p?_s1_l3 = bottom_elevation_l3.idf
    rivssmdens_p?_s1_l1 = density_l1.idf
    rivssmdens_p?_s1_l2 = density_l2.idf
    rivssmdens_p?_s1_l3 = density_l3.idf"""

    assert riv._render(directory, globaltimes=["?"], system_index=1) == compare


@pytest.mark.parametrize(
    "varname", ["stage", "conductance", "bottom_elevation", "concentration", "density"]
)
def test_render__timemap(river, varname):
    riv = river
    directory = pathlib.Path(".")
    da = riv[varname]
    datetimes = pd.date_range("2000-01-01", "2000-01-03")
    da_transient = xr.concat(
        [da.assign_coords(time=t) for t in datetimes[:-1]], dim="time"
    )
    riv[varname] = da_transient

    timemap = {datetimes[-1]: datetimes[0]}
    riv.add_timemap(**{varname: timemap})
    actual = riv._render(directory, globaltimes=datetimes, system_index=1)
