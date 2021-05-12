import pathlib

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from imod.wq import GeneralHeadBoundary


@pytest.fixture(scope="function")
def headboundary():
    layer = np.arange(1, 4)
    y = np.arange(4.5, 0.0, -1.0)
    x = np.arange(0.5, 5.0, 1.0)
    head = xr.DataArray(
        np.full((3, 5, 5), 1.0),
        coords={"layer": layer, "y": y, "x": x, "dx": 1.0, "dy": -1.0},
        dims=("layer", "y", "x"),
    )

    ghb = GeneralHeadBoundary(
        head=head,
        conductance=head.copy(),
        concentration=head.copy(),
        density=head.copy(),
    )
    return ghb


def test_render(headboundary):
    ghb = headboundary
    directory = pathlib.Path(".")

    compare = """
    bhead_p?_s1_l$ = head_l$.idf
    cond_p?_s1_l$ = conductance_l$.idf
    ghbssmdens_p?_s1_l$ = density_l$.idf"""

    assert (
        ghb._render(directory, globaltimes=["?"], system_index=1, nlayer=3) == compare
    )


@pytest.mark.parametrize("varname", ["head", "conductance", "concentration", "density"])
def test_render__timemap(headboundary, varname):
    ghb = headboundary
    directory = pathlib.Path(".")
    da = ghb[varname]
    datetimes = pd.date_range("2000-01-01", "2000-01-03")
    da_transient = xr.concat(
        [da.assign_coords(time=t) for t in datetimes[:-1]], dim="time"
    )
    ghb[varname] = da_transient

    timemap = {datetimes[-1]: datetimes[0]}
    ghb.repeat_stress(**{varname: timemap})
    actual = ghb._render(directory, globaltimes=datetimes, system_index=1, nlayer=3)
    # TODO check result
