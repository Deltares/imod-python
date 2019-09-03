import pathlib

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from imod.wq import GeneralHeadBoundary


@pytest.fixture(scope="function")
def headboundary(request):
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

    compare = (
        "\n"
        "    bhead_p?_s1_l1 = head_l1.idf\n"
        "    bhead_p?_s1_l2 = head_l2.idf\n"
        "    bhead_p?_s1_l3 = head_l3.idf\n"
        "    cond_p?_s1_l1 = conductance_l1.idf\n"
        "    cond_p?_s1_l2 = conductance_l2.idf\n"
        "    cond_p?_s1_l3 = conductance_l3.idf\n"
        "    ghbssmdens_p?_s1_l1 = density_l1.idf\n"
        "    ghbssmdens_p?_s1_l2 = density_l2.idf\n"
        "    ghbssmdens_p?_s1_l3 = density_l3.idf"
    )

    assert ghb._render(directory, globaltimes=["?"], system_index=1) == compare


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
    ghb.add_timemap(**{varname: timemap})
    actual = ghb._render(directory, globaltimes=datetimes, system_index=1)
