import pathlib
import textwrap

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from imod.wq import ConstantHead


@pytest.fixture(scope="function")
def constanthead():
    layer = np.arange(1, 4)
    y = np.arange(4.5, 0.0, -1.0)
    x = np.arange(0.5, 5.0, 1.0)
    head = xr.DataArray(
        np.full((3, 5, 5), 1.0),
        coords={"layer": layer, "y": y, "x": x, "dx": 1.0, "dy": -1.0},
        dims=("layer", "y", "x"),
    )

    chd = ConstantHead(head_start=head, head_end=head.copy(), concentration=head.copy())
    return chd


def test_render(constanthead):
    chd = constanthead
    directory = pathlib.Path(".")

    compare = """
    shead_p?_s1_l$ = head_start_l$.idf
    ehead_p?_s1_l$ = head_end_l$.idf"""

    assert (
        chd._render(directory, globaltimes=["?"], system_index=1, nlayer=3) == compare
    )


@pytest.mark.parametrize("varname", ["head_start", "head_end"])
def test_render__timemap(constanthead, varname):
    chd = constanthead
    directory = pathlib.Path(".")
    da = chd[varname]
    datetimes = pd.date_range("2000-01-01", "2000-01-03")
    da_transient = xr.concat(
        [da.assign_coords(time=t) for t in datetimes[:-1]], dim="time"
    )
    chd[varname] = da_transient

    timemap = {datetimes[-1]: datetimes[0]}
    chd.add_timemap(**{varname: timemap})
    actual = chd._render(directory, globaltimes=datetimes, system_index=1, nlayer=3)
