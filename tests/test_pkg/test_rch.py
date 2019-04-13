from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from imod.pkg import RechargeHighestActive


@pytest.fixture(scope="module")
def recharge_ha(request):
    datetimes = pd.date_range("2000-01-01", "2000-01-05")
    y = np.arange(4.5, 0.0, -1.0)
    x = np.arange(0.5, 5.0, 1.0)
    rate = xr.DataArray(
        np.full((5, 5, 5), 1.0),
        coords={"time": datetimes, "y": y, "x": x, "dx": 1.0, "dy": -1.0},
        dims=("time", "y", "x"),
    )

    rch = RechargeHighestActive(rate=rate, concentration=rate.copy(), save_budget=False)
    return rch


def test_render(recharge_ha):
    rch = recharge_ha
    directory = Path(".")
    compare = (
        "[rch]\n"
        "    nrchop = 3\n"
        "    irchcb = 0\n"
        "    rech_p1 = rate_20000101000000.idf\n"
        "    rech_p2 = rate_20000102000000.idf\n"
        "    rech_p3 = rate_20000103000000.idf\n"
        "    rech_p4 = rate_20000104000000.idf\n"
        "    rech_p5 = rate_20000105000000.idf"
    )

    assert rch._render(directory, globaltimes=rch.time.values) == compare
