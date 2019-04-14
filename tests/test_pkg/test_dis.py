import numpy as np
import pandas as pd
import pytest
import xarray as xr
from imod.pkg import TimeDiscretization


@pytest.fixture(scope="module")
def discret(request):
    datetimes = pd.date_range("2000-01-01", "2000-01-05")
    timestep_duration = xr.DataArray(np.full(5, 1.0), coords={"time": datetimes}, dims=("time",))
 
    dis = TimeDiscretization(
        time=datetimes,
        timestep_duration=timestep_duration,
        n_timesteps=xr.full_like(timestep_duration, 1, dtype=np.int),
        transient=xr.full_like(timestep_duration, True),
        timestep_multiplier=xr.full_like(timestep_duration, 1.0),
        max_n_transport_timestep=xr.full_like(timestep_duration, 10, dtype=np.int),
        transport_timestep_multiplier=xr.full_like(timestep_duration, 1.0),
        transport_initial_timestep=0,
    )
    return dis


def test_render_dis(discret):
    dis = discret
    globaltimes = dis.time.values

    compare = (
    "\n"
    "    perlen_p1 = 1.0\n"
    "    perlen_p2 = 1.0\n"
    "    perlen_p3 = 1.0\n"
    "    perlen_p4 = 1.0\n"
    "    perlen_p5 = 1.0\n"
    "    nstp_p1 = 1\n"
    "    nstp_p2 = 1\n"
    "    nstp_p3 = 1\n"
    "    nstp_p4 = 1\n"
    "    nstp_p5 = 1\n"
    "    sstr_p1 = TR\n"
    "    sstr_p2 = TR\n"
    "    sstr_p3 = TR\n"
    "    sstr_p4 = TR\n"
    "    sstr_p5 = TR\n"
    "    tsmult_p1 = 1.0\n"
    "    tsmult_p2 = 1.0\n"
    "    tsmult_p3 = 1.0\n"
    "    tsmult_p4 = 1.0\n"
    "    tsmult_p5 = 1.0"
    )

    assert dis._render(globaltimes) == compare


def test_render_dis__notime(discret):
    dis = discret.isel(time=0).drop("time")
    globaltimes = ["?"]

    compare = (
    "\n"
    "    perlen_p? = 1.0\n"
    "    nstp_p? = 1\n"
    "    sstr_p? = TR\n"
    "    tsmult_p? = 1.0"
    )

    assert dis._render(globaltimes) == compare


def test_render_btn(discret):
    dis = discret.isel(time=0).drop("time")
    globaltimes = ["?"]

    compare = (
    "\n"
    "    dt0_p? = 0\n"
    "    ttsmult_p? = 1.0\n"
    "    mxstrn_p? = 10"
    )
    assert dis._render_btn(globaltimes) == compare