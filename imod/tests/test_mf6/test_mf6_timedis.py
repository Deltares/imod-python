import textwrap

import pandas as pd
import pytest
import xarray as xr

import imod
from imod.schemata import ValidationError


def test_render():
    timestep_duration = xr.DataArray(
        data=[0.001, 7.0, 365.0],
        coords={"time": pd.date_range("2000-01-01", "2000-01-03")},
        dims=["time"],
    )
    timedis = imod.mf6.TimeDiscretization(
        timestep_duration, n_timesteps=2, timestep_multiplier=1.1
    )
    actual = timedis.render()
    expected = textwrap.dedent(
        """\
        begin options
          time_units days
          start_date_time 2000-01-01T00:00:00.000000000
        end options

        begin dimensions
          nper 3
        end dimensions

        begin perioddata
          0.001 2 1.1
          7.0 2 1.1
          365.0 2 1.1
        end perioddata
        """
    )
    assert actual == expected


def test_wrong_dtype():
    timestep_duration = xr.DataArray(
        data=[1, 7, 365],
        coords={"time": pd.date_range("2000-01-01", "2000-01-03")},
        dims=["time"],
    )
    with pytest.raises(ValidationError):
        imod.mf6.TimeDiscretization(
            timestep_duration, n_timesteps=2, timestep_multiplier=1.1
        )


def test_wrong_dims():
    timestep_duration = xr.DataArray(
        data=[[0.001, 7.0, 365.0]],
        coords={"time": pd.date_range("2000-01-01", "2000-01-03"), "layer": [1]},
        dims=["layer", "time"],
    )

    with pytest.raises(ValidationError):
        imod.mf6.TimeDiscretization(
            timestep_duration, n_timesteps=2, timestep_multiplier=1.1
        )


def test_validate_false():
    timestep_duration = xr.DataArray(
        data=[[0.001, 7.0, 365.0]],
        coords={"time": pd.date_range("2000-01-01", "2000-01-03"), "layer": [1]},
        dims=["layer", "time"],
    )

    imod.mf6.TimeDiscretization(
        timestep_duration,
        n_timesteps=2,
        timestep_multiplier=1.1,
        validate=False,
    )
