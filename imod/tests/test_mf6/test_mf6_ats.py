from textwrap import dedent

import numpy as np
import pandas as pd
import xarray as xr

from imod.mf6.ats import AdaptiveTimeStepping


def _to_time_da(data):
    periods = len(data)
    coords = {"time": pd.date_range("2023-01-01", periods=periods, freq="D")}
    dims = ("time",)
    return xr.DataArray(
        data,
        coords=coords,
        dims=dims,
    )


def test_render():
    globaltimes = pd.date_range("2023-01-01", periods=10, freq="D")

    ats = AdaptiveTimeStepping(
        dt_init=_to_time_da(np.array([1.0, 2.0])),
        dt_min=_to_time_da(np.array([0.5, 1.0])),
        dt_max=_to_time_da(np.array([2.0, 4.0])),
        dt_multiplier=_to_time_da(np.array([1.5, 2.0])),
        dt_fail_multiplier=_to_time_da(np.array([2.0, 3.0])),
    )

    rendered = ats._render("test_dir", "test_pkg", globaltimes, False)

    expected = dedent("""\
        begin dimensions
          maxats 2
        end dimensions

        begin perioddata
          1 1.0 0.5 2.0 1.5 2.0
          2 2.0 1.0 4.0 2.0 3.0
        end perioddata
        """)

    assert isinstance(rendered, str)
    assert rendered == expected
