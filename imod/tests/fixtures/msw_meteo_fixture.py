import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy import nan

from imod.typing import GridDataArray


@pytest.fixture(scope="function")
def meteo_grids() -> tuple[GridDataArray, GridDataArray]:
    x = [1.0, 2.0, 3.0]
    y = [1.0, 2.0, 3.0]
    time = pd.date_range(start="2000-01-01", end="2000-01-02", freq="D")

    dx = 1.0
    dy = -1.0
    # fmt: off
    precipitation = xr.DataArray(
        np.array(
            [
                [[1.0, 1.0, 1.0],
                 [nan, nan, nan],
                 [1.0, 1.0, 1.0]],

                [[2.0, 2.0, 1.0],
                 [nan, nan, nan],
                 [1.0, 2.0, 1.0]],
            ]
        ),
        dims=("time", "y", "x"),
        coords={"time": time, "y": y, "x": x, "dx": dx, "dy": dy}
    )

    evapotranspiration = xr.DataArray(
        np.array(
            [1.0, 3.0]
        ),
        dims=("time",),
        coords={"time": time}
    )
    # fmt: on
    return precipitation, evapotranspiration
