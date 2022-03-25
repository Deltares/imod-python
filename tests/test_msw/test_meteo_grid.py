import csv
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy import nan
from numpy.testing import assert_equal

from imod.msw import MeteoGrid


def test_meteo_grid():
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

    meteo_grid = MeteoGrid(precipitation, evapotranspiration)

    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        meteo_grid.write(output_dir)

        results = pd.read_csv(
            output_dir / "mete_grid.inp", header=None, quoting=csv.QUOTE_NONE
        )
        gridnames = sorted([file.name for file in output_dir.glob("meteo_grids/*.asc")])

    expected_paths = [
        '"meteo_grids' + os.path.sep + 'precipitation_20000101000000.asc"',
        '"meteo_grids' + os.path.sep + 'precipitation_20000102000000.asc"',
    ]

    assert_equal(results[0].values, np.array([0.0, 1.0]))
    assert_equal(results[1].values, np.array([2000, 2000]))
    assert_equal(results[2].values, np.array(expected_paths, dtype=object))
    assert_equal(results[3].values, np.array(['"1.0"', '"3.0"'], dtype=object))

    # strip directory and quotes from filename
    expected_filenames = [
        f.replace('"', "").replace("meteo_grids" + os.path.sep, "")
        for f in expected_paths
    ]
    assert gridnames == expected_filenames


def test_meteo_no_time_grid():
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

    evapotranspiration = 3.0
    # fmt: on

    with pytest.raises(ValueError):
        MeteoGrid(precipitation, evapotranspiration)
