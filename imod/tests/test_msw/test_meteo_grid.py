import csv
import filecmp
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_equal

from imod.msw import MeteoGrid, MeteoGridCopy
from imod.util.regrid import RegridderWeightsCache


def test_meteo_grid_init(meteo_grids):
    meteo_grids = MeteoGrid(*meteo_grids)

    assert meteo_grids.dataset["precipitation"].dims == ("time", "y", "x")
    assert meteo_grids.dataset["evapotranspiration"].dims == ("time",)


def test_meteo_grid_write(meteo_grids):
    meteo_grid = MeteoGrid(*meteo_grids)

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


def test_meteo_no_time_grid(meteo_grids):
    precipitation, _ = meteo_grids

    evapotranspiration = 3.0
    # fmt: on

    with pytest.raises(ValueError):
        MeteoGrid(precipitation, evapotranspiration)


def test_regrid_meteo(meteo_grids, simple_2d_grid_with_subunits):
    meteo = MeteoGrid(*meteo_grids)
    new_grid = simple_2d_grid_with_subunits

    regrid_context = RegridderWeightsCache()

    regridded_ponding = meteo.regrid_like(new_grid, regrid_context)

    assert np.all(regridded_ponding.dataset["x"].values == new_grid["x"].values)
    assert np.all(regridded_ponding.dataset["y"].values == new_grid["y"].values)


def test_clip_box__spatial(meteo_grids):
    meteo = MeteoGrid(*meteo_grids)
    meteo_selected = meteo.clip_box(x_min=1.0, x_max=2.5, y_min=1.0, y_max=2.5)
    expected = meteo_grids[0].sel(x=slice(1.0, 2.5), y=slice(1.0, 2.5))
    xr.testing.assert_allclose(meteo_selected.dataset["precipitation"], expected)


def test_clip_box__time(meteo_grids):
    meteo = MeteoGrid(*meteo_grids)
    time_min = None
    time_max = "2000-01-01"
    meteo_selected = meteo.clip_box(time_min=time_min, time_max=time_max)
    expected = meteo_grids[0].sel(time=slice(time_min, time_max))
    xr.testing.assert_allclose(meteo_selected.dataset["precipitation"], expected)


def setup_written_meteo_grids(meteo_grids, tmp_path) -> Path:
    meteo_grid = MeteoGrid(*meteo_grids)
    grid_dir = tmp_path / "grid"
    grid_dir.mkdir(exist_ok=True, parents=True)
    meteo_grid.write(grid_dir)
    return grid_dir


def test_meteogridcopy_write(meteo_grids, tmp_path):
    # Arrange
    grid_dir = setup_written_meteo_grids(meteo_grids, tmp_path)
    meteo_grid_copy = MeteoGridCopy(grid_dir / "mete_grid.inp")
    copy_dir = tmp_path / "copied"
    copy_dir.mkdir(exist_ok=True, parents=True)
    # Act
    meteo_grid_copy.write(copy_dir)
    # Assert
    assert filecmp.cmp(grid_dir / "mete_grid.inp", copy_dir / "mete_grid.inp")


def test_meteogridcopy_from_imod5(meteo_grids, tmp_path):
    # Arrange
    grid_dir = setup_written_meteo_grids(meteo_grids, tmp_path)
    imod5_data = {}
    imod5_data["extra"] = {}
    imod5_data["extra"]["paths"] = [
        ["foo"],
        [(grid_dir / "mete_grid.inp").resolve()],
        ["bar"],
    ]
    meteo_grid_copy = MeteoGridCopy.from_imod5_data(imod5_data)
    copy_dir = tmp_path / "copied"
    copy_dir.mkdir(exist_ok=True, parents=True)
    # Act
    meteo_grid_copy.write(copy_dir)
    # Assert
    assert filecmp.cmp(grid_dir / "mete_grid.inp", copy_dir / "mete_grid.inp")
