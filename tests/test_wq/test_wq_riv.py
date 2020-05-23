import pathlib
import textwrap

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from imod.wq import River


@pytest.fixture(scope="function")
def river():
    layer = np.arange(1, 4)
    y = np.arange(4.5, 0.0, -1.0)
    x = np.arange(0.5, 5.0, 1.0)
    stage = xr.DataArray(
        np.full((3, 5, 5), 1.0),
        coords={"layer": layer, "y": y, "x": x, "dx": 1.0, "dy": -1.0},
        dims=("layer", "y", "x"),
    )

    riv = River(
        stage=stage,
        conductance=stage.copy(),
        bottom_elevation=stage.copy(),
        concentration=stage.copy(),
        density=stage.copy(),
    )
    return riv


@pytest.fixture(scope="function")
def river_multiple_species():
    layer = np.arange(1, 4)
    y = np.arange(4.5, 0.0, -1.0)
    x = np.arange(0.5, 5.0, 1.0)
    stage = xr.DataArray(
        np.full((3, 5, 5), 1.0),
        coords={"layer": layer, "y": y, "x": x, "dx": 1.0, "dy": -1.0},
        dims=("layer", "y", "x"),
    )

    riv = River(
        stage=stage,
        conductance=stage.copy(),
        bottom_elevation=stage.copy(),
        concentration=stage.copy(),
        density=stage.copy(),
    )
    conc1 = riv["concentration"].assign_coords(species=1)
    conc2 = riv["concentration"].assign_coords(species=2)
    riv["concentration"] = xr.concat([conc1, conc2], dim="species")
    return riv


def test_render(river):
    riv = river
    directory = pathlib.Path(".")

    compare = """
    stage_p?_s1_l1:3 = stage_l:.idf
    cond_p?_s1_l1:3 = conductance_l:.idf
    rbot_p?_s1_l1:3 = bottom_elevation_l:.idf
    rivssmdens_p?_s1_l1:3 = density_l:.idf"""

    assert riv._render(directory, globaltimes=["?"], system_index=1) == compare


def test_render_multiple_scalar_concentration(river):
    riv = river
    directory = pathlib.Path(".")
    riv["concentration"] = xr.DataArray(
        [10.0, 20.0], coords={"species": [1, 2]}, dims=["species"]
    )

    compare = """
    criv_t1_p?_l? = 10.0
    criv_t2_p?_l? = 20.0"""
    actual = riv._render_ssm(directory, globaltimes=["?"])
    assert compare == actual


def test_render_multiple_array_concentration(river_multiple_species, tmp_path):
    riv = river_multiple_species
    directory = pathlib.Path(".")

    compare = """
    criv_t1_p?_l1:3 = concentration_c1_l:.idf
    criv_t2_p?_l1:3 = concentration_c2_l:.idf"""
    actual = riv._render_ssm(directory, globaltimes=["?"])
    assert compare == actual


def test_save_multiple_array_concentration(river_multiple_species, tmp_path):
    riv = river_multiple_species
    riv.save(tmp_path / "river")

    files = [
        "stage_l1.idf",
        "stage_l2.idf",
        "stage_l3.idf",
        "conductance_l1.idf",
        "conductance_l2.idf",
        "conductance_l3.idf",
        "bottom_elevation_l1.idf",
        "bottom_elevation_l2.idf",
        "bottom_elevation_l3.idf",
        "density_l1.idf",
        "density_l2.idf",
        "density_l3.idf",
        "concentration_c1_l1.idf",
        "concentration_c1_l2.idf",
        "concentration_c1_l3.idf",
        "concentration_c2_l1.idf",
        "concentration_c2_l2.idf",
        "concentration_c2_l3.idf",
    ]

    for file in files:
        assert (tmp_path / "river" / file).is_file()


@pytest.mark.parametrize(
    "varname", ["stage", "conductance", "bottom_elevation", "concentration", "density"]
)
def test_render__timemap(river, varname):
    riv = river
    directory = pathlib.Path(".")
    da = riv[varname]
    datetimes = pd.date_range("2000-01-01", "2000-01-03")
    da_transient = xr.concat(
        [da.assign_coords(time=t) for t in datetimes[:-1]], dim="time"
    )
    riv[varname] = da_transient

    timemap = {datetimes[-1]: datetimes[0]}
    riv.add_timemap(**{varname: timemap})
    actual = riv._render(directory, globaltimes=datetimes, system_index=1)
