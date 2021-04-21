import xarray as xr
import pytest
import numpy as np
import pandas as pd


@pytest.fixture(scope="module")
def basic_dis():
    """Basic model discretization"""

    shape = nlay, nrow, ncol = 3, 9, 9

    dx = 10.0
    dy = -10.0
    dz = np.array([5, 30, 100])
    xmin = 0.0
    xmax = dx * ncol
    ymin = 0.0
    ymax = abs(dy) * nrow
    dims = ("layer", "y", "x")

    layer = np.arange(1, nlay + 1)
    y = np.arange(ymax, ymin, dy) + 0.5 * dy
    x = np.arange(xmin, xmax, dx) + 0.5 * dx
    coords = {"layer": layer, "y": y, "x": x}

    ibound = xr.DataArray(np.ones(shape), coords=coords, dims=dims)

    surface = 0.0
    interfaces = np.insert((surface - np.cumsum(dz)), 0, surface)

    bottom = xr.DataArray(interfaces[1:], coords={"layer": layer}, dims="layer")
    top = xr.DataArray(interfaces[:-1], coords={"layer": layer}, dims="layer")

    return ibound, top, bottom


@pytest.fixture(scope="module")
def three_days():
    """Simple time discretization of three days"""

    return pd.date_range(start="1/1/2018", end="1/3/2018", freq="D")


@pytest.fixture(scope="module")
def two_days():
    """Simple time discretization of two days"""

    return pd.date_range(start="1/1/2018", end="1/2/2018", freq="D")


@pytest.fixture(scope="module")
def well_df(three_days):
    shape = nlay, nrow, ncol = 3, 9, 9
    dx = 10.0
    dy = -10.0
    xmin = 0.0
    xmax = dx * ncol
    ymin = 0.0
    ymax = abs(dy) * nrow

    layer = np.arange(1, nlay + 1)
    y = np.arange(ymax, ymin, dy) + 0.5 * dy
    x = np.arange(xmin, xmax, dx) + 0.5 * dx

    times = three_days
    n_repeats = int(len(x) / len(times)) + 1

    df = pd.DataFrame()
    df["id_name"] = np.arange(len(x)).astype(str)
    df["x"] = x
    df["y"] = y
    df["rate"] = dx * dy * -1 * 0.5
    df["time"] = np.tile(times, n_repeats)[: len(x)]
    df["layer"] = 2
    return df


@pytest.fixture(scope="module")
def get_render_dict():
    """
    Helper function to return dict to render.

    Fixture returns local helper function, so that the helper function
    is only evaluated when called.
    See: https://stackoverflow.com/a/51389067
    """

    def _get_render_dict(
        package, directory, globaltimes, nlayer, composition=None, system_index=1
    ):

        composition = package.compose(
            directory,
            globaltimes,
            nlayer,
            composition=composition,
            system_index=system_index,
        )

        return dict(
            pkg_id=package._pkg_id,
            name=package.__class__.__name__,
            variable_order=package._variable_order,
            package_data=composition[package._pkg_id],
        )

    return _get_render_dict


@pytest.fixture(scope="module")
def metaswap_dict(basic_dis):
    ibound, _, _ = basic_dis

    active = ibound.isel(layer=0)

    d = {}
    d["boundary"] = active
    d["landuse"] = active
    d["rootzone_thickness"] = 1.2
    d["soil_physical_unit"] = active
    d["meteo_station_number"] = active
    d["surface_elevation"] = 0.0
    d["sprinkling_type"] = active
    d["sprinkling_layer"] = active
    d["sprinkling_capacity"] = 1000.0
    d["wetted_area"] = 30.0
    d["urban_area"] = 30.0
    d["ponding_depth_urban"] = 0.02
    d["ponding_depth_rural"] = 0.005
    d["runoff_resistance_urban"] = 1.5
    d["runoff_resistance_rural"] = 1.5
    d["runon_resistance_urban"] = 1.5
    d["runon_resistance_rural"] = 1.5
    d["infiltration_capacity_urban"] = 10.0
    d["infiltration_capacity_rural"] = 2.0
    d["perched_water_table"] = 0.5
    d["soil_moisture_factor"] = 1.0
    d["conductivity_factor"] = 1.0

    d["extra_files"] = [
        "fact_svat.inp",
        "luse_svat.inp",
        "mete_grid.inp",
        "para_sim.inp",
        "tiop_sim.inp",
        "init_svat.inp",
        "comp_post.inp",
        "sel_key_svat_per.inp",
    ]

    return d