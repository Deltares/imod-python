from typing import Callable, NamedTuple

import numpy as np
import pandas as pd
import pytest
import xarray as xr


def make_basic_dis(dz, nrow, ncol):
    """Basic model discretization"""
    dx = 10.0
    dy = -10.0

    nlay = len(dz)

    shape = nlay, nrow, ncol

    xmin = 0.0
    xmax = dx * ncol
    ymin = 0.0
    ymax = abs(dy) * nrow
    dims = ("layer", "y", "x")

    layer = np.arange(1, nlay + 1)
    y = np.arange(ymax, ymin, dy) + 0.5 * dy
    x = np.arange(xmin, xmax, dx) + 0.5 * dx
    coords = {"layer": layer, "y": y, "x": x}

    ibound = xr.DataArray(np.ones(shape, dtype=np.int32), coords=coords, dims=dims)

    surface = 0.0
    interfaces = np.insert((surface - np.cumsum(dz)), 0, surface)

    bottom = xr.DataArray(interfaces[1:], coords={"layer": layer}, dims="layer")
    top = xr.DataArray(interfaces[:-1], coords={"layer": layer}, dims="layer")

    return ibound, top, bottom


@pytest.fixture(scope="module")
def basic_dis():
    return make_basic_dis(dz=[5, 30, 100], nrow=9, ncol=9)


@pytest.fixture(scope="function")
def basic_dis__topsystem():
    return make_basic_dis(dz=[1.0, 2.0, 4.0, 10.0], nrow=9, ncol=9)


class BasicDisSettings(NamedTuple):
    nlay: int = 1
    nrow: int = 1
    ncol: int = 1
    zstart: float = 0.0
    zstop: float = -1.0
    ystart: float = 0.0
    ystop: float = 1.0
    xstart: float = 0.0
    xstop: float = 1.0
    dx: float = 1.0
    dy: float = 1.0
    dz: float = 1.0
    space_generator: Callable = np.linspace


@pytest.fixture
def parameterizable_basic_dis(request):
    settings = request.param
    shape = (settings.nlay, settings.nrow, settings.ncol)

    x = (
        settings.space_generator(
            settings.xstart + 1, settings.xstop + 1, settings.ncol + 1, endpoint=True
        )
        - 1
    )
    y = (
        settings.space_generator(
            settings.ystop + 1, settings.ystart + 1, settings.nrow + 1, endpoint=True
        )
        - 1
    )
    z = (
        settings.space_generator(
            settings.zstart - 1, settings.zstop - 1, settings.nlay + 1, endpoint=True
        )
        + 1
    )

    xc = (x[:-1] + x[1:]) / 2
    yc = (y[:-1] + y[1:]) / 2

    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]

    layers = np.arange(settings.nlay) + 1

    idomain = xr.DataArray(
        np.ones(shape, dtype=np.int32),
        coords={"layer": layers, "y": yc, "x": xc},
        name="idomain",
    )

    # Assign dx and dy coordinates. They are needed for certain methods like 'coord_reference'
    if np.all(np.isclose(dx, dx[0])):
        idomain = idomain.assign_coords({"dx": dx[0]})
    else:
        idomain = idomain.assign_coords({"dx": ("x", dx)})
    if np.all(np.isclose(dy, dy[0])):
        idomain = idomain.assign_coords({"dy": dy[0]})
    else:
        idomain = idomain.assign_coords({"dy": ("y", dy)})

    top = xr.DataArray(z[:-1], coords={"layer": layers})
    bottom = xr.DataArray(z[1:], coords={"layer": layers})

    return idomain, top, bottom


@pytest.fixture(scope="module")
def three_days():
    """Simple time discretization of three days"""
    return pd.date_range(start="2018-01-01", end="2018-01-03", freq="D")


@pytest.fixture(scope="module")
def two_days():
    """Simple time discretization of two days"""
    return pd.date_range(start="2018-01-01", end="2018-01-02", freq="D")


@pytest.fixture(scope="module")
def well_df(three_days):
    nrow = 9
    ncol = 9
    dx = 10.0
    dy = -10.0
    xmin = 0.0
    xmax = dx * ncol
    ymin = 0.0
    ymax = abs(dy) * nrow

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

    def _get_render_dict(package, directory, globaltimes, nlayer, system_index=1):
        composition = package.compose(
            directory,
            globaltimes,
            nlayer,
            system_index=system_index,
        )

        return {
            "pkg_id": package._pkg_id,
            "name": package.__class__.__name__,
            "variable_order": package._variable_order,
            "package_data": composition[package._pkg_id],
        }

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

    d["lookup_and_forcing_files"] = [
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


@pytest.fixture(scope="module")
def horizontal_flow_barrier_gdf(basic_dis):
    """GeoDataframe that can be used to initiate HorizontalFlowBarriers"""
    import geopandas as gpd
    from shapely.geometry import LineString

    ibound, _, _ = basic_dis

    x = ibound.x.values
    y = ibound.y.values

    line1 = LineString([(x[1], y[1]), (x[1], y[-2])])
    line2 = LineString([(x[4], y[1]), (x[4], y[-2])])

    lines = np.array([line1, line2, line1, line2], dtype="object")
    hfb_layers = np.array([1, 1, 3, 3])
    id_name = ["left_upper", "right_upper", "left_lower", "right_lower"]

    hfb_gdf = gpd.GeoDataFrame(
        geometry=lines,
        data={"layer": hfb_layers, "resistance": 100.0, "id_name": id_name},
    )

    return hfb_gdf
