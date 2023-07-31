import numpy as np
import pytest
import xarray as xr

from imod.mf6 import (
    GroundwaterFlowModel,
    InitialConditions,
    NodePropertyFlow,
    OutputControl,
    River,
    SpecificStorage,
)

globaltimes = np.array(
    [
        "2000-01-01",
        "2000-01-02",
        "2000-01-03",
    ],
    dtype="datetime64[ns]",
)


class grid_dimensions:
    nlay = 3
    nrow = 15
    ncol = 15
    dx = 5000
    dy = -5000
    xmin = 0
    ymin = 0


def get_data_array(dimensions, globaltimes):
    ntimes = len(globaltimes)
    shape = (ntimes, dimensions.nlay, dimensions.nrow, dimensions.ncol)
    dims = ("time", "layer", "y", "x")

    layer = np.array([1, 2, 3])
    xmax = dimensions.dx * dimensions.ncol
    ymax = abs(dimensions.dy) * dimensions.nrow
    y = np.arange(ymax, dimensions.ymin, dimensions.dy) + 0.5 * dimensions.dy
    x = np.arange(dimensions.xmin, xmax, dimensions.dx) + 0.5 * dimensions.dx
    coords = {"time": globaltimes, "layer": layer, "y": y, "x": x}

    # Discretization data
    return xr.DataArray(
        np.ones(shape),
        coords=coords,
        dims=dims,
    )


@pytest.fixture(scope="session")
def head_fc():
    idomain = get_data_array(grid_dimensions(), globaltimes)

    # Constant head
    head = xr.full_like(idomain, np.nan)
    return head


@pytest.fixture(scope="session")
def concentration_fc():
    idomain = get_data_array(grid_dimensions(), globaltimes)
    idomain = idomain.expand_dims(species=["salinity", "temperature"])

    concentration = xr.full_like(idomain, np.nan)
    return concentration


@pytest.fixture(scope="session")
def conductance_fc():
    globaltimes = np.array(
        [
            "2000-01-01",
            "2000-01-02",
            "2000-01-03",
        ],
        dtype="datetime64[ns]",
    )
    idomain = get_data_array(grid_dimensions(), globaltimes)

    # Constant head
    conductance = xr.full_like(idomain, np.nan)
    return conductance


@pytest.fixture(scope="session")
def elevation_fc():
    idomain = get_data_array(grid_dimensions(), globaltimes)

    elevation = xr.full_like(idomain, np.nan)
    elevation[:, 0, 7, 7:9] = 1.0

    return elevation


@pytest.fixture(scope="session")
def rate_fc():
    idomain = get_data_array(grid_dimensions(), globaltimes)

    rate = xr.full_like(idomain, np.nan)
    rate[:, 0, 7, 7:9] = 0.001

    return rate


@pytest.fixture(scope="session")
def proportion_rate_fc():
    idomain = get_data_array(grid_dimensions(), globaltimes)

    proportion_rate = xr.full_like(idomain, np.nan)
    proportion_rate[:, 0, 7, 7:9] = 0.3
    return proportion_rate


@pytest.fixture(scope="session")
def proportion_depth_fc():
    idomain = get_data_array(grid_dimensions(), globaltimes)

    proportion_depth = xr.full_like(idomain, np.nan)
    proportion_depth[:, 0, 7, 7:9] = 0.4
    return proportion_depth


@pytest.fixture(scope="session")
def porosity_fc():
    idomain = get_data_array(grid_dimensions(), globaltimes)

    porosity_fc = xr.full_like(idomain, np.nan).isel(time=0)
    return porosity_fc


@pytest.fixture(scope="session")
def decay_fc():
    idomain = get_data_array(grid_dimensions(), globaltimes)

    decay_fc = xr.full_like(idomain, np.nan).isel(time=0)
    return decay_fc


@pytest.fixture(scope="session")
def decay_sorbed_fc():
    idomain = get_data_array(grid_dimensions(), globaltimes)

    decay_sorbed_fc = xr.full_like(idomain, np.nan).isel(time=0)
    return decay_sorbed_fc


@pytest.fixture(scope="session")
def bulk_density_fc():
    idomain = get_data_array(grid_dimensions(), globaltimes)

    bulk_density_fc = xr.full_like(idomain, np.nan).isel(time=0)
    return bulk_density_fc


@pytest.fixture(scope="session")
def distcoef_fc():
    idomain = get_data_array(grid_dimensions(), globaltimes)

    distcoef_fc = xr.full_like(idomain, np.nan).isel(time=0)
    return distcoef_fc


@pytest.fixture(scope="session")
def sp2_fc():
    idomain = get_data_array(grid_dimensions(), globaltimes)

    sp2_fc = xr.full_like(idomain, np.nan).isel(time=0)
    return sp2_fc


@pytest.fixture(scope="session")
@pytest.mark.usefixtures("concentration_fc")
def flow_model_with_concentration(concentration_fc):
    idomain = get_data_array(grid_dimensions(), globaltimes)
    cellType = xr.full_like(idomain.isel(time=0), 1, dtype=np.int32)
    k = xr.full_like(idomain.isel(time=0), 10.0)
    k33 = xr.full_like(idomain.isel(time=0), 10.0)

    # River
    riv_dict = dict(
        stage=idomain.sel(layer=1),
        conductance=idomain.sel(layer=1),
        bottom_elevation=idomain.sel(layer=1) - 1.0,
        concentration=concentration_fc.sel(layer=1),
    )

    gwf_model = GroundwaterFlowModel()

    gwf_model["npf"] = NodePropertyFlow(
        icelltype=cellType,
        k=k,
        k33=k33,
    )

    gwf_model["sto"] = SpecificStorage(
        specific_storage=1.0e-5,
        specific_yield=0.15,
        transient=False,
        convertible=0,
    )
    gwf_model["ic"] = InitialConditions(start=0.0)
    gwf_model["oc"] = OutputControl(save_head="all", save_budget="all")
    gwf_model["riv-1"] = River(
        concentration_boundary_type="AUX",
        **riv_dict,
    )

    return gwf_model
