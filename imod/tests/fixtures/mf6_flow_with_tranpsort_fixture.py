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

globaltimes = [
    np.datetime64("2000-01-01"),
    np.datetime64("2000-01-02"),
    np.datetime64("2000-01-03"),
]


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
    globaltimes = [
        np.datetime64("2000-01-01"),
        np.datetime64("2000-01-02"),
        np.datetime64("2000-01-03"),
    ]
    idomain = get_data_array(grid_dimensions(), globaltimes)

    # Constant head
    conductance = xr.full_like(idomain, np.nan)
    return conductance


@pytest.fixture(scope="session")
def elevation_fc():

    idomain = get_data_array(grid_dimensions(), globaltimes)

    elevation = xr.full_like(idomain, np.nan)
    return elevation


@pytest.fixture(scope="session")
def rate_fc():

    idomain = get_data_array(grid_dimensions(), globaltimes)

    elevation = xr.full_like(idomain, np.nan)
    return elevation


@pytest.fixture(scope="session")
def proportion_rate_fc():

    idomain = get_data_array(grid_dimensions(), globaltimes)

    proportion_rate = xr.full_like(idomain, np.nan)
    return proportion_rate


@pytest.fixture(scope="session")
def proportion_depth_fc():

    idomain = get_data_array(grid_dimensions(), globaltimes)

    proportion_rate = xr.full_like(idomain, np.nan)
    return proportion_rate

@pytest.fixture(scope="session")
def decay_fc():

    idomain = get_data_array(grid_dimensions(), globaltimes)

    decay = xr.full_like(idomain, np.nan)
    return decay

@pytest.fixture(scope="session")
def decay_sorbed_fc():

    idomain = get_data_array(grid_dimensions(), globaltimes)

    decay = xr.full_like(idomain, np.nan)
    return decay

@pytest.fixture(scope="session")
def bulk_density_fc():

    idomain = get_data_array(grid_dimensions(), globaltimes)

    bulk_density = xr.full_like(idomain, np.nan)
    return bulk_density

@pytest.fixture(scope="session")
def sp2_fc():

    idomain = get_data_array(grid_dimensions(), globaltimes)

    sp2 = xr.full_like(idomain, np.nan)
    return sp2

@pytest.fixture(scope="session")
def distcoef_fc():

    idomain = get_data_array(grid_dimensions(), globaltimes)

    distcoef = xr.full_like(idomain, np.nan)
    return distcoef

@pytest.fixture(scope="session")
def porosity_fc():

    idomain = get_data_array(grid_dimensions(), globaltimes)

    porosity = xr.full_like(idomain, np.nan)
    return porosity