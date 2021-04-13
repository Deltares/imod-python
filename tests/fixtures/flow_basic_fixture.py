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
