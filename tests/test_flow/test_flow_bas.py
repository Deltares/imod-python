from imod.flow import Top, Bottom, Boundary
import pytest
import numpy as np
import xarray as xr


@pytest.fixture(scope="module")
def basic():
    """Create basic model discretization"""

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
def get_package_attributes():
    """Helper function to return relevant package attributes
    for rendering as dictionary.

    Fixture returns local helper function, so that the helper function
    is only evaluated when called.
    See: https://stackoverflow.com/a/51389067
    """

    def _get_package_attributes(package):
        return dict(
            pkg_id=package._pkg_id,
            name=package.__class__.__name__,
            variable_order=package._variable_order,
        )

    return _get_package_attributes


@pytest.fixture(scope="module")
def get_render_dict(get_package_attributes):
    """
    Helper function to return dict to render.

    Fixture returns local helper function, so that the helper function
    is only evaluated when called.
    See: https://stackoverflow.com/a/51389067
    """

    def _get_render_dict(package):
        composition_args = (".", None, 3)
        composition = package.compose(*composition_args)
        to_render = get_package_attributes(package)
        to_render["package_data"] = composition[package._pkg_id]
        return to_render

    return _get_render_dict


def test_boundary(basic, get_render_dict):
    ibound, _, _ = basic
    boundary = Boundary(ibound=ibound)
    to_render = get_render_dict(boundary)
    to_render["n_entry"] = 3

    compare = (
        "0001, (bnd), 1, Boundary, ['ibound']\n"
        "001, 003\n"
        "1, 2, 001, 1.000, 0.000, -9999., ibound_l1.idf\n"
        "1, 2, 002, 1.000, 0.000, -9999., ibound_l2.idf\n"
        "1, 2, 003, 1.000, 0.000, -9999., ibound_l3.idf"
    )

    rendered = boundary._render_projectfile(**to_render)

    assert rendered == compare


def test_top(basic, get_render_dict):
    _, top, _ = basic
    top = Top(top=top)
    to_render = get_render_dict(top)
    to_render["n_entry"] = 3

    compare = (
        "0001, (top), 1, Top, ['top']\n"
        "001, 003\n"
        '1, 1, 001, 1.000, 0.000, 0.0, ""\n'
        '1, 1, 002, 1.000, 0.000, -5.0, ""\n'
        '1, 1, 003, 1.000, 0.000, -35.0, ""'
    )

    rendered = top._render_projectfile(**to_render)

    assert rendered == compare


def test_bot(basic, get_render_dict):
    _, _, bottom = basic
    bottom = Bottom(bottom=bottom)
    to_render = get_render_dict(bottom)
    to_render["n_entry"] = 3

    compare = (
        "0001, (bot), 1, Bottom, ['bottom']\n"
        "001, 003\n"
        '1, 1, 001, 1.000, 0.000, -5.0, ""\n'
        '1, 1, 002, 1.000, 0.000, -35.0, ""\n'
        '1, 1, 003, 1.000, 0.000, -135.0, ""'
    )

    rendered = bottom._render_projectfile(**to_render)

    assert rendered == compare
