import pytest
import xarray as xr

from imod.select.grid import active_grid_boundary_xy, grid_boundary_xy


@pytest.fixture()
def all_active_idomain(basic_dis):
    idomain, _, _ = basic_dis
    return idomain == 1


@pytest.fixture()
def inactive_idomain(basic_dis):
    # idomain with top row equal zero
    idomain, _, _ = basic_dis
    idomain_inactive = idomain.copy()
    idomain_inactive[:, 0, :] = 0
    return idomain_inactive == 1


@pytest.fixture()
def unstructured_idomain(circle_dis):
    idomain, _, _ = circle_dis
    return idomain == 1


def test_grid_boundary_xy__all_active(all_active_idomain):
    grid_boundary = grid_boundary_xy(all_active_idomain)

    grid_boundary_expected = xr.zeros_like(all_active_idomain.isel(layer=0), dtype=bool)
    grid_boundary_expected[0, :] = True
    grid_boundary_expected[-1, :] = True
    grid_boundary_expected[:, 0] = True
    grid_boundary_expected[:, -1] = True

    xr.testing.assert_equal(grid_boundary, grid_boundary_expected)


def test_grid_boundary_xy__inactive(inactive_idomain):
    grid_boundary = grid_boundary_xy(inactive_idomain)

    grid_boundary_expected = xr.zeros_like(inactive_idomain.isel(layer=0), dtype=bool)
    grid_boundary_expected[0, :] = True
    grid_boundary_expected[-1, :] = True
    grid_boundary_expected[:, 0] = True
    grid_boundary_expected[:, -1] = True

    xr.testing.assert_equal(grid_boundary, grid_boundary_expected)


def test_grid_boundary_xy__unstructured(unstructured_idomain):
    grid_boundary = grid_boundary_xy(unstructured_idomain)

    assert grid_boundary.shape == (127,)
    assert grid_boundary.sum() == 36


def test_active_grid_boundary_xy__all_active(all_active_idomain):
    grid_boundary = active_grid_boundary_xy(all_active_idomain)

    grid_boundary_expected = xr.zeros_like(all_active_idomain)
    grid_boundary_expected[:, 0, :] = 1
    grid_boundary_expected[:, -1, :] = 1
    grid_boundary_expected[:, :, 0] = 1
    grid_boundary_expected[:, :, -1] = 1

    xr.testing.assert_equal(grid_boundary, grid_boundary_expected)


def test_active_grid_boundary_xy__inactive(inactive_idomain):
    grid_boundary = active_grid_boundary_xy(inactive_idomain)

    grid_boundary_expected = xr.zeros_like(inactive_idomain)
    grid_boundary_expected[:, -1, :] = 1
    grid_boundary_expected[:, :, 0] = 1
    grid_boundary_expected[:, :, -1] = 1
    # Force top row to zero to force corners to zero as well
    grid_boundary_expected[:, 0, :] = 0

    xr.testing.assert_equal(grid_boundary, grid_boundary_expected)
