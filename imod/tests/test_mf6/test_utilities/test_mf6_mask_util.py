import numpy as np
import pandas as pd
import pytest_cases
import xarray as xr

from imod.mf6.utilities.mask import mask_topsystem, mask_topsystem_where_bc
from imod.typing.grid import ones_like, zeros_like


def test_mask_topsystem(twri_model):
    """
    Test the mask_topsystem utility function by deactivating the first cell in
    the grid.
    """
    # Arrange
    gwf_model = twri_model["GWF_1"]
    is_active = ones_like(gwf_model.domain)
    # Mask first cell
    is_active[0, 0, 0] = 0

    # Act
    mask_topsystem(gwf_model, is_active, True)
    # Assert
    for key in ["rch", "drn"]:
        pkg = gwf_model[key]
        gridded_var = pkg.dataset[pkg._period_data[0]].compute()
        first_cell = gridded_var.data.ravel()[0]
        assert np.isnan(first_cell).item()


def test_mask_topsystem__all_removed(twri_model):
    """
    Test the mask_topsystem utility function by deactivating all cells in the
    grid.
    """
    # Arrange
    gwf_model = twri_model["GWF_1"]
    is_active = zeros_like(gwf_model.domain)
    # Act
    mask_topsystem(gwf_model, is_active, True)
    # Assert
    for key in ["rch", "drn"]:
        assert key not in gwf_model.keys()


def test_mask_topsystem_where_bc__no_time(twri_model):
    """
    Test the mask_topsystem utility function with boundary conditions by deactivating the first cell in the grid.
    """
    # Arrange
    gwf_model = twri_model["GWF_1"]
    bc_condition = gwf_model["chd"]

    # Test if fixture still as expected, first cell should be active in the
    # boundary condition
    chd_active = np.isnan(bc_condition.dataset["head"])
    assert chd_active[0, 0, 0].item() is False
    assert chd_active[0, 0, 1].item() is True
    # Should be all active
    rch_active_prior = np.isnan(gwf_model["rch"].dataset["rate"])
    assert rch_active_prior[0, 0].item() is False
    assert rch_active_prior[0, 1].item() is False

    # Act
    mask_topsystem_where_bc(gwf_model, bc_condition, True)

    # Assert
    rch_active_post = np.isnan(gwf_model["rch"].dataset["rate"])
    assert rch_active_post[0, 0].item() is True
    assert rch_active_post[0, 1].item() is False


@pytest_cases.parametrize("ignore_time", [True, False])
def test_mask_topsystem_where_bc__time(twri_model, ignore_time):
    """
    Test the mask_topsystem utility function with boundary conditions by deactivating the first cell in the grid.
    """
    # Arrange
    gwf_model = twri_model["GWF_1"]
    bc_condition = gwf_model["chd"]

    # Make package transient
    time_coord = pd.date_range("2000-01-01", periods=3)
    time_da = xr.DataArray([1, 1, 1], dims=["time"], coords={"time": time_coord})
    ds = bc_condition.dataset.copy()
    bc_condition.dataset["head"] = time_da * ds["head"]

    # Test if fixture still as expected, first cell should be active in the
    # boundary condition
    chd_active = np.isnan(bc_condition.dataset["head"])
    assert chd_active[1, 0, 0, 0].item() is False
    assert chd_active[1, 0, 0, 1].item() is True
    # Should be all active
    rch_active_prior = np.isnan(gwf_model["rch"].dataset["rate"])
    assert rch_active_prior[0, 0].item() is False
    assert rch_active_prior[0, 1].item() is False

    # Act
    mask_topsystem_where_bc(gwf_model, bc_condition, ignore_time)

    # Assert
    rch_active_post = np.isnan(gwf_model["rch"].dataset["rate"])
    assert rch_active_post[0, 0].item() is True
    assert rch_active_post[0, 1].item() is False
