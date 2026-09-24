import numpy as np

from imod.mf6.utilities.mask import mask_topsystem
from imod.typing.grid import ones_like, zeros_like


def test_mask_topsystem(twri_model):
    """
    Test the mask_topsystem utility function by deactivating all cells in the
    grid.
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
