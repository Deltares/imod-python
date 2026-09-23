from imod.mf6.utilities.mask import mask_topsystem
from imod.typing.grid import zeros_like


def test_mask_topsystem(twri_model):
    """
    Test the mask_topsystem utility function by deactivating all cells in the
    grid.
    """
    # Arrange
    gwf_model = twri_model["GWF_1"]
    mask = zeros_like(gwf_model.domain)
    # Act
    mask_topsystem(gwf_model, mask)
    # Assert
    for key in ["rch", "drn"]:
        pkg = gwf_model[key]
        gridded_var = pkg.dataset[pkg._period_data[0]].compute()
        assert not gridded_var.notnull().any().item()
