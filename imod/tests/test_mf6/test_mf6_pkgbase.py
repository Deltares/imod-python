import numpy as np
import xarray as xr

import imod
from imod.tests.fixtures.mf6_small_models_fixture import (
    grid_data_structured,
    grid_data_unstructured,
)


def test_mask_structured():
    head = grid_data_structured(np.float64, 2.1, 2.0)
    pkg = imod.mf6.ConstantHead(head=head)
    mask = grid_data_structured(np.int32, 1, 2.0)
    inactive_cell_location = {"layer": 1, "x": 2.0, "y": 4.0}
    mask.loc[inactive_cell_location] = 0

    masked_package = pkg.mask(mask)

    masked_head = masked_package.dataset["head"]
    assert type(masked_head) is type(head)
    assert masked_head.dtype == head.dtype
    assert np.isnan(masked_head.sel(inactive_cell_location).values[()])
    masked_head.loc[inactive_cell_location] = 2.1
    assert (masked_head == head).all().values[()]


def test_mask_unstructured():
    head = grid_data_unstructured(np.float64, 2.1, 2.0)
    pkg = imod.mf6.ConstantHead(head=head)
    mask = grid_data_unstructured(np.int32, 1, 2.0)
    inactive_cell_location = {"layer": 1, "mesh2d_nFaces": 23}
    mask.loc[inactive_cell_location] = 0

    masked_package = pkg.mask(mask)

    masked_head = masked_package.dataset["head"]
    assert type(masked_head) is type(head)
    assert masked_head.dtype == head.dtype
    assert np.isnan(masked_head.sel(inactive_cell_location).values[()])
    masked_head.loc[inactive_cell_location] = 2.1
    assert (masked_head == head).all().values[()]


def test_mask_scalar_input():
    # Create a storage package with scalar input
    storage_pack = imod.mf6.SpecificStorage(
        specific_storage=1.0e-5,
        specific_yield=0.15,
        transient=False,
        convertible=0,
    )
    mask = grid_data_unstructured(np.int32, 1, 2.0)
    inactive_cell_location = {"layer": 1, "mesh2d_nFaces": 23}
    mask.loc[inactive_cell_location] = 0

    masked_package = storage_pack.mask(mask)
    ss = masked_package["specific_storage"]
    assert np.isscalar(ss.values[()])


def test_mask_layered_input():
    # Create a npf package with scalar input
    model_layers = np.array([1, 2, 3])
    k = xr.DataArray([1.0e-3, 1.0e-4, 2.0e-4], {"layer": model_layers}, ("layer",))
    icelltype = xr.DataArray([1, 0, 0], {"layer": model_layers}, ("layer",))
    npf_pack = imod.mf6.NodePropertyFlow(
        icelltype=icelltype,
        k=k,
        variable_vertical_conductance=True,
        dewatered=True,
        perched=False,
        save_flows=True,
    )

    # Create a mask
    mask = grid_data_unstructured(np.int32, 1, 2.0)
    inactive_cell_location = {"layer": 1, "mesh2d_nFaces": 23}
    mask.loc[inactive_cell_location] = 0

    # Apply the mask
    masked_package = npf_pack.mask(mask)

    # Check layered array intact after masking
    assert (masked_package["k"] == k).all()
