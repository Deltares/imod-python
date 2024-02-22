from copy import deepcopy
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
import xarray as xr

import imod
from imod.mf6.model_gwf import GroundwaterFlowModel
from imod.tests.fixtures.mf6_modelrun_fixture import assert_model_can_run


def test_masked_model_validation_inactive_cell_pillar(
    tmp_path: Path, unstructured_flow_model: GroundwaterFlowModel
):
    # create mask from idomain. Deactivate the same cell in all layers
    mask = deepcopy(unstructured_flow_model.domain)
    mask.loc[{"layer": 1, "mesh2d_nFaces": 23}] = 0
    mask.loc[{"layer": 2, "mesh2d_nFaces": 23}] = 0
    mask.loc[{"layer": 3, "mesh2d_nFaces": 23}] = 0

    # apply the mask to a model
    unstructured_flow_model.mask_all_packages(mask)

    # test output validity
    errors = unstructured_flow_model._validate("model")
    assert len(errors.errors) == 0
    assert_model_can_run(unstructured_flow_model, "disv", tmp_path)


@pytest.mark.parametrize("layer_and_face", [(1, 23), (2, 23), (3, 23)])
@pytest.mark.parametrize("inactivity_marker", [0, -1])
def test_masked_model_validation_one_inactive_cell(
    tmp_path: Path,
    unstructured_flow_model: GroundwaterFlowModel,
    layer_and_face: Tuple[int, int],
    inactivity_marker: int,
):
    # create mask from idomain. a single cell
    layer, face = layer_and_face
    mask = deepcopy(unstructured_flow_model.domain)
    mask.loc[{"layer": layer, "mesh2d_nFaces": face}] = inactivity_marker

    # apply the mask to a model
    unstructured_flow_model.mask_all_packages(mask)

    # test output validity
    errors = unstructured_flow_model._validate("model")
    assert len(errors.errors) == 0
    assert_model_can_run(unstructured_flow_model, "disv", tmp_path)


@pytest.mark.parametrize("layer_and_face", [(1, 23), (2, 23), (3, 23)])
def test_masked_model_layered_and_scalar_package_input(
    tmp_path: Path,
    unstructured_flow_model: GroundwaterFlowModel,
    layer_and_face: Tuple[int, int],
):
    # Create mask from idomain. a single cell
    layer, face = layer_and_face
    mask = deepcopy(unstructured_flow_model.domain)
    mask.loc[{"layer": layer, "mesh2d_nFaces": face}] = 0

    # Make one package layer-based
    model_layers = np.array([1, 2, 3])
    k = xr.DataArray([1.0e-3, 1.0e-4, 2.0e-4], {"layer": model_layers}, ("layer",))
    icelltype = xr.DataArray([1, 0, 0], {"layer": model_layers}, ("layer",))
    unstructured_flow_model["npf"] = imod.mf6.NodePropertyFlow(
        icelltype=icelltype,
        k=k,
        variable_vertical_conductance=True,
        dewatered=True,
        perched=False,
        save_flows=True,
    )

    # Make one packages scalar-based
    unstructured_flow_model["sto"] = imod.mf6.SpecificStorage(
        specific_storage=1.0e-5,
        specific_yield=0.15,
        transient=False,
        convertible=0,
    )

    # Apply the mask to a model
    unstructured_flow_model.mask_all_packages(mask)

    # Test output validity
    errors = unstructured_flow_model._validate("model")
    assert len(errors.errors) == 0
    assert_model_can_run(unstructured_flow_model, "disv", tmp_path)

@pytest.mark.parametrize("inactivity_marker", [0, -1])
def test_masking_preservers_inactive_cells(
    tmp_path: Path,
    unstructured_flow_model: GroundwaterFlowModel,
    inactivity_marker: int
):
    # Create mask from idomain. set some cells to inactive.
    mask = deepcopy(unstructured_flow_model.domain.sel(layer = 1))
    mask = mask.drop_vars("layer")
    mask.values[17:22] = inactivity_marker
    unstructured_flow_model.mask_all_packages(mask)

    # Now set these cells to active in the mask
    mask.values[17:22] = 1
    unstructured_flow_model.mask_all_packages(mask)

    # Masking should not activate inactive cells
    assert np.all(unstructured_flow_model.domain.values[:, 17: 22] == inactivity_marker)


def test_masking_masks_vertical_passthrough_cells(
    tmp_path: Path,
    unstructured_flow_model: GroundwaterFlowModel,
):
    # Create mask from idomain. set some cells to vertical passthrough.
    mask = deepcopy(unstructured_flow_model.domain.sel(layer = 1))
    mask = mask.drop_vars("layer")
    mask.values[17:22] = -1
    unstructured_flow_model.mask_all_packages(mask)

    # Now set these cells to inactive in the mask
    mask.values[17:22] = 0
    unstructured_flow_model.mask_all_packages(mask)

    # Masking should mask vertical passthrough cells
    assert np.all(unstructured_flow_model.domain.values[:, 17: 22] == 0)

    

