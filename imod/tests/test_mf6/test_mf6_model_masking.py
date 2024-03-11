from copy import deepcopy
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
import xarray as xr
import xugrid as xu

import imod
from imod.mf6.model_gwf import GroundwaterFlowModel
from imod.tests.fixtures.mf6_modelrun_fixture import assert_model_can_run


def test_masked_model_validation_inactive_cell_pillar(
    tmp_path: Path, unstructured_flow_model: GroundwaterFlowModel
):
    # create mask from idomain. Deactivate the same cell in all layers
    mask = deepcopy(unstructured_flow_model.domain)
    mask.loc[{"layer": 1, "mesh2d_nFaces": 23}] = -1
    mask.loc[{"layer": 2, "mesh2d_nFaces": 23}] = 0
    mask.loc[{"layer": 3, "mesh2d_nFaces": 23}] = 0

    # apply the mask to a model
    unstructured_flow_model.mask_all_packages(mask)

    # test output validity
    errors = unstructured_flow_model.validate("model")
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
    errors = unstructured_flow_model.validate("model")
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
    errors = unstructured_flow_model.validate("model")
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

 
@pytest.mark.parametrize("mask_cells", [( 2,1),  # case 1: disable a chd cell. These are indices, NOT coordinates. 
                                        ( 3,2)]) # case 2: disable all the cells the well ends up in
def test_mask_structured_xy_masks_across_all_layers(tmp_path: Path, structured_flow_model: GroundwaterFlowModel, mask_cells:tuple[int, int]):
    # Arrange 
    # add a well to the model
    well = imod.mf6.Well(
        x=[ 3.0],
        y=[ 3.0],
        screen_top=[0.0],
        screen_bottom=[ -3.0],
        rate=[1.0],
        print_flows=True,
        validate=True,
    )
    structured_flow_model["well"] = well
    
    mask = deepcopy(structured_flow_model.domain.sel(layer=1))
    mask = mask.drop_vars("layer")
    mask.values[*mask_cells] = 0

    cell_count = structured_flow_model.domain.size
    
    # Act    
    structured_flow_model.mask_all_packages( mask)
    
    # Assert
    assert all(structured_flow_model.domain.isel(y = mask_cells[0], x = mask_cells[1]).values == np.zeros(len(structured_flow_model.domain.layer)))
    unique, counts = np.unique(structured_flow_model.domain.values.reshape(cell_count), return_counts=True)
    assert counts[0] == len(structured_flow_model.domain.layer)
    assert counts[1] == cell_count - len(structured_flow_model.domain.layer)
    assert_model_can_run( structured_flow_model, "dis", tmp_path )    

   


@pytest.mark.parametrize("layer_mask", [[1,1,0], [0,1,1], [1,0,1]])
def test_mask_with_layer_array(
    tmp_path: Path,
    unstructured_flow_model: GroundwaterFlowModel,
    layer_mask: list[int]
):  
    """
    Specifying a mask as a dataset with only a layer coordinate is not necessarily something we want
    to maintain forever, so we should discourage external users from using this. 
    """
    nlayer = 3
    layer = np.arange(nlayer, dtype=int) + 1
    grid = unstructured_flow_model.domain.ugrid.grid
    mask =  xu.UgridDataArray(
        xr.DataArray(
            coords={"layer": layer},
            dims=["layer"],
        ),
        grid=grid,
    )
    mask.values = layer_mask

    unstructured_flow_model.mask_all_packages(mask)

    assert_model_can_run( unstructured_flow_model, "disv", tmp_path )

    

@pytest.mark.parametrize("mask_cell", [[1,1], [1,33]])
@pytest.mark.parametrize("inactivity_marker",[ 0, -1])  # 0 = inactive, -1 = vertical passthrough
def test_mask_unstructured(
    tmp_path: Path,
    unstructured_flow_model: GroundwaterFlowModel,
    mask_cell: list[int],
    inactivity_marker: int
):  

    # Arrange     
    layer_dim = len(unstructured_flow_model.domain.coords["layer"].values)
    planar_dim = len(unstructured_flow_model.domain.coords["mesh2d_nFaces"].values)
    cell_count = planar_dim * layer_dim
    mask = deepcopy(unstructured_flow_model.domain)
    mask.values[*mask_cell] = inactivity_marker

    # Act  
    unstructured_flow_model.mask_all_packages(mask)

    # Assert
    unique, counts = np.unique(unstructured_flow_model.domain.values.reshape(cell_count), return_counts=True)
    assert unstructured_flow_model.domain.values[*mask_cell] == inactivity_marker
    assert unique[0] == inactivity_marker
    assert counts[0] == 1
    assert counts[1] == cell_count - 1
    assert_model_can_run( unstructured_flow_model, "disv", tmp_path )


def test_mask_with_time_coordinate(
    tmp_path: Path,
    unstructured_flow_model: GroundwaterFlowModel,
):      

    nlayer = 3
    layer = np.arange(nlayer, dtype=int) + 1
    grid = unstructured_flow_model.domain.ugrid.grid
    mask =  xu.UgridDataArray(
        xr.DataArray(
            coords={"layer": layer, "time" : [1,2] },
            dims=["layer", "time"],
        ),
        grid=grid,
    )
    mask.sel(time=1).values = np.array([1,1,0])
    mask.sel(time=2).values = np.array([1,0,1])

    with pytest.raises(ValueError):
        unstructured_flow_model.mask_all_packages(mask)

@pytest.mark.parametrize("mask_cells",[ [(0, 2,1)],    # case 1: disable a chd cell 
                         [(0, 3,2),(1,3,2), (2,3,2)]]) # case 2: disable all the cells the well ends up in
@pytest.mark.parametrize("inactivity_marker",[ 0, -1]) # 0 = inactive, -1 = vertical passthrough
def test_mask_structured(tmp_path: Path, structured_flow_model: GroundwaterFlowModel, mask_cells: list[tuple[int, int, int]], inactivity_marker: int):

    # Arrange 
    # add a well to the model
    well = imod.mf6.Well(
        x=[ 3.0],
        y=[ 3.0],
        screen_top=[0.0],
        screen_bottom=[ -3.0],
        rate=[1.0],
        print_flows=True,
        validate=True,
    )
    structured_flow_model["well"] = well
    cell_count = structured_flow_model.domain.size

    mask = deepcopy(structured_flow_model.domain)
    for cell in mask_cells:
        mask.values[*cell ] = inactivity_marker

    # Act
    structured_flow_model.mask_all_packages( mask)

    # Assert
    unique, counts = np.unique(structured_flow_model.domain.values.reshape(cell_count), return_counts=True)
    assert unique[0] == inactivity_marker
    assert counts[0] == len(mask_cells)
    assert counts[1] == cell_count - len(mask_cells)
    assert_model_can_run( structured_flow_model, "dis", tmp_path )
