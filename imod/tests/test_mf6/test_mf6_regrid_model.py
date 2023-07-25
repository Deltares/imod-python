from copy import deepcopy

import numpy as np
import pytest

import imod
from imod.mf6.lak import Lake
from imod.tests.fixtures.mf6_regridding_fixture import (
    grid_data_structured,
    grid_data_unstructured,
)


def test_regrid_structured_model_to_structured_model(
    structured_flow_model: imod.mf6.GroundwaterFlowModel,
):
    finer_idomain = grid_data_structured(np.int32, 1, 0.4)

    new_gwf_model = structured_flow_model.regrid_like(finer_idomain)

    assert len(new_gwf_model.items()) == len(structured_flow_model.items())
    validation_result = new_gwf_model._validate("test_model")
    assert not validation_result.has_errors()


def test_regrid_structured_model_with_wells_to_structured_model(
    structured_flow_model: imod.mf6.GroundwaterFlowModel,
):
    well_package = imod.mf6.Well(
        screen_top=[0.0, 1],
        screen_bottom=[-5.0, -6.0],
        y=[1.0, 5.0],
        x=[2.0, 3.0],
        rate=[1.0, 3.0],
    )
    structured_flow_model["well"] = well_package

    finer_idomain = grid_data_structured(np.int32, 1, 0.4)

    new_gwf_model = structured_flow_model.regrid_like(finer_idomain)

    assert len(new_gwf_model.items()) == len(structured_flow_model.items())
    validation_result = new_gwf_model._validate("test_model")
    assert not validation_result.has_errors()


def test_regrid_unstructured_model_with_wells_to_unstructured_model(
    unstructured_flow_model: imod.mf6.GroundwaterFlowModel,
):
    well_package = imod.mf6.Well(
        screen_top=[0.0, 1],
        screen_bottom=[-5.0, -6.0],
        y=[1.0, 5.0],
        x=[2.0, 3.0],
        rate=[1.0, 3.0],
    )
    unstructured_flow_model["well"] = well_package

    finer_idomain = grid_data_unstructured(np.int32, 1, 0.4)

    new_gwf_model = unstructured_flow_model.regrid_like(finer_idomain)

    assert len(new_gwf_model.items()) == len(unstructured_flow_model.items())
    validation_result = new_gwf_model._validate("test_model")
    assert not validation_result.has_errors()


def test_regrid_unstructured_model_to_unstructured_model(
    unstructured_flow_model: imod.mf6.GroundwaterFlowModel,
):
    finer_idomain = grid_data_unstructured(np.int32, 1, 0.4)

    new_gwf_model = unstructured_flow_model.regrid_like(finer_idomain)

    assert len(new_gwf_model.items()) == len(unstructured_flow_model.items())
    validation_result = new_gwf_model._validate("test_model")
    assert not validation_result.has_errors()


def test_regrid_model_with_unsupported_package(
    unstructured_flow_model, naardermeer, ijsselmeer
):
    lake_pkg = Lake.from_lakes_and_outlets(
        [naardermeer(), ijsselmeer()],
    )
    unstructured_flow_model["lake"] = lake_pkg

    finer_idomain = grid_data_unstructured(np.int32, 1, 0.4)
    with pytest.raises(
        NotImplementedError,
        match="regridding is not implemented for package lake of type <class 'imod.mf6.lak.Lake'>",
    ):
        _ = unstructured_flow_model.regrid_like(finer_idomain)


@pytest.mark.parametrize("inactivity_marker", [0, -1])
def test_regrid_unstructured_model_with_inactive_cells(
    inactivity_marker: int,
    unstructured_flow_model: imod.mf6.GroundwaterFlowModel,
):
    """
    inactivity_marker = 0  -> inactive cell
    inactivity_marker = -1 -> vertical passthrough cell
    """
    inactive_cells = unstructured_flow_model.get_domain()
    inactive_cells.loc[{"layer": 1, "mesh2d_nFaces": 23}] = inactivity_marker
    inactive_cells.loc[{"layer": 3, "mesh2d_nFaces": 23}] = inactivity_marker

    unstructured_flow_model["disv"]["idomain"] = inactive_cells

    finer_idomain = grid_data_unstructured(np.int32, 1, 0.4)

    new_gwf_model = unstructured_flow_model.regrid_like(finer_idomain)

    assert len(new_gwf_model.items()) == len(unstructured_flow_model.items())
    validation_result = new_gwf_model._validate("test_model")
    assert not validation_result.has_errors()
    new_idomain = new_gwf_model.get_domain()
    assert new_idomain.max().values[()] == 1 and new_idomain.min().values[()] == 0
    # Check that write validation still fails for the regridded package
    new_bottom = deepcopy(new_grid)
    new_bottom.loc[{"layer": 1}] = 0.0
    new_bottom.loc[{"layer": 2}] = -1.0
    new_bottom.loc[{"layer": 3}] = -2.0

    regridded_package = regridded_model["sto"]
    pkg_errors = regridded_package._validate(
        schemata=imod.mf6.StorageCoefficient._write_schemata,
        idomain=new_grid,
        bottom=new_bottom,
    )

    # Check that the right errors were found
    assert len(pkg_errors) == 2
    assert (
        str(pkg_errors["storage_coefficient"])
        == "[ValidationError('not all values comply with criterion: >= 0.0')]"
    )
    assert (
        str(pkg_errors["specific_yield"])
        == "[ValidationError('not all values comply with criterion: >= 0.0')]"
    )

def test_model_regridding_can_skip_validation(
    structured_flow_model: imod.mf6.GroundwaterFlowModel,
):
    """
    This tests if an invalid model can be regridded by turning off validation
    """

    # create a sto package with a negative storage coefficient. This would trigger a validation error if it were turned on.
    storage_coefficient = grid_data_structured(np.float64, -20.0, 0.25)
    specific_yield = grid_data_structured(np.float64, -30.0, 0.25)
    sto_package = imod.mf6.StorageCoefficient(
        storage_coefficient,
        specific_yield,
        transient=True,
        convertible=False,
        save_flows=True,
        validate=False,
    )
    structured_flow_model["sto"] = sto_package

    # Regrid the package to a finer domain
    new_grid = grid_data_structured(np.float64, 1.0, 0.025)
    regridded_model = structured_flow_model.regrid_like(new_grid, validate=False)

    # Check that write validation still fails for the regridded package
    new_bottom = deepcopy(new_grid)
    new_bottom.loc[{"layer": 1}] = 0.0
    new_bottom.loc[{"layer": 2}] = -1.0
    new_bottom.loc[{"layer": 3}] = -2.0

    regridded_package = regridded_model["sto"]
    pkg_errors = regridded_package._validate(
        schemata=imod.mf6.StorageCoefficient._write_schemata,
        idomain=new_grid,
        bottom=new_bottom,
    )

    # Check that the right errors were found
    assert len(pkg_errors) == 2
    assert (
        str(pkg_errors["storage_coefficient"])
        == "[ValidationError('not all values comply with criterion: >= 0.0')]"
    )
    assert (
        str(pkg_errors["specific_yield"])
        == "[ValidationError('not all values comply with criterion: >= 0.0')]"
    )


def test_model_regridding_can_validate(
    structured_flow_model: imod.mf6.GroundwaterFlowModel,
):
    """
    This tests if an invalid model will throw a validation error on regridding if validation is turned on
    """

    # Create a storage package with a negative storage coefficient. This would trigger a validation error if it were turned on.
    storage_coefficient = grid_data_structured(np.float64, -20, 0.25)
    specific_yield = grid_data_structured(np.float64, -30, 0.25)
    sto_package = imod.mf6.StorageCoefficient(
        storage_coefficient,
        specific_yield,
        transient=True,
        convertible=False,
        save_flows=True,
        validate=False,
    )
    structured_flow_model["sto"] = sto_package

    # Create  a finer domain to regrid to
    new_grid = grid_data_structured(np.float64, 1, 0.025)

    # Check that a validation error is thrown while regridding
    with pytest.raises(imod.schemata.ValidationError):
        _ = structured_flow_model.regrid_like(new_grid, validate=True)
