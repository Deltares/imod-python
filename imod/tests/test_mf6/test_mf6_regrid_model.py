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
