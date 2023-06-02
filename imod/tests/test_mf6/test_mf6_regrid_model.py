import numpy as np
import pytest

import imod
from imod.tests.fixtures.mf6_regridding_fixture import (
    grid_data_structured,
    grid_data_unstructured,
)


def test_regrid_structured_model_to_structured_model(structured_flow_model):
    finer_idomain = grid_data_structured(np.int32, 1, 0.4)

    new_gwf_model = structured_flow_model.regrid_like(finer_idomain)

    assert len(new_gwf_model.items()) == len(structured_flow_model.items())
    validation_result = new_gwf_model._validate("test_model")
    assert not validation_result.has_errors()


def test_regrid_unstructured_model_to_unstructured_model(unstructured_flow_model):
    finer_idomain = grid_data_unstructured(np.int32, 1, 0.4)

    new_gwf_model = unstructured_flow_model.regrid_like(finer_idomain)

    assert len(new_gwf_model.items()) == len(unstructured_flow_model.items())
    validation_result = new_gwf_model._validate("test_model")
    assert not validation_result.has_errors()


def test_regrid_model_with_unsupported_package(unstructured_flow_model):
    unstructured_flow_model["well"] = imod.mf6.Well([0.0], [-10.0], [5.0], [5.0], [2.0])

    finer_idomain = grid_data_unstructured(np.int32, 1, 0.4)
    with pytest.raises(
        NotImplementedError,
        match="regridding is not implemented for package well of type <class 'imod.mf6.wel.Well'>",
    ):
        _ = unstructured_flow_model.regrid_like(finer_idomain)
