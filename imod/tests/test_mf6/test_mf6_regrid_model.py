
import imod
import xarray as xr
import xugrid as xu
import numpy as np
from imod.tests.fixtures.mf6_regridding_fixture import grid_data_structured, grid_data_unstructured

def test_regrid_structured_model_to_structured_model(structured_flow_model):
 
    finer_idomain = grid_data_structured(np.int32, 1, 0.4)

    new_gwf_model = structured_flow_model.regrid_like(finer_idomain)

    assert len(new_gwf_model.items()) == len (structured_flow_model.items())
    validation_result =  new_gwf_model._validate("test_model")
    assert not validation_result.has_errors()


def test_regrid_unstructured_model_to_unstructured_model(unstructured_flow_model):
 
    finer_idomain = grid_data_unstructured(np.int32, 1, 0.4)

    new_gwf_model = unstructured_flow_model.regrid_like(finer_idomain)

    assert len(new_gwf_model.items()) == len (unstructured_flow_model.items())
    validation_result = new_gwf_model._validate("test_model")
    assert not validation_result.has_errors()
