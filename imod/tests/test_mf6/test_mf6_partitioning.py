

import pytest
import copy

from imod.mf6.modelsplitter import partition_structured_slices, split_model_packages, split_model_unstructured_packages

@pytest.mark.usefixtures("twri_model")
def test_partition_structured(twri_model, tmp_path):

    labels = copy.deepcopy(twri_model["GWF_1"]["dis"]["idomain"].sel({"layer": 1}))

    #fill the first half of the array- in the x direction-  with 0's and the second half with 1

    ny,  nx = labels.shape
    for ix in range(nx):
        for iy in range(ny):
            if ix < round( nx/2): 
                label = 0
            else:
                label = 1
            labels.values[iy, ix] = label
    
    new_models = split_model_packages( labels, twri_model["GWF_1"])


@pytest.mark.usefixtures("circle_model")
def test_partition_unstructured(circle_model, tmp_path):
    labels = copy.deepcopy(circle_model["GWF_1"]["disv"]["idomain"].sel({"layer": 1}))

    #fill the first half of the array- in the x direction-  with 0's and the second half with 1

    ncell = labels.shape[0]
    for icell  in range (ncell ):
        if icell < round(ncell /2):
            label = 0
        else:
            label = 1
        labels.values[icell] = label
        
    _ =split_model_unstructured_packages(labels, circle_model["GWF_1"])