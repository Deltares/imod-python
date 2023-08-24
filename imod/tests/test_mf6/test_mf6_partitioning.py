

import pytest
import copy

from imod.mf6.modelsplitter import partition_structured_slices, split_model_packages, split_model_unstructured_packages
import imod

@pytest.mark.usefixtures("twri_model")
def test_partition_structured(twri_model, tmp_path):

    labels = copy.deepcopy(twri_model["GWF_1"]["dis"]["idomain"].sel({"layer": 1}))
    number_of_subdomains = 2
    #fill the first half of the array- in the x direction-  with 0's and the second half with 1

    ny,  nx = labels.shape
    
    for isub in range(1, number_of_subdomains+1):
        from_index =  round(ny/number_of_subdomains)*(isub  - 1) 
        to_index = round(ny/number_of_subdomains)*(isub) 
        for ix in range(nx):
            for iy in range(ny):
                if iy >= from_index and iy < to_index:
                    labels.values[iy, ix] = isub
    
    new_models = split_model_packages( labels, twri_model["GWF_1"])
    
    for isub in range(number_of_subdomains):
        submodel = new_models[isub ]
        new_simulation = imod.mf6.Modflow6Simulation("*")
        new_model_name = f"partial_GWF_{isub}"
        new_simulation[new_model_name] = submodel
        new_simulation[new_model_name].pop("wel")
        # Define solver settings
        new_simulation["solver"] = imod.mf6.Solution(
            modelnames=[new_model_name],
            print_option="summary",
            csv_output=False,
            no_ptc=True,
            outer_dvclose=1.0e-4,
            outer_maximum=500,
            under_relaxation=None,
            inner_dvclose=1.0e-4,
            inner_rclose=0.001,
            inner_maximum=100,
            linear_acceleration="cg",
            scaling_method=None,
            reordering_method=None,
            relaxation_factor=0.97,
        )        
        new_simulation.create_time_discretization(additional_times=["2000-01-01T00:00", "2000-01-01T06:00"])
        new_simulation.write(tmp_path, False, False)
        new_simulation.run()
 

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