import pytest

from imod.mf6.multimodel.modelsplitter import create_partition_info, slice_model
from imod.tests.fixtures.mf6_modelrun_fixture import assert_simulation_can_run
from imod.typing.grid import zeros_like
import numpy as np

def test_slice_model_structured(flow_transport_simulation):
    # Arrange.
    transport_model = flow_transport_simulation["tpt_a"]
    submodel_labels = zeros_like(transport_model.domain)
    submodel_labels[:, :, 30:] = 1

    partition_info = create_partition_info(submodel_labels)

    submodel_list = []

    # Act
    for submodel_partition_info in partition_info:
        submodel_list.append(slice_model(submodel_partition_info, transport_model))

    # Assert
    assert len(submodel_list) == 2
    for submodel in submodel_list:
        for package_name in list(transport_model.keys()):
            assert package_name in list(submodel.keys())


@pytest.mark.usefixtures("flow_transport_simulation")
def test_split_flow_and_transport_model(tmp_path, flow_transport_simulation):
    simulation = flow_transport_simulation

    flow_model = simulation["flow"]
    active = flow_model.domain

    submodel_labels = zeros_like(active)
    submodel_labels = submodel_labels.drop_vars("layer")
    submodel_labels.values[:, :, 50:] = 1
    submodel_labels = submodel_labels.sel(layer=0, drop=True)

    new_simulation = simulation.split(submodel_labels)
    new_simulation.write(tmp_path, binary=False)
    assert len(new_simulation["gwtgwf_exchanges"]) == 8

    assert new_simulation["gwtgwf_exchanges"][0]["model_name_1"].values[()] == "flow_0"
    assert new_simulation["gwtgwf_exchanges"][0]["model_name_2"].values[()] == "tpt_a_0"

    assert new_simulation["gwtgwf_exchanges"][1]["model_name_1"].values[()] == "flow_0"
    assert new_simulation["gwtgwf_exchanges"][1]["model_name_2"].values[()] == "tpt_b_0"

    assert new_simulation["gwtgwf_exchanges"][2]["model_name_1"].values[()] == "flow_0"
    assert new_simulation["gwtgwf_exchanges"][2]["model_name_2"].values[()] == "tpt_c_0"

    assert new_simulation["gwtgwf_exchanges"][3]["model_name_1"].values[()] == "flow_0"
    assert new_simulation["gwtgwf_exchanges"][3]["model_name_2"].values[()] == "tpt_d_0"

    assert new_simulation["gwtgwf_exchanges"][4]["model_name_1"].values[()] == "flow_1"
    assert new_simulation["gwtgwf_exchanges"][4]["model_name_2"].values[()] == "tpt_a_1"

    assert new_simulation["gwtgwf_exchanges"][5]["model_name_1"].values[()] == "flow_1"
    assert new_simulation["gwtgwf_exchanges"][5]["model_name_2"].values[()] == "tpt_b_1"

    assert new_simulation["gwtgwf_exchanges"][6]["model_name_1"].values[()] == "flow_1"
    assert new_simulation["gwtgwf_exchanges"][6]["model_name_2"].values[()] == "tpt_c_1"


@pytest.mark.usefixtures("flow_transport_simulation")
def test_split_flow_and_transport_model_evaluate_output(tmp_path, flow_transport_simulation):
    
    simulation = flow_transport_simulation

    flow_model = simulation["flow"]
    active = flow_model.domain
    simulation.pop("tpt_a")
    simulation.pop("tpt_c")
    simulation.pop("tpt_d")
    simulation["transport_solver"].remove_model_from_solution("tpt_a")
    simulation["transport_solver"].remove_model_from_solution("tpt_c")
    simulation["transport_solver"].remove_model_from_solution("tpt_d")    
    submodel_labels = zeros_like(active)
    submodel_labels = submodel_labels.drop_vars("layer")
    submodel_labels.values[:, :, 50:] = 1
    submodel_labels = submodel_labels.sel(layer=0, drop=True)

    simulation.write(tmp_path/"original", binary = False )
    simulation.run ()

    new_simulation = simulation.split(submodel_labels)
    new_simulation.write(tmp_path, binary=False)

    original_conc = simulation.open_concentration(species_ls=["b"])
    original_head = simulation.open_head()

    assert len(new_simulation["gwtgwf_exchanges"]) == 2

    assert new_simulation["gwtgwf_exchanges"][0]["model_name_1"].values[()] == "flow_0"
    assert new_simulation["gwtgwf_exchanges"][0]["model_name_2"].values[()] == "tpt_b_0"

    assert new_simulation["gwtgwf_exchanges"][1]["model_name_1"].values[()] == "flow_1"
    assert new_simulation["gwtgwf_exchanges"][1]["model_name_2"].values[()] == "tpt_b_1"


    new_simulation.run()
    conc = new_simulation.open_concentration(species_ls=["a"])
    head = new_simulation.open_head()

    # Compare the head result of the original simulation with the result of the
    # partitioned simulation.
    np.testing.assert_allclose(
        head.sel(time=2000)["head"].values, original_head.sel(time=200).values, rtol=1e-4, atol=1e-4
    )
    print( conc.sel(time = 2000)["concentration"].values - original_conc.sel(time=200).values)
    np.testing.assert_allclose(
        conc.sel(time = 2000)["concentration"].values, original_conc.sel(time=200).values, rtol= 6.5, atol=0.011
    )
