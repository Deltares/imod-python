import numpy as np
import pytest

from imod.mf6.adv import Advection
from imod.mf6.dsp import Dispersion
from imod.mf6.multimodel.modelsplitter import create_partition_info, slice_model
from imod.mf6.simulation import Modflow6Simulation
from imod.tests.fixtures.mf6_modelrun_fixture import assert_simulation_can_run
from imod.typing.grid import zeros_like


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
    submodel_labels.values[:, :, 15:] = 1
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

    assert new_simulation["gwtgwf_exchanges"][7]["model_name_1"].values[()] == "flow_1"
    assert new_simulation["gwtgwf_exchanges"][7]["model_name_2"].values[()] == "tpt_d_1"
    for species_name in ["a", "b", "c", "d"]:
        assert list(
            new_simulation[f"tpt_{species_name}_0"]["ssm"]
            .dataset["package_names"]
            .values
        ) == ["chd", "well"]
        assert list(
            new_simulation[f"tpt_{species_name}_1"]["ssm"]
            .dataset["package_names"]
            .values
        ) == ["chd", "rch", "well"]

    assert_simulation_can_run(new_simulation, "dis", tmp_path)

@pytest.mark.usefixtures("flow_transport_simulation")
def test_split_flow_and_transport_model_evaluate_output(
    tmp_path, flow_transport_simulation
):
    simulation = flow_transport_simulation

    flow_model = simulation["flow"]
    active = flow_model.domain

    # create label array
    submodel_labels = zeros_like(active)
    submodel_labels = submodel_labels.drop_vars("layer")
    submodel_labels.values[:, :, 30:] = 1
    submodel_labels = submodel_labels.sel(layer=0, drop=True)

    # for reference run the original model and load the results
    simulation.write(tmp_path / "original", binary=False)
    simulation.run()
    original_conc = simulation.open_concentration()
    original_head = simulation.open_head()

    # split the model , run the split model and load the results
    new_simulation = simulation.split(submodel_labels)
    new_simulation.write(tmp_path, binary=False)
    new_simulation.run()
    conc = new_simulation.open_concentration()
    head = new_simulation.open_head()

    # Compare
    np.testing.assert_allclose(
        head.sel(time=2000)["head"].values,
        original_head.sel(time=2000).values,
        rtol=1e-4,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        conc.sel(time=2000)["concentration"].values,
        original_conc.sel(time=2000).values,
        rtol=1e-4,
        atol=1e-6,
    )


@pytest.mark.usefixtures("flow_transport_simulation")
def test_split_flow_and_transport_model_evaluate_output_with_species(
    tmp_path, flow_transport_simulation
):
    simulation = flow_transport_simulation

    flow_model = simulation["flow"]
    active = flow_model.domain

    # create label array
    submodel_labels = zeros_like(active)
    submodel_labels = submodel_labels.drop_vars("layer")
    submodel_labels.values[:, :, 30:] = 1
    submodel_labels = submodel_labels.sel(layer=0, drop=True)

    # for reference run the original model and load the results
    simulation.write(tmp_path / "original", binary=False)
    simulation.run()
    original_conc = simulation.open_concentration(species_ls=["a", "b", "c", "d"])

    # split the model , run the split model and load the results
    new_simulation = simulation.split(submodel_labels)
    new_simulation.write(tmp_path, binary=False)
    new_simulation.run()
    conc = new_simulation.open_concentration(species_ls=["a", "b", "c", "d"])

    # Assert species coordinates
    assert all(original_conc.coords["species"] == ["a", "b", "c", "d"])
    assert all(conc.coords["species"] == ["a", "b", "c", "d"])

    # Compare
    np.testing.assert_allclose(
        conc.sel(time=2000)["concentration"].values,
        original_conc.sel(time=2000).values,
        rtol=1e-4,
        atol=1e-6,
    )

@pytest.mark.usefixtures("flow_transport_simulation")
@pytest.mark.parametrize("advection_scheme", ["TVD", "upstream", "central"])
@pytest.mark.parametrize("dsp_xt3d", [True, False])
def test_split_flow_and_transport_settings(
    tmp_path, flow_transport_simulation, advection_scheme, dsp_xt3d
):
    simulation = flow_transport_simulation

    flow_model = simulation["flow"]
    active = flow_model.domain

    simulation["tpt_b"]["dsp"] = Dispersion(
        diffusion_coefficient=1e-2,
        longitudinal_horizontal=20.0,
        transversal_horizontal1=2.0,
        xt3d_off=dsp_xt3d,
        xt3d_rhs=dsp_xt3d,
    )
    simulation["tpt_b"]["adv"] = Advection(scheme=advection_scheme)
    # create label array
    submodel_labels = zeros_like(active)
    submodel_labels = submodel_labels.drop_vars("layer")
    submodel_labels.values[:, :, 30:] = 1
    submodel_labels = submodel_labels.sel(layer=0, drop=True)

    new_simulation = simulation.split(submodel_labels)
    assert (
        new_simulation["split_exchanges"][1].dataset["adv_scheme"].values[()]
        == advection_scheme
    )
    assert (
        new_simulation["split_exchanges"][1].dataset["dsp_xt3d_off"].values[()]
        == dsp_xt3d
    )
    assert (
        new_simulation["split_exchanges"][1].dataset["dsp_xt3d_rhs"].values[()]
        == dsp_xt3d
    )
