import textwrap

import numpy as np
import pytest

import imod
from imod.mf6.adv import AdvectionCentral
from imod.mf6.write_context import WriteContext


def test_long_package_name():
    m = imod.mf6.GroundwaterTransportModel()
    with pytest.raises(
        KeyError,
        match="Received key with more than 16 characters: 'my_very_long_package_name'Modflow 6 has a character limit of 16.",
    ):
        m["my_very_long_package_name"] = imod.mf6.AdvectionCentral()


def test_transport_model_rendering():
    adv = imod.mf6.AdvectionCentral()
    disp = imod.mf6.Dispersion(1e-4, 1.0, 10.0, 1.0, 2.0, 3.0, True, True)
    m = imod.mf6.GroundwaterTransportModel(print_input=True, save_flows=True)
    m["dsp"] = disp
    m["adv"] = adv
    write_context = WriteContext()
    actual = m.render("GWF_1", write_context)

    actual = m.render("transport", write_context)
    expected = textwrap.dedent(
        """\
      begin options
        print_input
        save_flows
      end options

      begin packages
        dsp6 transport/dsp.dsp dsp
        adv6 transport/adv.adv adv
      end packages
      """
    )
    assert actual == expected


def test_assign_flow_discretization(basic_dis, concentration_fc):
    # define a grid
    idomain, _, bottom = basic_dis

    like = idomain.sel(layer=1).astype(np.float32)
    concentration = concentration_fc.sel(layer=1)

    gwf_model = imod.mf6.GroundwaterFlowModel()
    gwf_model["dis"] = imod.mf6.StructuredDiscretization(
        top=200.0, bottom=bottom, idomain=idomain
    )
    gwf_model["riv-1"] = imod.mf6.River(
        stage=like,
        conductance=like,
        bottom_elevation=like - 2.0,
        concentration=concentration,
        concentration_boundary_type="AUX",
    )

    tpt_model = imod.mf6.GroundwaterTransportModel()
    tpt_model["ssm"] = imod.mf6.SourceSinkMixing.from_flow_model(gwf_model, "salinity")
    tpt_model["advection"] = AdvectionCentral()

    # let the transport model take the discretization from the flow model
    tpt_model["dis"] = gwf_model["dis"]

    # check that the discretization was added to the transport model
    assert len(tpt_model.keys()) == 3
    assert "dis" in tpt_model.keys()
    assert isinstance(tpt_model["dis"], imod.mf6.StructuredDiscretization)


@pytest.mark.usefixtures("twri_model")
def test_flowmodel_validation(twri_model):
    # initialize transport with a flow model without concentration data
    flow_model = twri_model["GWF_1"]
    with pytest.raises(ValueError):
        imod.mf6.SourceSinkMixing.from_flow_model(flow_model, "salinity")


@pytest.mark.usefixtures("flow_transport_simulation")
def test_transport_concentration_loading(tmp_path, flow_transport_simulation):

    flow_transport_simulation.write(tmp_path)
    flow_transport_simulation.run()

    conc_notime = flow_transport_simulation.open_concentration(species_ls=["a", "b", "c", "d"])
    assert conc_notime.coords["time"].dtype == float

    conc_time = flow_transport_simulation.open_concentration(species_ls=["a", "b", "c", "d"] ,simulation_start_time = "2000-01-31", time_unit = "s")
    assert conc_time.coords["time"].dtype ==  np.dtype('datetime64[ns]')


@pytest.mark.usefixtures("flow_transport_simulation")
def test_transport_balance_loading(tmp_path, flow_transport_simulation):

    flow_transport_simulation.write(tmp_path)
    flow_transport_simulation.run()

    balance_notime = flow_transport_simulation.open_transport_budget(species_ls=["a", "b", "d"])
    assert balance_notime.coords["time"].dtype == float

    balance_time = flow_transport_simulation.open_transport_budget(species_ls=["a", "b", "d"],simulation_start_time = "2000-01-31", time_unit = "s")
    assert balance_time.coords["time"].dtype == np.dtype('datetime64[ns]')

    assert np.all(balance_notime.sel(species = "a")["ssm"].values == balance_time.sel(species = "a")["ssm"].values)