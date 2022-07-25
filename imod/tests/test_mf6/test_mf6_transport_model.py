import textwrap

import pytest

import imod
import imod.mf6.model
from imod.mf6.adv import AdvectionCentral


def test_transport_model_rendering():

    adv = imod.mf6.AdvectionCentral()
    disp = imod.mf6.Dispersion(1e-4, 1.0, 10.0, 1.0, 2.0, 3.0, True, True)
    m = imod.mf6.model.GroundwaterTransportModel(None, None)
    m["dsp"] = disp
    m["adv"] = adv
    actual = m.render("transport")
    expected = textwrap.dedent(
        """\
      begin options
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

    gwf_model = imod.mf6.GroundwaterFlowModel()
    gwf_model["dis"] = imod.mf6.StructuredDiscretization(
        top=200.0, bottom=bottom, idomain=idomain
    )
    gwf_model["riv-1"] = imod.mf6.River(
        stage=1.0,
        conductance=10.0,
        bottom_elevation=-1.0,
        concentration=concentration_fc,
        concentration_boundary_type="AUX",
    )

    # define a transport model, with the key "dis" not in use
    tpt_model = imod.mf6.model.GroundwaterTransportModel(gwf_model, "salinity")
    tpt_model["advection"] = AdvectionCentral()

    # let the transport model take the discretization from the flow model
    tpt_model["dis"] = gwf_model["dis"]

    # check that the discretization was added to the transport model
    assert len(tpt_model.keys()) == 3
    assert "dis" in tpt_model.keys()
    assert isinstance(tpt_model["dis"], imod.mf6.StructuredDiscretization)


def test_assign_flow_discretization2(basic_dis, concentration_fc):

    # define a grid
    idomain, _, bottom = basic_dis

    gwf_model = imod.mf6.GroundwaterFlowModel()
    gwf_model["dis"] = imod.mf6.StructuredDiscretization(
        top=200.0, bottom=bottom, idomain=idomain
    )
    gwf_model["riv-1"] = imod.mf6.River(
        stage=1.0,
        conductance=10.0,
        bottom_elevation=-1.0,
        concentration=concentration_fc,
        concentration_boundary_type="AUX",
    )
    # define a transport model, with the key "dis" in use
    tpt_model = imod.mf6.model.GroundwaterTransportModel(gwf_model, "salinity")
    tpt_model["adv"] = AdvectionCentral()

    # let the transport model take the discretization from the flow model
    tpt_model["dis"] = gwf_model["dis"]

    # check that the discretization was added to the transport model
    assert len(tpt_model.keys()) == 3
    assert "disX" in tpt_model.keys()
    assert isinstance(tpt_model["disX"], imod.mf6.StructuredDiscretization)


@pytest.mark.usefixtures("twri_model")
def test_flowmodel_validation(twri_model):
    # initialize transport with a flow model without concentration data
    flow_model = twri_model["GWF_1"]
    with pytest.raises(ValueError):
        imod.mf6.model.GroundwaterTransportModel(flow_model, "salinity")
