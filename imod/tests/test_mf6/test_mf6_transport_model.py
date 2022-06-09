import textwrap

import xarray as xr

import imod
from imod.mf6.adv import AdvectionCentral
import imod.mf6.model
import numpy as np


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

def test_assign_flow_discretization(basic_dis,):

    #define a grid
    idomain, _, bottom = basic_dis

    gwf_model = imod.mf6.GroundwaterFlowModel()
    gwf_model["dis"] = imod.mf6.StructuredDiscretization(
        top=200.0, bottom=bottom, idomain=idomain
    )

    #define a transport model, with the key "dis" not in use
    tpt_model = imod.mf6.model.GroundwaterTransportModel(gwf_model, "None")
    tpt_model["advection"] = AdvectionCentral()

    #let the transport model take the discretization from the flow model
    tpt_model.take_discretization_from_model(gwf_model)

    #check that the discretization was added to the transport model
    assert len(tpt_model.keys())==3
    assert "dis" in tpt_model.keys()
    assert isinstance( tpt_model["dis"], imod.mf6.StructuredDiscretization)

def test_assign_flow_discretization2(basic_dis,):

    #define a grid
    idomain, _, bottom = basic_dis

    gwf_model = imod.mf6.GroundwaterFlowModel()
    gwf_model["dis"] = imod.mf6.StructuredDiscretization(
        top=200.0, bottom=bottom, idomain=idomain
    )

    #define a transport model, with the key "dis" in use
    tpt_model = imod.mf6.model.GroundwaterTransportModel(gwf_model, "None")
    tpt_model["dis"] = AdvectionCentral()

    #let the transport model take the discretization from the flow model
    tpt_model.take_discretization_from_model(gwf_model)

    #check that the discretization was added to the transport model
    assert len(tpt_model.keys())==3
    assert "disX" in tpt_model.keys()
    assert  isinstance( tpt_model["disX"], imod.mf6.StructuredDiscretization)
