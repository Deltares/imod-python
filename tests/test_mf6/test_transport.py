import numpy as np
import pytest
import xarray as xr
import pathlib
import textwrap
import imod
from imod.mf6.adv import AdvectionSchemes
import imod.mf6.model


def test_advection_():

    a = imod.mf6.Advection(AdvectionSchemes.upstream)

    directory = pathlib.Path("mymodel")
    globaltimes = [np.datetime64("2000-01-01")]    
    actual = a.render(directory, "adv", globaltimes, True)
    expected = 'begin options\n  scheme upstream\n\nend options'
    assert actual == expected
    
def test_advection_default():

    a = imod.mf6.Advection() #call constructor without parameters

    directory = pathlib.Path("mymodel")
    globaltimes = [np.datetime64("2000-01-01")]    
    actual = a.render(directory, "adv", globaltimes, True)
    expected = 'begin options\n  scheme upstream\n\nend options'
    assert actual == expected

def test_transport_():
    m = imod.mf6.model.GroundwaterTransportModel()
    a = imod.mf6.Advection(AdvectionSchemes.upstream)
    directory = pathlib.Path("mymodel")
    globaltimes = [np.datetime64("2000-01-01")]        
    m["adv"] = a
    actual = m.render("dummy")
    expected = textwrap.dedent(
        """\
        begin options
        end options

        begin packages
          adv6 dummy/adv.adv adv
        end packages
       """)
    assert actual == expected

def test_difdisp_default():
    directory = pathlib.Path("mymodel")
    globaltimes = [np.datetime64("2000-01-01")]      
    dd = imod.mf6.DifDisp(1e-4, 1, 10) 
    m = imod.mf6.model.GroundwaterTransportModel()
    m["dsp"] = dd
    actual =dd.render(directory, "dsp", globaltimes, True)
    expected=textwrap.dedent(
      '''\
      begin options
      end options

      begin griddata
        diffc
          constant 0.0001

        alh
          constant 1

        ath1
          constant 10


      end griddata
      '''
    )
    assert actual == expected


test_difdisp_default()