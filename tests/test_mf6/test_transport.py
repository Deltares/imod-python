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


