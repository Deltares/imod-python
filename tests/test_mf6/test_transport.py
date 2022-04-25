import numpy as np
import pytest
import xarray as xr
import pathlib
import textwrap
import imod
from imod.mf6.adv import AdvectionSchemes
import imod.mf6.model


def test_tranport_():

    a = imod.mf6.Advection(AdvectionSchemes.upstream)

    directory = pathlib.Path("mymodel")
    globaltimes = [np.datetime64("2000-01-01")]    
    actual = a.render(directory, "adv", globaltimes, True)
    expected = 'begin options\n  scheme upstream\n\nend options'
    assert actual == expected
    


test_tranport_()  


