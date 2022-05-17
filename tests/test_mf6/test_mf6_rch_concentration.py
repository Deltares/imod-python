import textwrap
import numpy as np
import pytest
import imod
import pathlib

@pytest.mark.usefixtures( "concentration_fc", "rate_fc")
def test_render(concentration_fc, rate_fc ):
    directory = pathlib.Path("mymodel")
    globaltimes = [np.datetime64("2000-01-01"),np.datetime64("2000-01-02"),np.datetime64("2000-01-03")]

    rch = imod.mf6.Recharge(rate=rate_fc,boundary_concentration =concentration_fc, transport_boundary_type="AUX")

    actual= rch.render(directory, "rch", globaltimes, False)

    expected = textwrap.dedent(
        """\
        begin options
          auxiliary salinity  temperature 
        end options

        begin dimensions
          maxbound 0
        end dimensions

        begin period 1
          open/close mymodel/rch/rch-0.dat
        end period
        begin period 2
          open/close mymodel/rch/rch-1.dat
        end period
        begin period 3
          open/close mymodel/rch/rch-2.dat
        end period
        """
    )
    assert actual == expected