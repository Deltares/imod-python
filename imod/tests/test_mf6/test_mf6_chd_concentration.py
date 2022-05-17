import textwrap
import numpy as np
import pytest
import imod
import pathlib

@pytest.mark.usefixtures("head_fc", "concentration_fc")
def test_render(head_fc, concentration_fc):
    directory = pathlib.Path("mymodel")
    globaltimes = [np.datetime64("2000-01-01"),np.datetime64("2000-01-02"),np.datetime64("2000-01-03")]

    chd = imod.mf6.ConstantHead(
        head_fc,concentration_fc, "AUX",  print_input=True, print_flows=True, save_flows=True
    )


    actual = chd.render(directory, "chd", globaltimes, False)

    expected = textwrap.dedent(
        """\
        begin options
          auxiliary salinity  temperature 
          print_input
          print_flows
          save_flows
        end options

        begin dimensions
          maxbound 0
        end dimensions

        begin period 1
          open/close mymodel/chd/chd-0.dat
        end period
        begin period 2
          open/close mymodel/chd/chd-1.dat
        end period
        begin period 3
          open/close mymodel/chd/chd-2.dat
        end period
        """
    )
    assert actual == expected