import pathlib
import textwrap

import numpy as np
import pytest

import imod


@pytest.mark.usefixtures("concentration_fc")
def test_render(concentration_fc):
    directory = pathlib.Path("mymodel")
    globaltimes = [
        np.datetime64("2000-01-01"),
        np.datetime64("2000-01-02"),
        np.datetime64("2000-01-03"),
    ]

    riv = imod.mf6.River(
        stage=1.0,
        conductance=10.0,
        bottom_elevation=-1.0,
        boundary_concentration=concentration_fc,
        transport_boundary_type="AUX",
    )
    actual = riv.render(directory, "riv", globaltimes, False)

    expected = textwrap.dedent(
        """\
        begin options
          auxiliary salinity temperature
        end options

        begin dimensions
          maxbound 1
        end dimensions

        begin period 1
          open/close mymodel/riv/riv.dat
        end period
        """
    )
    assert actual == expected
