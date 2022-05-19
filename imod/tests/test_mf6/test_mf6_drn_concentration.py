import pathlib
import textwrap

import numpy as np
import pytest

import imod


@pytest.mark.usefixtures("concentration_fc", "elevation_fc", "conductance_fc")
def test_render(
    concentration_fc,
    elevation_fc,
    conductance_fc,
):
    directory = pathlib.Path("mymodel")
    globaltimes = [
        np.datetime64("2000-01-01"),
        np.datetime64("2000-01-02"),
        np.datetime64("2000-01-03"),
    ]

    drn = imod.mf6.Drainage(
        elevation=elevation_fc,
        conductance=conductance_fc,
        boundary_concentration=concentration_fc,
        transport_boundary_type="AUX",
    )

    actual = drn.render(directory, "drn", globaltimes, False)

    expected = textwrap.dedent(
        """\
        begin options
          auxiliary salinity  temperature 
        end options

        begin dimensions
          maxbound 0
        end dimensions

        begin period 1
          open/close mymodel/drn/drn-0.dat
        end period
        begin period 2
          open/close mymodel/drn/drn-1.dat
        end period
        begin period 3
          open/close mymodel/drn/drn-2.dat
        end period
        """
    )
    assert actual == expected
