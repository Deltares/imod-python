import pathlib
import textwrap

import numpy as np
import pytest

import imod


@pytest.mark.usefixtures(
    "concentration_fc", "proportion_rate_fc", "proportion_depth_fc"
)
def test_render(concentration_fc, proportion_rate_fc, proportion_depth_fc):
    directory = pathlib.Path("mymodel")
    globaltimes = [
        np.datetime64("2000-01-01"),
        np.datetime64("2000-01-02"),
        np.datetime64("2000-01-03"),
    ]

    evt = imod.mf6.Evapotranspiration(
        surface=25.0,
        rate=1e-3,
        depth=1.0,
        proportion_rate=proportion_rate_fc,
        proportion_depth=proportion_depth_fc,
        concentration=concentration_fc,
        concentration_boundary_type="AUX",
    )

    actual = evt.render(directory, "evt", globaltimes, False)

    expected = textwrap.dedent(
        """\
        begin options
          auxiliary salinity temperature

        end options

        begin dimensions
          maxbound 1
          nseg
        end dimensions


        begin period 1
          open/close mymodel/evt/evt-0.dat
        end period
        begin period 2
          open/close mymodel/evt/evt-1.dat
        end period
        begin period 3
          open/close mymodel/evt/evt-2.dat
        end period"""
    )
    assert actual == expected
