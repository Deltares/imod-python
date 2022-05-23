import pathlib
import textwrap

import numpy as np
import pytest

from imod.mf6.ssm import Transport_Sink_Sources


@pytest.mark.usefixtures("flow_model_with_concentration")
def test_transport_model_rendering(flow_model_with_concentration):
    directory = pathlib.Path("mymodel")
    globaltimes = [np.datetime64("2000-01-01")]
    m = Transport_Sink_Sources(flow_model_with_concentration, "salinity")
    actual = m.render(directory, "river", globaltimes, True)
    expected = textwrap.dedent(
        """\
        begin options
        end options

        begin sources
            # pname          srctype           auxname

            riv-1   AUX   salinity
        end sources"""
    )
    assert actual == expected
