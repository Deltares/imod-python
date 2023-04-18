import pathlib
import textwrap

import numpy as np
import pytest

from imod.mf6.ssm import SourceSinkMixing
from imod.schemata import ValidationError


@pytest.mark.usefixtures("flow_model_with_concentration")
def test_transport_model_rendering(flow_model_with_concentration):
    directory = pathlib.Path("mymodel")
    globaltimes = np.array(["2000-01-01"], dtype="datetime64[ns]")
    m = SourceSinkMixing.from_flow_model(flow_model_with_concentration, "salinity")
    actual = m.render(directory, "river", globaltimes, True)
    expected = textwrap.dedent(
        """\
        begin options
        end options

        begin sources
          riv-1 AUX salinity
        end sources"""
    )
    assert actual == expected


def test_wrong_dtype():
    SourceSinkMixing(
        np.array(["a", "b"]),
        np.array(["AUX", "AUX"]),
        np.array(["salinity", "salinity"]),
    )

    SourceSinkMixing(
        "a",
        "AUX",
        "salinity",
    )

    with pytest.raises(ValidationError):
        SourceSinkMixing(
            np.array([1, 1]),
            np.array(["AUX", "AUX"]),
            np.array(["salinity", "salinity"]),
        )
