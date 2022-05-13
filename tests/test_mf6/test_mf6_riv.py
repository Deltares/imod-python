import pathlib
import pytest
import textwrap

import numpy as np

import imod


def test_render():
    riv = imod.mf6.River(stage=1.0, conductance=10.0, bottom_elevation=-1.0)
    directory = pathlib.Path("mymodel")
    globaltimes = [np.datetime64("2000-01-01")]
    actual = riv.render(directory, "river", globaltimes, True)
    expected = textwrap.dedent(
        """\
        begin options
        end options

        begin dimensions
          maxbound 1
        end dimensions

        begin period 1
          open/close mymodel/river/riv.bin (binary)
        end period
        """
    )
    assert actual == expected


def test_wrong_dtype():
    with pytest.raises(TypeError):
        imod.mf6.River(stage=1, conductance=10.0, bottom_elevation=-1.0)
