import pathlib
import textwrap

import numpy as np

import imod


def test_render():
    riv = imod.mf6.River(
        stage=1.0,
        conductance=10.0,
        bottom_elevation=-1.0
    )
    directory = pathlib.Path("mymodel")
    globaltimes = [np.datetime64("2000-01-01")]
    actual = riv.render(directory, "river", globaltimes)
    expected = textwrap.dedent(
        """\
            begin options
              print_input
              print_flows
              save_flows
            end options

            begin dimensions
              maxbound 1
            end dimensions

            begin period 1
                open/close mymodel/river/riv.bin (binary)
            end period"""
    )
    assert actual == expected