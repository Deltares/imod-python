import pathlib
import textwrap

import numpy as np

import imod


def test_render():
    rch = imod.mf6.Recharge(rate=3.0e-8)
    directory = pathlib.Path("mymodel")
    globaltimes = [np.datetime64("2000-01-01")]
    actual = rch.render(directory, "recharge", globaltimes)
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
              open/close mymodel/recharge/rch.bin (binary)
            end period"""
    )
    assert actual == expected
