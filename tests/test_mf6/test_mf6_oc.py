import pathlib
import textwrap

import numpy as np

import imod


def test_render():
    oc = imod.mf6.OutputControl(save_head=True, save_budget=True)
    directory = pathlib.Path("mymodel")
    globaltimes = [np.datetime64("2000-01-01")]
    actual = oc.render(directory, "outputcontrol", globaltimes)
    expected = textwrap.dedent(
        """\
            begin options
              budget fileout mymodel/mymodel.cbb
              head fileout mymodel/mymodel.hds
            end options

            begin period 1
              save head all
              save budget all
            end period"""
    )
    assert actual == expected
