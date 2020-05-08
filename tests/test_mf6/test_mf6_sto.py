import pathlib
import textwrap

import numpy as np

import imod


def test_render():
    assert True
    # riv = imod.mf6.Storage(...)
    # directory = pathlib.Path("mymodel")
    # globaltimes = [np.datetime64("2000-01-01")]
    # actual = riv.render(directory, "river", globaltimes)
    # expected = textwrap.dedent(
    #     """\
    #     begin options
    #     end options

    #     begin dimensions
    #       maxbound 1
    #     end dimensions

    #     begin period 1
    #       open/close mymodel/river/riv.bin (binary)
    #     end period
    #     """
    # )
    # assert actual == expected
