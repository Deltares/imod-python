import pathlib
import textwrap

import numpy as np
import pytest

import imod


def test_render():
    rch = imod.mf6.Recharge(rate=3.0e-8)
    directory = pathlib.Path("mymodel")
    globaltimes = [np.datetime64("2000-01-01")]
    actual = rch.render(directory, "recharge", globaltimes, True)
    expected = textwrap.dedent(
        """\
        begin options
        end options

        begin dimensions
          maxbound 1
        end dimensions

        begin period 1
          open/close mymodel/recharge/rch.bin (binary)
        end period
        """
    )
    assert actual == expected


def test_wrong_dtype():
    with pytest.raises(TypeError):
        imod.mf6.Recharge(rate=3)
