import pathlib
import textwrap

import pytest

import imod
from imod.schemata import ValidationError


def test_render():
    ic_head = imod.mf6.InitialConditions(head=0.0)
    ic_start = imod.mf6.InitialConditions(start=0.0)
    directory = pathlib.Path("mymodel")
    actual_head = ic_head.render(directory, "ic", None, True)
    actual_start = ic_start.render(directory, "ic", None, True)
    expected = textwrap.dedent(
        """\
        begin options
        end options

        begin griddata
          strt
            constant 0.0
        end griddata
        """
    )
    assert actual_head == expected
    assert actual_start == expected


def test_wrong_dtype():
    with pytest.raises(ValidationError):
        imod.mf6.InitialConditions(head=0)

    with pytest.raises(TypeError):
        imod.mf6.InitialConditions(start=0)


def test_wrong_arguments():
    with pytest.raises(ValueError):
        imod.mf6.InitialConditions()

    with pytest.raises(ValueError):
        imod.mf6.InitialConditions(head=0.0, start=1.0)
