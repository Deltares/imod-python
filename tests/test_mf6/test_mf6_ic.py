import pathlib
import textwrap

import imod


def test_render():
    ic = imod.mf6.InitialConditions(head=0.0)
    directory = pathlib.Path("mymodel")
    actual = ic.render(directory, "ic")
    expected = textwrap.dedent(
        """\
            begin options
            end options

            begin griddata
              strt
                constant 0.0
            end griddata"""
    )
    assert actual == expected
