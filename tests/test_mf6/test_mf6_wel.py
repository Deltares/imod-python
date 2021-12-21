import pathlib
import textwrap

import numpy as np

import imod


def test_render():
    layer = np.array([3, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    row = np.array([5, 4, 6, 9, 9, 9, 9, 11, 11, 11, 11, 13, 13, 13, 13])
    column = np.array([11, 6, 12, 8, 10, 12, 14, 8, 10, 12, 14, 8, 10, 12, 14])
    rate = np.full(15, 5.0)
    wel = imod.mf6.WellDisStructured(
        layer=layer,
        row=row,
        column=column,
        rate=rate,
        print_input=False,
        print_flows=False,
        save_flows=False,
    )
    directory = pathlib.Path("mymodel")
    globaltimes = [np.datetime64("2000-01-01")]
    actual = wel.render(directory, "well", globaltimes, True)
    expected = textwrap.dedent(
        """\
        begin options
        end options

        begin dimensions
          maxbound 15
        end dimensions

        begin period 1
          open/close mymodel/well/wel.bin (binary)
        end period
        """
    )

    cell2d = (row - 1) * 15 + column
    wel = imod.mf6.WellDisVertices(
        layer=layer,
        cell2d=cell2d,
        rate=rate,
        print_input=False,
        print_flows=False,
        save_flows=False,
    )
    actual = wel.render(directory, "well", globaltimes, True)
    assert actual == expected
