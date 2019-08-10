import pathlib
import textwrap

import numpy as np
import xarray as xr

import imod


def test_render():
    layer = [3, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    row = [5, 4, 6, 9, 9, 9, 9, 11, 11, 11, 11, 13, 13, 13, 13]
    column = [11, 6, 12, 8, 10, 12, 14, 8, 10, 12, 14, 8, 10, 12, 14]
    rate = [
        -5.0,
        -5.0,
        -5.0,
        -5.0,
        -5.0,
        -5.0,
        -5.0,
        -5.0,
        -5.0,
        -5.0,
        -5.0,
        -5.0,
        -5.0,
        -5.0,
        -5.0,
    ]
    wel = imod.mf6.Well(
        layer=layer,
        row=row,
        column=column,
        rate=rate,
        print_input=True,
        print_flows=True,
        save_flows=True,
    )
    directory = pathlib.Path("mymodel")
    globaltimes = [np.datetime64("2000-01-01")]
    actual = wel.render(directory, "well", globaltimes)
    expected = textwrap.dedent(
        """\
            begin options
              print_input
              print_flows
              save_flows
            end options

            begin dimensions
              maxbound 15
            end dimensions

            begin period 1
              open/close mymodel/well/wel.bin (binary)
            end period"""
    )
    assert actual == expected
