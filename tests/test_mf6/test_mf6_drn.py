import textwrap
import pathlib

import imod

import numpy as np
import pandas as pd
import pytest
import xarray as xr


@pytest.fixture(scope="module")
def drainage(request):
    layer = np.arange(1, 4)
    y = np.arange(4.5, 0.0, -1.0)
    x = np.arange(0.5, 5.0, 1.0)
    elevation = xr.DataArray(
        np.full((3, 5, 5), 1.0),
        coords={"layer": layer, "y": y, "x": x, "dx": 1.0, "dy": -1.0},
        dims=("layer", "y", "x"),
    )
    conductance = elevation.copy()

    drn = imod.mf6.Drainage(elevation=elevation, conductance=conductance)
    return drn


def test_write(drainage):
    drn = drainage
    directory = pathlib.Path(".")
    drn.write(directory, "mydrn", [1])

    block_expected = textwrap.dedent(
        """\
            begin options
              print_input
              print_flows
              save_flows
            end options

            begin dimensions
              maxbound 75
            end dimensions

            begin period 1
              open/close mydrn/drn.bin (binary)
            end period
        """
    )

    with open(directory / "mydrn.drn") as f:
        block = f.read()

    assert block == block_expected
