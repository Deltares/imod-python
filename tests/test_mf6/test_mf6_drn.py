import textwrap

import numpy as np
import pytest
import xarray as xr

import imod


@pytest.fixture(scope="module")
def drainage():
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


def test_write(drainage, tmp_path):
    drn = drainage
    drn.write(tmp_path, "mydrn", [1], True)
    path = tmp_path.as_posix()
    block_expected = textwrap.dedent(
        f"""\
        begin options
        end options

        begin dimensions
          maxbound 75
        end dimensions

        begin period 1
          open/close {path}/mydrn/drn.bin (binary)
        end period
        """
    )

    with open(tmp_path / "mydrn.drn") as f:
        block = f.read()

    assert block == block_expected
