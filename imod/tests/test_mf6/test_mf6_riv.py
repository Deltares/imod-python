import pathlib
import textwrap

import numpy as np
import pytest

import imod
import xarray as xr


@pytest.fixture(scope="function")
def riv_dict():
    x = [5.0, 15.0, 25.0]
    y = [25.0, 15.0, 5.0]
    layer = [2, 3]
    dx = 10.0
    dy = -10.0

    da = xr.DataArray(
        data=np.ones((2, 3, 3), dtype=float),
        dims=("layer", "y", "x"),
        coords=dict(layer=layer, y=y, x=x, dx=dx, dy=dy),
    )

    da[:, 1, 1] = np.nan

    return dict(stage=da, conductance=da, bottom_elevation=da - 1.0)


def test_render(riv_dict):
    river = imod.mf6.River(**riv_dict)
    directory = pathlib.Path("mymodel")
    globaltimes = [np.datetime64("2000-01-01")]
    actual = river.render(directory, "river", globaltimes, True)
    expected = textwrap.dedent(
        """\
        begin options
        end options

        begin dimensions
          maxbound 16
        end dimensions

        begin period 1
          open/close mymodel/river/riv.bin (binary)
        end period
        """
    )
    assert actual == expected


def test_wrong_dtype(riv_dict):
    riv_dict["stage"] = riv_dict["stage"].astype(int)
    with pytest.raises(TypeError):
        imod.mf6.River(**riv_dict)
