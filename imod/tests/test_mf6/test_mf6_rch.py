import pathlib
import re
import textwrap

import numpy as np
import pytest
import xarray as xr

import imod


@pytest.fixture(scope="function")
def rch_dict():
    x = [5.0, 15.0, 25.0]
    y = [25.0, 15.0, 5.0]
    layer = [1]
    dx = 10.0
    dy = -10.0

    da = xr.DataArray(
        data=np.ones((1, 3, 3), dtype=float),
        dims=("layer", "y", "x"),
        coords=dict(layer=layer, y=y, x=x, dx=dx, dy=dy),
    )

    da[:, 1, 1] = np.nan

    return dict(rate=da)


@pytest.fixture(scope="function")
def rch_dict_transient():
    x = [5.0, 15.0, 25.0]
    y = [25.0, 15.0, 5.0]
    layer = [1]
    time = [np.datetime64("2000-01-01"), np.datetime64("2000-01-02")]
    dx = 10.0
    dy = -10.0

    da = xr.DataArray(
        data=np.ones((2, 1, 3, 3), dtype=float),
        dims=("time", "layer", "y", "x"),
        coords=dict(time=time, layer=layer, y=y, x=x, dx=dx, dy=dy),
    )

    da[..., 1, 1] = np.nan

    return dict(rate=da)


def test_render(rch_dict):
    rch = imod.mf6.Recharge(**rch_dict)
    directory = pathlib.Path("mymodel")
    globaltimes = [np.datetime64("2000-01-01")]
    actual = rch.render(directory, "recharge", globaltimes, True)
    expected = textwrap.dedent(
        """\
        begin options
        end options

        begin dimensions
          maxbound 8
        end dimensions

        begin period 1
          open/close mymodel/recharge/rch.bin (binary)
        end period
        """
    )
    assert actual == expected


def test_render_transient(rch_dict_transient):
    rch = imod.mf6.Recharge(**rch_dict_transient)
    directory = pathlib.Path("mymodel")
    globaltimes = [
        np.datetime64("2000-01-01"),
        np.datetime64("2000-01-02"),
        np.datetime64("2000-01-03"),
    ]
    actual = rch.render(directory, "recharge", globaltimes, True)
    expected = textwrap.dedent(
        """\
        begin options
        end options

        begin dimensions
          maxbound 8
        end dimensions

        begin period 1
          open/close mymodel/recharge/rch-0.bin (binary)
        end period
        begin period 2
          open/close mymodel/recharge/rch-1.bin (binary)
        end period
        """
    )
    assert actual == expected


def test_wrong_dtype(rch_dict):
    rch_dict["rate"] = rch_dict["rate"].astype(np.int32)
    with pytest.raises(TypeError):
        imod.mf6.Recharge(**rch_dict)


def test_no_layer_dim(rch_dict):
    rch_dict["rate"] = rch_dict["rate"].sel(layer=1, drop=False)
    rch = imod.mf6.Recharge(**rch_dict)
    directory = pathlib.Path("mymodel")
    globaltimes = [np.datetime64("2000-01-01")]
    actual = rch.render(directory, "recharge", globaltimes, True)
    expected = textwrap.dedent(
        """\
        begin options
        end options

        begin dimensions
          maxbound 8
        end dimensions

        begin period 1
          open/close mymodel/recharge/rch.bin (binary)
        end period
        """
    )
    assert actual == expected


def test_transient_no_layer_dim(rch_dict_transient):
    rch_dict_transient["rate"] = rch_dict_transient["rate"].sel(layer=1, drop=False)
    rch = imod.mf6.Recharge(**rch_dict_transient)
    directory = pathlib.Path("mymodel")
    globaltimes = [
        np.datetime64("2000-01-01"),
        np.datetime64("2000-01-02"),
        np.datetime64("2000-01-03"),
    ]
    actual = rch.render(directory, "recharge", globaltimes, True)
    expected = textwrap.dedent(
        """\
        begin options
        end options

        begin dimensions
          maxbound 8
        end dimensions

        begin period 1
          open/close mymodel/recharge/rch-0.bin (binary)
        end period
        begin period 2
          open/close mymodel/recharge/rch-1.bin (binary)
        end period
        """
    )
    assert actual == expected


def test_no_layer_coord(rch_dict):
    rch_dict["rate"] = rch_dict["rate"].sel(layer=1, drop=True)
    with pytest.raises(
        ValueError,
        match="No 'layer' coordinate assigned to rate in the Recharge package. 2D grids still require a 'layer' coordinate.",
    ):
        imod.mf6.Recharge(**rch_dict)


def test_scalar():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Boundary conditions should be specified as spatial grids. Instead, got () for rate in the Recharge package. "
        ),
    ):
        imod.mf6.Recharge(rate=0.001)
