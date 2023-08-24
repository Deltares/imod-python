import pathlib
import re
import tempfile
import textwrap

import numpy as np
import pytest
import xarray as xr

import imod
from imod.mf6.write_context import WriteContext
from imod.schemata import ValidationError


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
    time = np.array(["2000-01-01", "2000-01-02"], dtype="datetime64[ns]")
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
    globaltimes = np.array(["2000-01-01"], dtype="datetime64[ns]")
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
    globaltimes = np.array(
        [
            "2000-01-01",
            "2000-01-02",
            "2000-01-03",
        ],
        dtype="datetime64[ns]",
    )
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
    with pytest.raises(ValidationError):
        imod.mf6.Recharge(**rch_dict)


def test_no_layer_dim(rch_dict):
    rch_dict["rate"] = rch_dict["rate"].sel(layer=1, drop=False)
    rch = imod.mf6.Recharge(**rch_dict)
    directory = pathlib.Path("mymodel")
    globaltimes = np.array(["2000-01-01"], dtype="datetime64[ns]")
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
    globaltimes = np.array(
        [
            "2000-01-01",
            "2000-01-02",
            "2000-01-03",
        ],
        dtype="datetime64[ns]",
    )

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


@pytest.mark.usefixtures("concentration_fc", "rate_fc")
def test_render_concentration(concentration_fc, rate_fc):
    rch = imod.mf6.Recharge(
        rate=rate_fc,
        concentration=concentration_fc,
        concentration_boundary_type="AUX",
    )

    directory = pathlib.Path("mymodel")
    globaltimes = np.array(
        [
            "2000-01-01",
            "2000-01-02",
            "2000-01-03",
        ],
        dtype="datetime64[ns]",
    )

    actual = rch.render(directory, "rch", globaltimes, False)

    expected = textwrap.dedent(
        """\
        begin options
          auxiliary salinity temperature
        end options

        begin dimensions
          maxbound 2
        end dimensions

        begin period 1
          open/close mymodel/rch/rch-0.dat
        end period
        begin period 2
          open/close mymodel/rch/rch-1.dat
        end period
        begin period 3
          open/close mymodel/rch/rch-2.dat
        end period
        """
    )
    assert actual == expected


def test_no_layer_coord(rch_dict):
    message = textwrap.dedent(
        """
        * rate
        \t- coords has missing keys: {'layer'}"""
    )

    rch_dict["rate"] = rch_dict["rate"].sel(layer=1, drop=True)
    with pytest.raises(
        ValidationError,
        match=re.escape(message),
    ):
        imod.mf6.Recharge(**rch_dict)


def test_scalar():
    message = textwrap.dedent(
        """
        * rate
        \t- coords has missing keys: {'layer'}
        \t- No option succeeded:
        \tdim mismatch: expected ('time', 'layer', 'y', 'x'), got ()
        \tdim mismatch: expected ('layer', 'y', 'x'), got ()
        \tdim mismatch: expected ('time', 'layer', '{face_dim}'), got ()
        \tdim mismatch: expected ('layer', '{face_dim}'), got ()
        \tdim mismatch: expected ('time', 'y', 'x'), got ()
        \tdim mismatch: expected ('y', 'x'), got ()
        \tdim mismatch: expected ('time', '{face_dim}'), got ()
        \tdim mismatch: expected ('{face_dim}',), got ()"""
    )
    with pytest.raises(ValidationError, match=re.escape(message)):
        imod.mf6.Recharge(rate=0.001)


def test_validate_false():
    imod.mf6.Recharge(rate=0.001, validate=False)


@pytest.mark.usefixtures("rate_fc", "concentration_fc")
def test_write_concentration_period_data(rate_fc, concentration_fc):
    globaltimes = np.array(
        [
            "2000-01-01",
            "2000-01-02",
            "2000-01-03",
        ],
        dtype="datetime64",
    )
    rate_fc[:] = 1
    concentration_fc[:] = 2

    rch = imod.mf6.Recharge(
        rate=rate_fc,
        concentration=concentration_fc,
        concentration_boundary_type="AUX",
    )
    with tempfile.TemporaryDirectory() as output_dir:
        write_context = WriteContext(
            simulation_directory=output_dir, write_directory=output_dir
        )
        rch.write(pkgname="rch", globaltimes=globaltimes, write_context=write_context)

        with open(output_dir + "/rch/rch-0.dat", "r") as f:
            data = f.read()
            assert (
                data.count("2") == 1755
            )  # the number 2 is in the concentration data, and in the cell indices.


def test_clip_box(rch_dict):
    rch = imod.mf6.Recharge(**rch_dict)

    selection = rch.clip_box()
    assert isinstance(selection, imod.mf6.Recharge)
    assert selection.dataset.identical(rch.dataset)

    selection = rch.clip_box(x_min=10.0, x_max=20.0, y_min=10.0, y_max=20.0)
    assert selection["rate"].dims == ("layer", "y", "x")
    assert selection["rate"].shape == (1, 1, 1)

    # No layer dim
    rch_dict["rate"] = rch_dict["rate"].sel(layer=1, drop=False)
    rch = imod.mf6.Recharge(**rch_dict)
    selection = rch.clip_box(x_min=10.0, x_max=20.0, y_min=10.0, y_max=20.0)
    assert selection["rate"].dims == ("y", "x")
    assert selection["rate"].shape == (1, 1)
