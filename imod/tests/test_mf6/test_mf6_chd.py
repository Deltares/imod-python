import pathlib
import textwrap

import numpy as np
import pytest
import xarray as xr

import imod


@pytest.fixture()
def head():
    nlay = 3
    nrow = 15
    ncol = 15
    shape = (nlay, nrow, ncol)

    dx = 5000.0
    dy = -5000.0
    xmin = 0.0
    xmax = dx * ncol
    ymin = 0.0
    ymax = abs(dy) * nrow
    dims = ("layer", "y", "x")

    layer = np.array([1, 2, 3])
    y = np.arange(ymax, ymin, dy) + 0.5 * dy
    x = np.arange(xmin, xmax, dx) + 0.5 * dx
    coords = {"layer": layer, "y": y, "x": x}

    # Discretization data
    idomain = xr.DataArray(np.ones(shape), coords=coords, dims=dims)

    # Constant head
    head = xr.full_like(idomain, np.nan).sel(layer=[1, 2])
    head[...] = np.nan
    head[..., 0] = 0.0

    return head


def test_render(head):
    directory = pathlib.Path("mymodel")
    globaltimes = [np.datetime64("2000-01-01")]

    chd = imod.mf6.ConstantHead(
        head, print_input=True, print_flows=True, save_flows=True
    )
    actual = chd.render(directory, "chd", globaltimes, True)

    expected = textwrap.dedent(
        """\
        begin options
          print_input
          print_flows
          save_flows
        end options

        begin dimensions
          maxbound 30
        end dimensions

        begin period 1
          open/close mymodel/chd/chd.bin (binary)
        end period
        """
    )
    assert actual == expected


def test_from_file(head, tmp_path):
    directory = pathlib.Path("mymodel")
    globaltimes = [np.datetime64("2000-01-01")]

    chd = imod.mf6.ConstantHead(
        head, print_input=True, print_flows=True, save_flows=True
    )
    path = tmp_path / "chd.nc"
    chd.dataset.to_netcdf(path)
    chd2 = imod.mf6.ConstantHead.from_file(path)
    actual = chd2.render(directory, "chd", globaltimes, False)

    expected = textwrap.dedent(
        """\
        begin options
          print_input
          print_flows
          save_flows
        end options

        begin dimensions
          maxbound 30
        end dimensions

        begin period 1
          open/close mymodel/chd/chd.dat
        end period
        """
    )
    assert actual == expected


def test_wrong_dtype(head):
    with pytest.raises(TypeError):
        imod.mf6.ConstantHead(
            head.astype(np.int16), print_input=True, print_flows=True, save_flows=True
        )
