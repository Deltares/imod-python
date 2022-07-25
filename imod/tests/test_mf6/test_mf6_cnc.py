import pathlib
import tempfile
import textwrap

import numpy as np
import pytest
import xarray as xr

import imod


@pytest.fixture()
def concentration_steadystate():
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

    # Constant cocnentration
    concentration = xr.full_like(idomain, np.nan).sel(layer=[1, 2])
    concentration[...] = np.nan
    concentration[..., 0] = 0.0

    return concentration


@pytest.fixture()
def concentration_transient():
    nlay = 3
    nrow = 15
    ncol = 15
    ntimes = 3
    globaltimes = [
        np.datetime64("2000-01-01"),
        np.datetime64("2000-01-02"),
        np.datetime64("2000-01-03"),
    ]
    shape = (ntimes, nlay, nrow, ncol)

    dx = 5000.0
    dy = -5000.0
    xmin = 0.0
    xmax = dx * ncol
    ymin = 0.0
    ymax = abs(dy) * nrow
    dims = ("time", "layer", "y", "x")

    layer = np.array([1, 2, 3])
    y = np.arange(ymax, ymin, dy) + 0.5 * dy
    x = np.arange(xmin, xmax, dx) + 0.5 * dx
    coords = {"time": globaltimes, "layer": layer, "y": y, "x": x}

    # Discretization data
    idomain = xr.DataArray(np.ones(shape), coords=coords, dims=dims)

    # Constant cocnentration
    concentration = xr.full_like(idomain, np.nan)
    concentration[...] = np.nan
    concentration[..., 0] = 0.0

    return concentration


def test_render(concentration_steadystate):
    directory = pathlib.Path("mymodel")
    globaltimes = [np.datetime64("2000-01-01")]

    cnc = imod.mf6.ConstantConcentration(
        concentration_steadystate, print_input=True, print_flows=True, save_flows=True
    )
    actual = cnc.render(directory, "cnc", globaltimes, True)

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
          open/close mymodel/cnc/cnc.bin (binary)
        end period"""
    )
    assert actual == expected


def test_write_period_data(concentration_transient):
    globaltimes = [
        np.datetime64("2000-01-01"),
        np.datetime64("2000-01-02"),
        np.datetime64("2000-01-03"),
    ]
    concentration_transient[:] = 2
    cnc = imod.mf6.ConstantConcentration(
        concentration_transient,
        print_input=True,
        print_flows=True,
        save_flows=True,
    )
    with tempfile.TemporaryDirectory() as output_dir:
        cnc.write(output_dir, "cnc", globaltimes, False)
        with open(output_dir + "/cnc/cnc-0.dat", "r") as f:
            data = f.read()
            assert (
                data.count("2") == 1080
            )  # the number 2 is in the concentration data, and in the cell indices.
