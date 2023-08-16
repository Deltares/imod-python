import pathlib
import tempfile
import textwrap

import numpy as np
import pytest
import xarray as xr

import imod
from imod.mf6.write_context import WriteContext
from imod.schemata import ValidationError


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
    globaltimes = np.array(["2000-01-01"], dtype="datetime64[ns]")

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
    globaltimes = np.array(["2000-01-01"], dtype="datetime64[ns]")

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
    with pytest.raises(ValidationError):
        imod.mf6.ConstantHead(
            head.astype(np.int16), print_input=True, print_flows=True, save_flows=True
        )


pytest.mark.usefixtures("head_fc", "concentration_fc")


def test_render_concentration(head_fc, concentration_fc):
    directory = pathlib.Path("mymodel")
    globaltimes = np.array(
        [
            "2000-01-01",
            "2000-01-02",
            "2000-01-03",
        ],
        dtype="datetime64[ns]",
    )

    chd = imod.mf6.ConstantHead(
        head_fc,
        concentration_fc,
        "AUX",
        print_input=True,
        print_flows=True,
        save_flows=True,
    )

    actual = chd.render(directory, "chd", globaltimes, False)

    expected = textwrap.dedent(
        """\
        begin options
          auxiliary salinity temperature
          print_input
          print_flows
          save_flows
        end options

        begin dimensions
          maxbound 0
        end dimensions

        begin period 1
          open/close mymodel/chd/chd-0.dat
        end period
        begin period 2
          open/close mymodel/chd/chd-1.dat
        end period
        begin period 3
          open/close mymodel/chd/chd-2.dat
        end period
        """
    )
    assert actual == expected


pytest.mark.usefixtures("head_fc", "concentration_fc")


def test_write_concentration_period_data(head_fc, concentration_fc):
    globaltimes = np.array(
        [
            "2000-01-01",
            "2000-01-02",
            "2000-01-03",
        ],
        dtype="datetime64[ns]",
    )
    head_fc[:] = 1
    concentration_fc[:] = 2
    chd = imod.mf6.ConstantHead(
        head_fc,
        concentration_fc,
        "AUX",
        print_input=True,
        print_flows=True,
        save_flows=True,
    )
    with tempfile.TemporaryDirectory() as output_dir:
        write_context = WriteContext(simulation_directory=output_dir)
        write_context.current_output_directory = output_dir
        chd.write("chd", globaltimes, write_context)
        with open(output_dir + "/chd/chd-0.dat", "r") as f:
            data = f.read()
            assert (
                data.count("2") == 1755
            )  # the number 2 is in the concentration data, and in the cell indices.
