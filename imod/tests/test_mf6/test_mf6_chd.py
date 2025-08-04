import pathlib
import tempfile
import textwrap
from datetime import datetime

import numpy as np
import pytest
import xarray as xr

import imod
from imod.mf6.chd import ConstantHead
from imod.mf6.dis import StructuredDiscretization
from imod.mf6.utilities.chd_concat import concat_layered_chd_packages
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
    actual = chd._render(directory, "chd", globaltimes, True)

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


def test_copy(head):
    chd = imod.mf6.ConstantHead(
        head, print_input=True, print_flows=True, save_flows=True
    )
    chd2 = chd.copy()
    assert isinstance(chd2, ConstantHead)
    assert chd2.dataset.equals(chd.dataset)


def test_from_file(head, tmp_path):
    directory = pathlib.Path("mymodel")
    globaltimes = np.array(["2000-01-01"], dtype="datetime64[ns]")

    chd = imod.mf6.ConstantHead(
        head, print_input=True, print_flows=True, save_flows=True
    )
    path = tmp_path / "chd.nc"
    chd.dataset.to_netcdf(path)
    chd2 = imod.mf6.ConstantHead.from_file(path)
    actual = chd2._render(directory, "chd", globaltimes, False)

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

    actual = chd._render(directory, "chd", globaltimes, False)

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
        chd.write("chd", globaltimes, output_dir)
        with open(output_dir + "/chd/chd-0.dat", "r") as f:
            data = f.read()
            assert (
                data.count("2") == 1755
            )  # the number 2 is in the concentration data, and in the cell indices.


def test_from_imod5(imod5_dataset, tmp_path):
    imod5_data = imod5_dataset[0]
    period_data = imod5_dataset[1]

    target_dis = StructuredDiscretization.from_imod5_data(imod5_data)

    chd3 = imod.mf6.ConstantHead.from_imod5_data(
        "chd-3",
        imod5_data,
        period_data,
        target_dis=target_dis,
        time_min=datetime(2002, 2, 2),
        time_max=datetime(2022, 2, 2),
        regridder_types=None,
    )

    assert isinstance(chd3, imod.mf6.ConstantHead)
    assert np.count_nonzero(~np.isnan(chd3.dataset["head"].values)) == 589
    assert len(chd3.dataset["layer"].values) == 1

    # write the packages for write validation
    chd3.write("chd3", [1], tmp_path, use_binary=False)


def test_from_imod5_shd(imod5_dataset, tmp_path):
    imod5_data = imod5_dataset[0]
    period_data = imod5_dataset[1]

    target_dis = StructuredDiscretization.from_imod5_data(imod5_data)

    chd_shd = imod.mf6.ConstantHead.from_imod5_shd_data(
        imod5_data,
        period_data,
        target_dis,
        regridder_types=None,
    )

    assert isinstance(chd_shd, imod.mf6.ConstantHead)
    assert len(chd_shd.dataset["layer"].values) == 37
    # write the packages for write validation
    chd_shd.write("chd_shd", [1], tmp_path, use_binary=False)


@pytest.mark.unittest_jit
@pytest.mark.parametrize("remove_merged_packages", [True, False])
def test_concatenate_chd(imod5_dataset, tmp_path, remove_merged_packages):
    # Arrange
    imod5_data = imod5_dataset[0]
    period_data = imod5_dataset[1]

    target_dis = StructuredDiscretization.from_imod5_data(imod5_data)
    chd_packages = {}

    # import a few chd packages per layer
    for layer in range(1, 7):
        key = f"chd-{layer}"
        chd_packages[key] = imod.mf6.ConstantHead.from_imod5_data(
            key,
            imod5_data,
            period_data,
            target_dis,
            time_min=datetime(2000, 1, 1),
            time_max=datetime(2020, 1, 1),
        )

    # import a few chd packages per layer but store them under another key
    for layer in range(8, 16):
        key = f"chd-{layer}"
        other_key = f"other_chd-{layer}"
        chd_packages[other_key] = imod.mf6.ConstantHead.from_imod5_data(
            key,
            imod5_data,
            period_data,
            target_dis,
            time_min=datetime(2000, 1, 1),
            time_max=datetime(2020, 1, 1),
        )

    # Act
    merged_package = concat_layered_chd_packages(
        "chd", chd_packages, remove_merged_packages
    )

    # Assert
    assert isinstance(merged_package, ConstantHead)
    assert len(merged_package["layer"]) == 6
    if remove_merged_packages:
        assert len(chd_packages) == 8
    else:
        assert len(chd_packages) == 14
    # write the packages for write validation
    merged_package.write("merged_chd", [1], tmp_path, use_binary=False)
