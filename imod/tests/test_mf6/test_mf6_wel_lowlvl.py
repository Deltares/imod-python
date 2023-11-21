import textwrap
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import imod
import imod.mf6.mf6_wel_adapter
import imod.mf6.utilities.dataset
import imod.mf6.wel
from imod.mf6.write_context import WriteContext


@pytest.fixture()
def struct_array_expected():
    return np.array(
        [
            (1, 1, 9, 1.0),
            (1, 2, 9, 1.0),
            (1, 1, 8, 1.0),
            (1, 2, 8, 1.0),
            (2, 3, 7, 1.0),
            (2, 4, 7, 1.0),
            (2, 3, 6, 1.0),
            (2, 4, 6, 1.0),
        ],
        dtype=[("layer", "<i4"), ("row", "<i4"), ("column", "<i4"), ("rate", "<f8")],
    )


def test_mf6wel_to_struct_array__stationary(
    mf6wel_test_data_stationary, struct_array_expected
):
    # Arrange
    cellid, rate = mf6wel_test_data_stationary
    mf6wel = imod.mf6.mf6_wel_adapter.Mf6Wel(cellid=cellid, rate=rate)

    # Act
    arrdict = mf6wel._ds_to_arrdict(mf6wel.dataset)
    struct_array = mf6wel._to_struct_array(arrdict, None)

    # Assert
    np.testing.assert_array_equal(struct_array, struct_array_expected)


def test_mf6wel_to_struct_array__transient(
    mf6wel_test_data_transient, struct_array_expected
):
    # Arrange
    cellid, rate = mf6wel_test_data_transient
    mf6wel = imod.mf6.mf6_wel_adapter.Mf6Wel(cellid=cellid, rate=rate)
    ds = mf6wel.dataset.isel(time=0)

    # Act
    arrdict = mf6wel._ds_to_arrdict(ds)
    struct_array = mf6wel._to_struct_array(arrdict, None)

    # Assert
    np.testing.assert_array_equal(struct_array, struct_array_expected)


def test_mf6wel_write_datafile__stationary(
    tmp_path, mf6wel_test_data_stationary, struct_array_expected
):
    # Arrange
    cellid, rate = mf6wel_test_data_stationary
    mf6wel = imod.mf6.mf6_wel_adapter.Mf6Wel(cellid=cellid, rate=rate)

    ds = mf6wel.dataset
    file_path = Path(tmp_path) / "mf6wel.bin"

    # Act
    mf6wel._write_datafile(file_path, ds, True)
    arr = np.fromfile(file_path, dtype=struct_array_expected.dtype)

    # Assert
    np.testing.assert_array_equal(arr, struct_array_expected)


def test_mf6wel_write__stationary(
    tmp_path, mf6wel_test_data_stationary, struct_array_expected
):
    # Arrange
    cellid, rate = mf6wel_test_data_stationary
    mf6wel = imod.mf6.mf6_wel_adapter.Mf6Wel(cellid=cellid, rate=rate)
    globaltimes = pd.date_range("2000-01-01", "2000-01-06")
    pkgname = "wel"
    directory = Path(tmp_path) / "mf6wel"
    directory.mkdir(exist_ok=True)
    write_context = WriteContext(directory, use_binary=True)

    # Act
    mf6wel.write(pkgname, globaltimes, write_context)

    # Assert
    assert len(list(directory.glob("**/*"))) == 3
    arr = np.fromfile(
        directory / pkgname / f"{pkgname}.bin", dtype=struct_array_expected.dtype
    )
    np.testing.assert_array_equal(arr, struct_array_expected)


def test_mf6wel_write__transient(
    tmp_path, mf6wel_test_data_transient, struct_array_expected
):
    # Arrange
    cellid, rate = mf6wel_test_data_transient
    mf6wel = imod.mf6.mf6_wel_adapter.Mf6Wel(cellid=cellid, rate=rate)
    globaltimes = pd.date_range("2000-01-01", "2000-01-06")
    pkgname = "wel"
    directory = Path(tmp_path) / "mf6wel"
    directory.mkdir(exist_ok=True)
    write_context = WriteContext(directory, use_binary=True)
    # Act
    mf6wel.write(pkgname, globaltimes, write_context)

    # Assert
    assert len(list(directory.glob("**/*"))) == 7
    arr = np.fromfile(
        directory / pkgname / f"{pkgname}-0.bin", dtype=struct_array_expected.dtype
    )
    np.testing.assert_array_equal(arr, struct_array_expected)


def test_mf6wel_render__transient(
    tmp_path: Path, mf6wel_test_data_transient: Tuple[xr.DataArray, xr.DataArray]
):
    # Arrange
    cellid, rate = mf6wel_test_data_transient
    mf6wel = imod.mf6.mf6_wel_adapter.Mf6Wel(cellid=cellid, rate=rate)
    globaltimes = pd.date_range("2000-01-01", "2000-01-06")
    pkgname = "wel"
    directory = Path(tmp_path) / "mf6wel"
    directory.mkdir(exist_ok=True)

    # Act
    actual = mf6wel.render(directory.stem, pkgname, globaltimes, False)
    expected = textwrap.dedent(
        """\
    begin options
    end options

    begin dimensions
      maxbound 24
    end dimensions

    begin period 1
      open/close mf6wel/wel/wel-0.dat
    end period
    begin period 2
      open/close mf6wel/wel/wel-1.dat
    end period
    begin period 3
      open/close mf6wel/wel/wel-2.dat
    end period
    begin period 4
      open/close mf6wel/wel/wel-3.dat
    end period
    begin period 5
      open/close mf6wel/wel/wel-4.dat
    end period
    """
    )

    # Assert
    assert actual == expected


def test_remove_inactive__stationary(basic_dis, mf6wel_test_data_stationary):
    # Arrange
    cellid, rate = mf6wel_test_data_stationary
    ds = xr.Dataset()
    ds["cellid"] = cellid
    ds["rate"] = rate

    ibound, _, _ = basic_dis
    active = ibound == 1
    active[0, 0, :] = False

    # Act
    ds_removed = imod.mf6.utilities.dataset.remove_inactive(ds, active)

    # Assert
    assert dict(ds_removed.dims) == {"ncellid": 6, "nmax_cellid": 3}


def test_remove_inactive__transient(basic_dis, mf6wel_test_data_transient):
    # Arrange
    cellid, rate = mf6wel_test_data_transient
    ds = xr.Dataset()
    ds["cellid"] = cellid
    ds["rate"] = rate

    ibound, _, _ = basic_dis
    active = ibound == 1
    active[0, 0, :] = False

    # Act
    ds_removed = imod.mf6.utilities.dataset.remove_inactive(ds, active)

    # Assert
    assert dict(ds_removed.dims) == {"ncellid": 6, "time": 5, "nmax_cellid": 3}
