from copy import deepcopy
from typing import Callable

import numpy as np
import pytest
import xarray as xr
from pytest_cases import parametrize_with_cases

from imod.msw import GridData
from imod.msw.utilities.mask import (
    MetaSwapActive,
    mask_and_broadcast_cap_data,
    mask_and_broadcast_pkg_data,
)
from imod.typing import GridDataDict


@pytest.fixture(scope="function")
def coords_planar() -> dict:
    x = [1.0, 2.0, 3.0]
    y = [3.0, 2.0, 1.0]
    dx = 1.0
    dy = 1.0
    return {"y": y, "x": x, "dx": dx, "dy": dy}


@pytest.fixture(scope="function")
def coords_subunit(coords_planar: dict) -> dict:
    coords_subunit = deepcopy(coords_planar)
    coords_subunit["subunit"] = [0, 1]
    return coords_subunit


@pytest.fixture(scope="function")
def mask_fixture(coords_subunit: dict) -> MetaSwapActive:
    mask_per_subunit = xr.DataArray(
        np.array(
            [
                [[0, 0, 0], [0, 1, 1], [0, 0, 0]],
                [[1, 1, 1], [0, 1, 1], [0, 0, 0]],
            ]
        ).astype(bool),
        dims=("subunit", "y", "x"),
        coords=coords_subunit,
    )
    mask_all = mask_per_subunit.any(dim="subunit")

    return MetaSwapActive(mask_all, mask_per_subunit)


@pytest.fixture(scope="function")
def data_subunit(coords_subunit: dict):
    return xr.DataArray(
        np.array(
            [
                [[2.0, 2.0, 2.0], [2.0, 3.0, 3.0], [2.0, 2.0, 2.0]],
                [[3.0, 3.0, 3.0], [2.0, 3.0, 3.0], [2.0, 2.0, 2.0]],
            ]
        ),
        dims=("subunit", "y", "x"),
        coords=coords_subunit,
    )


@pytest.fixture(scope="function")
def data_planar(data_subunit: xr.DataArray) -> xr.DataArray:
    return data_subunit.sel(subunit=0, drop=True)


def integer_nan(da):
    return da == 0


def case_grid_float(
    data_subunit: xr.DataArray, data_planar: xr.DataArray
) -> tuple[GridDataDict, Callable]:
    return {"landuse": data_subunit, "elevation": data_planar}, np.isnan


def case_grid_integer(
    data_subunit: xr.DataArray, data_planar: xr.DataArray
) -> tuple[GridDataDict, Callable]:
    return {
        "landuse": data_subunit.astype(int),
        "elevation": data_planar.astype(int),
    }, integer_nan


def case_constant_float():
    data = xr.DataArray(2.0)
    return {"landuse": data, "elevation": data}, np.isnan


def case_constant_integer():
    data = xr.DataArray(2)
    return {"landuse": data, "elevation": data}, integer_nan


@parametrize_with_cases("imod5_grid_data, isnan", cases=".")
def test_mask_and_broadcast_pkg_data(
    imod5_grid_data: GridDataDict, isnan: Callable, mask_fixture: MetaSwapActive
):
    # Act
    masked_data = mask_and_broadcast_pkg_data(GridData, imod5_grid_data, mask_fixture)

    # Assert
    assert isnan(masked_data["landuse"]).to_numpy().sum() == 11
    assert isnan(masked_data["elevation"]).to_numpy().sum() == 4


@parametrize_with_cases("imod5_grid_data, isnan", cases=".")
def test_mask_and_broadcast_cap_data(
    imod5_grid_data: GridDataDict,
    isnan: Callable,
    mask_fixture: MetaSwapActive,
    coords_planar: dict,
):
    # Arrange, cap data doesn't have subunit coords
    imod5_grid_data["landuse"] = imod5_grid_data["landuse"].isel(
        subunit=0, drop=True, missing_dims="ignore"
    )

    # Act
    masked_data = mask_and_broadcast_cap_data(imod5_grid_data, mask_fixture)

    # Assert
    coords_expected = xr.Coordinates(coords_planar)
    xr.testing.assert_equal(masked_data["landuse"].coords, coords_expected)
    xr.testing.assert_equal(masked_data["elevation"].coords, coords_expected)
    assert isnan(masked_data["landuse"]).to_numpy().sum() == 4
    assert isnan(masked_data["elevation"]).to_numpy().sum() == 4
