from typing import Callable

import numpy as np
import pytest
import xugrid as xu
from pytest_cases import parametrize, parametrize_with_cases

from imod.prepare.cleanup import cleanup_drn, cleanup_ghb, cleanup_riv
from imod.tests.test_mf6.test_mf6_riv import RivDisCases
from imod.typing import GridDataArray


def _first(grid: GridDataArray):
    """
    helper function to get first value, regardless of unstructured or
    structured grid."""
    return grid.values.ravel()[0]


def _first_index(grid: GridDataArray) -> tuple:
    if isinstance(grid, xu.UgridDataArray):
        return (0, 0)
    else:
        return (0, 0, 0)


def _rename_data_dict(data: dict, func: Callable):
    renamed = data.copy()
    to_rename = _RENAME_DICT[func]
    for src, dst in to_rename.items():
        mv_data = renamed.pop(src)
        if dst is not None:
            renamed[dst] = mv_data
    return renamed


def _prepare_dis_dict(dis_dict: dict, func: Callable):
    """Keep required dis args for specific cleanup functions"""
    keep_vars = _KEEP_FROM_DIS_DICT[func]
    return {var: dis_dict[var] for var in keep_vars}


_RENAME_DICT = {
    cleanup_riv: {},
    cleanup_drn: {"stage": "elevation", "bottom_elevation": None},
    cleanup_ghb: {"stage": "head", "bottom_elevation": None},
}

_KEEP_FROM_DIS_DICT = {
    cleanup_riv: ["idomain", "bottom"],
    cleanup_drn: ["idomain"],
    cleanup_ghb: ["idomain"],
}


@parametrize_with_cases("riv_data, dis_data", cases=RivDisCases)
@parametrize("cleanup_func", [cleanup_drn, cleanup_ghb, cleanup_riv])
def test_cleanup__align_nodata(riv_data: dict, dis_data: dict, cleanup_func: Callable):
    dis_dict = _prepare_dis_dict(dis_data, cleanup_func)
    data_dict = _rename_data_dict(riv_data, cleanup_func)
    # Assure conductance not modified by previous tests.
    np.testing.assert_equal(_first(data_dict["conductance"]), 1.0)
    idx = _first_index(data_dict["conductance"])
    # Arrange: Deactivate one cell
    first_key = next(iter(data_dict.keys()))
    data_dict[first_key][idx] = np.nan
    # Act
    data_cleaned = cleanup_func(**dis_dict, **data_dict)
    # Assert
    for key in data_cleaned.keys():
        np.testing.assert_equal(_first(data_cleaned[key][idx]), np.nan)


@parametrize_with_cases("riv_data, dis_data", cases=RivDisCases)
@parametrize("cleanup_func", [cleanup_drn, cleanup_ghb, cleanup_riv])
def test_cleanup__zero_conductance(
    riv_data: dict, dis_data: dict, cleanup_func: Callable
):
    dis_dict = _prepare_dis_dict(dis_data, cleanup_func)
    data_dict = _rename_data_dict(riv_data, cleanup_func)
    # Assure conductance not modified by previous tests.
    np.testing.assert_equal(_first(data_dict["conductance"]), 1.0)
    idx = _first_index(data_dict["conductance"])
    # Arrange: Deactivate one cell
    data_dict["conductance"][idx] = 0.0
    # Act
    data_cleaned = cleanup_func(**dis_dict, **data_dict)
    # Assert
    for key in data_cleaned.keys():
        np.testing.assert_equal(_first(data_cleaned[key][idx]), np.nan)


@parametrize_with_cases("riv_data, dis_data", cases=RivDisCases)
@parametrize("cleanup_func", [cleanup_drn, cleanup_ghb, cleanup_riv])
def test_cleanup__negative_concentration(
    riv_data: dict, dis_data: dict, cleanup_func: Callable
):
    dis_dict = _prepare_dis_dict(dis_data, cleanup_func)
    data_dict = _rename_data_dict(riv_data, cleanup_func)
    first_key = next(iter(data_dict.keys()))
    # Create concentration data
    data_dict["concentration"] = data_dict[first_key].copy()
    # Assure conductance not modified by previous tests.
    np.testing.assert_equal(_first(data_dict["conductance"]), 1.0)
    idx = _first_index(data_dict["conductance"])
    # Arrange: Deactivate one cell
    data_dict["concentration"][idx] = -10.0
    # Act
    data_cleaned = cleanup_func(**dis_dict, **data_dict)
    # Assert
    np.testing.assert_equal(_first(data_cleaned["concentration"]), 0.0)


@parametrize_with_cases("riv_data, dis_data", cases=RivDisCases)
@parametrize("cleanup_func", [cleanup_drn, cleanup_ghb, cleanup_riv])
def test_cleanup__outside_active_domain(
    riv_data: dict, dis_data: dict, cleanup_func: Callable
):
    dis_dict = _prepare_dis_dict(dis_data, cleanup_func)
    data_dict = _rename_data_dict(riv_data, cleanup_func)
    # Assure conductance not modified by previous tests.
    np.testing.assert_equal(_first(data_dict["conductance"]), 1.0)
    idx = _first_index(data_dict["conductance"])
    # Arrange: Deactivate one cell
    dis_dict["idomain"][idx] = 0.0
    # Act
    data_cleaned = cleanup_func(**dis_dict, **data_dict)
    # Assert
    for key in data_cleaned.keys():
        np.testing.assert_equal(_first(data_cleaned[key][idx]), np.nan)


@parametrize_with_cases("riv_data, dis_data", cases=RivDisCases)
def test_cleanup_riv__fix_bottom_elevation_to_bottom(riv_data: dict, dis_data: dict):
    dis_dict = _prepare_dis_dict(dis_data, cleanup_riv)
    # Arrange: Set bottom elevation model layer bottom
    riv_data["bottom_elevation"] -= 3.0
    # Assure conductance not modified by previous tests.
    np.testing.assert_equal(_first(riv_data["conductance"]), 1.0)
    # Act
    riv_data_cleaned = cleanup_riv(**dis_dict, **riv_data)
    # Assert
    # Account for cells inactive river cells.
    riv_active = riv_data_cleaned["stage"].notnull()
    expected = dis_dict["bottom"].where(riv_active)

    np.testing.assert_equal(
        riv_data_cleaned["bottom_elevation"].values, expected.values
    )


@parametrize_with_cases("riv_data, dis_data", cases=RivDisCases)
def test_cleanup_riv__fix_bottom_elevation_to_stage(riv_data: dict, dis_data: dict):
    dis_dict = _prepare_dis_dict(dis_data, cleanup_riv)
    # Arrange: Set bottom elevation above stage
    riv_data["bottom_elevation"] += 3.0
    # Assure conductance not modified by previous tests.
    np.testing.assert_equal(_first(riv_data["conductance"]), 1.0)
    # Act
    riv_data_cleaned = cleanup_riv(**dis_dict, **riv_data)
    # Assert
    np.testing.assert_equal(
        riv_data_cleaned["bottom_elevation"].values, riv_data_cleaned["stage"].values
    )


@parametrize_with_cases("riv_data, dis_data", cases=RivDisCases)
def test_cleanup_riv__raise_error(riv_data: dict, dis_data: dict):
    """
    Test if error raised when stage below model layer bottom and see if user is
    guided to the right prepare function.
    """
    dis_dict = _prepare_dis_dict(dis_data, cleanup_riv)
    # Arrange: Set bottom elevation above stage
    riv_data["stage"] -= 10.0
    # Act
    with pytest.raises(ValueError, match="imod.prepare.topsystem.allocate_riv_cells"):
        cleanup_riv(**dis_dict, **riv_data)
