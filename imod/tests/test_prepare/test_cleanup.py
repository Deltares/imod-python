from typing import Callable

import numpy as np
import xugrid as xu
from pytest_cases import parametrize, parametrize_with_cases

from imod.prepare.cleanup import cleanup_drn, cleanup_ghb, cleanup_riv
from imod.tests.test_mf6.test_mf6_riv import RivCases
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


_RENAME_DICT = {
    cleanup_riv: {},
    cleanup_drn: {"stage": "elevation", "bottom_elevation": None},
    cleanup_ghb: {"stage": "head", "bottom_elevation": None},
}


@parametrize_with_cases("riv_data", cases=RivCases)
@parametrize("cleanup_func", [cleanup_drn, cleanup_ghb, cleanup_riv])
def test_cleanup__align_nodata(riv_data: dict, cleanup_func: Callable):
    data_dict = _rename_data_dict(riv_data, cleanup_func)
    # Assure conductance not modified by previous tests.
    np.testing.assert_equal(_first(data_dict["conductance"]), 1.0)
    idx = _first_index(data_dict["conductance"])
    # Arrange: Deactivate one cell
    first_key = next(iter(data_dict.keys()))
    data_dict[first_key][idx] = np.nan
    # Act
    data_cleaned = cleanup_func(**data_dict)
    # Assert
    for key in data_cleaned.keys():
        np.testing.assert_equal(_first(data_cleaned[key][idx]), np.nan)


@parametrize_with_cases("riv_data", cases=RivCases)
@parametrize("cleanup_func", [cleanup_drn, cleanup_ghb, cleanup_riv])
def test_cleanup__zero_conductance(riv_data: dict, cleanup_func: Callable):
    data_dict = _rename_data_dict(riv_data, cleanup_func)
    # Assure conductance not modified by previous tests.
    np.testing.assert_equal(_first(data_dict["conductance"]), 1.0)
    idx = _first_index(data_dict["conductance"])
    # Arrange: Deactivate one cell
    data_dict["conductance"][idx] = 0.0
    # Act
    data_cleaned = cleanup_func(**data_dict)
    # Assert
    for key in data_cleaned.keys():
        np.testing.assert_equal(_first(data_cleaned[key][idx]), np.nan)


@parametrize_with_cases("riv_data", cases=RivCases)
def test_cleanup_riv__fix_bottom_elevation(riv_data):
    # Arrange: Set bottom elevation above stage
    riv_data["bottom_elevation"] += 3.0
    # Assure conductance not modified by previous tests.
    np.testing.assert_equal(_first(riv_data["conductance"]), 1.0)
    # Act
    riv_data_cleaned = cleanup_riv(**riv_data)
    # Assert
    np.testing.assert_equal(
        riv_data_cleaned["bottom_elevation"].values, riv_data_cleaned["stage"].values
    )
