from typing import Callable

import geopandas as gpd
from shapely import linestrings
import numpy as np
import pandas as pd
import pytest
import xugrid as xu
from pytest_cases import parametrize, parametrize_with_cases

from imod.prepare.cleanup import cleanup_drn, cleanup_ghb, cleanup_riv, cleanup_wel, cleanup_hfb
from imod.tests.test_mf6.test_mf6_riv import DisCases, RivDisCases
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
    cleanup_wel: ["top", "bottom"],
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
def test_cleanup_riv__stage_equals_bottom_elevation(riv_data: dict, dis_data: dict):
    """Assure no cleanup accidentily takes place when stage equals bottom_elevation"""
    dis_dict = _prepare_dis_dict(dis_data, cleanup_riv)
    # Arrange: Set bottom elevation equal to stage
    riv_data["bottom_elevation"] = riv_data["stage"].copy()
    # Assure conductance not modified by previous tests.
    np.testing.assert_equal(_first(riv_data["conductance"]), 1.0)
    # Act
    riv_data_cleaned = cleanup_riv(**dis_dict, **riv_data)
    # Assert
    np.testing.assert_equal(
        riv_data_cleaned["bottom_elevation"].values, riv_data_cleaned["stage"].values
    )
    np.testing.assert_equal(riv_data["stage"].values, riv_data_cleaned["stage"].values)
    np.testing.assert_equal(
        riv_data["bottom_elevation"].values, riv_data_cleaned["bottom_elevation"].values
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


@parametrize_with_cases("dis_data", cases=DisCases)
def test_cleanup_wel(dis_data: dict):
    """
    Cleanup wells.

    Cases by id (on purpose not in order, to see if pandas'
    sorting results in any issues):

    a: filter completely above surface level -> point filter in top layer
    c: filter partly above surface level -> filter top set to surface level
    b: filter completely below model base -> well should be removed
    d: filter partly below model base -> filter bottom set to model base
    f: well outside grid bounds -> well should be removed
    e: utrathin filter -> filter should be forced to point filter
    g: filter screen_bottom above screen_top -> filter should be forced to point filter
    """
    # Arrange
    dis_dict = _prepare_dis_dict(dis_data, cleanup_wel)
    wel_dict = {
        "id": ["a", "c", "b", "d", "f", "e", "g"],
        "x": [17.0, 17.0, 17.0, 17.0, 40.0, 17.0, 17.0],
        "y": [15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0],
        "screen_top": [
            2.0,
            2.0,
            -7.0,
            -1.0,
            -1.0,
            1e-3,
            0.0,
        ],
        "screen_bottom": [
            1.5,
            0.0,
            -8.0,
            -8.0,
            -1.0,
            0.0,
            0.5,
        ],
    }
    well_df = pd.DataFrame(wel_dict)
    wel_expected = {
        "id": ["a", "c", "d", "e", "g"],
        "x": [17.0, 17.0, 17.0, 17.0, 17.0],
        "y": [15.0, 15.0, 15.0, 15.0, 15.0],
        "screen_top": [
            1.0,
            1.0,
            -1.0,
            1e-3,
            0.0,
        ],
        "screen_bottom": [
            1.0,
            0.0,
            -1.5,
            1e-3,
            0.0,
        ],
    }
    well_expected_df = pd.DataFrame(wel_expected).set_index("id")
    # Act
    well_cleaned = cleanup_wel(well_df, **dis_dict)
    # Assert
    pd.testing.assert_frame_equal(well_cleaned, well_expected_df)


@parametrize_with_cases("dis_data", cases=DisCases)
def test_cleanup_hfb(dis_data: dict):
    # Arrange
    barrier_y = [25.0, 15.0, -1.0]
    barrier_x = [16.0, 16.0, 16.0]

    geometry = gpd.GeoDataFrame(
        geometry=[linestrings(barrier_x, barrier_y)],
        data={
            "resistance": [1200.0],
            "layer": [1],
        },
    )
    y_max = 20.0
    y = dis_data["idomain"].coords["y"]
    idomain = dis_data["idomain"].where(y < y_max, 0)

    # Act
    with pytest.raises(ValueError):
        cleanup_hfb(geometry, idomain)
    
    clipped_geometry = cleanup_hfb(geometry, idomain.isel(layer=0))

    # Assert
    np.testing.assert_allclose(clipped_geometry.bounds.maxy, y_max)

