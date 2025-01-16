import numpy as np
import xarray as xr
from pytest_cases import parametrize_with_cases

from imod.prepare.topsystem import (
    allocate_drn_cells,
    allocate_ghb_cells,
    allocate_rch_cells,
    allocate_riv_cells,
    distribute_drn_conductance,
    distribute_ghb_conductance,
    distribute_riv_conductance,
    ALLOCATION_OPTION
)
from imod.typing import GridDataArray
from imod.typing.grid import is_unstructured, zeros_like
from imod.util.dims import enforce_dim_order


def take_nth_layer_column(grid: GridDataArray, n: int) -> GridDataArray:
    """
    Parameters
    ----------
    grid: DataArray | UgridDataArray
        grid to take values from. Must have dimensions (layer,y,x) for
        structured and (layer,{face_dim}) for unstructured grids.
    n: int
        index number in the xy plane where layer column is taken.

    Returns
    -------
    DataArray | UgridDataArray
        Column along the layer dimension at the nth cell in the xy plane.
    """
    if "time" in grid.dims:
        grid = grid.isel(time=-1)

    if is_unstructured(grid):
        return grid.values[:, n]
    else:
        return grid.values[:, n, n]


@parametrize_with_cases(
    argnames="active,top,bottom,stage,bottom_elevation",
    prefix="riv_",
)
@parametrize_with_cases(
    argnames="option,expected_riv,expected_drn", prefix="allocation_", has_tag="riv"
)
def test_riv_allocation(
    active, top, bottom, stage, bottom_elevation, option, expected_riv, expected_drn
):
    actual_riv_da, actual_drn_da = allocate_riv_cells(
        option, active, top, bottom, stage, bottom_elevation
    )

    actual_riv = take_nth_layer_column(actual_riv_da, 0)
    empty_riv = take_nth_layer_column(actual_riv_da, 1)

    if actual_drn_da is None:
        actual_drn = None
        empty_drn = None
    else:
        actual_drn = take_nth_layer_column(actual_drn_da, 0)
        empty_drn = take_nth_layer_column(actual_drn_da, 1)

    np.testing.assert_equal(actual_riv, expected_riv)
    np.testing.assert_equal(actual_drn, expected_drn)
    assert np.all(~empty_riv)
    if empty_drn is not None:
        assert np.all(~empty_drn)


@parametrize_with_cases(
    argnames="active,top,bottom,drn_elevation",
    prefix="drn_",
)
@parametrize_with_cases(
    argnames="option,expected,_", prefix="allocation_", has_tag="drn"
)
def test_drn_allocation(active, top, bottom, drn_elevation, option, expected, _):
    actual_da = allocate_drn_cells(option, active, top, bottom, drn_elevation)

    actual = take_nth_layer_column(actual_da, 0)
    empty = take_nth_layer_column(actual_da, 1)

    np.testing.assert_equal(actual, expected)
    assert np.all(~empty)


@parametrize_with_cases(
    argnames="active,top,bottom,head",
    prefix="ghb_",
)
@parametrize_with_cases(
    argnames="option,expected,_", prefix="allocation_", has_tag="ghb"
)
def test_ghb_allocation(active, top, bottom, head, option, expected, _):
    actual_da = allocate_ghb_cells(option, active, top, bottom, head)

    actual = take_nth_layer_column(actual_da, 0)
    empty = take_nth_layer_column(actual_da, 1)

    np.testing.assert_equal(actual, expected)
    assert np.all(~empty)


@parametrize_with_cases(
    argnames="active,rate",
    prefix="rch_",
)
@parametrize_with_cases(
    argnames="option,expected,_", prefix="allocation_", has_tag="rch"
)
def test_rch_allocation(active, rate, option, expected, _):
    actual_da = allocate_rch_cells(option, active, rate)

    actual = take_nth_layer_column(actual_da, 0)
    empty = take_nth_layer_column(actual_da, 1)

    np.testing.assert_equal(actual, expected)
    assert np.all(~empty)


@parametrize_with_cases(
    argnames="active,top,bottom,stage,bottom_elevation",
    prefix="riv_",
)
@parametrize_with_cases(
    argnames="option,allocated_layer,expected", prefix="distribution_", has_tag="riv"
)
def test_distribute_riv_conductance(
    active, top, bottom, stage, bottom_elevation, option, allocated_layer, expected
):
    allocated = enforce_dim_order(active & allocated_layer)
    k = xr.DataArray(
        [2.0, 2.0, 1.0, 1.0], coords={"layer": [1, 2, 3, 4]}, dims=("layer",)
    )

    conductance = zeros_like(bottom_elevation) + 1.0

    actual_da = distribute_riv_conductance(
        option, allocated, conductance, top, bottom, k, stage, bottom_elevation
    )
    actual = take_nth_layer_column(actual_da, 0)

    np.testing.assert_equal(actual, expected)


@parametrize_with_cases(
    argnames="active,top,bottom,elevation",
    prefix="drn_",
)
@parametrize_with_cases(
    argnames="option,allocated_layer,expected", prefix="distribution_", has_tag="drn"
)
def test_distribute_drn_conductance(
    active, top, bottom, elevation, option, allocated_layer, expected
):
    allocated = enforce_dim_order(active & allocated_layer)
    k = xr.DataArray(
        [2.0, 2.0, 1.0, 1.0], coords={"layer": [1, 2, 3, 4]}, dims=("layer",)
    )

    conductance = zeros_like(elevation) + 1.0

    actual_da = distribute_drn_conductance(
        option, allocated, conductance, top, bottom, k, elevation
    )
    actual = take_nth_layer_column(actual_da, 0)

    np.testing.assert_equal(actual, expected)


@parametrize_with_cases(
    argnames="active,top,bottom,elevation",
    prefix="ghb_",
)
@parametrize_with_cases(
    argnames="option,allocated_layer,expected", prefix="distribution_", has_tag="ghb"
)
def test_distribute_ghb_conductance(
    active, top, bottom, elevation, option, allocated_layer, expected
):
    allocated = enforce_dim_order(active & allocated_layer)
    k = xr.DataArray(
        [2.0, 2.0, 1.0, 1.0], coords={"layer": [1, 2, 3, 4]}, dims=("layer",)
    )

    conductance = zeros_like(elevation) + 1.0

    actual_da = distribute_ghb_conductance(
        option, allocated, conductance, top, bottom, k
    )
    actual = take_nth_layer_column(actual_da, 0)

    np.testing.assert_equal(actual, expected)


@parametrize_with_cases(
    argnames="active,top,bottom,stage,bottom_elevation",
    prefix="riv_",
)
@parametrize_with_cases(
    argnames="option,expected_riv,expected_drn", prefix="allocation_", has_tag="riv"
)
def test_riv_allocation__elevation_above_surface_level(
    active, top, bottom, stage, bottom_elevation, option, expected_riv, expected_drn
):
    # Put elevations a lot above surface level. Need to be allocated to first
    # layer.
    actual_riv_da, actual_drn_da = allocate_riv_cells(
        option, active, top, bottom, stage + 100.0, bottom_elevation + 100.0
    )

    # Override expected values
    expected_riv = [True, False, False, False]
    if expected_drn:
        expected_drn = [False, False, False, False]

    actual_riv = take_nth_layer_column(actual_riv_da, 0)
    empty_riv = take_nth_layer_column(actual_riv_da, 1)

    if actual_drn_da is None:
        actual_drn = None
        empty_drn = None
    else:
        actual_drn = take_nth_layer_column(actual_drn_da, 0)
        empty_drn = take_nth_layer_column(actual_drn_da, 1)

    np.testing.assert_equal(actual_riv, expected_riv)
    np.testing.assert_equal(actual_drn, expected_drn)
    assert np.all(~empty_riv)
    if empty_drn is not None:
        assert np.all(~empty_drn)


@parametrize_with_cases(
    argnames="active,top,bottom,stage,bottom_elevation",
    prefix="riv_",
)
@parametrize_with_cases(
    argnames="option,expected_riv,expected_drn", prefix="allocation_", has_tag="riv"
)
def test_riv_allocation__stage_equals_bottom_elevation(
    active, top, bottom, stage, bottom_elevation, option, expected_riv, expected_drn
):
    # Put elevations a lot above surface level. Need to be allocated to first
    # layer.
    actual_riv_da, actual_drn_da = allocate_riv_cells(
        option, active, top, bottom, stage, stage
    )

    # Override expected values
    if option is ALLOCATION_OPTION.first_active_to_elevation:
        expected_riv = [True, True, False, False]
    elif option is not ALLOCATION_OPTION.at_first_active:
        expected_riv = [False, True, False, False]
    if expected_drn:
        expected_drn = [True, False, False, False]

    actual_riv = take_nth_layer_column(actual_riv_da, 0)
    empty_riv = take_nth_layer_column(actual_riv_da, 1)

    if actual_drn_da is None:
        actual_drn = None
        empty_drn = None
    else:
        actual_drn = take_nth_layer_column(actual_drn_da, 0)
        empty_drn = take_nth_layer_column(actual_drn_da, 1)

    np.testing.assert_equal(actual_riv, expected_riv)
    np.testing.assert_equal(actual_drn, expected_drn)
    assert np.all(~empty_riv)
    if empty_drn is not None:
        assert np.all(~empty_drn)


@parametrize_with_cases(
    argnames="active,top,bottom,elevation",
    prefix="drn_",
)
@parametrize_with_cases(
    argnames="option,expected,_", prefix="allocation_", has_tag="drn"
)
def test_drn_allocation__elevation_above_surface_level(
    active, top, bottom, elevation, option, expected, _
):
    # Put elevations a lot above surface level. Need to be allocated to first
    # layer.
    actual_da = allocate_drn_cells(
        option,
        active,
        top,
        bottom,
        elevation + 100.0,
    )

    # Override expected
    expected = [True, False, False, False]

    actual = take_nth_layer_column(actual_da, 0)
    empty = take_nth_layer_column(actual_da, 1)

    np.testing.assert_equal(actual, expected)
    assert np.all(~empty)
    if empty is not None:
        assert np.all(~empty)


@parametrize_with_cases(
    argnames="active,top,bottom,head",
    prefix="ghb_",
)
@parametrize_with_cases(
    argnames="option,expected,_", prefix="allocation_", has_tag="ghb"
)
def test_ghb_allocation__elevation_above_surface_level(
    active, top, bottom, head, option, expected, _
):
    # Put elevations a lot above surface level. Need to be allocated to first
    # layer.
    actual_da = allocate_ghb_cells(
        option,
        active,
        top,
        bottom,
        head + 100.0,
    )

    # Override expected
    expected = [True, False, False, False]

    actual = take_nth_layer_column(actual_da, 0)
    empty = take_nth_layer_column(actual_da, 1)

    np.testing.assert_equal(actual, expected)
    assert np.all(~empty)
    if empty is not None:
        assert np.all(~empty)


@parametrize_with_cases(
    argnames="active,top,bottom,elevation",
    prefix="drn_",
)
@parametrize_with_cases(
    argnames="option,allocated_layer,_", prefix="distribution_", has_tag="drn"
)
def test_distribute_drn_conductance__above_surface_level(
    active, top, bottom, elevation, option, allocated_layer, _
):
    allocated_layer.data = np.array([True, False, False, False])
    expected = [1.0, np.nan, np.nan, np.nan]
    allocated = enforce_dim_order(active & allocated_layer)
    k = xr.DataArray(
        [2.0, 2.0, 1.0, 1.0], coords={"layer": [1, 2, 3, 4]}, dims=("layer",)
    )

    conductance = zeros_like(elevation) + 1.0

    actual_da = distribute_drn_conductance(
        option, allocated, conductance, top, bottom, k, elevation + 100.0
    )
    actual = take_nth_layer_column(actual_da, 0)

    np.testing.assert_equal(actual, expected)


@parametrize_with_cases(
    argnames="active,top,bottom,elevation",
    prefix="drn_",
)
@parametrize_with_cases(
    argnames="option,allocated_layer,_", prefix="distribution_", has_tag="drn"
)
def test_distribute_drn_conductance__equal_to_surface_level(
    active, top, bottom, elevation, option, allocated_layer, _
):
    allocated_layer.data = np.array([True, False, False, False])
    expected = [1.0, np.nan, np.nan, np.nan]
    allocated = enforce_dim_order(active & allocated_layer)
    k = xr.DataArray(
        [2.0, 2.0, 1.0, 1.0], coords={"layer": [1, 2, 3, 4]}, dims=("layer",)
    )

    conductance = zeros_like(elevation) + 1.0
    elevation = zeros_like(elevation) + top

    actual_da = distribute_drn_conductance(
        option, allocated, conductance, top, bottom, k, elevation
    )
    actual = take_nth_layer_column(actual_da, 0)

    np.testing.assert_equal(actual, expected)


@parametrize_with_cases(
    argnames="active,top,bottom,stage,bottom_elevation",
    prefix="riv_",
)
@parametrize_with_cases(
    argnames="option,allocated_layer,_", prefix="distribution_", has_tag="riv"
)
def test_distribute_riv_conductance__above_surface_level(
    active, top, bottom, stage, bottom_elevation, option, allocated_layer, _
):
    allocated_layer.data = np.array([True, False, False, False])
    expected = [1.0, np.nan, np.nan, np.nan]
    allocated = enforce_dim_order(active & allocated_layer)
    k = xr.DataArray(
        [2.0, 2.0, 1.0, 1.0], coords={"layer": [1, 2, 3, 4]}, dims=("layer",)
    )

    conductance = zeros_like(bottom_elevation) + 1.0

    actual_da = distribute_riv_conductance(
        option,
        allocated,
        conductance,
        top,
        bottom,
        k,
        stage + 100.0,
        bottom_elevation + 100.0,
    )
    actual = take_nth_layer_column(actual_da, 0)

    np.testing.assert_equal(actual, expected)


@parametrize_with_cases(
    argnames="active,top,bottom,stage,bottom_elevation",
    prefix="riv_",
)
@parametrize_with_cases(
    argnames="option,allocated_layer,_", prefix="distribution_", has_tag="riv"
)
def test_distribute_riv_conductance__equal_to_surface_level(
    active, top, bottom, stage, bottom_elevation, option, allocated_layer, _
):
    allocated_layer.data = np.array([True, False, False, False])
    expected = [1.0, np.nan, np.nan, np.nan]
    allocated = enforce_dim_order(active & allocated_layer)
    k = xr.DataArray(
        [2.0, 2.0, 1.0, 1.0], coords={"layer": [1, 2, 3, 4]}, dims=("layer",)
    )

    conductance = zeros_like(bottom_elevation) + 1.0
    elevation = zeros_like(bottom_elevation) + top

    actual_da = distribute_riv_conductance(
        option,
        allocated,
        conductance,
        top,
        bottom,
        k,
        elevation,
        elevation,
    )
    actual = take_nth_layer_column(actual_da, 0)

    np.testing.assert_equal(actual, expected)


@parametrize_with_cases(
    argnames="active,top,bottom,elevation",
    prefix="ghb_",
)
@parametrize_with_cases(
    argnames="option,allocated_layer,_", prefix="distribution_", has_tag="ghb"
)
def test_distribute_ghb_conductance__above_surface_level(
    active, top, bottom, elevation, option, allocated_layer, _
):
    allocated_layer.data = np.array([True, False, False, False])
    expected = [1.0, np.nan, np.nan, np.nan]
    allocated = enforce_dim_order(active & allocated_layer)
    k = xr.DataArray(
        [2.0, 2.0, 1.0, 1.0], coords={"layer": [1, 2, 3, 4]}, dims=("layer",)
    )

    conductance = zeros_like(elevation) + 1.0

    actual_da = distribute_ghb_conductance(
        option, allocated, conductance, top, bottom, k
    )
    actual = take_nth_layer_column(actual_da, 0)

    np.testing.assert_equal(actual, expected)
