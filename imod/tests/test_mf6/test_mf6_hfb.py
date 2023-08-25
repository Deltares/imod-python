from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pytest
import shapely
import xarray as xr
import xugrid as xu
from numpy.testing import assert_array_equal

from imod.mf6 import (
    HorizontalFlowBarrierHydraulicCharacteristic,
    HorizontalFlowBarrierMultiplier,
    HorizontalFlowBarrierResistance,
)
from imod.mf6.hfb import to_connected_cells_dataset
from imod.typing.grid import ones_like


@pytest.mark.parametrize("dis", ["basic_unstructured_dis", "basic_dis"])
@pytest.mark.parametrize(
    "barrier_class, barrier_value_name, barrier_value, expected_hydraulic_characteristic",
    [
        (HorizontalFlowBarrierResistance, "resistance", 1e3, 1e-3),
        (HorizontalFlowBarrierMultiplier, "multiplier", 1.5, -1.5),
        (
            HorizontalFlowBarrierHydraulicCharacteristic,
            "hydraulic_characteristic",
            1e-3,
            1e-3,
        ),
    ],
)
@patch("imod.mf6.mf6_hfb_adapter.Mf6HorizontalFlowBarrier.__new__", autospec=True)
def test_to_mf6_creates_mf6_adapter(
    mf6_flow_barrier_mock,
    dis,
    barrier_class,
    barrier_value_name,
    barrier_value,
    expected_hydraulic_characteristic,
    request,
):
    # Arrange.
    idomain, top, bottom = request.getfixturevalue(dis)
    k = ones_like(top)

    print_input = False

    barrier_y = [5.5, 5.5, 5.5]
    barrier_x = [82.0, 40.0, 0.0]

    geometry = gpd.GeoDataFrame(
        geometry=[shapely.linestrings(barrier_x, barrier_y)],
        data={
            barrier_value_name: [barrier_value],
            "ztop": [0.0],
            "zbottom": [min(bottom.values)],
        },
    )

    hfb = barrier_class(geometry, print_input)

    # Act.
    _ = hfb.to_mf6_pkg(idomain, top, bottom, k)

    # Assert.
    snapped, _ = xu.snap_to_grid(geometry, grid=idomain, max_snap_distance=0.5)
    edge_index = np.argwhere(snapped[barrier_value_name].notnull().values).ravel()

    grid = (
        idomain.ugrid.grid
        if isinstance(idomain, xu.UgridDataArray)
        else xu.UgridDataArray.from_structured(idomain).ugrid.grid
    )

    expected_barrier_values = xr.DataArray(
        np.ones((idomain.coords["layer"].size, edge_index.size)) * barrier_value,
        dims=("layer", "mesh2d_nFaces"),
        coords={"layer": idomain.coords["layer"]},
    )

    expected_values = to_connected_cells_dataset(
        idomain, grid, edge_index, {barrier_value_name: expected_barrier_values}
    )

    mf6_flow_barrier_mock.assert_called_once()

    _, args = mf6_flow_barrier_mock.call_args
    assert args["cell_id1"].equals(expected_values["cell_id1"])
    assert args["cell_id2"].equals(expected_values["cell_id2"])
    assert args["layer"].equals(expected_values["layer"])
    assert (
        args["hydraulic_characteristic"].values.max()
        == expected_hydraulic_characteristic
    )


@pytest.mark.parametrize(
    "ztop, zbottom, expected_values",
    [
        (-5.0, -35.0, np.array([1, 1e3, 1])),  # 2nd layer
        (0.0, -35.0, np.array([1e3, 1e3, 1])),  # 1st and 2nd layer
        (-5.0, -135.0, np.array([1, 1e3, 1e3])),  # 2nd and 3th layer
        (-5.0, -135.0, np.array([1, 1e3, 1e3])),  # 2nd and 3th layer
        (100.0, -135.0, np.array([1e3, 1e3, 1e3])),  # ztop out of bounds
        (0.0, -200.0, np.array([1e3, 1e3, 1e3])),  # zbottom out of bounds
        (100.0, 50.0, np.array([1, 1, 1])),  # z-range has no overlap with the domain
        (
            0.0,
            -2.5,
            np.array([1 / ((0.5 / 1e3) + ((1.0 - 0.5) / 1)), 1, 1]),
        ),  # test effective resistance: 1/((fraction / resistance) + ((1.0 - fraction) / c_aquifer))
    ],
)
@patch("imod.mf6.mf6_hfb_adapter.Mf6HorizontalFlowBarrier.__new__", autospec=True)
def test_to_mf6_different_z_boundaries(
    mf6_flow_barrier_mock, basic_dis, ztop, zbottom, expected_values
):
    # Arrange.
    idomain, top, bottom = basic_dis
    k = ones_like(top)

    print_input = False

    barrier_y = [5.5, 5.5, 5.5]
    barrier_x = [82.0, 40.0, 0.0]

    geometry = gpd.GeoDataFrame(
        geometry=[shapely.linestrings(barrier_x, barrier_y)],
        data={
            "resistance": [1e3],
            "ztop": [ztop],
            "zbottom": [zbottom],
        },
    )

    hfb = HorizontalFlowBarrierResistance(geometry, print_input)

    # Act.
    _ = hfb.to_mf6_pkg(idomain, top, bottom, k)

    # Assert.
    _, args = mf6_flow_barrier_mock.call_args
    barrier_values = args["hydraulic_characteristic"].values.reshape(3, 8)
    max_values_per_layer = barrier_values.max(axis=1)
    assert_array_equal(max_values_per_layer, 1.0 / expected_values)


@pytest.mark.parametrize(
    "barrier_x_loc, expected_number_barriers",
    [
        (0.0, 0),  # barrier lies on left boundary
        (1.0, 1),  # barrier lies between active cells
        (2.0, 0),  # barrier lies between an active and inactive cell
    ],
)
@patch("imod.mf6.mf6_hfb_adapter.Mf6HorizontalFlowBarrier.__new__", autospec=True)
def test_to_mf6_remove_invalid_edges(
    mf6_flow_barrier_mock, barrier_x_loc, expected_number_barriers
):
    # Arrange.
    shape = nlay, nrow, ncol = 1, 1, 3

    dx = dy = dz = 1.0

    x = np.arange(0, ncol + 1) * dx
    y = np.arange(0, nrow + 1)[::-1] * dy
    z = -np.arange(0, nlay + 1) * dz

    xc = (x[:-1] + x[1:]) / 2
    yc = (y[:-1] + y[1:]) / 2

    layers = np.arange(nlay) + 1

    idomain = xr.DataArray(
        np.ones(shape, dtype=np.int32), coords={"layer": layers, "y": yc, "x": xc}
    )
    idomain = idomain.assign_coords({"dy": -dy})
    idomain.loc[{"x": xc[-1]}] = -1  # make cells inactive

    top = xr.DataArray([z[0]], coords={"layer": layers})
    bottom = xr.DataArray(z[1:], coords={"layer": layers})
    k = ones_like(top)

    barrier_y = [0.0, 2.0]
    barrier_x = [barrier_x_loc, barrier_x_loc]

    geometry = gpd.GeoDataFrame(
        geometry=[shapely.linestrings(barrier_x, barrier_y)],
        data={
            "resistance": [1e3],
            "ztop": [0],
            "zbottom": [-5],
        },
    )

    hfb = HorizontalFlowBarrierResistance(geometry)

    # Act.
    _ = hfb.to_mf6_pkg(idomain, top, bottom, k)

    # Assert.
    _, args = mf6_flow_barrier_mock.call_args
    assert args["cell_id1"][0, :].size == expected_number_barriers
    assert args["cell_id2"][0, :].size == expected_number_barriers
    assert args["layer"].size == expected_number_barriers


@pytest.mark.parametrize(
    "barrier_x_loc, expected_number_barriers",
    [
        # barrier lies between active cells
        (1.0, 2),
        # barrier lies between active cells in the top layer but between an inactive and active cell in the bottom layer
        (2.0, 1),
    ],
)
@patch("imod.mf6.mf6_hfb_adapter.Mf6HorizontalFlowBarrier.__new__", autospec=True)
def test_to_mf6_remove_barrier_parts_adjacent_to_inactive_cells(
    mf6_flow_barrier_mock, barrier_x_loc, expected_number_barriers
):
    # Arrange.
    shape = nlay, nrow, ncol = 2, 1, 3

    dx = dy = dz = 1.0

    x = np.arange(0, ncol + 1) * dx
    y = np.arange(0, nrow + 1)[::-1] * dy
    z = -np.arange(0, nlay + 1) * dz

    xc = (x[:-1] + x[1:]) / 2
    yc = (y[:-1] + y[1:]) / 2

    layers = np.arange(nlay) + 1

    idomain = xr.DataArray(
        np.ones(shape, dtype=np.int32), coords={"layer": layers, "y": yc, "x": xc}
    )
    idomain = idomain.assign_coords({"dy": -dy})
    idomain.loc[{"x": xc[-1], "layer": layers[-1]}] = -1  # make cell inactive

    top = xr.DataArray(z[:-1], coords={"layer": layers})
    bottom = xr.DataArray(z[1:], coords={"layer": layers})
    k = ones_like(top)

    barrier_y = [0.0, 2.0]
    barrier_x = [barrier_x_loc, barrier_x_loc]

    geometry = gpd.GeoDataFrame(
        geometry=[shapely.linestrings(barrier_x, barrier_y)],
        data={
            "resistance": [1e3],
            "ztop": [0],
            "zbottom": [-5],
        },
    )

    hfb = HorizontalFlowBarrierResistance(geometry)

    # Act.
    _ = hfb.to_mf6_pkg(idomain, top, bottom, k)

    # Assert.
    _, args = mf6_flow_barrier_mock.call_args
    assert args["cell_id1"][0, :].size == expected_number_barriers
    assert args["cell_id2"][0, :].size == expected_number_barriers
    assert args["layer"].size == expected_number_barriers
