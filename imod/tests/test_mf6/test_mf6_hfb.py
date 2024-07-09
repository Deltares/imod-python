from copy import deepcopy
from itertools import pairwise
from typing import List, Tuple
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
import xugrid as xu
from numpy.testing import assert_array_equal
from shapely import Polygon, get_coordinates, linestrings

from imod.mf6 import (
    HorizontalFlowBarrierHydraulicCharacteristic,
    HorizontalFlowBarrierMultiplier,
    HorizontalFlowBarrierResistance,
    SingleLayerHorizontalFlowBarrierHydraulicCharacteristic,
    SingleLayerHorizontalFlowBarrierMultiplier,
    SingleLayerHorizontalFlowBarrierResistance,
)
from imod.mf6.dis import StructuredDiscretization
from imod.mf6.hfb import (
    _extract_hfb_bounds_from_dataframe,
    _make_linestring_from_polygon,
    _prepare_barrier_dataset_for_mf6_adapter,
    to_connected_cells_dataset,
)
from imod.mf6.ims import SolutionPresetSimple
from imod.mf6.npf import NodePropertyFlow
from imod.mf6.simulation import Modflow6Simulation
from imod.mf6.utilities.regrid import RegridderWeightsCache
from imod.tests.fixtures.flow_basic_fixture import BasicDisSettings
from imod.typing.grid import nan_like, ones_like


def line_to_square_zpolygon(
    x: Tuple[float, float], y: Tuple[float, float], z: Tuple[float, float]
) -> Polygon:
    """
    Creates polygon as follows:

    xy0,z0 -- xy1,z0
       |        |
       |        |
    xy0,z1 -- xy1,z1
    """
    return Polygon(
        (
            (x[0], y[0], z[0]),
            (x[0], y[0], z[1]),
            (x[1], y[1], z[1]),
            (x[1], y[1], z[0]),
        ),
    )


def linestring_to_square_zpolygons(
    barrier_x: List[float],
    barrier_y: List[float],
    barrier_ztop: List[float],
    barrier_zbottom: List[float],
) -> List[Polygon]:
    x_pairs = pairwise(barrier_x)
    y_pairs = pairwise(barrier_y)
    z_pairs = zip(barrier_ztop, barrier_zbottom)
    return [line_to_square_zpolygon(x, y, z) for x, y, z in zip(x_pairs, y_pairs, z_pairs)]


def line_to_trapezoid_zpolygon(
    x: Tuple[float, float], y: Tuple[float, float], zt: Tuple[float, float], zb: Tuple[float, float],
) -> Polygon:
    """
    Creates polygon as follows:

    xy0,zt0 
       |    \
       |     \
       |      xy1,zt1
       |         |
       |         |
       |      xy1,zb1
       |     /   
       |    /    
    xy0,zb0  
    """
    return Polygon(
        (
            (x[0], y[0], zt[0]),
            (x[0], y[0], zb[1]),
            (x[1], y[1], zt[1]),
            (x[1], y[1], zb[0]),
        ),
    )

def linestring_to_trapezoid_zpolygons(
    barrier_x: List[float],
    barrier_y: List[float],
    barrier_ztop: List[float],
    barrier_zbottom: List[float],
) -> List[Polygon]:
    x_pairs = pairwise(barrier_x)
    y_pairs = pairwise(barrier_y)
    zt_pairs = pairwise(barrier_ztop) 
    zb_pairs = pairwise(barrier_zbottom)
    return [line_to_trapezoid_zpolygon(x, y, zt, zb) for x, y, zt, zb in zip(x_pairs, y_pairs, zt_pairs, zb_pairs)]


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
def test_to_mf6_creates_mf6_adapter_init(
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

    barrier_ztop = [0.0, 0.0]
    barrier_zbottom = [min(bottom.values), min(bottom.values)]
    barrier_y = [5.5, 5.5, 5.5]
    barrier_x = [82.0, 40.0, 0.0]

    polygons = linestring_to_square_zpolygons(
        barrier_x, barrier_y, barrier_ztop, barrier_zbottom
    )

    geometry = gpd.GeoDataFrame(
        geometry=polygons,
        data={
            barrier_value_name: [barrier_value, barrier_value],
        },
    )

    hfb = barrier_class(geometry, print_input)

    # Act.
    _ = hfb.to_mf6_pkg(idomain, top, bottom, k)

    # Assert.
    lines = _make_linestring_from_polygon(geometry)
    gdf_line = deepcopy(geometry)
    gdf_line["geometry"] = lines
    snapped, _ = xu.snap_to_grid(gdf_line, grid=idomain, max_snap_distance=0.5)
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
    expected_values = _prepare_barrier_dataset_for_mf6_adapter(expected_values)

    mf6_flow_barrier_mock.assert_called_once()

    _, args = mf6_flow_barrier_mock.call_args
    assert args["cell_id1"].equals(expected_values["cell_id1"])
    assert args["cell_id2"].equals(expected_values["cell_id2"])
    assert args["layer"].equals(expected_values["layer"])
    assert (
        args["hydraulic_characteristic"].values.max()
        == expected_hydraulic_characteristic
    )


@pytest.mark.parametrize("dis", ["basic_unstructured_dis", "basic_dis"])
def test_hfb_regrid(
    dis,
    request,
):
    # Arrange
    idomain, _, _ = request.getfixturevalue(dis)

    print_input = False

    barrier_ztop = [0.0, 0.0]
    barrier_zbottom = [-100.0, -100.0]
    barrier_y = [5.5, 5.5, 5.5]
    barrier_x = [82.0, 40.0, 0.0]

    polygons = linestring_to_square_zpolygons(
        barrier_x, barrier_y, barrier_ztop, barrier_zbottom
    )

    geometry = gpd.GeoDataFrame(
        geometry=polygons,
        data={
            "resistance": [1e3, 1e3],
        },
    )

    hfb = HorizontalFlowBarrierResistance(geometry, print_input)

    # Act
    if isinstance(idomain, xu.UgridDataArray):
        idomain_clipped = idomain.ugrid.sel(x=slice(None, 54.0))
    else:
        idomain_clipped = idomain.sel(x=slice(None, 54.0))

    regrid_context = RegridderWeightsCache()

    hfb_clipped = hfb.regrid_like(idomain_clipped.sel(layer=1), regrid_context)

    # Assert
    x, y = hfb_clipped.dataset["geometry"].values[1].xy  # 2nd polygon is clipped
    np.testing.assert_allclose(x, [50.0, 40.0])
    np.testing.assert_allclose(y, [5.5, 5.5])


@pytest.mark.parametrize("dis", ["basic_unstructured_dis", "basic_dis"])
@pytest.mark.parametrize(
    "barrier_class, barrier_value_name, barrier_value, expected_hydraulic_characteristic",
    [
        (SingleLayerHorizontalFlowBarrierResistance, "resistance", 1e3, 1e-3),
        (SingleLayerHorizontalFlowBarrierMultiplier, "multiplier", 1.5, -1.5),
        (
            SingleLayerHorizontalFlowBarrierHydraulicCharacteristic,
            "hydraulic_characteristic",
            1e-3,
            1e-3,
        ),
    ],
)
@patch("imod.mf6.mf6_hfb_adapter.Mf6HorizontalFlowBarrier.__new__", autospec=True)
def test_to_mf6_creates_mf6_adapter_layered(
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
        geometry=[
            linestrings(barrier_x, barrier_y),
        ],
        data={
            barrier_value_name: [barrier_value],
            "layer": [1],  # , 2
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

    data = np.full((idomain.coords["layer"].size, edge_index.size), fill_value=np.nan)
    data[0, :] = barrier_value
    expected_barrier_values = xr.DataArray(
        data=data,
        dims=("layer", "mesh2d_nFaces"),
        coords={"layer": idomain.coords["layer"]},
    )

    expected_values = to_connected_cells_dataset(
        idomain, grid, edge_index, {barrier_value_name: expected_barrier_values}
    )
    expected_values = _prepare_barrier_dataset_for_mf6_adapter(expected_values)

    mf6_flow_barrier_mock.assert_called_once()

    _, args = mf6_flow_barrier_mock.call_args
    assert args["cell_id1"].equals(expected_values["cell_id1"])
    assert args["cell_id2"].equals(expected_values["cell_id2"])
    assert args["layer"].equals(expected_values["layer"])
    assert (
        args["hydraulic_characteristic"].values.min()
        == expected_hydraulic_characteristic
    )


@pytest.mark.parametrize(
    "ztop, zbottom, expected_values",
    [
        (-5.0, -35.0, np.array([1, 1e3, 1])),  # 2nd layer
        (0.0, -35.0, np.array([1e3, 1e3, 1])),  # 1st and 2nd layer
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
def test_to_mf6_different_constant_z_boundaries(
    mf6_flow_barrier_mock, basic_dis, ztop, zbottom, expected_values
):
    # Arrange.
    idomain, top, bottom = basic_dis
    k = ones_like(top)

    print_input = False

    barrier_ztop = [ztop, ztop]
    barrier_zbottom = [zbottom, zbottom]
    barrier_y = [5.5, 5.5, 5.5]
    barrier_x = [82.0, 40.0, 0.0]

    polygons = linestring_to_square_zpolygons(
        barrier_x, barrier_y, barrier_ztop, barrier_zbottom
    )

    geometry = gpd.GeoDataFrame(
        geometry=polygons,
        data={
            "resistance": [1e3, 1e3],
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
    "barrier_ztop, barrier_zbottom, expected_values",
    [
        (
            [0.0, -5.0],
            [
                -35.0,
                -35.0,
            ],
            np.array([[1e3, 1e3, 1], [1, 3e3, 1]]),
        ),  # 1st and 2nd layer, 2nd layer
        (
            [100.0, 0.0],
            [-135.0, -35.0],
            np.array([[1e3, 1e3, 1e3], [3e3, 3e3, 1]]),
        ),  # ztop out of bounds, 1st and 2nd layer,
        (
            [0.0, 100.0],
            [-200.0, 50.0],
            np.array([[1e3, 1e3, 1e3], [1, 1, 1]]),
        ),  # zbottom out of bounds, z-range has no overlap with the domain
    ],
)
@patch("imod.mf6.mf6_hfb_adapter.Mf6HorizontalFlowBarrier.__new__", autospec=True)
def test_to_mf6_different_varying_square_z_boundaries(
    mf6_flow_barrier_mock, basic_dis, barrier_ztop, barrier_zbottom, expected_values
):
    """
    Test with square zpolygons with varying bounds. The second barrier is a
    barrier that is so short it should be ignored.
    """
    # Arrange.
    idomain, top, bottom = basic_dis
    k = ones_like(top)

    print_input = False

    # Insert second barrier values, which need to be ignored
    barrier_ztop.insert(1, min(barrier_ztop))
    barrier_zbottom.insert(1, max(barrier_zbottom))
    barrier_y = [5.5, 5.5, 5.5, 5.5]
    barrier_x = [0.0, 40.0, 41.0, 82.0]

    polygons = linestring_to_square_zpolygons(
        barrier_x, barrier_y, barrier_ztop, barrier_zbottom
    )

    geometry = gpd.GeoDataFrame(
        geometry=polygons,
        data={
            "resistance": [1e3, 2e3, 3e3],
        },
    )

    hfb = HorizontalFlowBarrierResistance(geometry, print_input)

    # Act.
    _ = hfb.to_mf6_pkg(idomain, top, bottom, k)

    # Assert.
    _, args = mf6_flow_barrier_mock.call_args
    barrier_values = args["hydraulic_characteristic"].values.reshape(3, 8)
    assert_array_equal(barrier_values[:, 3:5], 1.0 / expected_values.T)



@pytest.mark.parametrize(
    "barrier_ztop, barrier_zbottom, expected_values",
    [
        (
            [2.5, -2.5, -7.5],
            [
                -35.0,
                -35.0,
                -35.0,
            ],
            np.array([[1e3, 1e3, 1], [1, 3e3, 1]]),
        ),  # 1st and 2nd layer, 2nd layer
        (
            [200.0, 0.0, 0.0],
            [-270.0, -35.0, -35.0],
            np.array([[1e3, 1e3, 1e3], [3e3, 3e3, 1]]),
        ),  # ztop out of bounds, 1st and 2nd layer,
        (
            [0.0, 200.0, 200.0],
            [-400.0, 50.0, 50.0],
            np.array([[1e3, 1e3, 1e3], [1, 1, 1]]),
        ),  # zbottom out of bounds, z-range has no overlap with the domain
    ],
)
@patch("imod.mf6.mf6_hfb_adapter.Mf6HorizontalFlowBarrier.__new__", autospec=True)
def test_to_mf6_different_trapezoid_z_boundaries(
    mf6_flow_barrier_mock, basic_dis, barrier_ztop, barrier_zbottom, expected_values
):
    # Arrange.
    idomain, top, bottom = basic_dis
    k = ones_like(top)

    print_input = False

    barrier_y = [5.5, 5.5, 5.5]
    barrier_x = [0.0, 40.0, 82.0]

    polygons = linestring_to_trapezoid_zpolygons(
        barrier_x, barrier_y, barrier_ztop, barrier_zbottom
    )

    geometry = gpd.GeoDataFrame(
        geometry=polygons,
        data={
            "resistance": [1e3, 3e3],
        },
    )

    hfb = HorizontalFlowBarrierResistance(geometry, print_input)

    # Act.
    _ = hfb.to_mf6_pkg(idomain, top, bottom, k)

    # Assert.
    _, args = mf6_flow_barrier_mock.call_args
    barrier_values = args["hydraulic_characteristic"].values.reshape(3, 8)
    assert_array_equal(barrier_values[:, 3:5], 1.0 / expected_values.T)




@pytest.mark.parametrize(
    "layer, expected_values",
    [
        (2, np.array([1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3])),  # 2nd layer
    ],
)
@patch("imod.mf6.mf6_hfb_adapter.Mf6HorizontalFlowBarrier.__new__", autospec=True)
def test_to_mf6_layered_hfb(mf6_flow_barrier_mock, basic_dis, layer, expected_values):
    # Arrange.
    idomain, top, bottom = basic_dis
    k = ones_like(top)

    print_input = False

    barrier_y = [5.5, 5.5, 5.5]
    barrier_x = [82.0, 40.0, 0.0]

    geometry = gpd.GeoDataFrame(
        geometry=[linestrings(barrier_x, barrier_y)],
        data={
            "resistance": [1e3],
            "layer": [layer],
        },
    )

    hfb = SingleLayerHorizontalFlowBarrierResistance(geometry, print_input)

    # Act.
    _ = hfb.to_mf6_pkg(idomain, top, bottom, k)

    # Assert.
    _, args = mf6_flow_barrier_mock.call_args
    barrier_values = args["hydraulic_characteristic"].values
    assert_array_equal(barrier_values, 1.0 / expected_values)
    expected_layer = np.full((8,), layer)
    barrier_layer = args["layer"].values
    assert_array_equal(barrier_layer, expected_layer)


def test_to_mf6_layered_hfb__error():
    """Throws error because multiple layers attached to one object."""
    # Arrange.
    print_input = False

    barrier_y = [5.5, 5.5, 5.5]
    barrier_x = [82.0, 40.0, 0.0]

    linestring = linestrings(barrier_x, barrier_y)

    geometry = gpd.GeoDataFrame(
        geometry=[linestring, linestring],
        data={
            "resistance": [1e3, 1e3],
            "layer": [1, 2],
        },
    )

    hfb = SingleLayerHorizontalFlowBarrierResistance(geometry, print_input)
    errors = hfb._validate(hfb._write_schemata)

    assert len(errors) > 0


@pytest.mark.parametrize(
    "barrier_x_loc, expected_number_barriers",
    [
        # barrier lies between active cells
        (1.0, 1),
        # barrier lies between active cells in the top layer but between an inactive and active cell in the bottom layer
        (2.0, 0),
    ],
)
@pytest.mark.parametrize(
    "parameterizable_basic_dis",
    [BasicDisSettings(nlay=1, nrow=1, ncol=3, xstart=0, xstop=3)],
    indirect=True,
)
@pytest.mark.parametrize("inactivity_marker", [0, -1])
@patch("imod.mf6.mf6_hfb_adapter.Mf6HorizontalFlowBarrier.__new__", autospec=True)
def test_to_mf6_remove_invalid_edges(
    mf6_flow_barrier_mock,
    parameterizable_basic_dis,
    inactivity_marker,
    barrier_x_loc,
    expected_number_barriers,
):
    # Arrange.
    idomain, top, bottom = parameterizable_basic_dis
    idomain.loc[{"x": idomain.coords["x"][-1]}] = (
        inactivity_marker  # make cells inactive
    )
    k = ones_like(top)

    barrier_ztop = [0.0]
    barrier_zbottom = [-5.0]
    barrier_y = [0.0, 2.0]
    barrier_x = [barrier_x_loc, barrier_x_loc]

    polygons = linestring_to_square_zpolygons(
        barrier_x, barrier_y, barrier_ztop, barrier_zbottom
    )

    geometry = gpd.GeoDataFrame(
        geometry=polygons,
        data={
            "resistance": [1e3],
        },
    )

    hfb = HorizontalFlowBarrierResistance(geometry)

    # Act.
    if expected_number_barriers == 0:
        pytest.xfail("Test expected to fail if expected number barriers = 0")

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
@pytest.mark.parametrize("inactivity_marker", [0, -1])
@pytest.mark.parametrize(
    "parameterizable_basic_dis",
    [BasicDisSettings(nlay=2, nrow=1, ncol=3, xstart=0, xstop=3)],
    indirect=True,
)
@patch("imod.mf6.mf6_hfb_adapter.Mf6HorizontalFlowBarrier.__new__", autospec=True)
def test_to_mf6_remove_barrier_parts_adjacent_to_inactive_cells(
    mf6_flow_barrier_mock,
    parameterizable_basic_dis,
    inactivity_marker,
    barrier_x_loc,
    expected_number_barriers,
):
    # Arrange.
    idomain, top, bottom = parameterizable_basic_dis
    idomain.loc[
        {"x": idomain.coords["x"][-1], "layer": idomain.coords["layer"][-1]}
    ] = inactivity_marker  # make cell inactive
    k = ones_like(top)

    barrier_ztop = [
        0.0,
    ]
    barrier_zbottom = [
        -5.0,
    ]
    barrier_y = [0.0, 2.0]
    barrier_x = [barrier_x_loc, barrier_x_loc]

    polygons = linestring_to_square_zpolygons(
        barrier_x, barrier_y, barrier_ztop, barrier_zbottom
    )

    geometry = gpd.GeoDataFrame(
        geometry=polygons,
        data={
            "resistance": [1e3],
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


def test_is_empty():
    geometry = gpd.GeoDataFrame(
        geometry=[Polygon()],
        data={
            "resistance": [],
        },
    )
    hfb = HorizontalFlowBarrierResistance(geometry)
    assert hfb.is_empty()

    barrier_ztop = [0.0, 0.0]
    barrier_zbottom = [-5.0, -5.0]
    barrier_y = [0.0, 2.0, 3.0]
    barrier_x = [0.0, 0.0, 0.0]

    polygons = linestring_to_square_zpolygons(
        barrier_x, barrier_y, barrier_ztop, barrier_zbottom
    )

    geometry = gpd.GeoDataFrame(
        geometry=polygons,
        data={
            "resistance": [1.0, 1.0],
        },
    )

    hfb = HorizontalFlowBarrierResistance(geometry)
    assert not hfb.is_empty()


@pytest.mark.parametrize(
    "parameterizable_basic_dis",
    [BasicDisSettings(nlay=2, nrow=3, ncol=3, xstart=0, xstop=3)],
    indirect=True,
)
@pytest.mark.parametrize("print_input", [True, False])
def test_set_options(print_input, parameterizable_basic_dis):
    idomain, top, bottom = parameterizable_basic_dis

    barrier_x = [-1000.0, 1000.0]
    barrier_y = [0.3, 0.3]
    barrier_ztop = [top.values[0]]
    barrier_zbottom = [bottom.values[-1]]

    polygons = linestring_to_square_zpolygons(
        barrier_x, barrier_y, barrier_ztop, barrier_zbottom
    )

    hfb = HorizontalFlowBarrierResistance(
        geometry=gpd.GeoDataFrame(
            geometry=polygons,
            data={
                "resistance": [1e3],
            },
        ),
        print_input=print_input,
    )
    k = ones_like(top)
    mf6_package = hfb.to_mf6_pkg(idomain, top, bottom, k)
    assert mf6_package.dataset["print_input"].values[()] == print_input


@pytest.mark.usefixtures("imod5_dataset")
def test_hfb_from_imod5(imod5_dataset, tmp_path):
    target_dis = StructuredDiscretization.from_imod5_data(imod5_dataset)
    target_npf = NodePropertyFlow.from_imod5_data(
        imod5_dataset, target_dis.dataset["idomain"]
    )

    hfb = SingleLayerHorizontalFlowBarrierResistance.from_imod5_dataset(
        "hfb-3", imod5_dataset
    )
    hfb_package = hfb.to_mf6_pkg(
        target_dis["idomain"], target_dis["top"], target_dis["bottom"], target_npf["k"]
    )
    assert list(np.unique(hfb_package["layer"].values)) == [7]


@pytest.mark.usefixtures("structured_flow_model")
def test_run_multiple_hfbs(tmp_path, structured_flow_model):
    # Single layered model
    structured_flow_model = structured_flow_model.clip_box(layer_max=1)
    structured_flow_model["dis"]["bottom"] = structured_flow_model["dis"][
        "bottom"
    ].isel(x=0, y=0, drop=True)
    # Arrange boundary conditions into something simple:
    # A linear decline from left to right, forced by chd
    structured_flow_model.pop("rch")
    chd_head = nan_like(structured_flow_model["chd"].dataset["head"])
    chd_head[:, :, 0] = 10.0
    chd_head[:, :, -1] = 0.0
    structured_flow_model["chd"].dataset["head"] = chd_head

    barrier_y = [11.0, 5.0, -1.0]
    barrier_x = [5.0, 5.0, 5.0]

    geometry = gpd.GeoDataFrame(
        geometry=[linestrings(barrier_x, barrier_y)],
        data={
            "resistance": [1200.0],
            "layer": [1],
        },
    )

    simulation_single = Modflow6Simulation("single_hfb")
    structured_flow_model["hfb"] = SingleLayerHorizontalFlowBarrierResistance(geometry)
    simulation_single["GWF"] = structured_flow_model
    simulation_single["solver"] = SolutionPresetSimple(["GWF"])
    simulation_single.create_time_discretization(["2000-01-01", "2000-01-02"])
    simulation_single.write(tmp_path / "single")
    simulation_single.run()
    head_single = simulation_single.open_head()

    geometry = gpd.GeoDataFrame(
        geometry=[linestrings(barrier_x, barrier_y)],
        data={
            "resistance": [400.0],
            "layer": [1],
        },
    )

    simulation_triple = Modflow6Simulation("triple_hfb")
    structured_flow_model.pop("hfb")  # Remove high resistance HFB package now.
    structured_flow_model["hfb-1"] = SingleLayerHorizontalFlowBarrierResistance(
        geometry
    )
    structured_flow_model["hfb-2"] = SingleLayerHorizontalFlowBarrierResistance(
        geometry
    )
    structured_flow_model["hfb-3"] = SingleLayerHorizontalFlowBarrierResistance(
        geometry
    )
    simulation_triple["GWF"] = structured_flow_model
    simulation_triple["solver"] = SolutionPresetSimple(["GWF"])
    simulation_triple.create_time_discretization(["2000-01-01", "2000-01-02"])
    simulation_triple.write(tmp_path / "triple")
    simulation_triple.run()
    head_triple = simulation_triple.open_head()

    xr.testing.assert_equal(head_single, head_triple)


def test_make_linestring_from_polygon():
    barrier_x = [-1000.0, 1000.0]
    barrier_y = [0.3, 0.3]
    barrier_ztop = [10.0]
    barrier_zbottom = [-10.0]

    polygons = linestring_to_square_zpolygons(
        barrier_x, barrier_y, barrier_ztop, barrier_zbottom
    )

    gdf_polygons = gpd.GeoDataFrame(
        geometry=polygons,
        data={
            "resistance": [1e3],
        },
    )

    linestrings = _make_linestring_from_polygon(gdf_polygons)

    coordinates = get_coordinates(linestrings)

    np.testing.assert_allclose(barrier_x, coordinates[:, 0])
    np.testing.assert_allclose(barrier_y, coordinates[:, 1])


def test_extract_hfb_bounds_from_dataframe():
    barrier_x = [-1000.0, 0.0, 1000.0]
    barrier_y = [0.3, 0.3, 0.3]
    barrier_ztop = [10.0, 10.0]
    barrier_zbottom = [-10.0, -10.0]

    polygons = linestring_to_square_zpolygons(
        barrier_x, barrier_y, barrier_ztop, barrier_zbottom
    )

    gdf_polygons = gpd.GeoDataFrame(
        geometry=polygons,
        data={
            "resistance": [1e3, 1e3],
        },
    )

    zmin, zmax = _extract_hfb_bounds_from_dataframe(gdf_polygons)

    np.testing.assert_equal(zmin, barrier_zbottom)
    np.testing.assert_equal(zmax, barrier_ztop)


def test_extract_hfb_bounds_from_dataframe__fails():
    """Test if function throws error when providing a line."""
    barrier_x = [-1000.0, 0.0, 1000.0]
    barrier_y = [0.3, 0.3, 0.3]

    line_data = linestrings(barrier_x, barrier_y)

    gdf_polygons = gpd.GeoDataFrame(
        geometry=[line_data],
        data={
            "resistance": [1e3],
        },
    )

    with pytest.raises(TypeError):
        _extract_hfb_bounds_from_dataframe(gdf_polygons)
