from unittest.mock import patch

import geopandas as gpd
import pytest
import shapely
import xugrid as xu
from numpy.testing import assert_array_equal

from imod.mf6 import (
    BarrierType,
    HorizontalFlowBarrierHydraulicCharacteristic,
    HorizontalFlowBarrierMultiplier,
    HorizontalFlowBarrierResistance,
)
from imod.typing.grid import ones_like


@pytest.mark.parametrize("dis", ["basic_unstructured_dis", "basic_dis"])
@pytest.mark.parametrize(
    "barrier_class, barrier_type, barrier_value_name, barrier_value",
    [
        (HorizontalFlowBarrierResistance, BarrierType.Resistance, "resistance", 1e3),
        (HorizontalFlowBarrierMultiplier, BarrierType.Multiplier, "multiplier", 1e-3),
        (
            HorizontalFlowBarrierHydraulicCharacteristic,
            BarrierType.HydraulicCharacteristic,
            "hydraulic_characteristic",
            1.0,
        ),
    ],
)
@patch("imod.mf6.mf6_hfb_adapter.Mf6HorizontalFlowBarrier.__new__", autospec=True)
def test_to_mf6_creates_mf6_adapter(
    mf6_flow_barrier_mock,
    dis,
    barrier_class,
    barrier_type,
    barrier_value_name,
    barrier_value,
    request,
):
    # Arrange.
    idomain, top, bottom = request.getfixturevalue(dis)
    k = ones_like(top)

    print_input = False

    barrier_y = [2.2, 2.2, 2.2]
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
    expected_barrier = snapped[barrier_value_name].broadcast_like(
        idomain.coords["layer"]
    )

    mf6_flow_barrier_mock.assert_called_once()

    args, _ = mf6_flow_barrier_mock.call_args
    assert args[1] is barrier_type
    assert args[2].equals(expected_barrier)
    assert args[3] is idomain
    assert args[4] is print_input


@pytest.mark.parametrize(
    "ztop, zbottom, expected_values",
    [
        (-5.0, -35.0, [1, 1e3, 1]),  # 2nd layer
        (0.0, -35.0, [1e3, 1e3, 1]),  # 1st and 2nd layer
        (-5.0, -135.0, [1, 1e3, 1e3]),  # 2nd and 3th layer
        (-5.0, -135.0, [1, 1e3, 1e3]),  # 2nd and 3th layer
        (100.0, -135.0, [1e3, 1e3, 1e3]),  # ztop out of bounds
        (0.0, -200.0, [1e3, 1e3, 1e3]),  # zbottom out of bounds
        (100.0, 50.0, [1, 1, 1]),  # z-range has no overlap with the domain
        (
            0.0,
            -2.5,
            [1 / ((0.5 / 1e3) + ((1.0 - 0.5) / 1)), 1, 1],
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

    barrier_y = [2.2, 2.2, 2.2]
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
    args, _ = mf6_flow_barrier_mock.call_args
    barrier_values = args[2]
    max_values_per_layer = barrier_values.max("mesh2d_nEdges").values
    assert_array_equal(max_values_per_layer, expected_values)
