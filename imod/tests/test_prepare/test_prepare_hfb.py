import geopandas as gpd
import numpy as np
import pytest
import shapely

import imod
from imod.prepare.hfb import (
    linestring_to_square_zpolygons,
    linestring_to_trapezoid_zpolygons,
)


def test_linestring_to_square_zpolygons():
    barrier_x = [-10.0, 0.0, 10.0]
    barrier_y = [10.0, 0.0, -10.0]
    barrier_ztop = [10.0, 20.0]
    barrier_zbot = [-10.0, -20.0]

    polygons = linestring_to_square_zpolygons(
        barrier_x, barrier_y, barrier_ztop, barrier_zbot
    )

    assert len(polygons) == 2

    coordinates_0 = shapely.get_coordinates(polygons[0], include_z=True)
    expected_0 = np.array(
        [
            [-10.0, 10.0, 10.0],
            [-10.0, 10.0, -10.0],
            [0.0, 0.0, -10.0],
            [0.0, 0.0, 10.0],
            [-10.0, 10.0, 10.0],
        ]
    )

    coordinates_1 = shapely.get_coordinates(polygons[1], include_z=True)
    expected_1 = np.array(
        [
            [0.0, 0.0, 20.0],
            [0.0, 0.0, -20.0],
            [10.0, -10.0, -20.0],
            [10.0, -10.0, 20.0],
            [0.0, 0.0, 20.0],
        ]
    )

    np.testing.assert_equal(coordinates_0, expected_0)
    np.testing.assert_equal(coordinates_1, expected_1)


def test_linestring_to_trapezoid_zpolygons():
    barrier_x = [-10.0, 0.0, 10.0]
    barrier_y = [10.0, 0.0, -10.0]
    barrier_ztop = [10.0, 20.0, 15.0]
    barrier_zbot = [-10.0, -20.0, 0.0]

    polygons = linestring_to_trapezoid_zpolygons(
        barrier_x, barrier_y, barrier_ztop, barrier_zbot
    )

    assert len(polygons) == 2

    coordinates_0 = shapely.get_coordinates(polygons[0], include_z=True)
    expected_0 = np.array(
        [
            [-10.0, 10.0, 10.0],
            [-10.0, 10.0, -10.0],
            [0.0, 0.0, -20.0],
            [0.0, 0.0, 20.0],
            [-10.0, 10.0, 10.0],
        ]
    )

    coordinates_1 = shapely.get_coordinates(polygons[1], include_z=True)
    expected_1 = np.array(
        [
            [0.0, 0.0, 20.0],
            [0.0, 0.0, -20.0],
            [10.0, -10.0, 0.0],
            [10.0, -10.0, 15.0],
            [0.0, 0.0, 20.0],
        ]
    )

    np.testing.assert_equal(coordinates_0, expected_0)
    np.testing.assert_equal(coordinates_1, expected_1)


def test_linestring_to_trapezoid_zpolygons__clip():
    polygons = linestring_to_trapezoid_zpolygons(
        [0.0, 10.0], [0.5, 0.5], [10.0, 20.0], [-10.0, -20.0]
    )
    geometry = gpd.GeoDataFrame(geometry=polygons, data={"resistance": [1e3]})
    hfb = imod.mf6.HorizontalFlowBarrierResistance(geometry)

    clipped = hfb.clip_box(x_max=2.0)

    coordinates = shapely.get_coordinates(
        clipped.dataset["geometry"].values[0], include_z=True
    )
    x = coordinates[:, 0]
    z = coordinates[:, 2]
    np.testing.assert_allclose(z[x == 0.0].max(), 10.0)
    np.testing.assert_allclose(z[x == 0.0].min(), -10.0)
    np.testing.assert_allclose(z[np.isclose(x, 2.0)].max(), 12.0)
    np.testing.assert_allclose(z[np.isclose(x, 2.0)].min(), -12.0)


def test_linestring_to_trapezoid_zpolygons__fails():
    barrier_x = [-10.0, 0.0, 10.0]
    barrier_y = [10.0, 0.0, -10.0]
    barrier_ztop = [10.0, 20.0]
    barrier_zbot = [-10.0, -20.0]

    with pytest.raises(ValueError):
        linestring_to_trapezoid_zpolygons(
            barrier_x, barrier_y, barrier_ztop, barrier_zbot
        )


def test_linestring_to_square_zpolygons__fails():
    barrier_x = [-10.0, 0.0, 10.0]
    barrier_y = [10.0, 0.0, -10.0]
    barrier_ztop = [10.0, 20.0, 15.0]
    barrier_zbot = [-10.0, -20.0, 0.0]

    with pytest.raises(ValueError):
        linestring_to_square_zpolygons(barrier_x, barrier_y, barrier_ztop, barrier_zbot)
