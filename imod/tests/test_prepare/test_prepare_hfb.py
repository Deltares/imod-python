import numpy as np
import pytest
import shapely

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
            [-10.0, 10.0, -20.0],
            [0.0, 0.0, 20.0],
            [0.0, 0.0, -10.0],
            [-10.0, 10.0, 10.0],
        ]
    )

    coordinates_1 = shapely.get_coordinates(polygons[1], include_z=True)
    expected_1 = np.array(
        [
            [0.0, 0.0, 20.0],
            [0.0, 0.0, 0.0],
            [10.0, -10.0, 15.0],
            [10.0, -10.0, -20.0],
            [0.0, 0.0, 20.0],
        ]
    )

    np.testing.assert_equal(coordinates_0, expected_0)
    np.testing.assert_equal(coordinates_1, expected_1)


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
