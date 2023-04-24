import numpy as np
import pandas as pd
import pytest
import xarray as xr
import xugrid as xu

import imod


@pytest.fixture(scope="module")
def test_da():
    nrow, ncol = 3, 4
    dx, dy = 1.0, -1.0
    xmin, xmax = 0.0, 4.0
    ymin, ymax = 0.0, 3.0
    coords = imod.util._xycoords((xmin, xmax, ymin, ymax), (dx, dy))
    kwargs = {"coords": coords, "dims": ("y", "x")}
    data = np.arange(nrow * ncol).reshape((nrow, ncol))
    return xr.DataArray(data, **kwargs)


@pytest.fixture(scope="module")
def test_da_nonequidistant():
    nrow, ncol = 3, 4
    dx = np.array([0.9, 1.1, 0.8, 1.2])
    dy = np.array([-1.5, -0.5, -1.0])
    xmin, xmax = 0.0, 4.0
    ymin, ymax = 0.0, 3.0
    coords = imod.util._xycoords((xmin, xmax, ymin, ymax), (dx, dy))
    kwargs = {"coords": coords, "dims": ("y", "x")}
    data = np.arange(nrow * ncol).reshape((nrow, ncol))
    return xr.DataArray(data, **kwargs)


@pytest.fixture(scope="module")
def test_uda(test_da):
    grid = xu.Ugrid2d.from_structured(test_da)

    da = xr.DataArray(
        data=[1.0, 2.0] * 6,
        dims=[grid.face_dimension],
    )
    return xu.UgridDataArray(da, grid)


@pytest.fixture(scope="module")
def test_uda_elevation():
    elevation = xu.data.elevation_nl()
    return xu.concat(
        [elevation, elevation - 10.0, elevation - 20.0], dim="z"
    ).assign_coords(z=[0.0, -10.0, -20.0])


def xy_indices(indices):
    """A little helper function to return tuples of np.arrays"""
    return indices["y"].values, indices["x"].values


# TODO: Use Pytest cases for this
def test_in_bounds(test_da_nonequidistant, test_uda):
    for case in [test_da_nonequidistant, test_uda]:
        x = 2.0
        y = 2.0
        expected = np.array([True])
        actual = imod.select.points_in_bounds(case, x=x, y=y)
        assert (expected == actual).all()

        x = -2.0
        y = 2.0
        expected = np.array([False])
        actual = imod.select.points_in_bounds(case, x=x, y=y)
        assert (expected == actual).all()

        # Lower inclusive
        x = 0.0
        y = 0.0
        expected = np.array([True])
        actual = imod.select.points_in_bounds(case, x=x, y=y)
        assert (expected == actual).all()

        # Upper exclusive
        x = 4.0
        y = 3.0
        expected = np.array([False])
        actual = imod.select.points_in_bounds(case, x=x, y=y)
        assert (expected == actual).all()


def test_get_indices__nonequidistant(test_da_nonequidistant):
    x = 3.0
    y = 2.5
    expected = (np.array([0]), np.array([3]))
    actual = imod.select.points_indices(test_da_nonequidistant, x=x, y=y)
    assert expected == xy_indices(actual)

    # Lower inclusive
    x = 2.8
    y = 2.5
    expected = (np.array([0]), np.array([3]))
    actual = imod.select.points_indices(test_da_nonequidistant, x=x, y=y)
    assert expected == xy_indices(actual)

    # Lower inclusive
    x = 0.0
    y = 0.0
    expected = (np.array([2]), np.array([0]))
    actual = imod.select.points_indices(test_da_nonequidistant, x=x, y=y)
    assert expected == xy_indices(actual)

    # Upper exclusive
    x = 4.0
    y = 2.5
    with pytest.raises(ValueError):
        actual = imod.select.points_indices(test_da_nonequidistant, x=x, y=y)

    # Arrays
    x = [3.0, 0.0]
    y = [2.5, 0.0]
    rr_e, cc_e = (np.array([0, 2]), np.array([3, 0]))
    rr_a, cc_a = xy_indices(
        imod.select.points_indices(test_da_nonequidistant, x=x, y=y)
    )
    assert (rr_e == rr_a).all()
    assert (cc_e == cc_a).all()

    # Arrays; upper exclusive
    x = [4.0, 0.0]
    y = [2.5, 0.0]
    with pytest.raises(ValueError):
        imod.select.points_indices(test_da_nonequidistant, x=x, y=y)


def test_get_indices__equidistant(test_da):
    x = 3.0
    y = 2.5
    expected = (np.array([0]), np.array([3]))
    actual = imod.select.points_indices(test_da, x=x, y=y)
    assert expected == xy_indices(actual)


def test_get_indices__unstructured(test_uda):
    x = 3.0
    y = 2.5
    expected = np.array([3])
    indices = imod.select.points_indices(test_uda, x=x, y=y)
    actual = indices["mesh2d_nFaces"].values
    assert expected == actual

    # Upper exclusive
    x = 4.0
    y = 2.5
    with pytest.raises(ValueError):
        actual = imod.select.points_indices(test_uda, x=x, y=y)


def test_get_values__unstructured(test_uda):
    x = 3.0
    y = 2.5
    expected = np.array([3])
    indices = imod.select.points_values(test_uda, x=x, y=y)
    actual = indices["mesh2d_nFaces"].values
    assert expected == actual


def test_get_indices__uda_elevation(test_uda_elevation):
    """
    Test for layered unstructured grid, x and y need to be looked up on
    Ugrid2d, whereas z should be looked in the regular coordinates.

    Point 3 is outside the grid and should be removed.
    """
    x = np.array([30_000.0, 31_000.0, 30_000.0, 1.0, 31_000.0])
    y = np.array([400_000.0, 390_000.0, 400_000.0, 1.0, 390_000.0])
    z = np.array([-1.0, -2.0, -15.0, 1.0, -23.0])

    points_indices = imod.select.points_indices(
        test_uda_elevation, x=x, y=y, z=z, out_of_bounds="ignore"
    )

    data_expected = {}
    data_expected["mesh2d_nFaces"] = np.array([2033, 2063, 2033, 2063])
    data_expected["z"] = np.array([0, 0, 1, 2])

    assert list(points_indices.keys()) == ["mesh2d_nFaces", "z"]
    for key in points_indices.keys():
        index = points_indices[key].coords["index"]
        np.testing.assert_equal(index, np.array([0, 1, 2, 4]))
        np.testing.assert_equal(points_indices[key], data_expected[key])


def test_get_values__uda_elevation(test_uda_elevation):
    x = np.array([30_000.0, 31_000.0, 30_000.0, 1.0, 31_000.0])
    y = np.array([400_000.0, 390_000.0, 400_000.0, 1.0, 390_000.0])
    z = np.array([-1.0, -2.0, -15.0, 1.0, -23.0])

    points_values = imod.select.points_values(
        test_uda_elevation, x=x, y=y, z=z, out_of_bounds="ignore"
    )
    data_expected = np.array([3.04, -0.74, -6.96, -20.74])

    np.testing.assert_allclose(points_values, data_expected)


def test_set_values(test_da_nonequidistant):
    da = xr.full_like(test_da_nonequidistant, 0.0)
    x = 0.0
    y = 0.0
    out = imod.select.points_set_values(da, 1.0, x=x, y=y)
    expected = np.array(
        [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
    )
    assert (out.values == expected).all()

    # paint diagonal
    x = [0.45, 1.45, 2.4]
    y = [2.25, 1.25, 0.5]
    out = imod.select.points_set_values(da, 1.0, x=x, y=y)
    expected = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    )
    assert (out.values == expected).all()

    values = [1.0, 2.0, 3.0]
    out = imod.select.points_set_values(da, values, x=x, y=y)
    expected = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 3.0, 0.0]]
    )
    assert (out.values == expected).all()


def test_set_values__error():
    da = xr.DataArray(
        np.random.rand(3, 2, 1),
        {"x": [1, 2, 3], "y": [2, 1], "z": [1]},
        dims=("x", "y", "z"),
    )
    with pytest.raises(ValueError):
        imod.select.points_set_values(da, 1.0, x=0, y=0)


def test_get_values(test_da_nonequidistant):
    x = [0.45, 1.45, 2.4]
    y = [2.25, 1.25, 0.5]
    actual = imod.select.points_values(test_da_nonequidistant, x=x, y=y)
    actual = actual.drop_vars(["dx", "dy"])
    data = [0, 5, 10]
    expected = xr.DataArray(
        data,
        coords={"index": [0, 1, 2], "x": ("index", x), "y": ("index", y)},
        dims=["index"],
    )
    assert actual.identical(expected)


def test_get_values__index(test_da_nonequidistant):
    x = pd.Series(data=[0.45, 1.45, 2.4], index=[11, 12, 13])
    y = [2.25, 1.25, 0.5]
    actual = imod.select.points_values(test_da_nonequidistant, x=x, y=y)
    actual = actual.drop_vars(["dx", "dy"])
    data = [0, 5, 10]
    expected = xr.DataArray(
        data,
        coords={"index": [0, 1, 2], "x": ("index", x), "y": ("index", y)},
        dims=["index"],
    )
    assert actual.identical(expected)


def test_get_values__scalar(test_da_nonequidistant):
    x = 0.45
    y = 2.25
    actual = imod.select.points_values(test_da_nonequidistant, x=x, y=y)
    actual = actual.drop_vars(["dx", "dy"])
    data = [0]
    expected = xr.DataArray(
        data,
        coords={"index": [0], "x": ("index", [x]), "y": ("index", [y])},
        dims=["index"],
    )
    assert actual.identical(expected)
