import numpy as np
import pytest
import xarray as xr

import imod


@pytest.fixture(scope="module")
def test_da(request):
    nrow, ncol = 3, 4
    dx, dy = 1.0, -1.0
    xmin, xmax = 0.0, 4.0
    ymin, ymax = 0.0, 3.0
    coords = imod.util._xycoords((xmin, xmax, ymin, ymax), (dx, dy))
    kwargs = {"coords": coords, "dims": ("y", "x")}
    data = np.arange(nrow * ncol).reshape((nrow, ncol))
    return xr.DataArray(data, **kwargs)


@pytest.fixture(scope="module")
def test_da_nonequidistant(request):
    nrow, ncol = 3, 4
    dx = np.array([0.9, 1.1, 0.8, 1.2])
    dy = np.array([-1.5, -0.5, -1.0])
    xmin, xmax = 0.0, 4.0
    ymin, ymax = 0.0, 3.0
    coords = imod.util._xycoords((xmin, xmax, ymin, ymax), (dx, dy))
    kwargs = {"coords": coords, "dims": ("y", "x")}
    data = np.arange(nrow * ncol).reshape((nrow, ncol))
    return xr.DataArray(data, **kwargs)


def test_in_bounds(test_da_nonequidistant):
    x = 2.0
    y = 2.0
    expected = np.array([True])
    actual = imod.select.points.in_bounds(test_da_nonequidistant, x, y)
    assert (expected == actual).all()

    x = -2.0
    y = 2.0
    expected = np.array([False])
    actual = imod.select.points.in_bounds(test_da_nonequidistant, x, y)
    assert (expected == actual).all()

    # Lower inclusive
    x = 0.0
    y = 0.0
    expected = np.array([True])
    actual = imod.select.points.in_bounds(test_da_nonequidistant, x, y)
    assert (expected == actual).all()

    # Upper exclusive
    x = 4.0
    y = 3.0
    expected = np.array([False])
    actual = imod.select.points.in_bounds(test_da_nonequidistant, x, y)
    assert (expected == actual).all()


def test_get_xy_indices__nonequidistant(test_da_nonequidistant):
    x = 3.0
    y = 2.5
    expected = (np.array([0]), np.array([3]))
    actual = imod.select.points.get_xy_indices(test_da_nonequidistant, x, y)
    assert expected == actual

    # Lower inclusive
    x = 2.8
    y = 2.5
    expected = (np.array([0]), np.array([3]))
    actual = imod.select.points.get_xy_indices(test_da_nonequidistant, x, y)
    assert expected == actual

    # Lower inclusive
    x = 0.0
    y = 0.0
    expected = (np.array([2]), np.array([0]))
    actual = imod.select.points.get_xy_indices(test_da_nonequidistant, x, y)
    assert expected == actual

    # Upper exclusive
    x = 4.0
    y = 2.5
    with pytest.raises(ValueError):
        actual = imod.select.points.get_xy_indices(test_da_nonequidistant, x, y)

    # Arrays
    x = [3.0, 0.0]
    y = [2.5, 0.0]
    rr_e, cc_e = (np.array([0, 2]), np.array([3, 0]))
    rr_a, cc_a = imod.select.points.get_xy_indices(test_da_nonequidistant, x, y)
    assert (rr_e == rr_a).all()
    assert (cc_e == cc_a).all()

    # Arrays; upper exclusive
    x = [4.0, 0.0]
    y = [2.5, 0.0]
    with pytest.raises(ValueError):
        rr_a, cc_a = imod.select.points.get_xy_indices(test_da_nonequidistant, x, y)


def test_get_xy_indices__equidistant(test_da):
    x = 3.0
    y = 2.5
    expected = (np.array([0]), np.array([3]))
    actual = imod.select.points.get_xy_indices(test_da, x, y)
    assert expected == actual


def test_set_xy_values(test_da_nonequidistant):
    out = xr.full_like(test_da_nonequidistant, 0.0)
    x = 0.0
    y = 0.0
    imod.select.points.set_xy_values(out, x, y, 1.0)
    expected = np.array(
        [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
    )
    assert (out.values == expected).all()

    # reset values
    out[:] = 0.0
    # paint diagonal
    x = [0.45, 1.45, 2.4]
    y = [2.25, 1.25, 0.5]
    imod.select.points.set_xy_values(out, x, y, 1.0)
    expected = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    )
    assert (out.values == expected).all()

    values = [1.0, 2.0, 3.0]
    imod.select.points.set_xy_values(out, x, y, values)
    expected = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 3.0, 0.0]]
    )
    assert (out.values == expected).all()


def test_set_xy_values__xyerror():
    da = xr.DataArray(
        np.random.rand(3, 2, 1),
        {"x": [1, 2, 3], "y": [2, 1], "z": [1]},
        dims=("x", "y", "z"),
    )
    with pytest.raises(ValueError):
        imod.select.points.set_xy_values(da, 0, 0, 1.0)


def test_get_xy_values(test_da_nonequidistant):
    x = [0.45, 1.45, 2.4]
    y = [2.25, 1.25, 0.5]
    actual = imod.select.points.get_xy_values(test_da_nonequidistant, x, y)
    actual = actual.drop("dx")
    actual = actual.drop("dy")
    data = [0, 5, 10]
    expected = xr.DataArray(
        data,
        coords={"index": [0, 1, 2], "x": ("index", x), "y": ("index", y)},
        dims=["index"],
    )
    assert actual.identical(expected)
