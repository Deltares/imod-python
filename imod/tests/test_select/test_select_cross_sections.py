import numpy as np
import pytest
import xarray as xr

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


def test_draw_line__diagonal():
    # Grid definition
    xmin = 0.0
    xmax = 4.0
    ymin = 0.0
    ymax = 3.0
    xs = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    ys = np.array([0.0, 1.0, 2.0, 3.0])

    # Simple diagonal
    x0, y0 = 0.0, 0.0
    x1, y1 = 3.0, 3.0
    a_ixs, a_iys, a_s, a_dxs, a_dys = imod.select.cross_sections._draw_line(
        xs, ys, x0, x1, y0, y1, xmin, xmax, ymin, ymax
    )
    e_ixs = np.array([0, 1, 2])
    e_iys = np.array([0, 1, 2])
    assert (e_ixs == a_ixs).all()
    assert (e_iys == a_iys).all()
    assert np.allclose(a_s, np.sqrt(2.0))
    assert np.allclose(a_dxs, 1.0)
    assert np.allclose(a_dys, 1.0)

    # Simple diagonal, flip it around
    x0, y0 = 3.0, 3.0
    x1, y1 = 0.0, 0.0
    a_ixs, a_iys, a_s, a_dxs, a_dys = imod.select.cross_sections._draw_line(
        xs, ys, x0, x1, y0, y1, xmin, xmax, ymin, ymax
    )
    assert (e_ixs == a_ixs[::-1]).all()
    assert (e_iys == a_iys[::-1]).all()
    assert np.allclose(a_s, np.sqrt(2.0))
    assert np.allclose(a_dxs, -1.0)
    assert np.allclose(a_dys, -1.0)

    # Deal with out of bounds, just one
    x0, y0 = -1.0, -1.0
    x1, y1 = 3.0, 3.0
    a_ixs, a_iys, a_s, a_dxs, a_dys = imod.select.cross_sections._draw_line(
        xs, ys, x0, x1, y0, y1, xmin, xmax, ymin, ymax
    )
    e_ixs = np.array([-1, 0, 1, 2])
    e_iys = np.array([-1, 0, 1, 2])
    assert (e_ixs == a_ixs).all()
    assert (e_iys == a_iys).all()
    assert np.allclose(a_s, np.sqrt(2.0))
    assert np.allclose(a_dxs, 1.0)
    assert np.allclose(a_dys, 1.0)

    # Deal with out of bounds, both
    x0, y0 = -1.0, -1.0
    x1, y1 = 4.0, 4.0
    a_ixs, a_iys, a_s, a_dxs, a_dys = imod.select.cross_sections._draw_line(
        xs, ys, x0, x1, y0, y1, xmin, xmax, ymin, ymax
    )
    e_ixs = np.array([-1, 0, 1, 2, -1])
    e_iys = np.array([-1, 0, 1, 2, -1])
    assert (e_ixs == a_ixs).all()
    assert (e_iys == a_iys).all()
    assert np.allclose(a_s, np.sqrt(2.0))
    assert np.allclose(a_dxs, 1.0)
    assert np.allclose(a_dys, 1.0)

    # And flipped around
    x0, y0 = 4.0, 4.0
    x1, y1 = -1.0, -1.0
    a_ixs, a_iys, a_s, a_dxs, a_dys = imod.select.cross_sections._draw_line(
        xs, ys, x0, x1, y0, y1, xmin, xmax, ymin, ymax
    )
    assert (e_ixs == a_ixs[::-1]).all()
    assert (e_iys == a_iys[::-1]).all()
    assert np.allclose(a_s, np.sqrt(2.0))
    assert np.allclose(a_dxs, -1.0)
    assert np.allclose(a_dys, -1.0)


def test_cross_section(test_da):
    start = (0.0, 0.0)
    end = (3.0, 3.0)
    ds = np.array([np.sqrt(2.0) for _ in range(3)])
    s = np.cumsum(ds) - 0.5 * ds
    expected = xr.DataArray(
        data=[8, 5, 2],
        coords={
            "x": ("s", [0.5, 1.5, 2.5]),
            "y": ("s", [0.5, 1.5, 2.5]),
            "dx": ("s", [1.0, 1.0, 1.0]),
            "dy": ("s", [1.0, 1.0, 1.0]),
            "s": np.round(s, 3),
            "ds": ("s", np.round(ds, 3)),
        },
        dims=("s",),
    )
    actual = imod.select.cross_section_line(test_da, start, end)
    actual["s"] = actual["s"].round(3)
    actual["ds"] = actual["ds"].round(3)
    assert actual.identical(expected)
