import pathlib
import textwrap

import numpy as np
import pytest
import xarray as xr
import xugrid as xu

import imod


def get_structured_grid_da(dtype, value=1):
    """
    This function creates a dataarray with scalar values for a grid of 3 layers and 9 rows and columns.
    """
    shape = nlay, nrow, ncol = 3, 9, 9
    dims = ("layer", "y", "x")

    dx = 10.0
    dy = -10.0
    xmin = 0.0
    xmax = dx * ncol
    ymin = 0.0
    ymax = abs(dy) * nrow

    layer = np.arange(1, nlay + 1)
    y = np.arange(ymax, ymin, dy) + 0.5 * dy
    x = np.arange(xmin, xmax, dx) + 0.5 * dx
    coords = {"layer": layer, "y": y, "x": x}

    da = xr.DataArray(np.ones(shape, dtype=dtype) * value, coords=coords, dims=dims)
    return da


def get_unstructured_grid_da(dtype, value=1):
    """
    This function creates an xugrid dataarray with scalar values for an unstructured grid
    """
    grid = imod.data.circle()
    nface = grid.n_face
    nlayer = 2

    dims = ("layer", grid.face_dimension)
    shape = (nlayer, nface)

    uda = xu.UgridDataArray(
        xr.DataArray(
            np.ones(shape, dtype=dtype) * value,
            coords={"layer": [1, 2]},
            dims=dims,
        ),
        grid=grid,
    )
    return uda


def get_grid_da(is_unstructured, dtype, value=1):
    """
    helper function for creating an xarray dataset of a given type
    Depending on the is_unstructured input parameter, it will create an array for a
    structured grid or for an unstructured grid.
    """

    if is_unstructured:
        return get_unstructured_grid_da(dtype, value)
    else:
        return get_structured_grid_da(dtype, value)


def test_dispersion_default():
    directory = pathlib.Path("mymodel")
    globaltimes = [np.datetime64("2000-01-01")]
    disp = imod.mf6.Dispersion(1e-4, 1.0, 10.0)

    actual = disp.render(directory, "dsp", globaltimes, True)
    expected = textwrap.dedent(
        """\
      begin options
      end options

      begin griddata
        diffc
          constant 0.0001
        alh
          constant 1.0
        ath1
          constant 10.0
      end griddata"""
    )

    assert actual == expected


@pytest.mark.parametrize("is_unstructured", [False, True])
def test_dispersion_init(is_unstructured):
    imod.mf6.Dispersion(
        diffusion_coefficient=get_grid_da(is_unstructured, np.float32, 1e-4),
        longitudinal_horizontal=get_grid_da(is_unstructured, np.float32, 10),
        transversal_horizontal1=get_grid_da(is_unstructured, np.float32, 10),
        longitudinal_vertical=get_grid_da(is_unstructured, np.float32, 5),
        transversal_horizontal2=get_grid_da(is_unstructured, np.float32, 2),
        transversal_vertical=get_grid_da(is_unstructured, np.float32, 4),
    ),


def test_dispersion_options():
    directory = pathlib.Path("mymodel")
    globaltimes = [np.datetime64("2000-01-01")]
    disp = imod.mf6.Dispersion(1e-4, 1.0, 10.0, 1.0, 2.0, 3.0, True, True)
    actual = disp.render(directory, "dsp", globaltimes, True)
    expected = textwrap.dedent(
        """\
      begin options
        xt3d_off
        xt3d_rhs
      end options

      begin griddata
        diffc
          constant 0.0001
        alh
          constant 1.0
        ath1
          constant 10.0
        alv
          constant 1.0
        ath2
          constant 2.0
        atv
          constant 3.0
      end griddata"""
    )

    assert actual == expected


def test_dispersion_layered():
    def layered(data):
        return xr.DataArray(data, {"layer": [1, 2, 3]}, ("layer",))

    directory = pathlib.Path("mymodel")
    globaltimes = [np.datetime64("2000-01-01")]
    disp = imod.mf6.Dispersion(
        diffusion_coefficient=layered([1.0, 2.0, 3.0]),
        longitudinal_horizontal=layered([4.0, 5.0, 6.0]),
        transversal_horizontal1=layered([7.0, 8.0, 9.0]),
        longitudinal_vertical=layered([10.0, 11.0, 12.0]),
        transversal_horizontal2=layered([13.0, 14.0, 15.0]),
        transversal_vertical=layered([16.0, 17.0, 18.0]),
        xt3d_off=True,
        xt3d_rhs=True,
    )

    actual = disp.render(directory, "dsp", globaltimes, True)
    expected = textwrap.dedent(
        """\
      begin options
        xt3d_off
        xt3d_rhs
      end options

      begin griddata
        diffc layered
          constant 1.0
          constant 2.0
          constant 3.0
        alh layered
          constant 4.0
          constant 5.0
          constant 6.0
        ath1 layered
          constant 7.0
          constant 8.0
          constant 9.0
        alv layered
          constant 10.0
          constant 11.0
          constant 12.0
        ath2 layered
          constant 13.0
          constant 14.0
          constant 15.0
        atv layered
          constant 16.0
          constant 17.0
          constant 18.0
      end griddata"""
    )
    assert actual == expected
