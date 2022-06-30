import pathlib
import re
import textwrap

import numpy as np
import pytest
import xarray as xr

import imod


@pytest.fixture(scope="function")
def riv_dict():
    x = [5.0, 15.0, 25.0]
    y = [25.0, 15.0, 5.0]
    layer = [2, 3]
    dx = 10.0
    dy = -10.0

    da = xr.DataArray(
        data=np.ones((2, 3, 3), dtype=float),
        dims=("layer", "y", "x"),
        coords=dict(layer=layer, y=y, x=x, dx=dx, dy=dy),
    )

    da[:, 1, 1] = np.nan

    return dict(stage=da, conductance=da, bottom_elevation=da - 1.0)


def test_render(riv_dict):
    river = imod.mf6.River(**riv_dict)
    directory = pathlib.Path("mymodel")
    globaltimes = [np.datetime64("2000-01-01")]
    actual = river.render(directory, "river", globaltimes, True)
    expected = textwrap.dedent(
        """\
        begin options
        end options

        begin dimensions
          maxbound 16
        end dimensions

        begin period 1
          open/close mymodel/river/riv.bin (binary)
        end period
        """
    )
    assert actual == expected


def test_wrong_dtype(riv_dict):
    riv_dict["stage"] = riv_dict["stage"].astype(int)

    with pytest.raises(TypeError):
        imod.mf6.River(**riv_dict)


def test_all_nan(riv_dict):
    # Use where to set everything to np.nan
    riv_dict["stage"] = riv_dict["stage"].where(False)

    with pytest.raises(ValueError, match="Provided grid with only nans in River."):
        imod.mf6.River(**riv_dict)


def test_inconsistent_nan(riv_dict):
    riv_dict["stage"][:, 1, 2] = np.nan

    with pytest.raises(
        ValueError,
        match="Detected inconsistent data in River, some variables contain nan, but others do not.",
    ):
        imod.mf6.River(**riv_dict)


def test_check_layer(riv_dict):
    """
    Test for error thrown if variable has no layer dim
    """
    riv_dict["stage"] = riv_dict["stage"].sel(layer=2, drop=True)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "No 'layer' coordinate assigned to stage in the River package. 2D grids still require a 'layer' coordinate. You can assign one with `da.assign_coordinate(layer=1)`"
        ),
    ):
        imod.mf6.River(**riv_dict)


def test_check_dimsize_zero():
    """
    Test that error is thrown for layer dim size 0.
    """
    x = [5.0, 15.0, 25.0]
    y = [25.0, 15.0, 5.0]
    dx = 10.0
    dy = -10.0

    da = xr.DataArray(
        data=np.ones((0, 3, 3), dtype=float),
        dims=("layer", "y", "x"),
        coords=dict(layer=[], y=y, x=x, dx=dx, dy=dy),
    )

    da[:, 1, 1] = np.nan

    with pytest.raises(
        ValueError, match="Provided dimension layer in River with size 0"
    ):
        imod.mf6.River(stage=da, conductance=da, bottom_elevation=da - 1.0)


def test_check_zero_conductance(riv_dict):
    """
    Test for zero conductance
    """
    riv_dict["conductance"] = riv_dict["conductance"] * 0.0

    with pytest.raises(
        ValueError, match="Detected conductance with value 0.0 in River"
    ):
        imod.mf6.River(**riv_dict)


def test_check_bottom_above_stage(riv_dict):
    """
    Check that river bottom is not above stage.
    """

    riv_dict["bottom_elevation"] = riv_dict["bottom_elevation"] + 10.0

    with pytest.raises(ValueError, match="Bottom elevation above stage in River."):
        imod.mf6.River(**riv_dict)


def test_check_dim_monotonicity(riv_dict):
    """
    Test if dimensions are monotonically increasing or, in case of the y coord,
    decreasing
    """
    riv_ds = xr.merge([riv_dict])

    with pytest.raises(
        ValueError, match="y coordinate in River not monotonically decreasing"
    ):
        imod.mf6.River(**riv_ds.sel(y=slice(None, None, -1)))

    with pytest.raises(
        ValueError, match="x coordinate in River not monotonically increasing"
    ):
        imod.mf6.River(**riv_ds.sel(x=slice(None, None, -1)))

    with pytest.raises(
        ValueError, match="layer coordinate in River not monotonically increasing"
    ):
        imod.mf6.River(**riv_ds.sel(layer=slice(None, None, -1)))
