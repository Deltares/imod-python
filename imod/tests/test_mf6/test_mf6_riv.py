import pathlib
import re
import tempfile
import textwrap

import numpy as np
import pytest
import xarray as xr

import imod
from imod.schemata import ValidationError


@pytest.fixture(scope="function")
def make_da():
    x = [5.0, 15.0, 25.0]
    y = [25.0, 15.0, 5.0]
    layer = [2, 3]
    dx = 10.0
    dy = -10.0

    return xr.DataArray(
        data=np.ones((2, 3, 3), dtype=float),
        dims=("layer", "y", "x"),
        coords=dict(layer=layer, y=y, x=x, dx=dx, dy=dy),
    )


@pytest.fixture(scope="function")
def dis_dict(make_da):
    da = make_da
    bottom = da - xr.DataArray(
        data=[1.5, 2.5], dims=("layer",), coords={"layer": [2, 3]}
    )

    return dict(idomain=da.astype(int), top=da.sel(layer=2), bottom=bottom)


@pytest.fixture(scope="function")
def riv_dict(make_da):
    da = make_da
    da[:, 1, 1] = np.nan

    bottom = da - xr.DataArray(
        data=[1.0, 2.0], dims=("layer",), coords={"layer": [2, 3]}
    )

    return dict(stage=da, conductance=da, bottom_elevation=bottom)


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

    with pytest.raises(ValidationError):
        imod.mf6.River(**riv_dict)


def test_all_nan(riv_dict, dis_dict):
    # Use where to set everything to np.nan
    for var in ["stage", "conductance", "bottom_elevation"]:
        riv_dict[var] = riv_dict[var].where(False)

    river = imod.mf6.River(**riv_dict)

    errors = river._validate(river._write_schemata, **dis_dict)

    assert len(errors) == 1

    for var, var_errors in errors.items():
        assert var == "stage"


def test_inconsistent_nan(riv_dict, dis_dict):
    riv_dict["stage"][:, 1, 2] = np.nan
    river = imod.mf6.River(**riv_dict)

    errors = river._validate(river._write_schemata, **dis_dict)

    assert len(errors) == 1


def test_check_layer(riv_dict):
    """
    Test for error thrown if variable has no layer dim
    """
    riv_dict["stage"] = riv_dict["stage"].sel(layer=2, drop=True)

    message = textwrap.dedent(
        """
        * stage
        \t- coords has missing keys: {'layer'}"""
    )

    with pytest.raises(
        ValidationError,
        match=re.escape(message),
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

    message = textwrap.dedent(
        """
        * stage
        \t- provided dimension layer with size 0
        * conductance
        \t- provided dimension layer with size 0
        * bottom_elevation
        \t- provided dimension layer with size 0"""
    )

    with pytest.raises(ValidationError, match=re.escape(message)):
        imod.mf6.River(stage=da, conductance=da, bottom_elevation=da - 1.0)


def test_check_zero_conductance(riv_dict, dis_dict):
    """
    Test for zero conductance
    """
    riv_dict["conductance"] = riv_dict["conductance"] * 0.0

    river = imod.mf6.River(**riv_dict)

    errors = river._validate(river._write_schemata, **dis_dict)

    assert len(errors) == 1
    for var, var_errors in errors.items():
        assert var == "conductance"


def test_check_bottom_above_stage(riv_dict, dis_dict):
    """
    Check that river bottom is not above stage.
    """

    riv_dict["bottom_elevation"] = riv_dict["bottom_elevation"] + 10.0

    river = imod.mf6.River(**riv_dict)

    errors = river._validate(river._write_schemata, **dis_dict)

    assert len(errors) == 1
    for var, var_errors in errors.items():
        assert var == "stage"


def test_check_riv_bottom_above_dis_bottom(riv_dict, dis_dict):
    """
    Check that river bottom not above dis bottom.
    """

    river = imod.mf6.River(**riv_dict)

    river._validate(river._write_schemata, **dis_dict)

    dis_dict["bottom"] += 2.0

    errors = river._validate(river._write_schemata, **dis_dict)

    assert len(errors) == 1
    for var, var_errors in errors.items():
        assert var == "bottom_elevation"


def test_check_boundary_outside_active_domain(riv_dict, dis_dict):
    """
    Check that river not outside idomain
    """

    river = imod.mf6.River(**riv_dict)

    errors = river._validate(river._write_schemata, **dis_dict)

    assert len(errors) == 0

    dis_dict["idomain"][0, 0, 0] = 0

    errors = river._validate(river._write_schemata, **dis_dict)

    assert len(errors) == 1


def test_check_dim_monotonicity(riv_dict):
    """
    Test if dimensions are monotonically increasing or, in case of the y coord,
    decreasing
    """
    riv_ds = xr.merge([riv_dict])

    message = textwrap.dedent(
        """
        * stage
        \t- coord y which is not monotonically decreasing
        * conductance
        \t- coord y which is not monotonically decreasing
        * bottom_elevation
        \t- coord y which is not monotonically decreasing"""
    )

    with pytest.raises(ValidationError, match=re.escape(message)):
        imod.mf6.River(**riv_ds.sel(y=slice(None, None, -1)))

    message = textwrap.dedent(
        """
        * stage
        \t- coord x which is not monotonically increasing
        * conductance
        \t- coord x which is not monotonically increasing
        * bottom_elevation
        \t- coord x which is not monotonically increasing"""
    )

    with pytest.raises(ValidationError, match=re.escape(message)):
        imod.mf6.River(**riv_ds.sel(x=slice(None, None, -1)))

    message = textwrap.dedent(
        """
        * stage
        \t- coord layer which is not monotonically increasing
        * conductance
        \t- coord layer which is not monotonically increasing
        * bottom_elevation
        \t- coord layer which is not monotonically increasing"""
    )

    with pytest.raises(ValidationError, match=re.escape(message)):
        imod.mf6.River(**riv_ds.sel(layer=slice(None, None, -1)))


@pytest.mark.usefixtures("concentration_fc")
def test_render_concentration(concentration_fc):
    directory = pathlib.Path("mymodel")
    globaltimes = [np.datetime64("2000-01-01")]

    riv = imod.mf6.River(
        stage=1.0,
        conductance=10.0,
        bottom_elevation=-1.0,
        concentration=concentration_fc.sel(
            time=np.datetime64("2000-01-01")
        ).reset_coords(drop=True),
        concentration_boundary_type="AUX",
    )
    actual = riv.render(directory, "riv", globaltimes, False)

    expected = textwrap.dedent(
        """\
        begin options
          auxiliary salinity temperature
        end options

        begin dimensions
          maxbound 1
        end dimensions

        begin period 1
          open/close mymodel/riv/riv.dat
        end period
        """
    )
    assert actual == expected


@pytest.mark.usefixtures("concentration_fc")
def test_write_concentration_period_data(concentration_fc):
    globaltimes = [np.datetime64("2000-01-01")]
    concentration_fc[:] = 2
    stage = xr.full_like(concentration_fc.sel({"species": "salinity"}), 13)
    conductance = xr.full_like(stage, 13)
    bottom_elevation = xr.full_like(stage, 13)
    riv = imod.mf6.River(
        stage=stage,
        conductance=conductance,
        bottom_elevation=bottom_elevation,
        concentration=concentration_fc,
        concentration_boundary_type="AUX",
    )
    with tempfile.TemporaryDirectory() as output_dir:
        riv.write(output_dir, "riv", globaltimes, False)
        with open(output_dir + "/riv/riv-0.dat", "r") as f:
            data = f.read()
            assert (
                data.count("2") == 1755
            )  # the number 2 is in the concentration data, and in the cell indices.
