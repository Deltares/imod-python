import pathlib
import re
import tempfile
import textwrap
from copy import deepcopy

import numpy as np
import pytest
import xarray as xr

import imod
from imod.mf6.dis import StructuredDiscretization
from imod.mf6.write_context import WriteContext
from imod.schemata import ValidationError
from imod.typing.grid import is_planar_grid, is_transient_data_grid, nan_like


@pytest.fixture(scope="function")
def rch_dict():
    x = [5.0, 15.0, 25.0]
    y = [25.0, 15.0, 5.0]
    layer = [1]
    dx = 10.0
    dy = -10.0

    da = xr.DataArray(
        data=np.ones((1, 3, 3), dtype=float),
        dims=("layer", "y", "x"),
        coords={"layer": layer, "y": y, "x": x, "dx": dx, "dy": dy},
    )

    da[:, 1, 1] = np.nan

    return {"rate": da}


@pytest.fixture(scope="function")
def rch_dict_transient():
    x = [5.0, 15.0, 25.0]
    y = [25.0, 15.0, 5.0]
    layer = [1]
    time = np.array(["2000-01-01", "2000-01-02"], dtype="datetime64[ns]")
    dx = 10.0
    dy = -10.0

    da = xr.DataArray(
        data=np.ones((2, 1, 3, 3), dtype=float),
        dims=("time", "layer", "y", "x"),
        coords={"time": time, "layer": layer, "y": y, "x": x, "dx": dx, "dy": dy},
    )

    da[..., 1, 1] = np.nan

    return {"rate": da}


def test_render(rch_dict):
    rch = imod.mf6.Recharge(**rch_dict)
    directory = pathlib.Path("mymodel")
    globaltimes = np.array(["2000-01-01"], dtype="datetime64[ns]")
    actual = rch.render(directory, "recharge", globaltimes, True)
    expected = textwrap.dedent(
        """\
        begin options
        end options

        begin dimensions
          maxbound 8
        end dimensions

        begin period 1
          open/close mymodel/recharge/rch.bin (binary)
        end period
        """
    )
    assert actual == expected


def test_render_fixed_cell(rch_dict):
    rch_dict["fixed_cell"] = True
    rch = imod.mf6.Recharge(**rch_dict)
    directory = pathlib.Path("mymodel")
    globaltimes = np.array(["2000-01-01"], dtype="datetime64[ns]")
    actual = rch.render(directory, "recharge", globaltimes, True)
    expected = textwrap.dedent(
        """\
        begin options
          fixed_cell
        end options

        begin dimensions
          maxbound 8
        end dimensions

        begin period 1
          open/close mymodel/recharge/rch.bin (binary)
        end period
        """
    )
    assert actual == expected


def test_render_transient(rch_dict_transient):
    rch = imod.mf6.Recharge(**rch_dict_transient)
    directory = pathlib.Path("mymodel")
    globaltimes = np.array(
        [
            "2000-01-01",
            "2000-01-02",
            "2000-01-03",
        ],
        dtype="datetime64[ns]",
    )
    actual = rch.render(directory, "recharge", globaltimes, True)
    expected = textwrap.dedent(
        """\
        begin options
        end options

        begin dimensions
          maxbound 8
        end dimensions

        begin period 1
          open/close mymodel/recharge/rch-0.bin (binary)
        end period
        begin period 2
          open/close mymodel/recharge/rch-1.bin (binary)
        end period
        """
    )
    assert actual == expected


def test_wrong_dtype(rch_dict):
    rch_dict["rate"] = rch_dict["rate"].astype(np.int32)
    with pytest.raises(ValidationError):
        imod.mf6.Recharge(**rch_dict)


def test_no_layer_dim(rch_dict):
    rch_dict["rate"] = rch_dict["rate"].sel(layer=1, drop=False)
    rch = imod.mf6.Recharge(**rch_dict)
    directory = pathlib.Path("mymodel")
    globaltimes = np.array(["2000-01-01"], dtype="datetime64[ns]")
    actual = rch.render(directory, "recharge", globaltimes, True)
    expected = textwrap.dedent(
        """\
        begin options
        end options

        begin dimensions
          maxbound 8
        end dimensions

        begin period 1
          open/close mymodel/recharge/rch.bin (binary)
        end period
        """
    )
    assert actual == expected


def test_transient_no_layer_dim(rch_dict_transient):
    rch_dict_transient["rate"] = rch_dict_transient["rate"].sel(layer=1, drop=False)
    rch = imod.mf6.Recharge(**rch_dict_transient)

    directory = pathlib.Path("mymodel")
    globaltimes = np.array(
        [
            "2000-01-01",
            "2000-01-02",
            "2000-01-03",
        ],
        dtype="datetime64[ns]",
    )

    actual = rch.render(directory, "recharge", globaltimes, True)
    expected = textwrap.dedent(
        """\
        begin options
        end options

        begin dimensions
          maxbound 8
        end dimensions

        begin period 1
          open/close mymodel/recharge/rch-0.bin (binary)
        end period
        begin period 2
          open/close mymodel/recharge/rch-1.bin (binary)
        end period
        """
    )

    assert actual == expected


@pytest.mark.usefixtures("concentration_fc", "rate_fc")
def test_render_concentration(concentration_fc, rate_fc):
    rch = imod.mf6.Recharge(
        rate=rate_fc,
        concentration=concentration_fc,
        concentration_boundary_type="AUX",
    )

    directory = pathlib.Path("mymodel")
    globaltimes = np.array(
        [
            "2000-01-01",
            "2000-01-02",
            "2000-01-03",
        ],
        dtype="datetime64[ns]",
    )

    actual = rch.render(directory, "rch", globaltimes, False)

    expected = textwrap.dedent(
        """\
        begin options
          auxiliary salinity temperature
        end options

        begin dimensions
          maxbound 2
        end dimensions

        begin period 1
          open/close mymodel/rch/rch-0.dat
        end period
        begin period 2
          open/close mymodel/rch/rch-1.dat
        end period
        begin period 3
          open/close mymodel/rch/rch-2.dat
        end period
        """
    )
    assert actual == expected


def test_no_layer_coord(rch_dict):
    message = textwrap.dedent(
        """
        * rate
        \t- coords has missing keys: {'layer'}"""
    )

    rch_dict["rate"] = rch_dict["rate"].sel(layer=1, drop=True)
    with pytest.raises(
        ValidationError,
        match=re.escape(message),
    ):
        imod.mf6.Recharge(**rch_dict)


def test_scalar():
    message = textwrap.dedent(
        """
        * rate
        \t- coords has missing keys: {'layer'}
        \t- No option succeeded:
        \tdim mismatch: expected ('time', 'layer', 'y', 'x'), got ()
        \tdim mismatch: expected ('layer', 'y', 'x'), got ()
        \tdim mismatch: expected ('time', 'layer', '{face_dim}'), got ()
        \tdim mismatch: expected ('layer', '{face_dim}'), got ()
        \tdim mismatch: expected ('time', 'y', 'x'), got ()
        \tdim mismatch: expected ('y', 'x'), got ()
        \tdim mismatch: expected ('time', '{face_dim}'), got ()
        \tdim mismatch: expected ('{face_dim}',), got ()"""
    )
    with pytest.raises(ValidationError, match=re.escape(message)):
        imod.mf6.Recharge(rate=0.001)


def test_validate_false():
    imod.mf6.Recharge(rate=0.001, validate=False)


@pytest.mark.usefixtures("rate_fc", "concentration_fc")
def test_write_concentration_period_data(rate_fc, concentration_fc):
    globaltimes = np.array(
        [
            "2000-01-01",
            "2000-01-02",
            "2000-01-03",
        ],
        dtype="datetime64",
    )
    rate_fc[:] = 1
    concentration_fc[:] = 2

    rch = imod.mf6.Recharge(
        rate=rate_fc,
        concentration=concentration_fc,
        concentration_boundary_type="AUX",
    )
    with tempfile.TemporaryDirectory() as output_dir:
        write_context = WriteContext(
            simulation_directory=output_dir, write_directory=output_dir
        )
        rch._write(pkgname="rch", globaltimes=globaltimes, write_context=write_context)

        with open(output_dir + "/rch/rch-0.dat", "r") as f:
            data = f.read()
            assert (
                data.count("2") == 1755
            )  # the number 2 is in the concentration data, and in the cell indices.


def test_clip_box(rch_dict):
    rch = imod.mf6.Recharge(**rch_dict)

    selection = rch.clip_box()
    assert isinstance(selection, imod.mf6.Recharge)
    assert selection.dataset.identical(rch.dataset)

    selection = rch.clip_box(x_min=10.0, x_max=20.0, y_min=10.0, y_max=20.0)
    assert selection["rate"].dims == ("layer", "y", "x")
    assert selection["rate"].shape == (1, 1, 1)

    # No layer dim
    rch_dict["rate"] = rch_dict["rate"].sel(layer=1, drop=False)
    rch = imod.mf6.Recharge(**rch_dict)
    selection = rch.clip_box(x_min=10.0, x_max=20.0, y_min=10.0, y_max=20.0)
    assert selection["rate"].dims == ("y", "x")
    assert selection["rate"].shape == (1, 1)


@pytest.mark.usefixtures("imod5_dataset")
def test_planar_rch_from_imod5_constant(imod5_dataset, tmp_path):
    data = deepcopy(imod5_dataset[0])
    target_discretization = StructuredDiscretization.from_imod5_data(data)

    # create a planar grid with time-independent recharge
    data["rch"]["rate"]["layer"].values[0] = -1
    assert not is_transient_data_grid(data["rch"]["rate"])
    assert is_planar_grid(data["rch"]["rate"])

    # Act
    rch = imod.mf6.Recharge.from_imod5_data(data, target_discretization)
    rendered_rch = rch.render(tmp_path, "rch", None, None)

    # Assert
    np.testing.assert_allclose(
        data["rch"]["rate"].mean().values / 1e3,
        rch.dataset["rate"].mean().values,
        atol=1e-5,
    )
    assert "maxbound 33856" in rendered_rch
    assert rendered_rch.count("begin period") == 1
    # teardown
    data["rch"]["rate"]["layer"].values[0] = 1


@pytest.mark.usefixtures("imod5_dataset")
def test_planar_rch_from_imod5_transient(imod5_dataset, tmp_path):
    data = deepcopy(imod5_dataset[0])
    target_discretization = StructuredDiscretization.from_imod5_data(data)

    # create a grid with recharge for 3 timesteps
    input_recharge = data["rch"]["rate"].copy(deep=True)
    input_recharge = input_recharge.expand_dims({"time": [0, 1, 2]})

    # make it planar by setting the layer coordinate to -1
    input_recharge = input_recharge.assign_coords({"layer": [-1]})

    # update the data set
    data["rch"]["rate"] = input_recharge
    assert is_transient_data_grid(data["rch"]["rate"])
    assert is_planar_grid(data["rch"]["rate"])

    # act
    rch = imod.mf6.Recharge.from_imod5_data(data, target_discretization)
    rendered_rch = rch.render(tmp_path, "rch", [0, 1, 2], None)

    # assert
    np.testing.assert_allclose(
        data["rch"]["rate"].mean().values / 1e3,
        rch.dataset["rate"].mean().values,
        atol=1e-5,
    )
    assert rendered_rch.count("begin period") == 3
    assert "maxbound 33856" in rendered_rch


@pytest.mark.usefixtures("imod5_dataset")
def test_non_planar_rch_from_imod5_constant(imod5_dataset, tmp_path):
    data = deepcopy(imod5_dataset[0])
    target_discretization = StructuredDiscretization.from_imod5_data(data)

    # make the first layer of the target grid inactive
    target_grid = target_discretization.dataset["idomain"]
    target_grid.loc[{"layer": 1}] = 0

    # the input for recharge is on the second layer of the targetgrid
    original_rch = data["rch"]["rate"].copy(deep=True)
    data["rch"]["rate"] = data["rch"]["rate"].assign_coords({"layer": [-1]})
    input_recharge = nan_like(data["khv"]["kh"])
    input_recharge.loc[{"layer": 2}] = data["rch"]["rate"].isel(layer=0)

    # update the data set

    data["rch"]["rate"] = input_recharge
    assert not is_planar_grid(data["rch"]["rate"])
    assert not is_transient_data_grid(data["rch"]["rate"])

    # act
    rch = imod.mf6.Recharge.from_imod5_data(data, target_discretization)
    rendered_rch = rch.render(tmp_path, "rch", None, None)

    # assert
    np.testing.assert_allclose(
        data["rch"]["rate"].mean().values / 1e3,
        rch.dataset["rate"].mean().values,
        atol=1e-5,
    )
    assert rendered_rch.count("begin period") == 1
    assert "maxbound 33856" in rendered_rch

    # teardown
    data["rch"]["rate"] = original_rch


@pytest.mark.usefixtures("imod5_dataset")
def test_non_planar_rch_from_imod5_transient(imod5_dataset, tmp_path):
    data = deepcopy(imod5_dataset[0])
    target_discretization = StructuredDiscretization.from_imod5_data(data)
    # make the first layer of the target grid inactive
    target_grid = target_discretization.dataset["idomain"]
    target_grid.loc[{"layer": 1}] = 0

    # the input for recharge is on the second layer of the targetgrid
    input_recharge = nan_like(data["rch"]["rate"])
    input_recharge = input_recharge.assign_coords({"layer": [2]})
    input_recharge.loc[{"layer": 2}] = data["rch"]["rate"].sel(layer=1)
    input_recharge = input_recharge.expand_dims({"time": [0, 1, 2]})

    # update the data set
    data["rch"]["rate"] = input_recharge
    assert not is_planar_grid(data["rch"]["rate"])
    assert is_transient_data_grid(data["rch"]["rate"])

    # act
    rch = imod.mf6.Recharge.from_imod5_data(data, target_discretization)
    rendered_rch = rch.render(tmp_path, "rch", [0, 1, 2], None)

    # assert
    np.testing.assert_allclose(
        data["rch"]["rate"].mean().values / 1e3,
        rch.dataset["rate"].mean().values,
        atol=1e-5,
    )
    assert rendered_rch.count("begin period") == 3
    assert "maxbound 33856" in rendered_rch


@pytest.mark.usefixtures("imod5_dataset")
def test_from_imod5_cap_data(imod5_dataset):
    # Arrange
    data = deepcopy(imod5_dataset[0])
    target_discretization = StructuredDiscretization.from_imod5_data(data)
    data["cap"] = {}
    msw_bound = data["bnd"]["ibound"].isel(layer=0, drop=False)
    data["cap"]["boundary"] = msw_bound
    data["cap"]["wetted_area"] = xr.ones_like(msw_bound) * 100
    data["cap"]["urban_area"] = xr.ones_like(msw_bound) * 200
    # Set to total cellsize, cell needs to be deactivated.
    data["cap"]["wetted_area"][100, 100] = 625.0
    # Set to zero, cell needs to be deactivated.
    data["cap"]["urban_area"][100, 100] = 0.0
    # Act
    rch = imod.mf6.Recharge.from_imod5_cap_data(data, target_discretization)
    rate = rch.dataset["rate"]
    # Assert
    np.testing.assert_array_equal(np.unique(rate), np.array([0.0, np.nan]))
    # Boundaries inactive in MetaSWAP
    assert np.isnan(rate[0, :, 0]).all()
    assert np.isnan(rate[0, :, -1]).all()
    assert np.isnan(rate[0, 0, :]).all()
    assert np.isnan(rate[0, -1, :]).all()
    assert np.isnan(rate[:, 100, 100]).all()
