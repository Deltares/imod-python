import pathlib
import re
import tempfile
import textwrap
from copy import deepcopy
from datetime import datetime

import numpy as np
import pytest
import xarray as xr
import xugrid as xu
from pytest_cases import parametrize_with_cases

import imod
from imod.mf6.dis import StructuredDiscretization
from imod.mf6.disv import VerticesDiscretization
from imod.mf6.npf import NodePropertyFlow
from imod.mf6.write_context import WriteContext
from imod.prepare.topsystem.allocation import ALLOCATION_OPTION
from imod.prepare.topsystem.conductance import DISTRIBUTING_OPTION
from imod.prepare.topsystem.default_allocation_methods import (
    SimulationAllocationOptions,
    SimulationDistributingOptions,
)
from imod.schemata import ValidationError
from imod.typing.grid import (
    enforce_dim_order,
    has_negative_layer,
    is_planar_grid,
    ones_like,
    zeros_like,
)

TYPE_DIS_PKG = {
    xu.UgridDataArray: VerticesDiscretization,
    xr.DataArray: StructuredDiscretization,
}


def make_da():
    x = [5.0, 15.0, 25.0]
    y = [25.0, 15.0, 5.0]
    layer = [2, 3]
    dx = 10.0
    dy = -10.0

    return xr.DataArray(
        data=np.ones((2, 3, 3), dtype=float),
        dims=("layer", "y", "x"),
        coords={"layer": layer, "y": y, "x": x, "dx": dx, "dy": dy},
    )


def dis_dict():
    da = make_da()
    bottom = da - xr.DataArray(
        data=[1.5, 2.5], dims=("layer",), coords={"layer": [2, 3]}
    )

    return {"idomain": da.astype(int), "top": da.sel(layer=2), "bottom": bottom}


def riv_dict():
    da = make_da()
    da[:, 1, 1] = np.nan

    bottom = da - xr.DataArray(
        data=[1.0, 2.0], dims=("layer",), coords={"layer": [2, 3]}
    )

    return {"stage": da, "conductance": da.copy(), "bottom_elevation": bottom}


def make_dict_unstructured(d):
    return {key: xu.UgridDataArray.from_structured2d(value) for key, value in d.items()}


class RivCases:
    def case_structured(self):
        return riv_dict()

    def case_unstructured(self):
        return make_dict_unstructured(riv_dict())


class DisCases:
    def case_structured(self):
        return dis_dict()

    def case_unstructured(self):
        return make_dict_unstructured(dis_dict())


class RivDisCases:
    def case_structured(self):
        return riv_dict(), dis_dict()

    def case_unstructured(self):
        return make_dict_unstructured(riv_dict()), make_dict_unstructured(dis_dict())


@parametrize_with_cases("riv_data", cases=RivCases)
def test_render(riv_data):
    river = imod.mf6.River(**riv_data)
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


@parametrize_with_cases("riv_data", cases=RivCases)
def test_render_repeat_stress(riv_data):
    """
    Test that rendering a river with a repeated stress period does not raise an error.
    """
    globaltimes = [
        np.datetime64("2000-04-01"),
        np.datetime64("2000-10-01"),
        np.datetime64("2001-04-01"),
        np.datetime64("2001-10-01"),
    ]

    seasonal_factors = [0.8, 1.2]
    seasonal_da = xr.DataArray(
        seasonal_factors, dims=["time"], coords={"time": globaltimes[:2]}
    )

    riv_data["stage"] = enforce_dim_order(riv_data["stage"] * seasonal_da)
    riv_data["conductance"] = enforce_dim_order(riv_data["conductance"] * seasonal_da)
    riv_data["bottom_elevation"] = enforce_dim_order(
        riv_data["bottom_elevation"] * seasonal_da
    )
    repeat_stress = {
        globaltimes[2]: globaltimes[0],
        globaltimes[3]: globaltimes[1],
    }
    river = imod.mf6.River(repeat_stress=repeat_stress, **riv_data)
    directory = pathlib.Path("mymodel")
    actual = river.render(directory, "river", globaltimes, True)

    expected = textwrap.dedent(
        """\
        begin options
        end options

        begin dimensions
          maxbound 16
        end dimensions

        begin period 1
          open/close mymodel/river/riv-0.bin (binary)
        end period
        begin period 2
          open/close mymodel/river/riv-1.bin (binary)
        end period
        begin period 3
          open/close mymodel/river/riv-0.bin (binary)
        end period
        begin period 4
          open/close mymodel/river/riv-1.bin (binary)
        end period
        """
    )
    assert actual == expected


@parametrize_with_cases("riv_data", cases=RivCases)
def test_wrong_dtype(riv_data):
    riv_data["stage"] = riv_data["stage"].astype(int)

    with pytest.raises(ValidationError):
        imod.mf6.River(**riv_data)


@parametrize_with_cases("riv_data,dis_data", cases=RivDisCases)
def test_all_nan(riv_data, dis_data):
    # Use where to set everything to np.nan
    for var in ["stage", "conductance", "bottom_elevation"]:
        riv_data[var] = riv_data[var].where(False)

    river = imod.mf6.River(**riv_data)

    errors = river._validate(river._write_schemata, **dis_data)

    assert len(errors) == 1
    assert "stage" in errors.keys()


@parametrize_with_cases("riv_data,dis_data", cases=RivDisCases)
def test_validate_inconsistent_nan(riv_data, dis_data):
    riv_data["stage"][..., 2] = np.nan
    river = imod.mf6.River(**riv_data)

    errors = river._validate(river._write_schemata, **dis_data)

    assert len(errors) == 2
    assert "bottom_elevation" in errors.keys()
    assert "conductance" in errors.keys()


@parametrize_with_cases("riv_data,dis_data", cases=RivDisCases)
def test_cleanup_inconsistent_nan(riv_data, dis_data):
    riv_data["stage"][..., 2] = np.nan
    river = imod.mf6.River(**riv_data)
    type_grid = type(riv_data["stage"])
    dis_pkg = TYPE_DIS_PKG[type_grid](**dis_data)

    river.cleanup(dis_pkg)
    errors = river._validate(river._write_schemata, **dis_data)

    assert len(errors) == 0


@parametrize_with_cases("riv_data,dis_data", cases=RivDisCases)
def test_layer_as_coord_in_active_cells(riv_data, dis_data):
    # Test if no bugs like https://github.com/Deltares/imod-python/issues/830
    river = imod.mf6.River(**riv_data)
    river.dataset = river.dataset.sel(layer=2, drop=False)

    dis_data["idomain"][1, ...] = 0

    errors = river._validate(river._write_schemata, **dis_data)

    assert len(errors) == 0


@parametrize_with_cases("riv_data,dis_data", cases=RivDisCases)
def test_layer_as_coord_in_inactive_cells(riv_data, dis_data):
    river = imod.mf6.River(**riv_data)
    river.dataset = river.dataset.sel(layer=2, drop=False)

    dis_data["idomain"][0, ...] = 0

    errors = river._validate(river._write_schemata, **dis_data)

    assert len(errors) == 1


@parametrize_with_cases("riv_data", cases=RivCases)
def test_check_layer(riv_data):
    """
    Test for error thrown if variable has no layer coord
    """
    riv_data["stage"] = riv_data["stage"].sel(layer=2, drop=True)

    message = textwrap.dedent(
        """
        - stage
            - coords has missing keys: {'layer'}"""
    )

    with pytest.raises(
        ValidationError,
        match=re.escape(message),
    ):
        imod.mf6.River(**riv_data)


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
        coords={"layer": [], "y": y, "x": x, "dx": dx, "dy": dy},
    )

    da[:, 1, 1] = np.nan

    message = textwrap.dedent(
        """
        - stage
            - provided dimension layer with size 0
        - conductance
            - provided dimension layer with size 0
        - bottom_elevation
            - provided dimension layer with size 0"""
    )

    with pytest.raises(ValidationError, match=re.escape(message)):
        imod.mf6.River(stage=da, conductance=da, bottom_elevation=da - 1.0)


@parametrize_with_cases("riv_data,dis_data", cases=RivDisCases)
def test_validate_zero_conductance(riv_data, dis_data):
    """
    Test for validation zero conductance
    """
    riv_data["conductance"][..., 2] = 0.0

    river = imod.mf6.River(**riv_data)

    errors = river._validate(river._write_schemata, **dis_data)

    assert len(errors) == 1
    for var, var_errors in errors.items():
        assert var == "conductance"


@parametrize_with_cases("riv_data,dis_data", cases=RivDisCases)
def test_cleanup_zero_conductance(riv_data, dis_data):
    """
    Cleanup zero conductance
    """
    riv_data["conductance"][..., 2] = 0.0
    type_grid = type(riv_data["stage"])
    dis_pkg = TYPE_DIS_PKG[type_grid](**dis_data)

    river = imod.mf6.River(**riv_data)
    river.cleanup(dis_pkg)

    errors = river._validate(river._write_schemata, **dis_data)
    assert len(errors) == 0


@parametrize_with_cases("riv_data,dis_data", cases=RivDisCases)
def test_validate_bottom_above_stage(riv_data, dis_data):
    """
    Validate that river bottom is not above stage.
    """

    riv_data["bottom_elevation"] = riv_data["bottom_elevation"] + 10.0

    river = imod.mf6.River(**riv_data)

    errors = river._validate(river._write_schemata, **dis_data)

    assert len(errors) == 1
    assert "stage" in errors.keys()


@parametrize_with_cases("riv_data,dis_data", cases=RivDisCases)
def test_cleanup_bottom_above_stage(riv_data, dis_data):
    """
    Cleanup river bottom above stage.
    """

    riv_data["bottom_elevation"] = riv_data["bottom_elevation"] + 10.0
    type_grid = type(riv_data["stage"])
    dis_pkg = TYPE_DIS_PKG[type_grid](**dis_data)

    river = imod.mf6.River(**riv_data)
    river.cleanup(dis_pkg)

    errors = river._validate(river._write_schemata, **dis_data)

    assert len(errors) == 0
    assert river.dataset["bottom_elevation"].equals(river.dataset["stage"])


@parametrize_with_cases("riv_data,dis_data", cases=RivDisCases)
def test_check_riv_bottom_above_dis_bottom(riv_data, dis_data):
    """
    Check that river bottom not above dis bottom.
    """

    river = imod.mf6.River(**riv_data)

    river._validate(river._write_schemata, **dis_data)

    dis_data["bottom"] += 2.0

    errors = river._validate(river._write_schemata, **dis_data)

    assert len(errors) == 1
    for var, var_errors in errors.items():
        assert var == "bottom_elevation"


@parametrize_with_cases("riv_data,dis_data", cases=RivDisCases)
def test_check_boundary_outside_active_domain(riv_data, dis_data):
    """
    Check that river not outside idomain
    """

    river = imod.mf6.River(**riv_data)

    errors = river._validate(river._write_schemata, **dis_data)

    assert len(errors) == 0

    dis_data["idomain"][..., 0] = 0

    errors = river._validate(river._write_schemata, **dis_data)

    assert len(errors) == 1


def test_check_dim_monotonicity():
    """
    Test if dimensions are monotonically increasing or, in case of the y coord,
    decreasing
    """
    riv_ds = xr.merge([riv_dict()])

    message = textwrap.dedent(
        """
        - stage
            - coord y which is not monotonically decreasing
        - conductance
            - coord y which is not monotonically decreasing
        - bottom_elevation
            - coord y which is not monotonically decreasing"""
    )

    with pytest.raises(ValidationError, match=re.escape(message)):
        imod.mf6.River(**riv_ds.sel(y=slice(None, None, -1)))

    message = textwrap.dedent(
        """
        - stage
            - coord x which is not monotonically increasing
        - conductance
            - coord x which is not monotonically increasing
        - bottom_elevation
            - coord x which is not monotonically increasing"""
    )

    with pytest.raises(ValidationError, match=re.escape(message)):
        imod.mf6.River(**riv_ds.sel(x=slice(None, None, -1)))

    message = textwrap.dedent(
        """
        - stage
            - coord layer which is not monotonically increasing
        - conductance
            - coord layer which is not monotonically increasing
        - bottom_elevation
            - coord layer which is not monotonically increasing"""
    )

    with pytest.raises(ValidationError, match=re.escape(message)):
        imod.mf6.River(**riv_ds.sel(layer=slice(None, None, -1)))


def test_validate_false():
    """
    Test turning off validation
    """

    riv_ds = xr.merge([riv_dict()])

    imod.mf6.River(validate=False, **riv_ds.sel(layer=slice(None, None, -1)))


def test_render_concentration(concentration_fc):
    riv_ds = xr.merge([riv_dict()])

    concentration = concentration_fc.sel(
        layer=[2, 3], time=np.datetime64("2000-01-01"), drop=True
    )
    riv_ds["concentration"] = concentration.where(~np.isnan(riv_ds["stage"]))

    directory = pathlib.Path("mymodel")
    globaltimes = [np.datetime64("2000-01-01")]

    riv = imod.mf6.River(concentration_boundary_type="AUX", **riv_ds)
    actual = riv.render(directory, "riv", globaltimes, False)

    expected = textwrap.dedent(
        """\
        begin options
          auxiliary salinity temperature
        end options

        begin dimensions
          maxbound 16
        end dimensions

        begin period 1
          open/close mymodel/riv/riv.dat
        end period
        """
    )
    assert actual == expected


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
        write_context = WriteContext(simulation_directory=output_dir)
        riv._write("riv", globaltimes, write_context)
        with open(output_dir + "/riv/riv-0.dat", "r") as f:
            data = f.read()
            assert (
                data.count("2") == 1755
            )  # the number 2 is in the concentration data, and in the cell indices.


def test_import_river_from_imod5(imod5_dataset, tmp_path):
    imod5_data = imod5_dataset[0]
    period_data = imod5_dataset[1]
    globaltimes = [np.datetime64("2000-01-01")]
    target_dis = StructuredDiscretization.from_imod5_data(imod5_data)
    grid = target_dis.dataset["idomain"]
    target_npf = NodePropertyFlow.from_imod5_data(imod5_data, grid)

    (riv, drn) = imod.mf6.River.from_imod5_data(
        "riv-1",
        imod5_data,
        period_data,
        target_dis,
        target_npf,
        time_min=datetime(2000, 1, 1),
        time_max=datetime(2002, 1, 1),
        allocation_option=SimulationAllocationOptions.riv,
        distributing_option=SimulationDistributingOptions.riv,
        regridder_types=None,
    )

    write_context = WriteContext(simulation_directory=tmp_path)
    riv._write("riv", globaltimes, write_context)
    drn._write("drn", globaltimes, write_context)

    errors = riv._validate(
        imod.mf6.River._write_schemata,
        idomain=target_dis.dataset["idomain"],
        bottom=target_dis.dataset["bottom"],
    )
    assert len(errors) == 0

    errors = drn._validate(
        imod.mf6.Drainage._write_schemata,
        idomain=target_dis.dataset["idomain"],
        bottom=target_dis.dataset["bottom"],
    )
    assert len(errors) == 0


def test_import_river_from_imod5__negative_layer(imod5_dataset, tmp_path):
    # Arrange
    imod5_data = imod5_dataset[0]
    period_data = imod5_dataset[1]
    globaltimes = [np.datetime64("2000-01-01")]
    target_dis = StructuredDiscretization.from_imod5_data(imod5_data)
    grid = target_dis.dataset["idomain"]
    target_npf = NodePropertyFlow.from_imod5_data(imod5_data, grid)

    # Gather reference packages (for negative layers, allocation option
    # "at_first_active" should be taken)
    (riv_reference, drn_reference) = imod.mf6.River.from_imod5_data(
        "riv-1",
        imod5_data,
        period_data,
        target_dis,
        target_npf,
        time_min=datetime(2000, 1, 1),
        time_max=datetime(2002, 1, 1),
        allocation_option=ALLOCATION_OPTION.at_first_active,
        distributing_option=DISTRIBUTING_OPTION.by_crosscut_thickness,
        regridder_types=None,
    )

    # Set layer to -1
    original_riv_1 = deepcopy(imod5_data["riv-1"])
    imod5_data["riv-1"] = {
        key: da.assign_coords(layer=[-1]) for key, da in imod5_data["riv-1"].items()
    }

    (riv, drn) = imod.mf6.River.from_imod5_data(
        "riv-1",
        imod5_data,
        period_data,
        target_dis,
        target_npf,
        time_min=datetime(2000, 1, 1),
        time_max=datetime(2002, 1, 1),
        allocation_option=ALLOCATION_OPTION.at_elevation,
        distributing_option=DISTRIBUTING_OPTION.by_crosscut_thickness,
        regridder_types=None,
    )

    write_context = WriteContext(simulation_directory=tmp_path)
    riv._write("riv", globaltimes, write_context)
    drn._write("drn", globaltimes, write_context)

    # Assert
    # Test if arrangement is correctly set up
    assert is_planar_grid(imod5_data["riv-1"]["conductance"])
    assert has_negative_layer(imod5_data["riv-1"]["conductance"])

    errors = riv._validate(
        imod.mf6.River._write_schemata,
        idomain=target_dis.dataset["idomain"],
        bottom=target_dis.dataset["bottom"],
    )
    assert len(errors) == 0
    errors = drn._validate(
        imod.mf6.Drainage._write_schemata,
        idomain=target_dis.dataset["idomain"],
        bottom=target_dis.dataset["bottom"],
    )
    assert len(errors) == 0

    assert riv.dataset.identical(riv_reference.dataset)
    assert drn.dataset.identical(drn_reference.dataset)

    # teardown
    imod5_data["riv-1"] = original_riv_1


def test_import_river_from_imod5__infiltration_factors(imod5_dataset):
    imod5_data = imod5_dataset[0]
    period_data = imod5_dataset[1]
    target_dis = StructuredDiscretization.from_imod5_data(imod5_data)
    grid = target_dis.dataset["idomain"]
    target_npf = NodePropertyFlow.from_imod5_data(imod5_data, grid)

    original_infiltration_factor = imod5_data["riv-1"]["infiltration_factor"]
    imod5_data["riv-1"]["infiltration_factor"] = ones_like(original_infiltration_factor)

    (riv, drn) = imod.mf6.River.from_imod5_data(
        "riv-1",
        imod5_data,
        period_data,
        target_dis,
        target_npf,
        time_min=datetime(2000, 1, 1),
        time_max=datetime(2002, 1, 1),
        allocation_option=ALLOCATION_OPTION.at_elevation,
        distributing_option=DISTRIBUTING_OPTION.by_crosscut_thickness,
        regridder_types=None,
    )

    assert riv is not None
    assert drn is None

    imod5_data["riv-1"]["infiltration_factor"] = zeros_like(
        original_infiltration_factor
    )
    (riv, drn) = imod.mf6.River.from_imod5_data(
        "riv-1",
        imod5_data,
        period_data,
        target_dis,
        target_npf,
        time_min=datetime(2000, 1, 1),
        time_max=datetime(2002, 1, 1),
        allocation_option=ALLOCATION_OPTION.at_elevation,
        distributing_option=DISTRIBUTING_OPTION.by_crosscut_thickness,
        regridder_types=None,
    )

    assert riv is None
    assert drn is not None

    # teardown
    imod5_data["riv-1"]["infiltration_factor"] = original_infiltration_factor


def test_import_river_from_imod5__constant(imod5_dataset):
    """Test importing river with a constant infiltration factor."""
    imod5_data = imod5_dataset[0]
    period_data = imod5_dataset[1]
    target_dis = StructuredDiscretization.from_imod5_data(imod5_data)
    grid = target_dis.dataset["idomain"]
    target_npf = NodePropertyFlow.from_imod5_data(imod5_data, grid)

    original_infiltration_factor = imod5_data["riv-1"]["infiltration_factor"]
    layer = original_infiltration_factor.coords["layer"]
    imod5_data["riv-1"]["infiltration_factor"] = xr.DataArray(
        [1.0], coords={"layer": layer}, dims=("layer",)
    )

    (riv, drn) = imod.mf6.River.from_imod5_data(
        "riv-1",
        imod5_data,
        period_data,
        target_dis,
        target_npf,
        time_min=datetime(2000, 1, 1),
        time_max=datetime(2002, 1, 1),
        allocation_option=ALLOCATION_OPTION.at_elevation,
        distributing_option=DISTRIBUTING_OPTION.by_crosscut_thickness,
        regridder_types=None,
    )

    assert riv is not None
    assert drn is None

    # teardown
    imod5_data["riv-1"]["infiltration_factor"] = original_infiltration_factor


def test_import_river_from_imod5__period_data(imod5_dataset_periods, tmp_path):
    imod5_data = imod5_dataset_periods[0]
    imod5_periods = imod5_dataset_periods[1]
    globaltimes = [np.datetime64("2000-01-01"), np.datetime64("2001-01-01")]
    target_dis = StructuredDiscretization.from_imod5_data(imod5_data, validate=False)
    grid = target_dis.dataset["idomain"]
    target_npf = NodePropertyFlow.from_imod5_data(imod5_data, grid)

    original_infiltration_factor = imod5_data["riv-1"]["infiltration_factor"]
    imod5_data["riv-1"]["infiltration_factor"] = ones_like(original_infiltration_factor)

    (riv, drn) = imod.mf6.River.from_imod5_data(
        "riv-1",
        imod5_data,
        imod5_periods,
        target_dis,
        target_npf,
        datetime(2002, 2, 2),
        datetime(2022, 2, 2),
        ALLOCATION_OPTION.stage_to_riv_bot_drn_above,
        SimulationDistributingOptions.riv,
        regridder_types=None,
    )

    assert riv is not None
    assert drn is not None

    errors = riv._validate(
        imod.mf6.River._write_schemata,
        idomain=target_dis.dataset["idomain"],
        bottom=target_dis.dataset["bottom"],
    )
    assert len(errors) == 0

    errors = drn._validate(
        imod.mf6.Drainage._write_schemata,
        idomain=target_dis.dataset["idomain"],
        bottom=target_dis.dataset["bottom"],
    )
    assert len(errors) == 0

    riv_time = riv.dataset.coords["time"].data
    drn_time = drn.dataset.coords["time"].data
    expected_times = np.array(
        [
            np.datetime64("2002-02-02"),
            np.datetime64("2002-04-01"),
            np.datetime64("2002-10-01"),
        ]
    )
    np.testing.assert_array_equal(riv_time, expected_times)
    np.testing.assert_array_equal(drn_time, expected_times)

    riv_repeat_stress = riv.dataset["repeat_stress"].data
    drn_repeat_stress = drn.dataset["repeat_stress"].data
    assert np.all(riv_repeat_stress[:, 1][::2] == np.datetime64("2002-04-01"))
    assert np.all(riv_repeat_stress[:, 1][1::2] == np.datetime64("2002-10-01"))
    assert np.all(drn_repeat_stress[:, 1][::2] == np.datetime64("2002-04-01"))
    assert np.all(drn_repeat_stress[:, 1][1::2] == np.datetime64("2002-10-01"))

    write_context = WriteContext(simulation_directory=tmp_path)
    riv._write("riv", globaltimes, write_context)
    drn._write("drn", globaltimes, write_context)


def test_import_river_from_imod5__transient_data(imod5_dataset_transient):
    """
    Test if importing a river from an IMOD5 dataset with transient data works
    correctly and that the time data is clipped to the specified time range.
    """
    imod5_data = imod5_dataset_transient[0]
    imod5_periods = imod5_dataset_transient[1]
    target_dis = StructuredDiscretization.from_imod5_data(imod5_data, validate=False)
    grid = target_dis.dataset["idomain"]
    target_npf = NodePropertyFlow.from_imod5_data(imod5_data, grid)

    original_infiltration_factor = imod5_data["riv-1"]["infiltration_factor"]
    imod5_data["riv-1"]["infiltration_factor"] = ones_like(original_infiltration_factor)

    (riv, drn) = imod.mf6.River.from_imod5_data(
        "riv-1",
        imod5_data,
        imod5_periods,
        target_dis,
        target_npf,
        datetime(2000, 4, 1),
        datetime(2010, 1, 1),
        ALLOCATION_OPTION.stage_to_riv_bot_drn_above,
        SimulationDistributingOptions.riv,
        regridder_types=None,
    )

    assert riv is not None
    assert drn is not None

    riv_time = riv.dataset.coords["time"].data
    drn_time = drn.dataset.coords["time"].data
    assert riv_time[0] == np.datetime64("2000-04-01")
    assert riv_time[-1] == np.datetime64("2003-01-01")
    assert drn_time[0] == np.datetime64("2000-04-01")
    assert drn_time[-1] == np.datetime64("2003-01-01")
