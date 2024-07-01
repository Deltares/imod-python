import pathlib
import re
import tempfile
import textwrap
from datetime import datetime

import numpy as np
import pytest
import xarray as xr
import xugrid as xu
from pytest_cases import parametrize_with_cases

import imod
from imod.formats.prj.prj import open_projectfile_data
from imod.mf6.dis import StructuredDiscretization
from imod.mf6.write_context import WriteContext
from imod.prepare.topsystem.allocation import ALLOCATION_OPTION
from imod.prepare.topsystem.conductance import DISTRIBUTING_OPTION
from imod.schemata import ValidationError
from imod.typing.grid import ones_like, zeros_like


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
        coords={"layer": layer, "y": y, "x": x, "dx": dx, "dy": dy},
    )


@pytest.fixture(scope="function")
def dis_dict(make_da):
    da = make_da
    bottom = da - xr.DataArray(
        data=[1.5, 2.5], dims=("layer",), coords={"layer": [2, 3]}
    )

    return {"idomain": da.astype(int), "top": da.sel(layer=2), "bottom": bottom}


@pytest.fixture(scope="function")
def riv_dict(make_da):
    da = make_da
    da[:, 1, 1] = np.nan

    bottom = da - xr.DataArray(
        data=[1.0, 2.0], dims=("layer",), coords={"layer": [2, 3]}
    )

    return {"stage": da, "conductance": da, "bottom_elevation": bottom}


def make_dict_unstructured(d):
    return {key: xu.UgridDataArray.from_structured(value) for key, value in d.items()}


class RivCases:
    def case_structured(self, riv_dict):
        return riv_dict

    def case_unstructured(self, riv_dict):
        return make_dict_unstructured(riv_dict)


class RivDisCases:
    def case_structured(self, riv_dict, dis_dict):
        return riv_dict, dis_dict

    def case_unstructured(self, riv_dict, dis_dict):
        return make_dict_unstructured(riv_dict), make_dict_unstructured(dis_dict)


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

    for var, var_errors in errors.items():
        assert var == "stage"


@parametrize_with_cases("riv_data,dis_data", cases=RivDisCases)
def test_inconsistent_nan(riv_data, dis_data):
    riv_data["stage"][..., 2] = np.nan
    river = imod.mf6.River(**riv_data)

    errors = river._validate(river._write_schemata, **dis_data)

    assert len(errors) == 1


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
        * stage
        \t- coords has missing keys: {'layer'}"""
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
        * stage
        \t- provided dimension layer with size 0
        * conductance
        \t- provided dimension layer with size 0
        * bottom_elevation
        \t- provided dimension layer with size 0"""
    )

    with pytest.raises(ValidationError, match=re.escape(message)):
        imod.mf6.River(stage=da, conductance=da, bottom_elevation=da - 1.0)


@parametrize_with_cases("riv_data,dis_data", cases=RivDisCases)
def test_check_zero_conductance(riv_data, dis_data):
    """
    Test for zero conductance
    """
    riv_data["conductance"] = riv_data["conductance"] * 0.0

    river = imod.mf6.River(**riv_data)

    errors = river._validate(river._write_schemata, **dis_data)

    assert len(errors) == 1
    for var, var_errors in errors.items():
        assert var == "conductance"


@parametrize_with_cases("riv_data,dis_data", cases=RivDisCases)
def test_check_bottom_above_stage(riv_data, dis_data):
    """
    Check that river bottom is not above stage.
    """

    riv_data["bottom_elevation"] = riv_data["bottom_elevation"] + 10.0

    river = imod.mf6.River(**riv_data)

    errors = river._validate(river._write_schemata, **dis_data)

    assert len(errors) == 1
    for var, var_errors in errors.items():
        assert var == "stage"


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


def test_validate_false(riv_dict):
    """
    Test turning off validation
    """

    riv_ds = xr.merge([riv_dict])

    imod.mf6.River(validate=False, **riv_ds.sel(layer=slice(None, None, -1)))


@pytest.mark.usefixtures("concentration_fc")
def test_render_concentration(riv_dict, concentration_fc):
    riv_ds = xr.merge([riv_dict])

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
        write_context = WriteContext(simulation_directory=output_dir)
        riv.write("riv", globaltimes, write_context)
        with open(output_dir + "/riv/riv-0.dat", "r") as f:
            data = f.read()
            assert (
                data.count("2") == 1755
            )  # the number 2 is in the concentration data, and in the cell indices.


@pytest.mark.usefixtures("imod5_dataset")
def test_import_river_from_imod5(imod5_dataset, tmp_path):
    globaltimes = [np.datetime64("2000-01-01")]
    target_dis = StructuredDiscretization.from_imod5_data(imod5_dataset)

    (riv, drn) = imod.mf6.River.from_imod5_data(
        "riv-1",
        imod5_dataset,
        target_dis,
        ALLOCATION_OPTION.at_elevation,
        DISTRIBUTING_OPTION.by_crosscut_thickness,
        regridder_types=None,
    )

    write_context = WriteContext(simulation_directory=tmp_path)
    riv.write("riv", globaltimes, write_context)
    drn.write("drn", globaltimes, write_context)

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


@pytest.mark.usefixtures("imod5_dataset")
def test_import_river_from_imod5_infiltration_factors(imod5_dataset):
    target_dis = StructuredDiscretization.from_imod5_data(imod5_dataset)

    original_infiltration_factor = imod5_dataset["riv-1"]["infiltration_factor"]
    imod5_dataset["riv-1"]["infiltration_factor"] = ones_like(
        original_infiltration_factor
    )

    (riv, drn) = imod.mf6.River.from_imod5_data(
        "riv-1",
        imod5_dataset,
        target_dis,
        ALLOCATION_OPTION.at_elevation,
        DISTRIBUTING_OPTION.by_crosscut_thickness,
        regridder_types=None,
    )

    assert riv is not None
    assert drn is None

    imod5_dataset["riv-1"]["infiltration_factor"] = zeros_like(
        original_infiltration_factor
    )
    (riv, drn) = imod.mf6.River.from_imod5_data(
        "riv-1",
        imod5_dataset,
        target_dis,
        ALLOCATION_OPTION.at_elevation,
        DISTRIBUTING_OPTION.by_crosscut_thickness,
        regridder_types=None,
    )

    assert riv is None
    assert drn is not None

    # teardown
    imod5_dataset["riv-1"]["infiltration_factor"] = original_infiltration_factor


def test_import_river_from_imod5_period_data():
    testdir = (
        "D:\\dev\\imod_python-gh\\imod-python\\imod\\tests\\imod5_data\\iMOD5_model.prj"
    )
    imod5_dataset = open_projectfile_data(testdir)
    imod5_data = imod5_dataset[0]
    imod5_periods = imod5_dataset[1]
    target_dis = StructuredDiscretization.from_imod5_data(imod5_data, validate=False)

    original_infiltration_factor = imod5_data["riv-1"]["infiltration_factor"]
    imod5_data["riv-1"]["infiltration_factor"] = ones_like(original_infiltration_factor)

    (riv, drn) = imod.mf6.River.from_imod5_data(
        "riv-1",
        imod5_data,
        imod5_periods,
        target_dis,
        datetime(2002, 2, 2),
        datetime(2022, 2, 2),
        ALLOCATION_OPTION.at_elevation,
        DISTRIBUTING_OPTION.by_crosscut_thickness,
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
        imod5_periods,
        target_dis,
        datetime(2002, 2, 2),
        datetime(2022, 2, 2),
        ALLOCATION_OPTION.at_elevation,
        DISTRIBUTING_OPTION.by_crosscut_thickness,
        regridder_types=None,
    )

    assert riv is None
    assert drn is not None

    # teardown
    imod5_dataset[0]["riv-1"]["infiltration_factor"] = original_infiltration_factor
