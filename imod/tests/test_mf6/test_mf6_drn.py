import pathlib
import textwrap
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pytest_cases import parametrize_with_cases

import imod
import imod.mf6.drn
from imod.logging import LoggerType, LogLevel
from imod.mf6.dis import StructuredDiscretization
from imod.mf6.npf import NodePropertyFlow
from imod.mf6.utilities.package import get_repeat_stress
from imod.mf6.write_context import WriteContext
from imod.prepare.topsystem.allocation import ALLOCATION_OPTION
from imod.prepare.topsystem.conductance import DISTRIBUTING_OPTION
from imod.prepare.topsystem.default_allocation_methods import (
    SimulationAllocationOptions,
    SimulationDistributingOptions,
)
from imod.schemata import ValidationError


@pytest.fixture(scope="function")
def drainage():
    layer = np.arange(1, 4)
    y = np.arange(4.5, 0.0, -1.0)
    x = np.arange(0.5, 5.0, 1.0)
    elevation = xr.DataArray(
        np.full((3, 5, 5), 1.0),
        coords={"layer": layer, "y": y, "x": x, "dx": 1.0, "dy": -1.0},
        dims=("layer", "y", "x"),
    )
    conductance = elevation.copy()

    drn = {"elevation": elevation, "conductance": conductance}
    return drn


@pytest.fixture(scope="function")
def transient_drainage():
    layer = np.arange(1, 4)
    y = np.arange(4.5, 0.0, -1.0)
    x = np.arange(0.5, 5.0, 1.0)
    elevation = xr.DataArray(
        np.full((3, 5, 5), 1.0),
        coords={"layer": layer, "y": y, "x": x, "dx": 1.0, "dy": -1.0},
        dims=("layer", "y", "x"),
    )
    time_multiplier = xr.DataArray(
        data=np.arange(1.0, 7.0, 1.0),
        coords={"time": pd.date_range("2000-01-01", "2005-01-01", freq="YS")},
        dims=("time",),
    )
    conductance = time_multiplier * elevation

    drn = {"elevation": elevation, "conductance": conductance}
    return drn


@pytest.fixture(scope="function")
def transient_concentration_drainage():
    layer = np.arange(1, 4)
    y = np.arange(4.5, 0.0, -1.0)
    x = np.arange(0.5, 5.0, 1.0)
    elevation = xr.DataArray(
        np.full((3, 5, 5), 1.0),
        coords={"layer": layer, "y": y, "x": x, "dx": 1.0, "dy": -1.0},
        dims=("layer", "y", "x"),
    )
    time_multiplier = xr.DataArray(
        data=np.arange(1.0, 7.0, 1.0),
        coords={"time": pd.date_range("2000-01-01", "2005-01-01", freq="YS")},
        dims=("time",),
    )
    species_multiplier = xr.DataArray(
        data=[35.0, 1.0],
        coords={"species": ["salinity", "temperature"]},
        dims=("species",),
    )
    conductance = time_multiplier * elevation
    concentration = species_multiplier * conductance

    drn = {
        "elevation": elevation,
        "conductance": conductance,
        "concentration": concentration,
    }
    return drn


def test_write(drainage, tmp_path):
    imod.logging.configure(
        LoggerType.PYTHON,
        log_level=LogLevel.DEBUG,
        add_default_file_handler=True,
        add_default_stream_handler=False,
    )

    drn = imod.mf6.Drainage(**drainage)
    write_context = WriteContext(simulation_directory=tmp_path, use_binary=True)
    drn._write("mydrn", [1], write_context)

    block_expected = textwrap.dedent(
        """\
        begin options
        end options

        begin dimensions
          maxbound 75
        end dimensions

        begin period 1
          open/close mydrn/drn.bin (binary)
        end period
        """
    )

    with open(tmp_path / "mydrn.drn") as f:
        block = f.read()

    assert block == block_expected


def test_wrong_dtype(drainage):
    drainage["elevation"] = drainage["elevation"].astype(np.int32)

    with pytest.raises(ValidationError):
        imod.mf6.Drainage(**drainage)


def test_validate_false(drainage):
    drainage["elevation"] = drainage["elevation"].astype(np.int32)

    imod.mf6.Drainage(validate=False, **drainage)


def test_check_conductance_zero(drainage):
    drainage["conductance"] = drainage["conductance"] * 0.0

    idomain = drainage["elevation"].astype(np.int16)
    top = 1.0
    bottom = top - idomain.coords["layer"]

    dis = imod.mf6.StructuredDiscretization(top=top, bottom=bottom, idomain=idomain)
    drn = imod.mf6.Drainage(**drainage)
    errors = drn._validate(drn._write_schemata, **dis.dataset)
    assert len(errors) == 1
    for var, error in errors.items():
        assert var == "conductance"


@pytest.mark.parametrize("nodata_idomain", [0, -1])
def test_validate_inside_nodata(drainage, nodata_idomain):
    idomain = drainage["elevation"].astype(np.int16)
    top = 1.0
    bottom = top - idomain.coords["layer"]

    idomain[:, 2, 2] = nodata_idomain

    dis = imod.mf6.StructuredDiscretization(top=top, bottom=bottom, idomain=idomain)
    drn = imod.mf6.Drainage(**drainage)
    errors = drn._validate(drn._write_schemata, **dis.dataset)
    assert len(errors) == 1
    for var, error in errors.items():
        assert var == "elevation"


def test_validate_concentration(transient_concentration_drainage):
    idomain = transient_concentration_drainage["elevation"].astype(np.int16)
    top = 1.0
    bottom = top - idomain.coords["layer"]

    dis = imod.mf6.StructuredDiscretization(top=top, bottom=bottom, idomain=idomain)
    drn = imod.mf6.Drainage(**transient_concentration_drainage)

    # No errors at start
    errors = drn._validate(drn._write_schemata, **dis.dataset)
    assert len(errors) == 0

    # Error with incongruent data
    # Rivers are located everywhere in the grid.
    drn.dataset["concentration"][0, 2, 2] = np.nan
    errors = drn._validate(drn._write_schemata, **dis.dataset)
    assert len(errors) == 1
    for var, error in errors.items():
        assert var == "concentration"

    # Error with smaller than zero
    drn.dataset["concentration"] = idomain.where(
        False, -200.0
    )  # Set concentrations negative
    errors = drn._validate(drn._write_schemata, **dis.dataset)
    assert len(errors) == 1
    for var, error in errors.items():
        assert var == "concentration"


def test_discontinuous_layer(drainage):
    drn = imod.mf6.Drainage(**drainage)
    drn["layer"] = [1, 3, 5]
    bin_ds = drn[list(drn._period_data)]
    layer = bin_ds["layer"].values
    arrdict = drn._ds_to_arrdict(bin_ds)
    struct_array = drn._to_struct_array(arrdict, layer)
    assert np.array_equal(np.unique(struct_array["layer"]), [1, 3, 5])


def test_3d_singelayer():
    # Introduced because of Issue #224
    layer = [1]
    y = np.arange(4.5, 0.0, -1.0)
    x = np.arange(0.5, 5.0, 1.0)
    elevation = xr.DataArray(
        np.full((1, 5, 5), 1.0),
        coords={"layer": layer, "y": y, "x": x, "dx": 1.0, "dy": -1.0},
        dims=("layer", "y", "x"),
    )
    conductance = elevation.copy()
    drn = imod.mf6.Drainage(elevation=elevation, conductance=conductance)

    bin_ds = drn[list(drn._period_data)]
    layer = bin_ds["layer"].values
    arrdict = drn._ds_to_arrdict(bin_ds)
    struct_array = drn._to_struct_array(arrdict, layer)
    assert isinstance(struct_array, np.ndarray)


@pytest.mark.usefixtures("concentration_fc", "elevation_fc", "conductance_fc")
def test_render_concentration(
    concentration_fc,
    elevation_fc,
    conductance_fc,
):
    directory = pathlib.Path("mymodel")
    globaltimes = np.array(
        [
            "2000-01-01",
            "2000-01-02",
            "2000-01-03",
        ],
        dtype="datetime64[ns]",
    )

    drn = imod.mf6.Drainage(
        elevation=elevation_fc,
        conductance=conductance_fc,
        concentration=concentration_fc,
        concentration_boundary_type="AUX",
    )

    actual = drn.render(directory, "drn", globaltimes, False)

    expected = textwrap.dedent(
        """\
        begin options
          auxiliary salinity temperature
        end options

        begin dimensions
          maxbound 2
        end dimensions

        begin period 1
          open/close mymodel/drn/drn-0.dat
        end period
        begin period 2
          open/close mymodel/drn/drn-1.dat
        end period
        begin period 3
          open/close mymodel/drn/drn-2.dat
        end period
        """
    )
    assert actual == expected


@pytest.mark.usefixtures("elevation_fc", "conductance_fc")
def test_repeat_stress(
    elevation_fc,
    conductance_fc,
):
    directory = pathlib.Path("mymodel")
    globaltimes = np.array(
        [
            "2000-01-01",
            "2000-01-02",
            "2000-01-03",
            "2000-01-04",
            "2000-01-05",
        ],
        dtype="datetime64[ns]",
    )

    repeat_stress = xr.DataArray(
        [
            [globaltimes[3], globaltimes[0]],
            [globaltimes[4], globaltimes[1]],
        ],
        dims=("repeat", "repeat_items"),
    )

    expected = textwrap.dedent(
        """\
        begin options
        end options

        begin dimensions
          maxbound 2
        end dimensions

        begin period 1
          open/close mymodel/drn/drn-0.dat
        end period
        begin period 2
          open/close mymodel/drn/drn-1.dat
        end period
        begin period 3
          open/close mymodel/drn/drn-2.dat
        end period
        begin period 4
          open/close mymodel/drn/drn-0.dat
        end period
        begin period 5
          open/close mymodel/drn/drn-1.dat
        end period
        """
    )

    drn = imod.mf6.Drainage(
        elevation=elevation_fc,
        conductance=conductance_fc,
        repeat_stress=repeat_stress,
    )
    actual = drn.render(directory, "drn", globaltimes, False)
    assert actual == expected

    drn = imod.mf6.Drainage(
        elevation=elevation_fc,
        conductance=conductance_fc,
    )
    drn.dataset["repeat_stress"] = get_repeat_stress(
        times={
            globaltimes[3]: globaltimes[0],
            globaltimes[4]: globaltimes[1],
        },
    )
    actual = drn.render(directory, "drn", globaltimes, False)
    assert actual == expected


@pytest.mark.usefixtures("elevation_fc", "conductance_fc")
def test_repeat_stress_old_style(
    elevation_fc,
    conductance_fc,
):
    directory = pathlib.Path("mymodel")
    globaltimes = np.array(
        [
            "2000-01-01",
            "2000-01-02",
            "2000-01-03",
            "2000-01-04",
            "2000-01-05",
        ],
        dtype="datetime64[ns]",
    )

    expected = textwrap.dedent(
        """\
        begin options
        end options

        begin dimensions
          maxbound 2
        end dimensions

        begin period 1
          open/close mymodel/drn/drn-0.dat
        end period
        begin period 2
          open/close mymodel/drn/drn-1.dat
        end period
        begin period 3
          open/close mymodel/drn/drn-2.dat
        end period
        begin period 4
          open/close mymodel/drn/drn-0.dat
        end period
        begin period 5
          open/close mymodel/drn/drn-1.dat
        end period
        """
    )

    drn = imod.mf6.Drainage(
        elevation=elevation_fc,
        conductance=conductance_fc,
    )
    drn.set_repeat_stress(
        times={
            globaltimes[3]: globaltimes[0],
            globaltimes[4]: globaltimes[1],
        }
    )
    actual = drn.render(directory, "drn", globaltimes, False)
    assert actual == expected


def test_clip_box(drainage):
    drn = imod.mf6.Drainage(**drainage)

    selection = drn.clip_box()
    assert isinstance(selection, imod.mf6.Drainage)
    assert selection.dataset.identical(drn.dataset)

    selection = drn.clip_box(x_min=None, x_max=None)
    assert isinstance(selection, imod.mf6.Drainage)
    assert selection.dataset.identical(drn.dataset)

    selection = drn.clip_box(
        layer_min=1,
        layer_max=2,
        y_min=1.0,
        y_max=4.0,
        x_min=1.0,
        x_max=4.0,
    )
    assert isinstance(selection, imod.mf6.Drainage)
    assert selection["conductance"].dims == ("layer", "y", "x")
    assert selection["conductance"].shape == (2, 3, 3)


def test_clip_box_transient(transient_drainage):
    drn = imod.mf6.Drainage(**transient_drainage)

    # First test the standard case: clip into existing times.
    selection = drn.clip_box(time_min="2001-01-01", time_max="2004-01-01")
    expected = np.array(
        [
            "2001-01-01T00:00:00.000000000",
            "2002-01-01T00:00:00.000000000",
            "2003-01-01T00:00:00.000000000",
            "2004-01-01T00:00:00.000000000",
        ],
        dtype="datetime64[ns]",
    )
    assert isinstance(selection, imod.mf6.Drainage)
    assert selection["elevation"].dims == ("layer", "y", "x")
    assert selection["conductance"].dims == ("time", "layer", "y", "x")
    assert np.array_equal(selection.dataset["time"], expected)

    # Now test a succesfull forward fill.
    selection = drn.clip_box(time_min="2000-06-01", time_max="2002-06-01")
    expected = np.array(
        [
            "2000-06-01T00:00:00.000000000",
            "2001-01-01T00:00:00.000000000",
            "2002-01-01T00:00:00.000000000",
        ],
        dtype="datetime64[ns]",
    )
    assert np.array_equal(selection.dataset["time"], expected)
    assert (selection["conductance"].sel(time="2000-06-01") == 1.0).all()

    # And a backfill.
    selection = drn.clip_box(time_min="1990-06-01", time_max="2002-06-01")
    expected = np.array(
        [
            "1990-06-01T00:00:00.000000000",
            "2000-01-01T00:00:00.000000000",
            "2001-01-01T00:00:00.000000000",
            "2002-01-01T00:00:00.000000000",
        ],
        dtype="datetime64[ns]",
    )
    assert np.array_equal(selection.dataset["time"].values, expected)
    assert (selection["conductance"].sel(time="1990-06-01") == 1.0).all()
    assert (selection["conductance"].sel(time="2000-01-01") == 1.0).all()


def test_repr(drainage):
    repr_string = imod.mf6.Drainage(**drainage).__repr__()
    assert isinstance(repr_string, str)
    assert repr_string.split("\n")[0] == "Drainage"


def test_html_repr(drainage):
    html_string = imod.mf6.Drainage(**drainage)._repr_html_()
    assert isinstance(html_string, str)
    assert html_string.split("</div>")[0] == "<div>Drainage"


class AllocationSettings:
    def case_default(self):
        return SimulationAllocationOptions.drn, SimulationDistributingOptions.drn

    def case_custom(self):
        return ALLOCATION_OPTION.at_elevation, DISTRIBUTING_OPTION.by_crosscut_thickness


@parametrize_with_cases(
    ["allocation_setting", "distribution_setting"], cases=AllocationSettings
)
def test_from_imod5(
    imod5_dataset_periods, tmp_path, allocation_setting, distribution_setting
):
    period_data = imod5_dataset_periods[1]
    imod5_dataset = imod5_dataset_periods[0]
    target_dis = StructuredDiscretization.from_imod5_data(imod5_dataset, validate=False)
    target_npf = NodePropertyFlow.from_imod5_data(
        imod5_dataset, target_dis.dataset["idomain"]
    )

    drn_2 = imod.mf6.Drainage.from_imod5_data(
        "drn-2",
        imod5_dataset,
        period_data,
        target_dis,
        target_npf,
        allocation_option=allocation_setting,
        distributing_option=distribution_setting,
        time_min=datetime(2002, 2, 2),
        time_max=datetime(2022, 2, 2),
        regridder_types=None,
    )

    assert isinstance(drn_2, imod.mf6.Drainage)

    pkg_errors = drn_2._validate(
        schemata=drn_2._write_schemata,
        idomain=target_dis["idomain"],
        bottom=target_dis["bottom"],
    )
    assert len(pkg_errors) == 0

    # write the packages for write validation
    write_context = WriteContext(simulation_directory=tmp_path, use_binary=False)
    drn_2._write("mydrn", [1], write_context)
