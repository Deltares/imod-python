import pathlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pytest_cases import parametrize_with_cases

import imod
import imod.tests
import imod.tests.fixtures
import imod.tests.fixtures.mf6_circle_fixture
import imod.tests.fixtures.mf6_twri_fixture
from imod.mf6.validation_settings import ValidationSettings
from imod.mf6.write_context import WriteContext
from imod.schemata import ValidationError
from imod.tests.fixtures.mf6_small_models_fixture import (
    grid_data_structured,
    grid_data_structured_layered,
    grid_data_unstructured,
    grid_data_unstructured_layered,
)
from imod.typing.grid import ones_like


@pytest.fixture(scope="function")
def twri_simulation(transient_twri_model):
    # remove old-style well
    transient_twri_model["GWF_1"].pop("wel", None)
    return transient_twri_model


class WriteWell:
    @staticmethod
    def case_write_wel_structured():
        expected_output = np.array(
            [
                # [layer, yind,, xind, rate]
                ["1 2 1 1"],
                ["1 2 1 2"],
                ["1 2 2 2"],
                ["2 2 1 1"],
                ["2 2 2 2"],
                ["3 2 2 1"],
            ]
        )
        return grid_data_structured, grid_data_structured_layered, expected_output

    @staticmethod
    def case_write_wel_unstructured():
        expected_output = np.array(
            [
                # [layer, faceid, rate]
                ["1 3 1"],
                ["1 3 2"],
                ["1 4 2"],
                ["2 3 1"],
                ["2 4 2"],
                ["3 4 1"],
            ]
        )
        return grid_data_unstructured, grid_data_unstructured_layered, expected_output


@parametrize_with_cases(
    "grid_data, grid_data_layered, reference_output", cases=WriteWell
)
def test_write_well(tmp_path: Path, grid_data, grid_data_layered, reference_output):
    well = imod.mf6.Well(
        x=[1.0, 3.0, 6.0],
        y=[3.0, 3.0, 3.0],
        screen_top=[0.0, 0.0, 0.0],
        screen_bottom=[-1, -3.0, -5.0],
        rate=[1.0, 3.0, 5.0],
        print_flows=True,
        validate=True,
    )
    globaltimes = [np.datetime64("2000-01-01")]
    active = grid_data(int, 1, 10)
    k = 100.0
    top = ones_like(active.sel(layer=1), dtype=np.float64)
    bottom = grid_data_layered(np.float64, -2.0, 10)
    validation_context = ValidationSettings(False)
    write_context = WriteContext(tmp_path)
    mf6_pkg = well._to_mf6_pkg(active, top, bottom, k, validation_context)
    mf6_pkg._write("packagename", globaltimes, write_context)
    assert pathlib.Path.exists(tmp_path / "packagename.wel")
    assert pathlib.Path.exists(tmp_path / "packagename" / "wel.dat")
    df = pd.read_csv(
        tmp_path / "packagename" / "wel.dat", sep=":-\\s*", engine="python"
    )

    assert (df.values == reference_output).all()


def test_write_well_from_model(
    tmp_path: Path, twri_simulation: imod.mf6.Modflow6Simulation
):
    twri_simulation["GWF_1"]["well"] = imod.mf6.Well(
        x=[1.0, 6002.0],
        y=[3.0, 5004.0],
        screen_top=[0.0, 0.0],
        screen_bottom=[-10.0, -10.0],
        rate=[1.0, 3.0],
        print_flows=True,
        validate=True,
    )

    # for this test, we increase the hydraulic conductivities a bit, because otherwise the wells will be filtered out as belonging to too impermeable layers
    twri_simulation["GWF_1"]["npf"]["k"] *= 20000
    twri_simulation["GWF_1"]["npf"]["k33"] *= 20000

    twri_simulation.write(tmp_path, binary=False)
    assert pathlib.Path.exists(tmp_path / "GWF_1" / "well.wel")
    assert pathlib.Path.exists(tmp_path / "GWF_1" / "well" / "wel.dat")
    assert twri_simulation.run() is None


def test_write_well_from_model_transient_rate(
    tmp_path: Path, twri_simulation: imod.mf6.Modflow6Simulation
):
    times = twri_simulation["time_discretization"]["time"]
    rate = xr.DataArray(dims=("index", "time"), coords={"index": [0, 1], "time": times})
    rate.sel(index=0).values[:] = 5.0
    rate.sel(index=1).values[:] = 4.0
    twri_simulation["GWF_1"]["well"] = imod.mf6.Well(
        x=[1.0, 6002.0],
        y=[3.0, 5004.0],
        screen_top=[0.0, 0.0],
        screen_bottom=[-10.0, -10.0],
        id=[103, -101],
        rate=rate,
        print_flows=True,
        validate=True,
    )

    # for this test, we increase the hydraulic conductivities a bit, because otherwise the wells will be filtered out as belonging to too impermeable layers
    twri_simulation["GWF_1"]["npf"]["k"] *= 20000
    twri_simulation["GWF_1"]["npf"]["k33"] *= 20000

    twri_simulation.write(tmp_path, binary=False)
    assert pathlib.Path.exists(tmp_path / "GWF_1" / "well.wel")
    for i in range(0, len(times)):
        file = Path(f"{tmp_path}/GWF_1/well/wel-{i}.dat")
        assert pathlib.Path.exists(file)
    assert twri_simulation.run() is None


def test_write_all_wells_filtered_out(
    tmp_path: Path, twri_simulation: imod.mf6.Modflow6Simulation
):
    # for this test, we leave the low conductivity of the twri model as is, so
    # all wells get filtered out
    # set minimum_k and minimum_thickness to force filtering
    twri_simulation["GWF_1"]["well"] = imod.mf6.Well(
        x=[1.0, 6002.0],
        y=[3.0, 5004.0],
        screen_top=[0.0, 0.0],
        screen_bottom=[-10.0, -10.0],
        rate=[1.0, 3.0],
        minimum_k=0.1,
        minimum_thickness=0.05,
        print_flows=True,
        validate=True,
    )

    with pytest.raises(ValidationError):
        twri_simulation.write(tmp_path, binary=False)


def test_write_one_well_filtered_out(
    tmp_path: Path, twri_simulation: imod.mf6.Modflow6Simulation
):
    # for this test, we increase the low conductivity of the twri model . But
    # one of the wells violates the thickness constraint (the second one) and
    # gets filtered out alltogether
    twri_simulation["GWF_1"]["npf"]["k"] *= 20000
    # set minimum_k and minimum_thickness to force filtering
    twri_simulation["GWF_1"]["well"] = imod.mf6.Well(
        x=[1.0, 6002.0],
        y=[3.0, 5004.0],
        screen_top=[0.0, 0.0],
        screen_bottom=[-10.0, -0.01],
        rate=[1.0, 3.0],
        minimum_k=0.1,
        minimum_thickness=0.05,
        print_flows=True,
        validate=True,
    )

    with pytest.raises(ValidationError):
        twri_simulation.write(tmp_path, binary=False)


def test_write_one_layer_filtered_out(
    tmp_path: Path, twri_simulation: imod.mf6.Modflow6Simulation
):
    # for this test, we assign a high conductivity to layer 1 and 3 and a low one to layer 2
    k = twri_simulation["GWF_1"]["npf"]["k"]
    k.loc[{"layer": 1}] = 12.0
    k.loc[{"layer": 2}] = 1e-10
    k.loc[{"layer": 3}] = 10.0

    # define 2 wells penetrating all layers
    # set minimum_k and minimum_thickness to force filtering
    twri_simulation["GWF_1"]["well"] = imod.mf6.Well(
        x=[1.0, 6002.0],
        y=[3.0, 5004.0],
        screen_top=[0.0, 0.0],
        screen_bottom=[-400.0, -400],
        rate=[1.0, 3.0],
        minimum_k=0.1,
        minimum_thickness=0.05,
        print_flows=True,
        validate=True,
    )

    twri_simulation.write(tmp_path, binary=False)
    assert pathlib.Path.exists(tmp_path / "GWF_1" / "well.wel")
    assert pathlib.Path.exists(tmp_path / "GWF_1" / "well" / "wel.dat")

    # we should have 4 active sections left
    df = pd.read_csv(
        tmp_path / "GWF_1" / "well" / "wel.dat", sep=":-\\s*", engine="python"
    )
    assert len(df) == 4


def test_constraints_are_configurable(
    tmp_path: Path, twri_simulation: imod.mf6.Modflow6Simulation
):
    # define 2 wells penetrating all layers
    twri_simulation["GWF_1"]["well"] = imod.mf6.Well(
        x=[1.0, 6002.0],
        y=[3.0, 5004.0],
        screen_top=[0.0, 0.0],
        screen_bottom=[-400.0, -400],
        rate=[1.0, 3.0],
        print_flows=True,
        validate=True,
    )

    # reset the constraints so that k-constraints are violated
    twri_simulation["GWF_1"]["well"]["minimum_k"] = 1.0
    twri_simulation["GWF_1"]["well"]["minimum_thickness"] = 1.0

    with pytest.raises(ValidationError):
        twri_simulation.write(tmp_path, binary=False)

    # reset the constraints again so that all constraints are met
    twri_simulation["GWF_1"]["well"]["minimum_k"] = 1e-9
    twri_simulation["GWF_1"]["well"]["minimum_thickness"] = 1.0

    twri_simulation.write(tmp_path, binary=False)

    # reset the constraints so that layer_thickness constraints are violated
    twri_simulation["GWF_1"]["well"]["minimum_k"] = 1e-9
    twri_simulation["GWF_1"]["well"]["minimum_thickness"] = 700.0

    with pytest.raises(ValidationError):
        twri_simulation.write(tmp_path, binary=False)


def test_non_unique_ids(twri_simulation: imod.mf6.Modflow6Simulation):
    times = twri_simulation["time_discretization"]["time"]
    rate = xr.DataArray(dims=("index", "time"), coords={"index": [0, 1], "time": times})
    rate.sel(index=0).values[:] = 5.0
    rate.sel(index=1).values[:] = 4.0
    with pytest.raises(ValueError):
        twri_simulation["GWF_1"]["well"] = imod.mf6.Well(
            x=[1.0, 6002.0],
            y=[3.0, 5004.0],
            screen_top=[0.0, 0.0],
            screen_bottom=[-10.0, -10.0],
            id=[103, 103],
            rate=rate,
            print_flows=True,
            validate=True,
        )


@pytest.mark.parametrize("fixture_name", ["twri_model", "circle_model"])
def test_error_message_wells_outside_grid(tmp_path: Path, fixture_name: str, request):
    simulation = request.getfixturevalue(fixture_name)

    # define wells inside the domain, and also  one outside
    in_domain_wells = {0: {"x": 1.0, "y": 2.0}, 1: {"x": 4.0, "y": 5.0}}
    out_of_domain_well = {"x": 500000, "y": 600000}
    simulation["GWF_1"]["well"] = imod.mf6.Well(
        x=[in_domain_wells[0]["x"], out_of_domain_well["x"], in_domain_wells[1]["x"]],
        y=[in_domain_wells[0]["y"], out_of_domain_well["y"], in_domain_wells[1]["y"]],
        screen_top=[0.0, 0.0, 0],
        screen_bottom=[-1400.0, -1300, -400],
        rate=[1.0, 2.0, 3.0],
        print_flows=True,
        validate=True,
    )

    asserted = False
    try:
        simulation.write(tmp_path, binary=False)
    except Exception as e:
        asserted = True

        # ensure the coordinates of the first offending well are present in the error message
        assert str(out_of_domain_well["x"]) in str(e)
        assert str(out_of_domain_well["y"]) in str(e)

    # ensure a problem was detected
    assert asserted
