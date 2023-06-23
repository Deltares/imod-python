import pathlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import imod
from imod.tests.fixtures.mf6_regridding_fixture import (
    grid_data_structured,
    grid_data_structured_layered,
)


@pytest.fixture(scope="function")
def twri_simulation(transient_twri_model):
    # remove old-style well
    transient_twri_model["GWF_1"].pop("wel", None)
    return transient_twri_model


def test_write_well(tmp_path: Path):
    well = imod.mf6.Well(
        screen_top=[0.0, 0.0, 0.0],
        screen_bottom=[-1, -3.0, -5.0],
        x=[1.0, 3.0, 6.0],
        y=[3.0, 3.0, 3.0],
        rate=[1.0, 3.0, 5.0],
        print_flows=True,
        validate=True,
    )
    globaltimes = [np.datetime64("2000-01-01")]
    active = grid_data_structured(int, 1, 10)
    k = 100.0
    top = xr.ones_like(active.sel(layer=1), dtype=np.float64)
    bottom = grid_data_structured_layered(np.float64, -2.0, 10)

    well.write(
        tmp_path, "packagename", globaltimes, False, True, active, top, bottom, k
    )
    assert pathlib.Path.exists(tmp_path / "packagename.wel")
    assert pathlib.Path.exists(tmp_path / "packagename" / "wel.dat")
    df = pd.read_csv(
        tmp_path / "packagename" / "wel.dat", sep=":-\\s*", engine="python"
    )

    reference_output = np.array(
        [["1 2 1 1"], ["1 2 1 2"], ["1 2 2 2"], ["2 2 1 1"], ["2 2 2 2"], ["3 2 2 1"]],
        dtype=object,
    )
    assert (df.values == reference_output).all()


def test_write_well_from_model(
    tmp_path: Path, twri_simulation: imod.mf6.Modflow6Simulation
):
    twri_simulation["GWF_1"]["well"] = imod.mf6.Well(
        screen_top=[0.0, 0.0],
        screen_bottom=[-10.0, -10.0],
        x=[1.0, 6002.0],
        y=[3.0, 5004.0],
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


def test_write_all_wells_filtered_out(
    tmp_path: Path, twri_simulation: imod.mf6.Modflow6Simulation
):
    # for this test, we leave the low conductivity of the twri model as is, so all wells get filtered out
    twri_simulation["GWF_1"]["well"] = imod.mf6.Well(
        screen_top=[0.0, 0.0],
        screen_bottom=[-10.0, -10.0],
        x=[1.0, 6002.0],
        y=[3.0, 5004.0],
        rate=[1.0, 3.0],
        print_flows=True,
        validate=True,
    )

    with pytest.raises(ValueError):
        twri_simulation.write(tmp_path, binary=False)


def test_write_one_well_filtered_out(
    tmp_path: Path, twri_simulation: imod.mf6.Modflow6Simulation
):
    # for this test, we increase the low conductivity of the twri model . But one of the wells violates the thickness constraint (the second one)
    # and gets filtered out alltogether
    twri_simulation["GWF_1"]["npf"]["k"] *= 20000
    twri_simulation["GWF_1"]["well"] = imod.mf6.Well(
        screen_top=[0.0, 0.0],
        screen_bottom=[-10.0, -0.01],
        x=[1.0, 6002.0],
        y=[3.0, 5004.0],
        rate=[1.0, 3.0],
        print_flows=True,
        validate=True,
    )

    with pytest.raises(ValueError):
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
    twri_simulation["GWF_1"]["well"] = imod.mf6.Well(
        screen_top=[0.0, 0.0],
        screen_bottom=[-400.0, -400],
        x=[1.0, 6002.0],
        y=[3.0, 5004.0],
        rate=[1.0, 3.0],
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
