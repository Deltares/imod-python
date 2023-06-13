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
    active = grid_data_structured(np.int, 1, 10)
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

    # TODO: The following output is wrong, but it is what the write function currently generates.
    # issue gitlab  #432 was created to fix this.
    reference_output = np.array(
        [["1 2 1 2"], ["1 2 1 2"], ["1 2 1 2"], ["1 2 1 2"], ["1 2 1 2"], ["1 2 1 2"]],
        dtype=object,
    )
    assert (df.values == reference_output).all()


def test_render__well_from_model(
    tmp_path: Path, transient_twri_model: imod.mf6.Modflow6Simulation
):
    transient_twri_model["GWF_1"]["well"] = imod.mf6.Well(
        screen_top=[0.0, 0.0],
        screen_bottom=[-10.0, -10.0],
        x=[1.0, 6002.0],
        y=[3.0, 5004.0],
        rate=[1.0, 3.0],
        print_flows=True,
        validate=True,
    )
    # remove old-style well
    transient_twri_model["GWF_1"].pop("wel", None)
    # for this test, we increase the hydraulic conductivities a bit, because otherwise the wells will be filtered out as belonging to too impermeable layers
    transient_twri_model["GWF_1"]["npf"]["k"] *= 20000
    transient_twri_model["GWF_1"]["npf"]["k33"] *= 20000

    transient_twri_model.write(tmp_path, binary=False)
    assert pathlib.Path.exists(tmp_path / "GWF_1" / "well.wel")
    assert pathlib.Path.exists(tmp_path / "GWF_1" / "well" / "wel.dat")
    assert transient_twri_model.run() is None


def test_render__wells_filtered_out(
    tmp_path: Path, transient_twri_model: imod.mf6.Modflow6Simulation
):
    # for this test, we leave the low conductivity of the twri model as is, so all wells get filtered out

    transient_twri_model["GWF_1"]["well"] = imod.mf6.Well(
        screen_top=[0.0, 0.0],
        screen_bottom=[-10.0, -10.0],
        x=[1.0, 6002.0],
        y=[3.0, 5004.0],
        rate=[1.0, 3.0],
        print_flows=True,
        validate=True,
    )
    # remove old-style well
    transient_twri_model["GWF_1"].pop("wel", None)

    with pytest.raises(ValueError):
        transient_twri_model.write(tmp_path, binary=False)
