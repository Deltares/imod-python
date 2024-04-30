import tempfile
from pathlib import Path

import pytest
import xarray as xr

from imod.mf6.utilities.regrid import (
    RegridderWeightsCache,
)
from imod.msw import (
    InitialConditionsEquilibrium,
    InitialConditionsPercolation,
    InitialConditionsRootzonePressureHead,
    InitialConditionsSavedState,
)
from imod.typing.grid import is_empty


def get_new_grid():
    x = [1.0, 1.5, 2.0, 2.5, 3.0]
    y = [3.0, 2.5, 2.0, 1.5, 1.0]
    subunit = [0, 1]
    dx = 0.5
    dy = 0.5
    # fmt: off
    new_grid = xr.DataArray(
        dims=("subunit", "y", "x"),
        coords={"subunit": subunit, "y": y, "x": x, "dx": dx, "dy": dy}
    )
    new_grid.values[:,:,:] = 1
    return new_grid


def test_initial_conditions_equilibrium():
    ic = InitialConditionsEquilibrium()
    dummy = None, None

    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        ic.write(output_dir, *dummy)

        with open(output_dir / ic._file_name) as f:
            lines = f.readlines()

    assert lines == ["Equilibrium\n"]


def test_initial_conditions_equilibrium_regrid():
    ic = InitialConditionsEquilibrium()

    # fmt: off
    new_grid = get_new_grid()

    regrid_context = RegridderWeightsCache(new_grid, new_grid)
    regridded = ic.regrid_like(new_grid, regrid_context )

    assert is_empty(regridded.dataset)


def test_initial_conditions_percolation():
    ic = InitialConditionsPercolation()
    dummy = None, None

    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        ic.write(output_dir, *dummy)

        with open(output_dir / ic._file_name) as f:
            lines = f.readlines()

    assert lines == ["MeteoInputP\n"]


def test_initial_conditions_percolation_regrid():
    ic = InitialConditionsPercolation()

    new_grid = get_new_grid()

    regrid_context = RegridderWeightsCache(new_grid, new_grid)
    regridded = ic.regrid_like(new_grid, regrid_context)
    assert is_empty(regridded.dataset)


def test_initial_conditions_rootzone_pressure_head():
    ic = InitialConditionsRootzonePressureHead(2.2)
    dummy = None, None

    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        ic.write(output_dir, *dummy)

        with open(output_dir / ic._file_name) as f:
            lines = f.readlines()

    assert lines == ["Rootzone_pF\n", " 2.200\n"]


def test_initial_conditions_rootzone_regrid():
    ic = InitialConditionsRootzonePressureHead(2.2)

    new_grid = get_new_grid()

    regrid_context = RegridderWeightsCache(new_grid, new_grid)
    regridded = ic.regrid_like(new_grid, regrid_context)
    assert regridded.dataset["initial_pF"] == 2.2


def test_initial_conditions_saved_state():
    dummy = None, None
    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        with open(output_dir / "foo.out", "w") as f:
            f.write("bar")

        ic = InitialConditionsSavedState(output_dir / "foo.out")

        ic.write(output_dir, *dummy)

        assert Path(output_dir / "init_svat.inp").exists()


def test_initial_conditions_saved_state_no_file():
    dummy = None, None
    ic = InitialConditionsSavedState(r"doesntexist.txt")

    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        with pytest.raises(FileNotFoundError):
            ic.write(output_dir, *dummy)


def test_initial_conditions_saved_state_regrid():
    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        ic = InitialConditionsSavedState(output_dir / "foo.out")

        assert not ic.is_regridding_supported()
