import tempfile
from pathlib import Path

import numpy as np
import pytest

from imod.msw import (
    InitialConditionsEquilibrium,
    InitialConditionsPercolation,
    InitialConditionsRootzonePressureHead,
    InitialConditionsSavedState,
)
from imod.typing.grid import is_empty
from imod.util.regrid import (
    RegridderWeightsCache,
)

DUMMY_ARGS = None, None, None, None


def test_initial_conditions_equilibrium():
    ic = InitialConditionsEquilibrium()

    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        ic.write(output_dir, *DUMMY_ARGS)

        with open(output_dir / ic._file_name) as f:
            lines = f.readlines()

    assert lines == ["Equilibrium\n"]


def test_initial_conditions_equilibrium_regrid(simple_2d_grid_with_subunits):
    ic = InitialConditionsEquilibrium()

    new_grid = simple_2d_grid_with_subunits

    regrid_context = RegridderWeightsCache()
    regridded = ic.regrid_like(new_grid, regrid_context)

    assert is_empty(regridded.dataset)


def test_initial_conditions_equilibrium_clip_box():
    ic = InitialConditionsEquilibrium()

    clipped = ic.clip_box(x_min=1.0, x_max=2.5, y_min=1.0, y_max=2.5)

    assert ic.dataset.identical(clipped.dataset)


def test_initial_conditions_percolation():
    ic = InitialConditionsPercolation()

    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        ic.write(output_dir, *DUMMY_ARGS)

        with open(output_dir / ic._file_name) as f:
            lines = f.readlines()

    assert lines == ["MeteoInputP\n"]


def test_initial_conditions_percolation_regrid(simple_2d_grid_with_subunits):
    ic = InitialConditionsPercolation()

    new_grid = simple_2d_grid_with_subunits

    regrid_context = RegridderWeightsCache()
    regridded = ic.regrid_like(new_grid, regrid_context)
    assert is_empty(regridded.dataset)


def test_initial_conditions_percolation_clip_box():
    ic = InitialConditionsPercolation()

    clipped = ic.clip_box(x_min=1.0, x_max=2.5, y_min=1.0, y_max=2.5)

    assert ic.dataset.identical(clipped.dataset)


def test_initial_conditions_rootzone_pressure_head():
    ic = InitialConditionsRootzonePressureHead(2.2)

    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        ic.write(output_dir, *DUMMY_ARGS)

        with open(output_dir / ic._file_name) as f:
            lines = f.readlines()

    assert lines == ["Rootzone_pF\n", " 2.200\n"]


def test_initial_conditions_rootzone_regrid(simple_2d_grid_with_subunits):
    ic = InitialConditionsRootzonePressureHead(2.2)

    new_grid = simple_2d_grid_with_subunits

    regrid_context = RegridderWeightsCache()
    regridded = ic.regrid_like(new_grid, regrid_context)
    np.testing.assert_almost_equal(regridded.dataset["initial_pF"], 2.2)


def test_initial_conditions_rootzone_clip_box():
    ic = InitialConditionsRootzonePressureHead(2.2)

    clipped = ic.clip_box(x_min=1.0, x_max=2.5, y_min=1.0, y_max=2.5)

    assert ic.dataset.identical(clipped.dataset)


def test_initial_conditions_saved_state():
    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        with open(output_dir / "foo.out", "w") as f:
            f.write("bar")

        ic = InitialConditionsSavedState(output_dir / "foo.out")

        ic.write(output_dir, *DUMMY_ARGS)

        assert Path(output_dir / "init_svat.inp").exists()


def test_initial_conditions_saved_state_no_file():
    ic = InitialConditionsSavedState(r"doesntexist.txt")

    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        with pytest.raises(FileNotFoundError):
            ic.write(output_dir, *DUMMY_ARGS)


def test_initial_conditions_saved_state_regrid():
    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        ic = InitialConditionsSavedState(output_dir / "foo.out")

        assert not ic.is_regridding_supported()


def test_initial_conditions_saved_state_clip_box():
    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        ic = InitialConditionsSavedState(output_dir / "foo.out")

        assert not ic.is_clipping_supported()
