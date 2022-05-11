import tempfile
from pathlib import Path

import pytest

from imod.msw import (
    InitialConditionsEquilibrium,
    InitialConditionsPercolation,
    InitialConditionsRootzonePressureHead,
    InitialConditionsSavedState,
)


def test_initial_conditions_equilibrium():
    ic = InitialConditionsEquilibrium()
    dummy = None, None

    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        ic.write(output_dir, *dummy)

        with open(output_dir / ic._file_name) as f:
            lines = f.readlines()

    assert lines == ["Equilibrium\n"]


def test_initial_conditions_percolation():
    ic = InitialConditionsPercolation()
    dummy = None, None

    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        ic.write(output_dir, *dummy)

        with open(output_dir / ic._file_name) as f:
            lines = f.readlines()

    assert lines == ["MeteoInputP\n"]


def test_initial_conditions_rootzone_pressure_head():
    ic = InitialConditionsRootzonePressureHead(2.2)
    dummy = None, None

    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        ic.write(output_dir, *dummy)

        with open(output_dir / ic._file_name) as f:
            lines = f.readlines()

    assert lines == ["Rootzone_pF\n", " 2.200\n"]


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
