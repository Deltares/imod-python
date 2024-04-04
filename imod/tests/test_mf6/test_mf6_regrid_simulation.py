from pathlib import Path

import numpy as np
import pytest

import imod
from imod.mf6.utilities.regridding_types import RegridderType
from imod.tests.fixtures.mf6_modelrun_fixture import assert_simulation_can_run
from imod.tests.fixtures.mf6_small_models_fixture import (
    grid_data_structured,
    grid_data_unstructured,
)


def test_regrid_structured_simulation_to_structured_simulation(
    tmp_path: Path,
    structured_flow_simulation: imod.mf6.Modflow6Simulation,
):
    finer_idomain = grid_data_structured(np.int32, 1, 0.4)

    new_simulation = structured_flow_simulation.regrid_like(
        "regridded_simulation", finer_idomain
    )

    assert_simulation_can_run(new_simulation, "dis", tmp_path)


def test_regrid_unstructured_simulation_to_unstructured_simulation(
    tmp_path: Path,
    unstructured_flow_simulation: imod.mf6.Modflow6Simulation,
):
    finer_idomain = grid_data_unstructured(np.int32, 1, 0.4)

    new_simulation = unstructured_flow_simulation.regrid_like(
        "regridded_simulation", finer_idomain
    )

    # Test that the newly regridded simulation can run
    assert_simulation_can_run(new_simulation, "disv", tmp_path)


def test_regridded_simulation_has_required_packages(
    unstructured_flow_simulation: imod.mf6.Modflow6Simulation,
):
    finer_idomain = grid_data_unstructured(np.int32, 1, 0.4)

    new_simulation = unstructured_flow_simulation.regrid_like(
        "regridded_simulation", finer_idomain
    )

    assert isinstance(new_simulation["solution"], imod.mf6.Solution)
    assert isinstance(
        new_simulation["time_discretization"], imod.mf6.TimeDiscretization
    )
    assert isinstance(new_simulation["flow"], imod.mf6.GroundwaterFlowModel)


@pytest.mark.usefixtures("circle_model")
def test_regrid_with_methods_without_functions(circle_model, tmp_path):
    simulation = circle_model
    idomain = circle_model["GWF_1"].domain
    # redefine the default regridding method for the constant head package,
    # assign a default method that does not have a function
    old_regrid_method = imod.mf6.ConstantHead._regrid_method
    imod.mf6.ConstantHead._regrid_method = {
        "head": (RegridderType.BARYCENTRIC,),
        "concentration": (RegridderType.BARYCENTRIC,),
    }
    regridding_succeeded = False

    # try regridding the simulation with the new default method
    try:
        simulation.regrid_like("regridded", idomain)
        regridding_succeeded = True
    finally:
        imod.mf6.ConstantHead._regrid_method = old_regrid_method

    assert regridding_succeeded
