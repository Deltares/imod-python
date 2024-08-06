from pathlib import Path

import numpy as np
import pytest

import imod
from imod.mf6.regrid.regrid_schemes import ConstantHeadRegridMethod
from imod.mf6.utilities.regrid import RegridderWeightsCache
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
def test_regrid_with_custom_method(circle_model):
    simulation = circle_model
    idomain = circle_model["GWF_1"].domain
    chd_pkg = circle_model["GWF_1"].pop("chd")

    simulation_regridded = simulation.regrid_like("regridded", idomain)
    regrid_method = ConstantHeadRegridMethod(
        head=(RegridderType.BARYCENTRIC,), concentration=(RegridderType.BARYCENTRIC,)
    )
    regrid_cache = RegridderWeightsCache()
    simulation_regridded["GWF_1"]["chd"] = chd_pkg.regrid_like(
        idomain, regrid_cache=regrid_cache, regridder_types=regrid_method
    )
