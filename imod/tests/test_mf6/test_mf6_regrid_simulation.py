import numpy as np
import pytest

import imod
from imod.tests.fixtures.mf6_regridding_fixture import (
    grid_data_structured,
    grid_data_unstructured,
)


def _assert_simulation_can_run(
    simulation: imod.mf6.Modflow6Simulation, discretization_name: str
):
    #  run simulation
    modeldir = imod.util.temporary_directory()
    simulation.write(modeldir, binary=False)
    simulation.run()

    # test that output was generated
    dis_outputfile = modeldir / f"flow/{discretization_name}.{discretization_name}.grb"
    head = imod.mf6.out.open_hds(
        modeldir / "flow/flow.hds",
        dis_outputfile,
    )

    # test that heads are not nan
    assert not np.any(np.isnan(head.values))


def test_regrid_structured_simulation_to_structured_simulation(
    structured_flow_simulation: imod.mf6.Modflow6Simulation,
):
    finer_idomain = grid_data_structured(np.int32, 1, 0.4)

    new_simulation = structured_flow_simulation.regrid_like(
        "regridded_simulation", finer_idomain
    )

    # assert number of pack equal
    assert len(structured_flow_simulation.items()) == len(new_simulation.items())

    _assert_simulation_can_run(new_simulation, "dis")


def test_regrid_unstructured_simulation_to_unstructured_simulation(
    unstructured_flow_simulation: imod.mf6.Modflow6Simulation,
):
    finer_idomain = grid_data_unstructured(np.int32, 1, 0.4)

    new_simulation = unstructured_flow_simulation.regrid_like(
        "regridded_simulation", finer_idomain
    )

    # assert number of pack equal
    assert len(unstructured_flow_simulation.items()) == len(new_simulation.items())

    # test that the newly regridded simulation can run
    _assert_simulation_can_run(new_simulation, "disv")
