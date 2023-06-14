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
    # Run simulation
    modeldir = imod.util.temporary_directory()
    simulation.write(modeldir, binary=False)
    simulation.run()

    # Test that output was generated
    dis_outputfile = modeldir / f"flow/{discretization_name}.{discretization_name}.grb"
    head = imod.mf6.out.open_hds(
        modeldir / "flow/flow.hds",
        dis_outputfile,
    )

    # Test that heads are not nan
    assert not np.any(np.isnan(head.values))


def test_regrid_structured_simulation_to_structured_simulation(
    structured_flow_simulation: imod.mf6.Modflow6Simulation,
):
    finer_idomain = grid_data_structured(np.int32, 1, 0.4)

    new_simulation = structured_flow_simulation.regrid_like(
        "regridded_simulation", finer_idomain
    )

    _assert_simulation_can_run(new_simulation, "dis")


def test_regrid_unstructured_simulation_to_unstructured_simulation(
    unstructured_flow_simulation: imod.mf6.Modflow6Simulation,
):
    finer_idomain = grid_data_unstructured(np.int32, 1, 0.4)

    new_simulation = unstructured_flow_simulation.regrid_like(
        "regridded_simulation", finer_idomain
    )

    # Test that the newly regridded simulation can run
    _assert_simulation_can_run(new_simulation, "disv")


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


def test_simulation_gives_exception_if_transport_model_is_regridded(
    unstructured_flow_simulation: imod.mf6.Modflow6Simulation,
):
    transport_model = imod.mf6.GroundwaterTransportModel()
    transport_model["adv"] = imod.mf6.AdvectionUpstream()
    transport_model["dsp"] = imod.mf6.Dispersion(
        diffusion_coefficient=0.0,
        longitudinal_horizontal=2.3,
        transversal_horizontal1=0.0,
        xt3d_off=False,
        xt3d_rhs=False,
    )
    flow_and_transport_simulation = unstructured_flow_simulation
    flow_and_transport_simulation["transport"] = transport_model

    finer_idomain = grid_data_unstructured(np.int32, 1, 0.4)

    with pytest.raises(NotImplementedError):
        _ = unstructured_flow_simulation.regrid_like(
            "regridded_simulation", finer_idomain
        )
