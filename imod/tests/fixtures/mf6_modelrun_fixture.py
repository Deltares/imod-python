from pathlib import Path

import numpy as np

import imod


def _assert_simulation_can_run(
    simulation: imod.mf6.Modflow6Simulation, discretization_name: str, modeldir: Path
):
    # Run simulation
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


def _assert_model_can_run(
    model: imod.mf6.GroundwaterFlowModel, discretization_name: str, modeldir: Path
):
    simulation = imod.mf6.Modflow6Simulation("simulation_name")
    simulation["GWF_1"] = model
    # Define solver settings
    simulation["solver"] = imod.mf6.Solution(
        modelnames=["GWF_1"],
        print_option="summary",
        csv_output=False,
        no_ptc=True,
        outer_dvclose=1.0e-4,
        outer_maximum=500,
        under_relaxation=None,
        inner_dvclose=1.0e-4,
        inner_rclose=0.001,
        inner_maximum=100,
        linear_acceleration="cg",
        scaling_method=None,
        reordering_method=None,
        relaxation_factor=0.97,
    )
    # Collect time discretization
    simulation.create_time_discretization(
        additional_times=["2000-01-01", "2000-01-02", "2000-01-03", "2000-01-04"]
    )

    _assert_simulation_can_run(simulation, discretization_name, modeldir)
