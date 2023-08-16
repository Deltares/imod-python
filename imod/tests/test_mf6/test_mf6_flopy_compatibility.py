from pathlib import Path

import flopy
import pytest

from imod.mf6 import Modflow6Simulation


@pytest.mark.usefixtures("twri_model")
@pytest.mark.parametrize("absolute_paths", [False])
def test_readable_by_flopy(
    twri_model: Modflow6Simulation, tmp_path: Path, absolute_paths
):
    # write model to files using imod-python
    twri_model.write(
        directory=tmp_path, binary=False, validate=True, absolute_paths=absolute_paths
    )

    # import the model using flopy
    sim = flopy.mf6.MFSimulation.load(sim_ws=tmp_path)

    # run the simulation.
    flopy_sim_result = sim.run_simulation(silent=False, report=True)

    # If these 3 conditions were met the simulation ended with a "Normal termination of simulation"
    assert flopy_sim_result[0] is True
    assert "Normal termination of simulation." in flopy_sim_result[1][-1]
    assert len(flopy_sim_result) == 2

@pytest.mark.usefixtures("twri_model")
@pytest.mark.parametrize("absolute_paths", [True, False])
def test_readable_by_mf6(
    twri_model: Modflow6Simulation, tmp_path: Path, absolute_paths
):
    # write model to files using imod-python
    twri_model.write(
        directory=tmp_path, binary=False, validate=True, absolute_paths=absolute_paths
    )

    #the run method will raise an exception if the run does not succeed
    twri_model.run()
