from pathlib import Path

import flopy
import pytest

from imod.mf6 import Modflow6Simulation


@pytest.mark.usefixtures("twri_model")
def test_readable_by_flopy(twri_model: Modflow6Simulation, tmp_path: Path):
    # write model to files using imod-python
    twri_model.write(
        directory=tmp_path, binary=False, validate=True, absolute_paths=True
    )

    # import the model using flopy
    sim = flopy.mf6.MFSimulation.load(sim_ws=tmp_path)

    # run the simulation.
    flopy_sim_result = sim.run_simulation(silent=False, report=True)

    # If these 3 conditions were met the simulation ended with a "Normal termination of simulation"
    assert flopy_sim_result[0] is True
    assert flopy_sim_result[1] == []
    assert len(flopy_sim_result) == 2
