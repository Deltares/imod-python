import sys

import numpy as np
import pytest
import xugrid as xu

import imod


@pytest.mark.usefixtures("twri_disv_model")
@pytest.mark.skipif(sys.version_info < (3, 7), reason="capture_output added in 3.7")
def test_simulation_write_and_run(twri_disv_model, tmp_path):
    simulation = twri_disv_model

    with pytest.raises(
        RuntimeError, match="Simulation ex01-twri-disv has not been written yet."
    ):
        twri_disv_model.run()

    modeldir = tmp_path / "ex01-twri-disv"
    simulation.write(modeldir, binary=False)
    simulation.run()

    head = imod.mf6.open_hds(
        modeldir / "GWF_1/GWF_1.hds", modeldir / "GWF_1/dis.disv.grb"
    )
    assert isinstance(head, xu.UgridDataArray)
    assert head.dims == ("time", "layer", "mesh2d_nFaces")
    assert head.shape == (1, 3, 225)
    meanhead_layer = head.groupby("layer").mean(dim="mesh2d_nFaces")
    mean_answer = np.array([59.79181509, 30.44132373, 24.88576811])
    assert np.allclose(meanhead_layer, mean_answer)
