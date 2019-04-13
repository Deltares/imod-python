from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from imod.pkg import River


@pytest.fixture(scope="module")
def river(request):
    layer = np.arange(1, 4)
    y = np.arange(4.5, 0.0, -1.0)
    x = np.arange(0.5, 5.0, 1.0)
    stage = xr.DataArray(
        np.full((3, 5, 5), 1.0),
        coords={"layer": layer, "y": y, "x": x, "dx": 1.0, "dy": -1.0},
        dims=("layer", "y", "x"),
    )

    riv = River(
        stage=stage,
        conductance=stage.copy(),
        bottom_elevation=stage.copy(),
        concentration=stage.copy(),
        density=stage.copy(),
    )
    return riv


def test_render(river):
    riv = river
    directory = Path(".")

    compare = (
        "\n"
        "    stage_p?_s1_l1 = stage_l1.idf\n"
        "    stage_p?_s1_l2 = stage_l2.idf\n"
        "    stage_p?_s1_l3 = stage_l3.idf\n"
        "    cond_p?_s1_l1 = conductance_l1.idf\n"
        "    cond_p?_s1_l2 = conductance_l2.idf\n"
        "    cond_p?_s1_l3 = conductance_l3.idf\n"
        "    rbot_p?_s1_l1 = bottom_elevation_l1.idf\n"
        "    rbot_p?_s1_l2 = bottom_elevation_l2.idf\n"
        "    rbot_p?_s1_l3 = bottom_elevation_l3.idf\n"
        "    rivssmdens_p?_s1_l1 = density_l1.idf\n"
        "    rivssmdens_p?_s1_l2 = density_l2.idf\n"
        "    rivssmdens_p?_s1_l3 = density_l3.idf"
    )

    assert riv._render(directory, globaltimes=["?"], system_index=1) == compare
