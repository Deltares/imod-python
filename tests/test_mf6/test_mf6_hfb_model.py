import pathlib
import shapely.geometry as sg
import geopandas as gpd

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import imod.mf6 as mf6
import imod
import subprocess
import sys


@pytest.fixture(scope="module")
def hfb_model():
    layer = [1, 2, 3]
    y = np.linspace(1.5, 0.5, num=9)
    x = np.linspace(0.5, 1.5, num=9)

    dy = y[1] - y[0]
    dx = x[1] - x[0]

    idomain = xr.DataArray(
        data=np.ones((3, 9, 9)),
        coords={"layer": layer, "y": y, "x": x, "dx": dx, "dy": dy},
        dims=["layer", "y", "x"],
    )

    line = sg.LineString([[1.25, 2.0], [0.75, 1.0]])
    line2 = sg.LineString([[1.25, 1.0], [0.75, 2.0]])

    line_gdf = gpd.GeoDataFrame({"resistance": [100.0, 10.0]}, geometry=[line, line2])

    starting_head = xr.full_like(idomain, 0.0)
    top = 1.0
    bot = xr.DataArray(data=[0, -1.0, -2.0], coords={"layer": layer}, dims=("layer",))

    constant_head = xr.full_like(idomain, np.nan)
    constant_head[:, :, 0] = 10.0
    constant_head[:, :, -1] = 0.0

    idomain = idomain.astype(np.int32)

    simulation = mf6.Modflow6Simulation("test")

    model = mf6.GroundwaterFlowModel()
    model["dis"] = mf6.StructuredDiscretization(top, bot, idomain)
    model["npf"] = mf6.NodePropertyFlow(0, 10.0)
    model["chd"] = mf6.ConstantHead(constant_head)

    model["ic"] = mf6.InitialConditions(starting_head)

    model["hfb"] = mf6.HorizontalFlowBarrier(idomain=idomain, **line_gdf)

    model["oc"] = mf6.OutputControl(save_head="last")

    simulation["GWF_1"] = model
    simulation["pcg"] = mf6.Solution(
        print_option="summary",
        csv_output=False,
        no_ptc=True,
        outer_hclose=1.0e-4,
        outer_maximum=500,
        under_relaxation=None,
        inner_hclose=1.0e-4,
        inner_rclose=0.001,
        inner_maximum=100,
        linear_acceleration="cg",
        scaling_method=None,
        reordering_method=None,
        relaxation_factor=0.97,
    )

    simulation.time_discretization(["2000-01-01", "2000-01-02"])

    return simulation


@pytest.mark.skipif(sys.version_info < (3, 7), reason="capture_output added in 3.7")
def test_simulation_write(hfb_model, tmp_path):
    simulation = hfb_model
    modeldir = tmp_path / "hfb_model"
    simulation.write(modeldir)
    with imod.util.cd(modeldir):
        p = subprocess.run("mf6", check=True, capture_output=True, text=True)
        assert p.stdout.endswith("Normal termination of simulation.\n")
        head = imod.mf6.open_hds("GWF_1/GWF_1.hds", "GWF_1/dis.dis.grb")
        assert head.dims == ("time", "layer", "y", "x")
        assert head.shape == (1, 3, 9, 9)
        meanhead = head.mean().values
        mean_answer = 5.04250203
        assert np.allclose(meanhead, mean_answer)
