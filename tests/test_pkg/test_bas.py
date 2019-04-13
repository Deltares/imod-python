from imod.pkg import BasicFlow
import xarray as xr
import numpy as np
import pytest
from pathlib import Path


@pytest.fixture(scope="module")
def basicflow(request):
    layer = np.arange(1, 4)
    y = np.arange(4.5, 0.0, -1.0)
    x = np.arange(0.5, 5.0, 1.0)
    ibound = xr.DataArray(
        np.full((3, 5, 5), 1.0),
        coords={"layer": layer, "y": y, "x": x, "dx": 1.0, "dy": -1.0},
        dims=("layer", "y", "x"),
    )
    starting_head = xr.full_like(ibound, 0.0)
    top = 30.0
    bot = xr.DataArray(
        np.arange(20.0, -10.0, -10.0), coords={"layer": layer}, dims=("layer",)
    )

    bas = BasicFlow(ibound=ibound, top=top, bottom=bot, starting_head=starting_head)
    return bas


def test_thickness(basicflow):
    bas = basicflow
    thickness = bas.thickness().values
    compare = np.array([10., 10., 10.])
    assert np.allclose(thickness, compare)


def test_render_bas(basicflow):
    bas = basicflow
    directory = Path(".")

    compare = (
        "[bas6]\n"
        "    ibound_l1 = ibound_l1.idf\n"
        "    ibound_l2 = ibound_l2.idf\n"
        "    ibound_l3 = ibound_l3.idf\n"
        "    hnoflo = 1e+30\n"
        "    strt_l1 = starting_head_l1.idf\n"
        "    strt_l2 = starting_head_l2.idf\n"
        "    strt_l3 = starting_head_l3.idf"
    )
    assert bas._render_bas(directory) == compare


def test_render_dis__scalartopbot(basicflow):
    bas = basicflow
    directory = Path(".")

    compare = (
        "[dis]\n"
        "    nlay = 3\n"
        "    nrow = 5\n"
        "    ncol = 5\n"
        "    delc_r? = 1.0\n"
        "    delr_c? = 1.0\n"
        "    top = 30.0\n"
        "    botm_l1 = 20.0\n"
        "    botm_l2 = 10.0\n"
        "    botm_l3 = 0.0"
    )
    assert bas._render_dis(directory) == compare


def test_render_dis__arraytopbot(basicflow):
    bas = basicflow
    bas["bottom"] = xr.full_like(bas["ibound"], 10.0)
    bas["top"] = bas["bottom"].isel(layer=0)
    directory = Path(".")

    compare = (
        "[dis]\n"
        "    nlay = 3\n"
        "    nrow = 5\n"
        "    ncol = 5\n"
        "    delc_r? = 1.0\n"
        "    delr_c? = 1.0\n"
        "    top = top.idf\n"
        "    botm_l1 = bottom_l1.idf\n"
        "    botm_l2 = bottom_l2.idf\n"
        "    botm_l3 = bottom_l3.idf"
    )
    assert bas._render_dis(directory) == compare

