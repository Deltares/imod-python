import pathlib
import textwrap

import numpy as np
import pytest
import xarray as xr

from imod.wq import BasicFlow


@pytest.fixture(scope="module")
def basicflow():
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
    compare = np.array([10.0, 10.0, 10.0])
    assert np.allclose(thickness, compare)


def test_render(basicflow):
    bas = basicflow
    directory = pathlib.Path(".")
    compare = textwrap.dedent(
        """\
        [bas6]
            ibound_l1 = ibound_l1.idf
            ibound_l2 = ibound_l2.idf
            ibound_l3 = ibound_l3.idf
            hnoflo = 1e+30
            strt_l1 = starting_head_l1.idf
            strt_l2 = starting_head_l2.idf
            strt_l3 = starting_head_l3.idf"""
    )
    assert bas._render(directory) == compare


def test_render_dis__scalartopbot(basicflow):
    bas = basicflow
    directory = pathlib.Path(".")
    confining_bed_below = 0
    compare = textwrap.dedent(
        """\
        [dis]
            nlay = 3
            nrow = 5
            ncol = 5
            delc_r? = 1.0
            delr_c? = 1.0
            top = 30.0
            botm_l1 = 20.0
            botm_l2 = 10.0
            botm_l3 = 0.0
            laycbd_l? = 0"""
    )
    assert bas._render_dis(directory, confining_bed_below) == compare


def test_render_dis__arraytopbot(basicflow):
    bas = basicflow
    bas["bottom"] = xr.full_like(bas["ibound"], 10.0)
    bas["top"] = bas["bottom"].isel(layer=0)
    directory = pathlib.Path(".")
    confining_bed_below = 1
    compare = textwrap.dedent(
        """\
        [dis]
            nlay = 3
            nrow = 5
            ncol = 5
            delc_r? = 1.0
            delr_c? = 1.0
            top = top.idf
            botm_l1 = bottom_l1.idf
            botm_l2 = bottom_l2.idf
            botm_l3 = bottom_l3.idf
            laycbd_l? = 1"""
    )
    assert bas._render_dis(directory, confining_bed_below) == compare
