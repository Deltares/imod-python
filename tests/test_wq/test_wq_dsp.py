import pathlib

import numpy as np
import pytest
import xarray as xr

from imod.wq import Dispersion


@pytest.fixture(scope="module")
def dispersion(request):
    layer = np.arange(1, 4)
    y = np.arange(4.5, 0.0, -1.0)
    x = np.arange(0.5, 5.0, 1.0)
    longitudinal = xr.DataArray(
        np.full((3, 5, 5), 1.0),
        coords={"layer": layer, "y": y, "x": x, "dx": 1.0, "dy": -1.0},
        dims=("layer", "y", "x"),
    )

    dsp = Dispersion(
        longitudinal=longitudinal,
        ratio_horizontal=longitudinal.copy(),
        ratio_vertical=longitudinal.copy(),
        diffusion_coefficient=longitudinal.copy(),
    )
    return dsp


def test_render_idf(dispersion):
    dsp = dispersion
    directory = pathlib.Path(".")

    compare = (
        "[dsp]\n"
        "    al_l1 = longitudinal_l1.idf\n"
        "    al_l2 = longitudinal_l2.idf\n"
        "    al_l3 = longitudinal_l3.idf\n"
        "    trpt_l1 = ratio_horizontal_l1.idf\n"
        "    trpt_l2 = ratio_horizontal_l2.idf\n"
        "    trpt_l3 = ratio_horizontal_l3.idf\n"
        "    trpv_l1 = ratio_vertical_l1.idf\n"
        "    trpv_l2 = ratio_vertical_l2.idf\n"
        "    trpv_l3 = ratio_vertical_l3.idf\n"
        "    dmcoef_l1 = diffusion_coefficient_l1.idf\n"
        "    dmcoef_l2 = diffusion_coefficient_l2.idf\n"
        "    dmcoef_l3 = diffusion_coefficient_l3.idf"
    )

    assert dsp._render(directory) == compare


def test_render_constant(dispersion):
    dsp = Dispersion(1.0, 1.0, 1.0, 1.0)
    directory = pathlib.Path(".")

    compare = (
        "[dsp]\n"
        "    al_l? = 1.0\n"
        "    trpt_l? = 1.0\n"
        "    trpv_l? = 1.0\n"
        "    dmcoef_l? = 1.0"
    )

    assert dsp._render(directory) == compare


def test_render_constant_per_layer(dispersion):
    dsp = dispersion
    # Get rid of x and y, so it's no longer an idf
    dsp = dsp.isel(x=0, y=0).drop(["y", "x", "dx", "dy"])
    directory = pathlib.Path(".")

    compare = (
        "[dsp]\n"
        "    al_l1 = 1.0\n"
        "    al_l2 = 1.0\n"
        "    al_l3 = 1.0\n"
        "    trpt_l1 = 1.0\n"
        "    trpt_l2 = 1.0\n"
        "    trpt_l3 = 1.0\n"
        "    trpv_l1 = 1.0\n"
        "    trpv_l2 = 1.0\n"
        "    trpv_l3 = 1.0\n"
        "    dmcoef_l1 = 1.0\n"
        "    dmcoef_l2 = 1.0\n"
        "    dmcoef_l3 = 1.0"
    )

    assert dsp._render(directory) == compare
