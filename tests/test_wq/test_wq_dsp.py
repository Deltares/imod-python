import pathlib
import textwrap

import numpy as np
import pytest
import xarray as xr

from imod.wq import Dispersion


@pytest.fixture(scope="module")
def dispersion():
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

    compare = textwrap.dedent(
        """\
        [dsp]
            al_l$ = longitudinal_l$.idf
            trpt_l$ = ratio_horizontal_l$.idf
            trpv_l$ = ratio_vertical_l$.idf
            dmcoef_l$ = diffusion_coefficient_l$.idf"""
    )

    assert dsp._render(directory, nlayer=3) == compare


def test_render_constant(dispersion):
    dsp = Dispersion(1.0, 1.0, 1.0, 1.0)
    directory = pathlib.Path(".")

    compare = textwrap.dedent(
        """\
        [dsp]
            al_l? = 1.0
            trpt_l? = 1.0
            trpv_l? = 1.0
            dmcoef_l? = 1.0"""
    )

    assert dsp._render(directory, nlayer=4) == compare


def test_render_constant_per_layer(dispersion):
    dsp = dispersion
    # Get rid of x and y, so it's no longer an idf
    dsp = Dispersion(**dsp.dataset.isel(x=0, y=0).drop_vars(["y", "x", "dx", "dy"]))
    directory = pathlib.Path(".")

    compare = textwrap.dedent(
        """\
        [dsp]
            al_l1:3 = 1.0
            trpt_l1:3 = 1.0
            trpv_l1:3 = 1.0
            dmcoef_l1:3 = 1.0"""
    )

    assert dsp._render(directory, nlayer=3) == compare
