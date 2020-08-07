import pathlib
import textwrap

import numpy as np
import pytest
import xarray as xr

from imod.wq import BasicTransport


@pytest.fixture(scope="module")
def basictransport():
    layer = np.arange(1, 4)
    y = np.arange(4.5, 0.0, -1.0)
    x = np.arange(0.5, 5.0, 1.0)
    icbund = xr.DataArray(
        np.full((3, 5, 5), 1.0),
        coords={"layer": layer, "y": y, "x": x, "dx": 1.0, "dy": -1.0},
        dims=("layer", "y", "x"),
    )
    starting_concentration = xr.full_like(icbund, 0.0)
    porosity = xr.DataArray(np.full(3, 0.3), {"layer": layer}, dims=("layer",))

    btn = BasicTransport(
        icbund=icbund,
        starting_concentration=starting_concentration,
        porosity=porosity,
        n_species=1,
        inactive_concentration=1.0e30,
        minimum_active_thickness=0.01,
    )
    return btn


def test_btn_render_arrays(basictransport):
    btn = basictransport
    directory = pathlib.Path(".")

    compare = textwrap.dedent(
        """\
            [btn]
                ncomp = 1
                mcomp = 1
                thkmin = 0.01
                cinact = 1e+30
                sconc_t1_l$ = starting_concentration_l$.idf
                icbund_l$ = icbund_l$.idf
                dz_l$ = thickness_l$.idf
                prsity_l1:3 = 0.3"""
    )

    shape = btn["icbund"].shape
    thickness = xr.full_like(btn["icbund"], 1.0)
    btn["thickness"] = thickness
    assert btn._render(directory, nlayer=3) == compare


def test_btn_render_constants(basictransport):
    btn = basictransport
    directory = pathlib.Path(".")
    layer = np.arange(1, 4)
    btn["starting_concentration"] = 0.0
    btn["porosity"] = 0.3
    thickness = xr.DataArray(np.full(3, 10.0), {"layer": layer}, dims=("layer",))

    compare = textwrap.dedent(
        """\
            [btn]
                ncomp = 1
                mcomp = 1
                thkmin = 0.01
                cinact = 1e+30
                sconc_t1_l? = 0.0
                icbund_l$ = icbund_l$.idf
                dz_l1:3 = 10.0
                prsity_l? = 0.3"""
    )

    thickness = xr.DataArray(np.full(3, 10.0), {"layer": layer}, dims=("layer",))
    btn["thickness"] = thickness
    assert btn._render(directory, nlayer=3) == compare
