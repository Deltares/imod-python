import pathlib
import textwrap

import numpy as np
import pytest
import xarray as xr

from imod.wq import GeneralHeadBoundary
from imod.wq.pkggroup import GeneralHeadBoundaryGroup


@pytest.fixture(scope="module")
def ghb_group():
    layer = np.arange(1, 3)
    y = np.arange(4.5, 0.0, -1.0)
    x = np.arange(0.5, 5.0, 1.0)
    head = xr.DataArray(
        np.full((2, 5, 5), 1.0),
        coords={"layer": layer, "y": y, "x": x, "dx": 1.0, "dy": -1.0},
        dims=("layer", "y", "x"),
    )

    headboundary = GeneralHeadBoundary(
        head=head,
        conductance=head.copy(),
        concentration=head.copy(),
        density=head.copy(),
        save_budget=False,
    )
    ghb1 = headboundary
    ghb2 = headboundary.copy().drop(["concentration", "density"])
    ghb3 = ghb2.copy()
    d = {"primary": ghb1, "secondary": ghb2, "tertiary": ghb3}

    ghb_group = GeneralHeadBoundaryGroup(**d)
    return ghb_group


def test_render(ghb_group):
    group = ghb_group
    directory = pathlib.Path(".")
    compare = textwrap.dedent(
        """\
        [ghb]
            mghbsys = 3
            mxactb = 150
            ighbcb = 0
            bhead_p?_s1_l1 = primary/head_l1.idf
            bhead_p?_s1_l2 = primary/head_l2.idf
            cond_p?_s1_l1 = primary/conductance_l1.idf
            cond_p?_s1_l2 = primary/conductance_l2.idf
            ghbssmdens_p?_s1_l1 = primary/density_l1.idf
            ghbssmdens_p?_s1_l2 = primary/density_l2.idf
            bhead_p?_s2_l1 = secondary/head_l1.idf
            bhead_p?_s2_l2 = secondary/head_l2.idf
            cond_p?_s2_l1 = secondary/conductance_l1.idf
            cond_p?_s2_l2 = secondary/conductance_l2.idf
            bhead_p?_s3_l1 = tertiary/head_l1.idf
            bhead_p?_s3_l2 = tertiary/head_l2.idf
            cond_p?_s3_l1 = tertiary/conductance_l1.idf
            cond_p?_s3_l2 = tertiary/conductance_l2.idf"""
    )
    assert group.render(directory, globaltimes=["?"], nlayer=3) == compare


def test_render_error__concentration_twice(ghb_group):
    layer = np.arange(1, 4)
    y = np.arange(4.5, 0.0, -1.0)
    x = np.arange(0.5, 5.0, 1.0)
    head = xr.DataArray(
        np.full((3, 5, 5), 1.0),
        coords={"layer": layer, "y": y, "x": x, "dx": 1.0, "dy": -1.0},
        dims=("layer", "y", "x"),
    )

    headboundary = GeneralHeadBoundary(
        head=head,
        conductance=head.copy(),
        concentration=head.copy(),
        density=head.copy(),
        save_budget=False,
    )
    ghb1 = headboundary
    ghb2 = headboundary.copy()
    d = {"primary": ghb1, "secondary": ghb2}

    with pytest.raises(ValueError):
        ghb_group = GeneralHeadBoundaryGroup(**d)


def test_render__count_nolayer():
    """
    Tests that when a 2D DataArray is given, mxactb is still correct.
    """
    y = np.arange(4.5, 0.0, -1.0)
    x = np.arange(0.5, 5.0, 1.0)
    head = xr.DataArray(
        np.full((5, 5), 1.0),
        coords={"y": y, "x": x, "dx": 1.0, "dy": -1.0},
        dims=("y", "x"),
    )

    headboundary = GeneralHeadBoundary(
        head=head,
        conductance=head.copy(),
        concentration=head.copy(),
        density=head.copy(),
        save_budget=False,
    )
    ghb1 = headboundary
    ghb2 = headboundary.copy().drop(["concentration", "density"])
    ghb3 = ghb2.copy()
    d = {"primary": ghb1, "secondary": ghb2, "tertiary": ghb3}
    group = GeneralHeadBoundaryGroup(**d)

    directory = pathlib.Path(".")

    # Since no layer is defined, they get appointed to every layer.
    compare = textwrap.dedent(
        """\
        [ghb]
            mghbsys = 3
            mxactb = 225
            ighbcb = 0
            bhead_p?_s1_l? = primary/head.idf
            cond_p?_s1_l? = primary/conductance.idf
            ghbssmdens_p?_s1_l? = primary/density.idf
            bhead_p?_s2_l? = secondary/head.idf
            cond_p?_s2_l? = secondary/conductance.idf
            bhead_p?_s3_l? = tertiary/head.idf
            cond_p?_s3_l? = tertiary/conductance.idf"""
    )
    assert group.render(directory, globaltimes=["?"], nlayer=3) == compare


# TODO add test for Well that has no time, to calculate max_n_active
