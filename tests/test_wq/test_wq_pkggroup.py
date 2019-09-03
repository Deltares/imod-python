import pathlib

import numpy as np
import pytest
import xarray as xr

from imod.wq import GeneralHeadBoundary
from imod.wq.pkggroup import GeneralHeadBoundaryGroup


@pytest.fixture(scope="module")
def ghb_group(request):
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
    d = {
        "ph": pathlib.Path("primary").joinpath("head"),
        "pc": pathlib.Path("primary").joinpath("conductance"),
        "pd": pathlib.Path("primary").joinpath("density"),
        "sh": pathlib.Path("secondary").joinpath("head"),
        "sc": pathlib.Path("secondary").joinpath("conductance"),
        "th": pathlib.Path("tertiary").joinpath("head"),
        "tc": pathlib.Path("tertiary").joinpath("conductance"),
    }

    compare = (
        "[ghb]\n"
        "    mghbsys = 3\n"
        "    mxactb = 150\n"
        "    ighbcb = 0\n"
        "    bhead_p?_s1_l1 = {ph}_l1.idf\n"
        "    bhead_p?_s1_l2 = {ph}_l2.idf\n"
        "    cond_p?_s1_l1 = {pc}_l1.idf\n"
        "    cond_p?_s1_l2 = {pc}_l2.idf\n"
        "    ghbssmdens_p?_s1_l1 = {pd}_l1.idf\n"
        "    ghbssmdens_p?_s1_l2 = {pd}_l2.idf\n"
        "    bhead_p?_s2_l1 = {sh}_l1.idf\n"
        "    bhead_p?_s2_l2 = {sh}_l2.idf\n"
        "    cond_p?_s2_l1 = {sc}_l1.idf\n"
        "    cond_p?_s2_l2 = {sc}_l2.idf\n"
        "    bhead_p?_s3_l1 = {th}_l1.idf\n"
        "    bhead_p?_s3_l2 = {th}_l2.idf\n"
        "    cond_p?_s3_l1 = {tc}_l1.idf\n"
        "    cond_p?_s3_l2 = {tc}_l2.idf"
    ).format(**d)
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
    d = {
        "ph": pathlib.Path("primary").joinpath("head"),
        "pc": pathlib.Path("primary").joinpath("conductance"),
        "pd": pathlib.Path("primary").joinpath("density"),
        "sh": pathlib.Path("secondary").joinpath("head"),
        "sc": pathlib.Path("secondary").joinpath("conductance"),
        "th": pathlib.Path("tertiary").joinpath("head"),
        "tc": pathlib.Path("tertiary").joinpath("conductance"),
    }

    # Since no layer is defined, they get appointed to every layer.
    compare = (
        "[ghb]\n"
        "    mghbsys = 3\n"
        "    mxactb = 225\n"
        "    ighbcb = 0\n"
        "    bhead_p?_s1_l? = {ph}.idf\n"
        "    cond_p?_s1_l? = {pc}.idf\n"
        "    ghbssmdens_p?_s1_l? = {pd}.idf\n"
        "    bhead_p?_s2_l? = {sh}.idf\n"
        "    cond_p?_s2_l? = {sc}.idf\n"
        "    bhead_p?_s3_l? = {th}.idf\n"
        "    cond_p?_s3_l? = {tc}.idf"
    ).format(**d)
    assert group.render(directory, globaltimes=["?"], nlayer=3) == compare


# TODO add test for Well that has no time, to calculate max_n_active
