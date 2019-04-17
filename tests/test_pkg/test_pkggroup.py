from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from imod.pkg import GeneralHeadBoundary
from imod.pkg.pkggroup import GeneralHeadBoundaryGroup


@pytest.fixture(scope="module")
def ghb_group(request):
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
    ghb2 = headboundary.copy().drop(["concentration", "density"])
    ghb3 = ghb2.copy()
    d = {"primary": ghb1, "secondary": ghb2, "tertiary": ghb3}

    ghb_group = GeneralHeadBoundaryGroup(**d)
    return ghb_group


def test_render(ghb_group):
    group = ghb_group
    directory = Path(".")

    compare = (
        "[ghb]\n"
        "    mghbsys = 3\n"
        "    mxactb = 225\n"
        "    ighbcb = False\n"
        "    bhead_p?_s1_l1 = head_l1.idf\n"
        "    bhead_p?_s1_l2 = head_l2.idf\n"
        "    bhead_p?_s1_l3 = head_l3.idf\n"
        "    cond_p?_s1_l1 = conductance_l1.idf\n"
        "    cond_p?_s1_l2 = conductance_l2.idf\n"
        "    cond_p?_s1_l3 = conductance_l3.idf\n"
        "    ghbssmdens_p?_s1_l1 = density_l1.idf\n"
        "    ghbssmdens_p?_s1_l2 = density_l2.idf\n"
        "    ghbssmdens_p?_s1_l3 = density_l3.idf\n"
        "    bhead_p?_s2_l1 = head_l1.idf\n"
        "    bhead_p?_s2_l2 = head_l2.idf\n"
        "    bhead_p?_s2_l3 = head_l3.idf\n"
        "    cond_p?_s2_l1 = conductance_l1.idf\n"
        "    cond_p?_s2_l2 = conductance_l2.idf\n"
        "    cond_p?_s2_l3 = conductance_l3.idf\n"
        "    bhead_p?_s3_l1 = head_l1.idf\n"
        "    bhead_p?_s3_l2 = head_l2.idf\n"
        "    bhead_p?_s3_l3 = head_l3.idf\n"
        "    cond_p?_s3_l1 = conductance_l1.idf\n"
        "    cond_p?_s3_l2 = conductance_l2.idf\n"
        "    cond_p?_s3_l3 = conductance_l3.idf"
    )
    assert group.render(directory, globaltimes=["?"]) == compare


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