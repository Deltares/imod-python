from imod.pkg import BasicTransport
import pytest
import xarray as xr
import numpy as np
from pathlib import Path


@pytest.fixture(scope="module")
def basictransport(request):
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
    directory = Path(".")

    compare = (
        "[btn]\n"
        "    thkmin = 0.01\n"
        "    cinact = 1e+30\n"
        "    sconc_t1_l1 = starting_concentration_l1.idf\n"
        "    sconc_t1_l2 = starting_concentration_l2.idf\n"
        "    sconc_t1_l3 = starting_concentration_l3.idf\n"
        "    icbund_l1 = icbund_l1.idf\n"
        "    icbund_l2 = icbund_l2.idf\n"
        "    icbund_l3 = icbund_l3.idf\n"
        "    dz_l1 = 10.0\n"
        "    dz_l2 = 10.0\n"
        "    dz_l3 = 10.0\n"
        "    prsity_l1 = 0.3\n"
        "    prsity_l2 = 0.3\n"
        "    prsity_l3 = 0.3"
    )

    layer = np.arange(1, 4)
    layer_type = xr.DataArray(np.array([1, 0, 0]), {"layer": layer}, dims=("layer",))
    thickness = xr.DataArray(np.full(3, 10.0), {"layer": layer}, dims=("layer",))

    assert btn._render(directory, thickness=thickness) == compare


def test_btn_render_constants(basictransport):
    btn = basictransport
    directory = Path(".")
    layer = np.arange(1, 4)
    layer_type = xr.DataArray(1)
    btn["starting_concentration"] = 0.0
    btn["porosity"] = 0.3
    thickness = xr.DataArray(np.full(3, 10.0), {"layer": layer}, dims=("layer",))

    compare = (
        "[btn]\n"
        "    thkmin = 0.01\n"
        "    cinact = 1e+30\n"
        "    sconc_t1_l? = 0.0\n"
        "    icbund_l1 = icbund_l1.idf\n"
        "    icbund_l2 = icbund_l2.idf\n"
        "    icbund_l3 = icbund_l3.idf\n"
        "    dz_l1 = 10.0\n"
        "    dz_l2 = 10.0\n"
        "    dz_l3 = 10.0\n"
        "    prsity_l? = 0.3"
    )

    assert btn._render(directory, thickness=thickness) == compare
