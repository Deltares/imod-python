from pathlib import Path

import imod
import imod.pkg
import numpy as np
import pandas as pd
import pytest
import xarray as xr


@pytest.fixture(scope="module")
def basicmodel(request):

    # Basic flow
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

    # LPF
    k_horizontal = ibound.copy()

    # GHB
    head = ibound.copy()

    # RCH
    datetimes = pd.date_range("2000-01-01", "2000-01-05")
    rate = xr.DataArray(
        np.full((5, 5, 5), 1.0),
        coords={"time": datetimes, "y": y, "x": x, "dx": 1.0, "dy": -1.0},
        dims=("time", "y", "x"),
    )
    # DSP
    longitudinal = ibound.copy()

    # Fill model
    m = imod.SeawatModel("test_model")
    m["bas"] = imod.pkg.BasicFlow(
        ibound=ibound, top=top, bottom=bot, starting_head=starting_head
    )
    m["lpf"] = imod.pkg.LayerPropertyFlow(
        k_horizontal=k_horizontal,
        k_vertical=k_horizontal.copy(),
        horizontal_anisotropy=k_horizontal.copy(),
        interblock=k_horizontal.copy(),
        layer_type=k_horizontal.copy(),
        specific_storage=k_horizontal.copy(),
        specific_yield=k_horizontal.copy(),
        save_budget=False,
        layer_wet=k_horizontal.copy(),
        interval_wet=0.01,
        method_wet="wetfactor",
        head_dry=1.0e20,
    )
    m["ghb"] = imod.pkg.GeneralHeadBoundary(
        head=head,
        conductance=head.copy(),
        concentration=head.copy(),
        density=head.copy(),
        save_budget=False,
    )
    m["rch"] = imod.pkg.RechargeHighestActive(
        rate=rate, concentration=rate.copy(), save_budget=False
    )
    m["pcg"] = imod.pkg.PreconditionedConjugateGradientSolver(
        max_iter=150, inner_iter=30, hclose=0.0001, rclose=1000.0, relax=0.98, damp=1.0
    )
    m["adv"] = imod.pkg.AdvectionTVD(courant=1.0)
    m["dsp"] = imod.pkg.Dispersion(
        longitudinal=longitudinal,
        ratio_horizontal=longitudinal.copy(),
        ratio_vertical=longitudinal.copy(),
        diffusion_coefficient=longitudinal.copy(),
    )
    m["vdf"] = imod.pkg.VariableDensityFlow(
        density_species=1,
        density_min=1000.0,
        density_max=1025.0,
        density_ref=1000.0,
        density_concentration_slope=0.71,
        density_criterion=0.01,
        read_density=False,
        internodal="central",
        coupling=1,
        correct_water_table=False,
    )
    m["gcg"] = imod.pkg.GeneralizedConjugateGradientSolver(
        max_iter=150,
        inner_iter=30,
        cclose=1.0e-6,
        preconditioner="mic",
        lump_dispersion=True,
    )
    m["oc"] = imod.pkg.OutputControl(save_head_idf=True, save_concentration_idf=True)

    return m


def test_get_pkgkey(basicmodel):
    m = basicmodel
    for key, package in m.items():
        assert key == package._pkg_id


def test_group(basicmodel):
    m = basicmodel
    g = m._group()
    # Contains only GHB group
    assert len(g) == 1
    # GHB group contains only one value
    assert len(g[0]) == 1
    assert list(g[0].keys())[0] == "ghb" 


def test_timediscretization(basicmodel):
    m = basicmodel
    m.time_discretization(endtime="2000-01-06")
    assert np.allclose(m["time_discretization"]["timestep_duration"].values, np.full(5, 1.0))


def test_render_gen(basicmodel):
    m = basicmodel
    m.time_discretization(endtime="2000-01-06")
    diskey = m._get_pkgkey("dis")
    globaltimes = m[diskey]["time"].values
    modelname = m.modelname

    compare = (
        "[gen]\n"
        "    modelname = test_model\n"
        "    writehelp = False\n"
        "    result_dir = test_model\n"
        "    packages = adv, bas, btn, dis, dsp, gcg, ghb, lpf, oc, pcg, rch, vdf\n"
        "    coord_xll = 0.0\n"
        "    coord_yll = 0.0\n"
        "    start_year = 2000\n"
        "    start_month = 01\n"
        "    start_day = 01"
    )
    with open("compare.txt", "w") as f:
        f.write(compare)
    with open("render.txt", "w") as f:
        f.write(m._render_gen(modelname=modelname, globaltimes=globaltimes))
    assert m._render_gen(modelname=modelname, globaltimes=globaltimes) == compare