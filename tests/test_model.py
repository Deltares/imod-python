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
    # BTN
    icbund = ibound.copy()

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
    m["btn"] = imod.pkg.BasicTransport(
        icbund=icbund,
        starting_concentration=icbund.copy(),
        porosity=icbund.copy(),
        n_species=1,
        inactive_concentration=1.0e30,
        minimum_active_thickness=0.01,
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
    assert np.allclose(
        m["time_discretization"]["timestep_duration"].values, np.full(5, 1.0)
    )


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
    assert m._render_gen(modelname=modelname, globaltimes=globaltimes) == compare


def test_render_pgk__gcg(basicmodel):
    m = basicmodel
    m.time_discretization(endtime="2000-01-06")
    diskey = m._get_pkgkey("dis")
    globaltimes = m[diskey]["time"].values
    modelname = m.modelname
    directory = Path(".")

    compare = (
        "[gcg]\n"
        "    mxiter = 150\n"
        "    iter1 = 30\n"
        "    isolve = 3\n"
        "    ncrs = 0\n"
        "    cclose = 1e-06\n"
        "    iprgcg = 0\n"
    )
    assert m._render_pkg("gcg", directory=directory, globaltimes=globaltimes) == compare


def test_render_pgk__rch(basicmodel):
    m = basicmodel
    m.time_discretization(endtime="2000-01-06")
    diskey = m._get_pkgkey("dis")
    globaltimes = m[diskey]["time"].values
    modelname = m.modelname
    directory = Path(".")

    compare = (
        "[rch]\n"
        "    nrchop = 3\n"
        "    irchcb = 0\n"
        "    rech_p1 = rate_20000101000000.idf\n"
        "    rech_p2 = rate_20000102000000.idf\n"
        "    rech_p3 = rate_20000103000000.idf\n"
        "    rech_p4 = rate_20000104000000.idf\n"
        "    rech_p5 = rate_20000105000000.idf"
    )
    assert m._render_pkg("rch", directory=directory, globaltimes=globaltimes) == compare


def test_render_dis(basicmodel):
    m = basicmodel
    m.time_discretization(endtime="2000-01-06")
    diskey = m._get_pkgkey("dis")
    globaltimes = m[diskey]["time"].values
    modelname = m.modelname
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
        "    botm_l3 = 0.0\n"
        "    perlen_p1 = 1.0\n"
        "    perlen_p2 = 1.0\n"
        "    perlen_p3 = 1.0\n"
        "    perlen_p4 = 1.0\n"
        "    perlen_p5 = 1.0\n"
        "    nstp_p? = 1\n"
        "    sstr_p? = tr\n"
        "    tsmult_p? = 1.0"
    )
    assert m._render_dis(directory=directory, globaltimes=globaltimes) == compare


def test_render_groups__ghb(basicmodel):
    m = basicmodel
    m.time_discretization(endtime="2000-01-06")
    diskey = m._get_pkgkey("dis")
    globaltimes = m[diskey]["time"].values
    modelname = m.modelname
    directory = Path(".")

    compare = (
        "[ghb]\n"
        "    mghbsys = 1\n"
        "    mxactb = 75\n"
        "    ighbcb = False\n"
        "    bhead_p?_s1_l1 = head_l1.idf\n"
        "    bhead_p?_s1_l2 = head_l2.idf\n"
        "    bhead_p?_s1_l3 = head_l3.idf\n"
        "    cond_p?_s1_l1 = conductance_l1.idf\n"
        "    cond_p?_s1_l2 = conductance_l2.idf\n"
        "    cond_p?_s1_l3 = conductance_l3.idf\n"
        "    ghbssmdens_p?_s1_l1 = density_l1.idf\n"
        "    ghbssmdens_p?_s1_l2 = density_l2.idf\n"
        "    ghbssmdens_p?_s1_l3 = density_l3.idf"
    )

    # TODO: fix stupid newline in the middle
    # check jinja2 documentation
    ssm_compare = (
        "[ssm]\n"
        "    mxss = 75\n"
        "\n"
        "    cghb_t1_p?_l1 = concentration_l1.idf\n"
        "    cghb_t1_p?_l2 = concentration_l2.idf\n"
        "    cghb_t1_p?_l3 = concentration_l3.idf"
    )
    content, ssm_content = m._render_groups(
        directory=directory, globaltimes=globaltimes
    )

    assert content == compare
    assert ssm_content == ssm_compare


def test_render_flowsolver(basicmodel):
    m = basicmodel

    compare = (
        "[pcg]\n"
        "    mxiter = 150\n"
        "    iter1 = 30\n"
        "    npcond = 1\n"
        "    hclose = 0.0001\n"
        "    rclose = 1000.0\n"
        "    relax = 0.98\n"
        "    iprpcg = 1\n"
        "    mutpcg = 0\n"
        "    damp = 1.0\n"
    )
    assert m._render_flowsolver() == compare


def test_render_btn(basicmodel):
    m = basicmodel
    m.time_discretization(endtime="2000-01-06")
    diskey = m._get_pkgkey("dis")
    globaltimes = m[diskey]["time"].values
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
        "    prsity_l3 = 0.3\n"
        "    laycon_l1 = 1\n"
        "    laycon_l2 = 0\n"
        "    laycon_l3 = 0\n"
        "    dt0_p? = 0\n"
        "    ttsmult_p? = 1.0\n"
        "    mxstrn_p? = 10"
    )
    assert m._render_btn(directory=directory, globaltimes=globaltimes)


def test_render_transportsolver(basicmodel):
    m = basicmodel

    compare = (
        "[gcg]\n"
        "    mxiter = 150\n"
        "    iter1 = 30\n"
        "    isolve = 3\n"
        "    ncrs = 0\n"
        "    cclose = 1e-06\n"
        "    iprgcg = 0\n"
    )
    assert m._render_transportsolver() == compare


def test_render(basicmodel):
    m = basicmodel
    m.time_discretization(endtime="2000-01-06")

    s = m.render()
