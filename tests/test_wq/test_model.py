import pathlib

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import imod
import imod.wq


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

    # WEL
    welly = np.arange(4.5, 0.0, -1.0)
    wellx = np.arange(0.5, 5.0, 1.0)

    # DSP
    longitudinal = ibound.copy()

    # Fill model
    m = imod.wq.SeawatModel("test_model")
    m["bas"] = imod.wq.BasicFlow(
        ibound=ibound, top=top, bottom=bot, starting_head=starting_head
    )
    m["lpf"] = imod.wq.LayerPropertyFlow(
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
    m["ghb"] = imod.wq.GeneralHeadBoundary(
        head=head,
        conductance=head.copy(),
        concentration=head.copy(),
        density=head.copy(),
        save_budget=False,
    )
    m["riv"] = imod.wq.River(
        stage=head,
        conductance=head.copy(),
        bottom_elevation=head.copy(),
        concentration=head.copy(),
        density=head.copy(),
        save_budget=False,
    )
    m["wel"] = imod.wq.Well(id_name="well", x=wellx, y=welly, rate=5.0, layer=2, time=datetimes)
    m["rch"] = imod.wq.RechargeHighestActive(
        rate=rate, concentration=rate.copy(), save_budget=False
    )
    m["pcg"] = imod.wq.PreconditionedConjugateGradientSolver(
        max_iter=150, inner_iter=30, hclose=0.0001, rclose=1000.0, relax=0.98, damp=1.0
    )
    m["btn"] = imod.wq.BasicTransport(
        icbund=icbund,
        starting_concentration=icbund.copy(),
        porosity=icbund.copy(),
        n_species=1,
        inactive_concentration=1.0e30,
        minimum_active_thickness=0.01,
    )
    m["adv"] = imod.wq.AdvectionTVD(courant=1.0)
    m["dsp"] = imod.wq.Dispersion(
        longitudinal=longitudinal,
        ratio_horizontal=longitudinal.copy(),
        ratio_vertical=longitudinal.copy(),
        diffusion_coefficient=longitudinal.copy(),
    )
    m["vdf"] = imod.wq.VariableDensityFlow(
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
    m["gcg"] = imod.wq.GeneralizedConjugateGradientSolver(
        max_iter=150,
        inner_iter=30,
        cclose=1.0e-6,
        preconditioner="mic",
        lump_dispersion=True,
    )
    m["oc"] = imod.wq.OutputControl(save_head_idf=True, save_concentration_idf=True)

    return m


def test_get_pkgkey(basicmodel):
    m = basicmodel
    for key, package in m.items():
        assert key == package._pkg_id


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
        "    result_dir = results\n"
        "    packages = adv, bas, btn, dis, dsp, gcg, ghb, lpf, oc, pcg, rch, riv, vdf, wel\n"
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
    directory = pathlib.Path(".")

    compare = (
        "[gcg]\n"
        "    mxiter = 150\n"
        "    iter1 = 30\n"
        "    isolve = 3\n"
        "    ncrs = 0\n"
        "    cclose = 1e-06\n"
        "    iprgcg = 0"
    )
    assert m._render_pkg("gcg", directory=directory, globaltimes=globaltimes) == compare


def test_render_pgk__rch(basicmodel):
    m = basicmodel
    m.time_discretization(endtime="2000-01-06")
    diskey = m._get_pkgkey("dis")
    globaltimes = m[diskey]["time"].values
    modelname = m.modelname
    directory = pathlib.Path(".")

    compare = (
        "[rch]\n"
        "    nrchop = 3\n"
        "    irchcb = 0\n"
        "    rech_p1 = rch\\rate_20000101000000.idf\n"
        "    rech_p2 = rch\\rate_20000102000000.idf\n"
        "    rech_p3 = rch\\rate_20000103000000.idf\n"
        "    rech_p4 = rch\\rate_20000104000000.idf\n"
        "    rech_p5 = rch\\rate_20000105000000.idf"
    )
    assert m._render_pkg("rch", directory=directory, globaltimes=globaltimes) == compare


def test_render_dis(basicmodel):
    m = basicmodel
    m.time_discretization(endtime="2000-01-06")
    diskey = m._get_pkgkey("dis")
    globaltimes = m[diskey]["time"].values
    modelname = m.modelname
    directory = pathlib.Path(".")

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


def test_render_groups__ghb_riv_wel(basicmodel):
    m = basicmodel
    m.time_discretization(endtime="2000-01-06")
    diskey = m._get_pkgkey("dis")
    globaltimes = m[diskey]["time"].values
    modelname = m.modelname
    directory = pathlib.Path(".")

    compare = (
        "[ghb]\n"
        "    mghbsys = 1\n"
        "    mxactb = 75\n"
        "    ighbcb = False\n"
        "    bhead_p?_s1_l1 = ghb\\head_l1.idf\n"
        "    bhead_p?_s1_l2 = ghb\\head_l2.idf\n"
        "    bhead_p?_s1_l3 = ghb\\head_l3.idf\n"
        "    cond_p?_s1_l1 = ghb\\conductance_l1.idf\n"
        "    cond_p?_s1_l2 = ghb\\conductance_l2.idf\n"
        "    cond_p?_s1_l3 = ghb\\conductance_l3.idf\n"
        "    ghbssmdens_p?_s1_l1 = ghb\\density_l1.idf\n"
        "    ghbssmdens_p?_s1_l2 = ghb\\density_l2.idf\n"
        "    ghbssmdens_p?_s1_l3 = ghb\\density_l3.idf\n"
        "\n"
        "[riv]\n"
        "    mrivsys = 1\n"
        "    mxactr = 75\n"
        "    irivcb = False\n"
        "    stage_p?_s1_l1 = riv\\stage_l1.idf\n"
        "    stage_p?_s1_l2 = riv\\stage_l2.idf\n"
        "    stage_p?_s1_l3 = riv\\stage_l3.idf\n"
        "    cond_p?_s1_l1 = riv\\conductance_l1.idf\n"
        "    cond_p?_s1_l2 = riv\\conductance_l2.idf\n"
        "    cond_p?_s1_l3 = riv\\conductance_l3.idf\n"
        "    rbot_p?_s1_l1 = riv\\bottom_elevation_l1.idf\n"
        "    rbot_p?_s1_l2 = riv\\bottom_elevation_l2.idf\n"
        "    rbot_p?_s1_l3 = riv\\bottom_elevation_l3.idf\n"
        "    rivssmdens_p?_s1_l1 = riv\\density_l1.idf\n"
        "    rivssmdens_p?_s1_l2 = riv\\density_l2.idf\n"
        "    rivssmdens_p?_s1_l3 = riv\\density_l3.idf\n"
        "\n"
        "[wel]\n"
        "    mwelsys = 1\n"
        "    mxactw = 1\n"
        "    iwelcb = False\n"
        "    wel_p1_s1_l2 = wel\\wel_20000101000000.ipf\n"
        "    wel_p2_s1_l2 = wel\\wel_20000102000000.ipf\n"
        "    wel_p3_s1_l2 = wel\\wel_20000103000000.ipf\n"
        "    wel_p4_s1_l2 = wel\\wel_20000104000000.ipf\n"
        "    wel_p5_s1_l2 = wel\\wel_20000105000000.ipf"
    )

    ssm_compare = (
        "[ssm]\n"
        "    mxss = 151\n"
        "    cghb_t1_p?_l1 = ghb\\concentration_l1.idf\n"
        "    cghb_t1_p?_l2 = ghb\\concentration_l2.idf\n"
        "    cghb_t1_p?_l3 = ghb\\concentration_l3.idf\n"
        "    criv_t1_p?_l1 = riv\\concentration_l1.idf\n"
        "    criv_t1_p?_l2 = riv\\concentration_l2.idf\n"
        "    criv_t1_p?_l3 = riv\\concentration_l3.idf"
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
        "    damp = 1.0"
    )
    assert m._render_flowsolver() == compare


def test_render_btn(basicmodel):
    m = basicmodel
    m.time_discretization(endtime="2000-01-06")
    diskey = m._get_pkgkey("dis")
    globaltimes = m[diskey]["time"].values
    directory = pathlib.Path(".")

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
        "    iprgcg = 0"
    )
    assert m._render_transportsolver() == compare


def test_render(basicmodel):
    m = basicmodel
    m.time_discretization(endtime="2000-01-06")

    s = m.render()
