import pathlib
import shutil

import imod
import imod.wq
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import cftime


@pytest.fixture(scope="module")
def basicmodel(request):

    # Basic flow
    layer = np.arange(1, 4)
    z = np.arange(25.0, 0.0, -10.0)
    y = np.arange(4.5, 0.0, -1.0)
    x = np.arange(0.5, 5.0, 1.0)
    ibound = xr.DataArray(
        np.full((3, 5, 5), 1.0),
        coords={"z": z, "layer": ("z", layer), "y": y, "x": x, "dx": 1.0, "dy": -1.0},
        dims=("z", "y", "x"),
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
    m["bas6"] = imod.wq.BasicFlow(
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
    m["wel"] = imod.wq.Well(id_name="well", x=wellx, y=welly, rate=5.0, time=datetimes)
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

    def teardown():
        try:
            shutil.rmtree("test_model")
        except FileNotFoundError:
            pass

    request.addfinalizer(teardown)

    return m


@pytest.fixture(scope="module")
def notime_model(basicmodel):
    m = basicmodel

    m_notime = imod.wq.SeawatModel("test_model_notime")
    m_notime["bas6"] = m["bas6"]
    m_notime["lpf"] = m["lpf"]
    m_notime["riv"] = m["riv"]
    m_notime["pcg"] = m["pcg"]
    m_notime["btn"] = m["btn"]
    m_notime["adv"] = m["adv"]
    m_notime["dsp"] = m["dsp"]
    m_notime["vdf"] = m["vdf"]
    m_notime["gcg"] = m["gcg"]
    m_notime["oc"] = m["oc"]
    return m_notime


@pytest.fixture(scope="module")
def cftime_model(basicmodel):
    m = basicmodel
    ibound = m["bas6"]["ibound"]

    m_cf = imod.wq.SeawatModel("test_model_cf")
    m_cf["lpf"] = m["lpf"]
    m_cf["riv"] = m["riv"]
    m_cf["pcg"] = m["pcg"]
    m_cf["btn"] = m["btn"]
    m_cf["adv"] = m["adv"]
    m_cf["dsp"] = m["dsp"]
    m_cf["vdf"] = m["vdf"]
    m_cf["gcg"] = m["gcg"]
    m_cf["oc"] = m["oc"]
    # We are not going to test for wells for now,
    # as cftime is not supported in WEL yet

    layer = np.arange(1, 4)
    top = 30.0
    bot = xr.DataArray(
        np.arange(20.0, -10.0, -10.0), coords={"layer": layer}, dims=("layer",)
    )

    m_cf["bas6"] = imod.wq.BasicFlow(
        ibound=ibound, top=top, bottom=bot, starting_head=ibound.copy()
    )

    times = np.array(
        [
            cftime.DatetimeProlepticGregorian(i, 1, 1)
            for i in [2000, 3000, 4000, 5000, 6000]
        ]
    )
    da_t = xr.DataArray(np.ones(len(times)), coords={"time": times}, dims=("time",))

    # Instead of WEL, we broadcast
    head = ibound * da_t

    m_cf["ghb"] = imod.wq.GeneralHeadBoundary(
        head=head,
        conductance=head.copy(),
        concentration=head.copy(),
        density=head.copy(),
        save_budget=False,
    )

    return m_cf


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
        "    runtype = SEAWAT\n"
        "    modelname = test_model\n"
        "    writehelp = False\n"
        "    result_dir = results\n"
        "    packages = adv, bas6, btn, dis, dsp, gcg, ghb, lpf, oc, pcg, rch, riv, ssm, vdf, wel\n"
        "    coord_xll = 0.0\n"
        "    coord_yll = 0.0\n"
        "    start_year = 2000\n"
        "    start_month = 01\n"
        "    start_day = 01"
    )
    assert (
        m._render_gen(
            modelname=modelname,
            globaltimes=globaltimes,
            writehelp=False,
            result_dir="results",
        )
        == compare
    )


def test_render_pkg__gcg(basicmodel):
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


def test_render_pkg__rch(basicmodel):
    m = basicmodel
    m.time_discretization(endtime="2000-01-06")
    diskey = m._get_pkgkey("dis")
    globaltimes = m[diskey]["time"].values
    modelname = m.modelname
    directory = pathlib.Path(".")
    path = pathlib.Path("rch").joinpath("rate")

    compare = (
        "[rch]\n"
        "    nrchop = 3\n"
        "    irchcb = 0\n"
        "    rech_p1 = {path}_20000101000000.idf\n"
        "    rech_p2 = {path}_20000102000000.idf\n"
        "    rech_p3 = {path}_20000103000000.idf\n"
        "    rech_p4 = {path}_20000104000000.idf\n"
        "    rech_p5 = {path}_20000105000000.idf"
    ).format(path=path)
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
        "    laycbd_l? = 0\n"
        "    nper = 5\n"
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
        "    ighbcb = 0\n"
        "    bhead_p?_s1_l1 = {gh}_l1.idf\n"
        "    bhead_p?_s1_l2 = {gh}_l2.idf\n"
        "    bhead_p?_s1_l3 = {gh}_l3.idf\n"
        "    cond_p?_s1_l1 = {gc}_l1.idf\n"
        "    cond_p?_s1_l2 = {gc}_l2.idf\n"
        "    cond_p?_s1_l3 = {gc}_l3.idf\n"
        "    ghbssmdens_p?_s1_l1 = {gd}_l1.idf\n"
        "    ghbssmdens_p?_s1_l2 = {gd}_l2.idf\n"
        "    ghbssmdens_p?_s1_l3 = {gd}_l3.idf\n"
        "\n"
        "[riv]\n"
        "    mrivsys = 1\n"
        "    mxactr = 75\n"
        "    irivcb = 0\n"
        "    stage_p?_s1_l1 = {rs}_l1.idf\n"
        "    stage_p?_s1_l2 = {rs}_l2.idf\n"
        "    stage_p?_s1_l3 = {rs}_l3.idf\n"
        "    cond_p?_s1_l1 = {rc}_l1.idf\n"
        "    cond_p?_s1_l2 = {rc}_l2.idf\n"
        "    cond_p?_s1_l3 = {rc}_l3.idf\n"
        "    rbot_p?_s1_l1 = {re}_l1.idf\n"
        "    rbot_p?_s1_l2 = {re}_l2.idf\n"
        "    rbot_p?_s1_l3 = {re}_l3.idf\n"
        "    rivssmdens_p?_s1_l1 = {rd}_l1.idf\n"
        "    rivssmdens_p?_s1_l2 = {rd}_l2.idf\n"
        "    rivssmdens_p?_s1_l3 = {rd}_l3.idf\n"
        "\n"
        "[wel]\n"
        "    mwelsys = 1\n"
        "    mxactw = 3\n"
        "    iwelcb = 0\n"
        "    wel_p1_s1_l? = {welpath}_20000101000000.ipf\n"
        "    wel_p2_s1_l? = {welpath}_20000102000000.ipf\n"
        "    wel_p3_s1_l? = {welpath}_20000103000000.ipf\n"
        "    wel_p4_s1_l? = {welpath}_20000104000000.ipf\n"
        "    wel_p5_s1_l? = {welpath}_20000105000000.ipf"
    ).format(
        gh=pathlib.Path("ghb").joinpath("head"),
        gc=pathlib.Path("ghb").joinpath("conductance"),
        gd=pathlib.Path("ghb").joinpath("density"),
        rs=pathlib.Path("riv").joinpath("stage"),
        rc=pathlib.Path("riv").joinpath("conductance"),
        re=pathlib.Path("riv").joinpath("bottom_elevation"),
        rd=pathlib.Path("riv").joinpath("density"),
        welpath=pathlib.Path("wel").joinpath("wel"),
    )  # Format is necessary because of Windows versus Unix paths

    ssm_compare = (
        "\n"
        "    cghb_t1_p?_l1 = {gc}_l1.idf\n"
        "    cghb_t1_p?_l2 = {gc}_l2.idf\n"
        "    cghb_t1_p?_l3 = {gc}_l3.idf\n"
        "    criv_t1_p?_l1 = {rc}_l1.idf\n"
        "    criv_t1_p?_l2 = {rc}_l2.idf\n"
        "    criv_t1_p?_l3 = {rc}_l3.idf"
    ).format(
        gc=pathlib.Path("ghb").joinpath("concentration"),
        rc=pathlib.Path("riv").joinpath("concentration"),
    )
    content, ssm_content, n_sinkssources = m._render_groups(
        directory=directory, globaltimes=globaltimes
    )

    assert n_sinkssources == 153
    assert content == compare
    assert ssm_content == ssm_compare


def test_render_flowsolver(basicmodel):
    m = basicmodel
    directory = pathlib.Path(".")

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
    assert m._render_flowsolver(directory) == compare


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
        "    sconc_t1_l1 = {sc}_concentration_l1.idf\n"
        "    sconc_t1_l2 = {sc}_concentration_l2.idf\n"
        "    sconc_t1_l3 = {sc}_concentration_l3.idf\n"
        "    icbund_l1 = {ic}_l1.idf\n"
        "    icbund_l2 = {ic}_l2.idf\n"
        "    icbund_l3 = {ic}_l3.idf\n"
        "    dz_l1 = 10.0\n"
        "    dz_l2 = 10.0\n"
        "    dz_l3 = 10.0\n"
        "    prsity_l1 = {pr}_l1.idf\n"
        "    prsity_l2 = {pr}_l2.idf\n"
        "    prsity_l3 = {pr}_l3.idf\n"
        "    tsmult_p? = 1.0\n"
        "    dt0_p? = 0.0\n"
        "    mxstrn_p? = 50000"
    ).format(
        sc=pathlib.Path("btn").joinpath("starting"),
        ic=pathlib.Path("btn").joinpath("icbund"),
        pr=pathlib.Path("btn").joinpath("porosity"),
    )
    assert m._render_btn(directory=directory, globaltimes=globaltimes) == compare


def test_render_ssm_rch(basicmodel):
    m = basicmodel
    m.time_discretization(endtime="2000-01-06")
    diskey = m._get_pkgkey("dis")
    globaltimes = m[diskey]["time"].values
    directory = pathlib.Path(".")

    compare = (
        "\n"
        "    crch_t1_p1_l? = concentration_20000101000000.idf\n"
        "    crch_t1_p2_l? = concentration_20000102000000.idf\n"
        "    crch_t1_p3_l? = concentration_20000103000000.idf\n"
        "    crch_t1_p4_l? = concentration_20000104000000.idf\n"
        "    crch_t1_p5_l? = concentration_20000105000000.idf"
    )
    assert m._render_ssm_rch(directory=directory, globaltimes=globaltimes) == compare


def test_render_transportsolver(basicmodel):
    m = basicmodel
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
    assert m._render_transportsolver(directory) == compare


def test_render(basicmodel):
    m = basicmodel
    m.time_discretization(endtime="2000-01-06")

    s = m.render()


def test_render_cf(cftime_model):
    m_cf = cftime_model
    m_cf.time_discretization(endtime="2000-01-06")

    s = m_cf.render()


def test_render_notime(notime_model):
    m = notime_model
    m.time_discretization(starttime="2000-01-01", endtime="2000-01-06")
    s = m.render()


def test_mxsscount_incongruent_icbund(basicmodel):
    """
    MT3D relies on the ICBUND to identify constant concentration cells. Seawat
    also always has an IBOUND for the flow. In Seawat, the IBOUND will be
    included to count the number of sources and sinks. I think Seawat will
    merge the IBOUND into the ICBUND (if IBOUND == -1, then ICBUND will bet set
    to -1 too).

    This tests makes sure that the IBOUND is also counted in the determination
    of the number of sinks and sources (MXSS).

    This test mutates the basicmodel provided by the fixture!
    """

    m = basicmodel
    m["bas6"]["ibound"][...] = -1.0
    m["btn"]["icbund"][...] = 0.0

    n_sinkssources = m._bas_btn_rch_sinkssources()
    assert n_sinkssources == 100


def test_write(basicmodel):
    basicmodel.write()
    assert pathlib.Path("test_model").exists()
    # TODO: more rigorous testing


def test_write__timemap(basicmodel):
    # fictitious timemap
    timemap = {basicmodel["rch"].time.values[4]: basicmodel["rch"].time.values[0]}
    basicmodel["rch"].add_timemap(rate=timemap)
    basicmodel.write()
    assert pathlib.Path("test_model").exists()
    # TODO: more rigorous testing


def test_write__error_stress_time_not_first(basicmodel):
    """
    In this case, the WEL package isn't specified for the first stress period.
    This should raise an error.
    """
    m = basicmodel
    datetimes = pd.date_range("2000-01-01", "2000-01-05")[1:]
    # WEL
    welly = np.arange(4.5, 0.0, -1.0)[1:]
    wellx = np.arange(0.5, 5.0, 1.0)[1:]
    m["wel"] = imod.wq.Well(id_name="well", x=wellx, y=welly, rate=5.0, time=datetimes)
    with pytest.raises(ValueError):
        m.time_discretization(endtime="2000-01-06")


def test_write_result_dir(basicmodel):
    basicmodel.write(result_dir="results")
    assert pathlib.Path("test_model").exists()
    # TODO: more rigorous testing
