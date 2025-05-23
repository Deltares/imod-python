import os
import pathlib
import textwrap

import cftime
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import imod
import imod.wq


@pytest.fixture(scope="function")
def basicmodel():
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

    # RIV
    head = ibound.copy()

    # GHB, only part of domain
    ghbhead = ibound.isel(z=slice(0, 2))

    # CHD
    constanthead = xr.full_like(ibound.isel(z=-1), 1.0)

    # MAL
    mal_concentration = constanthead.copy()

    # TVC
    tvc_concentration = constanthead.copy()

    # RCH
    datetimes = pd.date_range("2000-01-01", "2000-01-05")
    rate = xr.DataArray(
        np.full((5, 5, 5), 1.0),
        coords={"time": datetimes, "y": y, "x": x, "dx": 1.0, "dy": -1.0},
        dims=("time", "y", "x"),
    )

    # EVT
    maximum_rate = xr.DataArray(
        np.full((5, 5, 5), 1.0),
        coords={"time": datetimes, "y": y, "x": x, "dx": 1.0, "dy": -1.0},
        dims=("time", "y", "x"),
    )

    # WEL
    welly = np.arange(4.5, 0.0, -1.0)
    wellx = np.arange(0.5, 5.0, 1.0)
    id_name = [f"well_{i}" for i in range(wellx.size)]

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
        head=ghbhead,
        conductance=ghbhead.copy(),
        concentration=1.5,
        density=ghbhead.copy(),
        save_budget=False,
    )
    m["chd"] = imod.wq.ConstantHead(
        head_start=constanthead,
        head_end=constanthead,
        concentration=35.0,
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
    m["wel"] = imod.wq.Well(id_name=id_name, x=wellx, y=welly, rate=5.0, time=datetimes)
    m["rch"] = imod.wq.RechargeHighestActive(
        rate=rate, concentration=rate.copy(), save_budget=False
    )
    m["evt"] = imod.wq.EvapotranspirationTopLayer(
        maximum_rate=maximum_rate,
        surface=maximum_rate.copy(),
        extinction_depth=maximum_rate.copy(),
        save_budget=False,
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
    m["mal"] = imod.wq.MassLoading(concentration=mal_concentration)
    m["tvc"] = imod.wq.TimeVaryingConstantConcentration(concentration=tvc_concentration)
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


@pytest.fixture(scope="function")
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
    m_notime["mal"] = m["mal"]
    m_notime["tvc"] = m["tvc"]
    return m_notime


@pytest.fixture(scope="function")
def cftime_model(basicmodel):
    m = basicmodel
    ibound = m["bas6"]["ibound"]

    m_cf = imod.wq.SeawatModel("test_model_cf")
    m_cf["lpf"] = m["lpf"]
    m_cf["chd"] = m["chd"]
    m_cf["riv"] = m["riv"]
    m_cf["pcg"] = m["pcg"]
    m_cf["btn"] = m["btn"]
    m_cf["adv"] = m["adv"]
    m_cf["dsp"] = m["dsp"]
    m_cf["vdf"] = m["vdf"]
    m_cf["gcg"] = m["gcg"]
    m_cf["oc"] = m["oc"]
    m_cf["mal"] = m["mal"]
    m_cf["tvc"] = m["tvc"]
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


@pytest.fixture(scope="function")
def model_different_names(basicmodel):
    m = basicmodel

    m_other_name = imod.wq.SeawatModel("test_model_different_names")
    m_other_name["_bas6"] = m["bas6"]
    m_other_name["_lpf"] = m["lpf"]
    m_other_name["_ghb"] = m["ghb"]
    m_other_name["_chd"] = m["chd"]
    m_other_name["_riv"] = m["riv"]
    m_other_name["_wel"] = m["wel"]
    m_other_name["_rch"] = m["rch"]
    m_other_name["_evt"] = m["evt"]
    m_other_name["_pcg"] = m["pcg"]
    m_other_name["_btn"] = m["btn"]
    m_other_name["_adv"] = m["adv"]
    m_other_name["_dsp"] = m["dsp"]
    m_other_name["_vdf"] = m["vdf"]
    m_other_name["_gcg"] = m["gcg"]
    m_other_name["_oc"] = m["oc"]
    m_other_name["_mal"] = m["mal"]
    m_other_name["_tvc"] = m["tvc"]
    return m_other_name


def test_get_pkgkey(basicmodel):
    m = basicmodel
    for key, package in m.items():
        assert key == package._pkg_id


def test_timediscretization(basicmodel):
    # m.create_time_discretization() changes the basicmodel object if not deepcopied.
    m = basicmodel
    m.create_time_discretization("2000-01-06")
    assert np.allclose(
        m["time_discretization"]["timestep_duration"].values, np.full(5, 1.0)
    )


def test_render_gen(basicmodel):
    m = basicmodel
    m.create_time_discretization("2000-01-06")
    diskey = m._get_pkgkey("dis")
    globaltimes = m[diskey]["time"].values
    modelname = m.modelname

    compare = textwrap.dedent(
        """\
        [gen]
            runtype = SEAWAT
            modelname = test_model
            writehelp = False
            result_dir = results
            packages = adv, bas6, btn, chd, dis, dsp, evt, gcg, ghb, lpf, oc, pcg, rch, riv, ssm, vdf, wel
            coord_xll = 0.0
            coord_yll = 0.0
            start_year = 2000
            start_month = 01
            start_day = 01
            start_hour = 00
            start_minute = 00"""
    )
    actual = m._render_gen(
        modelname=modelname,
        globaltimes=globaltimes,
        writehelp=False,
        result_dir="results",
    )
    assert actual == compare


def test_render_pkg__gcg(basicmodel):
    m = basicmodel
    m.create_time_discretization("2000-01-06")
    diskey = m._get_pkgkey("dis")
    globaltimes = m[diskey]["time"].values
    directory = pathlib.Path(".")

    compare = textwrap.dedent(
        """\
        [gcg]
            mxiter = 150
            iter1 = 30
            isolve = 3
            ncrs = 0
            cclose = 1e-06
            iprgcg = 0"""
    )
    assert (
        m._render_pkg("gcg", directory=directory, globaltimes=globaltimes, nlayer=3)
        == compare
    )


def test_render_pkg__evt(basicmodel):
    m = basicmodel
    m.create_time_discretization("2000-01-06")
    diskey = m._get_pkgkey("dis")
    globaltimes = m[diskey]["time"].values
    directory = pathlib.Path(".")

    compare = textwrap.dedent(
        """\
        [evt]
            nevtop = 1
            ievtcb = 0
            evtr_p1 = evt/maximum_rate_20000101000000.idf
            evtr_p2 = evt/maximum_rate_20000102000000.idf
            evtr_p3 = evt/maximum_rate_20000103000000.idf
            evtr_p4 = evt/maximum_rate_20000104000000.idf
            evtr_p5 = evt/maximum_rate_20000105000000.idf
            surf_p1 = evt/surface_20000101000000.idf
            surf_p2 = evt/surface_20000102000000.idf
            surf_p3 = evt/surface_20000103000000.idf
            surf_p4 = evt/surface_20000104000000.idf
            surf_p5 = evt/surface_20000105000000.idf
            exdp_p1 = evt/extinction_depth_20000101000000.idf
            exdp_p2 = evt/extinction_depth_20000102000000.idf
            exdp_p3 = evt/extinction_depth_20000103000000.idf
            exdp_p4 = evt/extinction_depth_20000104000000.idf
            exdp_p5 = evt/extinction_depth_20000105000000.idf"""
    )
    assert (
        m._render_pkg("evt", directory=directory, globaltimes=globaltimes, nlayer=3)
        == compare
    )


def test_render_pkg__rch(basicmodel):
    m = basicmodel
    m.create_time_discretization("2000-01-06")
    diskey = m._get_pkgkey("dis")
    globaltimes = m[diskey]["time"].values
    directory = pathlib.Path(".")

    compare = textwrap.dedent(
        """\
        [rch]
            nrchop = 3
            irchcb = 0
            rech_p1 = rch/rate_20000101000000.idf
            rech_p2 = rch/rate_20000102000000.idf
            rech_p3 = rch/rate_20000103000000.idf
            rech_p4 = rch/rate_20000104000000.idf
            rech_p5 = rch/rate_20000105000000.idf"""
    )
    assert (
        m._render_pkg("rch", directory=directory, globaltimes=globaltimes, nlayer=3)
        == compare
    )


def test_render_dis(basicmodel):
    m = basicmodel
    m.create_time_discretization("2000-01-06")
    diskey = m._get_pkgkey("dis")
    globaltimes = m[diskey]["time"].values
    directory = pathlib.Path(".")

    compare = textwrap.dedent(
        """\
        [dis]
            nlay = 3
            nrow = 5
            ncol = 5
            delc_r? = 1.0
            delr_c? = 1.0
            top = 30.0
            botm_l1 = 20.0
            botm_l2 = 10.0
            botm_l3 = 0.0
            laycbd_l? = 0
            nper = 5
            perlen_p1:5 = 1.0
            nstp_p? = 1
            sstr_p? = tr
            tsmult_p? = 1.0"""
    )
    assert (
        m._render_dis(directory=directory, globaltimes=globaltimes, nlayer=3) == compare
    )


def test_render_groups__ghb_riv_wel(basicmodel):
    m = basicmodel
    m.create_time_discretization("2000-01-06")
    diskey = m._get_pkgkey("dis")
    globaltimes = m[diskey]["time"].values
    directory = pathlib.Path(".")

    compare = textwrap.dedent(
        """\
        [ghb]
            mghbsys = 1
            mxactb = 50
            ighbcb = 0
            bhead_p?_s1_l1:2 = ghb/head_l:.idf
            cond_p?_s1_l1:2 = ghb/conductance_l:.idf
            ghbssmdens_p?_s1_l1:2 = ghb/density_l:.idf

        [chd]
            mchdsys = 1
            mxactc = 25
            ichdcb = 0
            shead_p?_s1_l3 = chd/head_start_l3.idf
            ehead_p?_s1_l3 = chd/head_end_l3.idf

        [riv]
            mrivsys = 1
            mxactr = 75
            irivcb = 0
            stage_p?_s1_l$ = riv/stage_l$.idf
            cond_p?_s1_l$ = riv/conductance_l$.idf
            rbot_p?_s1_l$ = riv/bottom_elevation_l$.idf
            rivssmdens_p?_s1_l$ = riv/density_l$.idf

        [wel]
            mwelsys = 1
            mxactw = 15
            iwelcb = 0
            wel_p?_s1_l$ = wel/rate.ipf"""
    )

    ssm_compare = """
    cghb_t1_p?_l1:2 = 1.5
    cchd_t1_p?_l3 = 35.0
    criv_t1_p?_l$ = riv/concentration_l$.idf"""
    content, ssm_content, n_sinkssources = m._render_groups(
        directory=directory, globaltimes=globaltimes
    )

    assert n_sinkssources == 165
    assert content == compare
    assert ssm_content == ssm_compare


def test_render_groups__double_gbh(basicmodel):
    m = basicmodel
    ghbhead = m["ghb"]["head"].copy()
    m["ghb2"] = imod.wq.GeneralHeadBoundary(
        head=ghbhead,
        conductance=ghbhead.copy(),
        density=ghbhead.copy(),
        save_budget=False,
    )
    m.create_time_discretization("2000-01-06")
    diskey = m._get_pkgkey("dis")
    globaltimes = m[diskey]["time"].values
    directory = pathlib.Path(".")

    n_sinkssources = m._render_groups(directory=directory, globaltimes=globaltimes)[2]
    assert n_sinkssources == 215


def test_render_flowsolver(basicmodel):
    m = basicmodel
    directory = pathlib.Path(".")

    compare = textwrap.dedent(
        """\
        [pcg]
            mxiter = 150
            iter1 = 30
            npcond = 1
            hclose = 0.0001
            rclose = 1000.0
            relax = 0.98
            iprpcg = 1
            mutpcg = 0
            damp = 1.0"""
    )
    assert m._render_flowsolver(directory) == compare


def test_render_btn(basicmodel):
    m = basicmodel
    m.create_time_discretization("2000-01-06")
    diskey = m._get_pkgkey("dis")
    globaltimes = m[diskey]["time"].values
    directory = pathlib.Path(".")

    compare = textwrap.dedent(
        """\
        [btn]
            ncomp = 1
            mcomp = 1
            thkmin = 0.01
            cinact = 1e+30
            sconc_t1_l$ = btn/starting_concentration_l$.idf
            icbund_l$ = btn/icbund_l$.idf
            dz_l1:3 = 10.0
            prsity_l$ = btn/porosity_l$.idf
            tsmult_p? = 1.0
            dt0_p? = 0.0
            mxstrn_p? = 50000"""
    )
    assert (
        m._render_btn(directory=directory, globaltimes=globaltimes, nlayer=3) == compare
    )


def test_render_ssm_rch_evt_mal_tvc(basicmodel):
    m = basicmodel
    m.create_time_discretization("2000-01-06")
    diskey = m._get_pkgkey("dis")
    globaltimes = m[diskey]["time"].values
    directory = pathlib.Path(".")

    compare = """
    crch_t1_p1_l? = rch/concentration_20000101000000.idf
    crch_t1_p2_l? = rch/concentration_20000102000000.idf
    crch_t1_p3_l? = rch/concentration_20000103000000.idf
    crch_t1_p4_l? = rch/concentration_20000104000000.idf
    crch_t1_p5_l? = rch/concentration_20000105000000.idf
    cevt_t1_p?_l? = 0.0
    cmal_t1_p?_l3 = mal/concentration_l3.idf
    ctvc_t1_p?_l3 = tvc/concentration_l3.idf"""

    actual = m._render_ssm_rch_evt_mal_tvc(
        directory=directory, globaltimes=globaltimes, nlayer=3
    )
    assert actual == compare


def test_render_ssm_rch_constant(basicmodel):
    # Make sure it only writes crch for layers in which recharge are constant.
    m = basicmodel
    m["rch"] = imod.wq.RechargeHighestActive(
        rate=0.001, concentration=0.15, save_budget=False
    )
    m.create_time_discretization("2000-01-06")
    diskey = m._get_pkgkey("dis")
    globaltimes = m[diskey]["time"].values
    directory = pathlib.Path(".")

    compare = """
    crch_t1_p?_l1 = 0.15
    cevt_t1_p?_l1 = 0.0
    cmal_t1_p?_l3 = mal/concentration_l3.idf
    ctvc_t1_p?_l3 = tvc/concentration_l3.idf"""

    # Setup ssm_layers
    m._bas_btn_rch_evt_mal_tvc_sinkssources()
    assert hasattr(m["rch"], "_ssm_layers")

    actual = m._render_ssm_rch_evt_mal_tvc(
        directory=directory, globaltimes=globaltimes, nlayer=3
    )
    assert actual == compare

    compare = """
    crch_t1_p?_l1:2 = 0.15
    cevt_t1_p?_l1 = 0.0
    cmal_t1_p?_l3 = mal/concentration_l3.idf
    ctvc_t1_p?_l3 = tvc/concentration_l3.idf"""

    # Setup ssm_layers
    m["bas6"]["ibound"][0, 0, 0] = 0.0
    m._bas_btn_rch_evt_mal_tvc_sinkssources()
    assert hasattr(m["rch"], "_ssm_layers")

    actual = m._render_ssm_rch_evt_mal_tvc(
        directory=directory, globaltimes=globaltimes, nlayer=3
    )
    assert actual == compare


def test_render_transportsolver(basicmodel):
    m = basicmodel
    directory = pathlib.Path(".")

    compare = textwrap.dedent(
        """\
        [gcg]
            mxiter = 150
            iter1 = 30
            isolve = 3
            ncrs = 0
            cclose = 1e-06
            iprgcg = 0"""
    )
    assert m._render_transportsolver(directory) == compare


def test_render(basicmodel):
    m = basicmodel
    m.create_time_discretization("2000-01-06")
    d = pathlib.Path(".")
    r = pathlib.Path("results")
    _ = m.render(d, r, False)


def test_render_cf(cftime_model):
    m_cf = cftime_model
    m_cf.create_time_discretization("2000-01-06")
    d = pathlib.Path(".")
    r = pathlib.Path("results")
    _ = m_cf.render(d, r, False)


def test_render_notime(notime_model):
    m = notime_model
    m.create_time_discretization(additional_times=["2000-01-01", "2000-01-06"])
    d = pathlib.Path(".")
    r = pathlib.Path("results")
    _ = m.render(d, r, False)


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
    m["bas6"]["ibound"][1:, ...] = -1.0
    m["btn"]["icbund"][...] = 0.0

    n_sinkssources = m._bas_btn_rch_evt_mal_tvc_sinkssources()
    # 50 from ibound, 25 from recharge + 25 from evapotranspiration, 25 from mal, 25 from tvc
    assert n_sinkssources == 50 + 25 + 25 + 25 + 25


def test_highest_active_recharge(basicmodel):
    m = basicmodel
    n_sinkssources = m._bas_btn_rch_evt_mal_tvc_sinkssources()
    assert np.array_equal(m["rch"]._ssm_layers, np.array([1]))
    assert n_sinkssources == 25 + 25 + 25 + 25

    m["bas6"]["ibound"][0, 0, 0] = 0.0
    m["btn"]["icbund"][...] = 0.0
    n_sinkssources = m._bas_btn_rch_evt_mal_tvc_sinkssources()
    assert np.array_equal(m["rch"]._ssm_layers, np.array([1, 2]))
    assert n_sinkssources == 50 + 25 + 25 + 25

    m["bas6"]["ibound"][...] = -1.0
    m["btn"]["icbund"][...] = 0.0
    n_sinkssources = m._bas_btn_rch_evt_mal_tvc_sinkssources()
    assert np.array_equal(m["rch"]._ssm_layers, np.array([]))
    assert n_sinkssources == 75 + 25 + 25 + 25


def test_write(basicmodel, tmp_path):
    m = basicmodel
    m.create_time_discretization("2000-01-06")
    m.write(directory=tmp_path, result_dir=tmp_path / "results")
    # TODO: more rigorous testing


def test_write_pkgname_is_not_pkg_id(model_different_names, tmp_path):
    """
    Test if model with pkgnames which differ from _pkg_id writes without error.
    """
    m = model_different_names
    m.create_time_discretization("2000-01-06")
    m.write(directory=tmp_path, result_dir=tmp_path / "results")


def test_write__stress_repeats(basicmodel, tmp_path):
    # fictitious stress_repeats
    m = basicmodel

    # Remove last timestep of recharge package
    m["rch"] = imod.wq.RechargeHighestActive(
        **m["rch"].dataset.isel(time=slice(None, 4))
    )

    stress_repeats = {
        m["evt"].dataset["time"].values[4]: m["evt"].dataset["time"].values[0]
    }
    m["rch"].repeat_stress(rate=stress_repeats)

    m.create_time_discretization("2000-01-06")

    m.write(directory=tmp_path, result_dir=tmp_path / "results")

    # Test if test worked, only 8 files written, instead of 10
    assert len(os.listdir(tmp_path / "rch")) == 8

    # 'rch/rate_20000101000000.idf' should occur twice now in runfile
    runfile = open(tmp_path / "test_model.run").read()
    count = runfile.count("rch/rate_20000101000000.idf")
    assert count == 2


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
        m.create_time_discretization("2000-01-06")


def test_write_result_dir(basicmodel, tmp_path):
    m = basicmodel
    m.create_time_discretization("2000-01-06")
    m.write(directory=tmp_path, result_dir=tmp_path / "results")
    # TODO: more rigorous testing


def test_write_result_dir_is_workdir(basicmodel, tmp_path):
    m = basicmodel
    m.create_time_discretization("2000-01-06")

    m.write(
        directory=tmp_path, result_dir=tmp_path / "results", resultdir_is_workdir=True
    )
    with open(tmp_path / "test_model.run") as f:
        lines = f.readlines()

    for line in lines:
        if "result" in line:
            break

    assert line.split("=")[-1].strip() == "."
    # TODO: more rigorous testing


def test_write__error_space_in_path(basicmodel, tmp_path):
    """
    Test if error is raised when there is a space in the path.
    """
    m = basicmodel
    m.create_time_discretization("2000-01-06")

    txt_to_match = r"Spaces in directory names are not allowed"
    with pytest.raises(ValueError, match=txt_to_match):
        m.write(
            directory=tmp_path / "test model",
            result_dir=tmp_path / "results",
            resultdir_is_workdir=True,
        )

    with pytest.raises(ValueError, match=txt_to_match):
        m.write(
            directory=tmp_path / "test_model",
            result_dir=tmp_path / "re sults",
            resultdir_is_workdir=True,
        )
