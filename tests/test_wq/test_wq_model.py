import pathlib
import shutil
import textwrap

import imod
import imod.wq
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import cftime


@pytest.fixture(scope="module")
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
    m.time_discretization("2000-01-06")
    assert np.allclose(
        m["time_discretization"]["timestep_duration"].values, np.full(5, 1.0)
    )


def test_render_gen(basicmodel):
    m = basicmodel
    m.time_discretization("2000-01-06")
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
            packages = adv, bas6, btn, dis, dsp, gcg, ghb, lpf, oc, pcg, rch, riv, ssm, vdf, wel
            coord_xll = 0.0
            coord_yll = 0.0
            start_year = 2000
            start_month = 01
            start_day = 01"""
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
    m.time_discretization("2000-01-06")
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
    assert m._render_pkg("gcg", directory=directory, globaltimes=globaltimes) == compare


def test_render_pkg__rch(basicmodel):
    m = basicmodel
    m.time_discretization("2000-01-06")
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
    assert m._render_pkg("rch", directory=directory, globaltimes=globaltimes) == compare


def test_render_dis(basicmodel):
    m = basicmodel
    m.time_discretization("2000-01-06")
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
            perlen_p1 = 1.0
            perlen_p2 = 1.0
            perlen_p3 = 1.0
            perlen_p4 = 1.0
            perlen_p5 = 1.0
            nstp_p? = 1
            sstr_p? = tr
            tsmult_p? = 1.0"""
    )
    assert m._render_dis(directory=directory, globaltimes=globaltimes) == compare


def test_render_groups__ghb_riv_wel(basicmodel):
    m = basicmodel
    m.time_discretization("2000-01-06")
    diskey = m._get_pkgkey("dis")
    globaltimes = m[diskey]["time"].values
    directory = pathlib.Path(".")

    compare = textwrap.dedent(
        """\
        [ghb]
            mghbsys = 1
            mxactb = 75
            ighbcb = 0
            bhead_p?_s1_l1 = ghb/head_l1.idf
            bhead_p?_s1_l2 = ghb/head_l2.idf
            bhead_p?_s1_l3 = ghb/head_l3.idf
            cond_p?_s1_l1 = ghb/conductance_l1.idf
            cond_p?_s1_l2 = ghb/conductance_l2.idf
            cond_p?_s1_l3 = ghb/conductance_l3.idf
            ghbssmdens_p?_s1_l1 = ghb/density_l1.idf
            ghbssmdens_p?_s1_l2 = ghb/density_l2.idf
            ghbssmdens_p?_s1_l3 = ghb/density_l3.idf
        
        [riv]
            mrivsys = 1
            mxactr = 75
            irivcb = 0
            stage_p?_s1_l1 = riv/stage_l1.idf
            stage_p?_s1_l2 = riv/stage_l2.idf
            stage_p?_s1_l3 = riv/stage_l3.idf
            cond_p?_s1_l1 = riv/conductance_l1.idf
            cond_p?_s1_l2 = riv/conductance_l2.idf
            cond_p?_s1_l3 = riv/conductance_l3.idf
            rbot_p?_s1_l1 = riv/bottom_elevation_l1.idf
            rbot_p?_s1_l2 = riv/bottom_elevation_l2.idf
            rbot_p?_s1_l3 = riv/bottom_elevation_l3.idf
            rivssmdens_p?_s1_l1 = riv/density_l1.idf
            rivssmdens_p?_s1_l2 = riv/density_l2.idf
            rivssmdens_p?_s1_l3 = riv/density_l3.idf
        
        [wel]
            mwelsys = 1
            mxactw = 3
            iwelcb = 0
            wel_p1_s1_l? = wel/wel_20000101000000.ipf
            wel_p2_s1_l? = wel/wel_20000102000000.ipf
            wel_p3_s1_l? = wel/wel_20000103000000.ipf
            wel_p4_s1_l? = wel/wel_20000104000000.ipf
            wel_p5_s1_l? = wel/wel_20000105000000.ipf"""
    )

    ssm_compare = """
    cghb_t1_p?_l1 = ghb/concentration_l1.idf
    cghb_t1_p?_l2 = ghb/concentration_l2.idf
    cghb_t1_p?_l3 = ghb/concentration_l3.idf
    criv_t1_p?_l1 = riv/concentration_l1.idf
    criv_t1_p?_l2 = riv/concentration_l2.idf
    criv_t1_p?_l3 = riv/concentration_l3.idf"""
    content, ssm_content, n_sinkssources = m._render_groups(
        directory=directory, globaltimes=globaltimes
    )

    assert n_sinkssources == 153
    assert content == compare
    assert ssm_content == ssm_compare


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
    m.time_discretization("2000-01-06")
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
            sconc_t1_l1 = btn/starting_concentration_l1.idf
            sconc_t1_l2 = btn/starting_concentration_l2.idf
            sconc_t1_l3 = btn/starting_concentration_l3.idf
            icbund_l1 = btn/icbund_l1.idf
            icbund_l2 = btn/icbund_l2.idf
            icbund_l3 = btn/icbund_l3.idf
            dz_l1 = 10.0
            dz_l2 = 10.0
            dz_l3 = 10.0
            prsity_l1 = btn/porosity_l1.idf
            prsity_l2 = btn/porosity_l2.idf
            prsity_l3 = btn/porosity_l3.idf
            tsmult_p? = 1.0
            dt0_p? = 0.0
            mxstrn_p? = 50000"""
    )
    assert m._render_btn(directory=directory, globaltimes=globaltimes) == compare


def test_render_ssm_rch(basicmodel):
    m = basicmodel
    m.time_discretization("2000-01-06")
    diskey = m._get_pkgkey("dis")
    globaltimes = m[diskey]["time"].values
    directory = pathlib.Path(".")

    compare = """
    crch_t1_p1_l? = concentration_20000101000000.idf
    crch_t1_p2_l? = concentration_20000102000000.idf
    crch_t1_p3_l? = concentration_20000103000000.idf
    crch_t1_p4_l? = concentration_20000104000000.idf
    crch_t1_p5_l? = concentration_20000105000000.idf"""

    assert m._render_ssm_rch(directory=directory, globaltimes=globaltimes) == compare


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
    m.time_discretization("2000-01-06")
    d = pathlib.Path(".")
    r = pathlib.Path("results")
    s = m.render(d, r, False)


def test_render_cf(cftime_model):
    m_cf = cftime_model
    m_cf.time_discretization("2000-01-06")
    d = pathlib.Path(".")
    r = pathlib.Path("results")
    s = m_cf.render(d, r, False)


def test_render_notime(notime_model):
    m = notime_model
    m.time_discretization(times=["2000-01-01", "2000-01-06"])
    d = pathlib.Path(".")
    r = pathlib.Path("results")
    s = m.render(d, r, False)


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


def test_write(basicmodel, tmp_path):
    basicmodel.write(directory=tmp_path, result_dir=tmp_path / "results")
    # TODO: more rigorous testing


def test_write__timemap(basicmodel, tmp_path):
    # fictitious timemap
    timemap = {basicmodel["rch"].time.values[4]: basicmodel["rch"].time.values[0]}
    basicmodel["rch"].add_timemap(rate=timemap)
    basicmodel.write(directory=tmp_path, result_dir=tmp_path / "results")
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
        m.time_discretization("2000-01-06")


def test_write_result_dir(basicmodel, tmp_path):
    basicmodel.write(directory=tmp_path, result_dir=tmp_path / "results")
    # TODO: more rigorous testing


def test_write_result_dir_is_workdir(basicmodel, tmp_path):
    basicmodel.write(
        directory=tmp_path, result_dir=tmp_path / "results", resultdir_is_workdir=True
    )
    with open(tmp_path / "test_model.run") as f:
        lines = f.readlines()

    for line in lines:
        if "result" in line:
            break

    assert line.split("=")[-1].strip() == "."
    # TODO: more rigorous testing
