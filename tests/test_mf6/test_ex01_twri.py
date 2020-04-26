import pathlib
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import imod


@pytest.fixture(scope="module")
def twri_model():
    nlay = 3
    nrow = 15
    ncol = 15
    shape = (nlay, nrow, ncol)

    dx = 5000.0
    dy = -5000.0
    xmin = 0.0
    xmax = dx * ncol
    ymin = 0.0
    ymax = abs(dy) * nrow
    dims = ("layer", "y", "x")

    layer = np.array([1, 2, 3])
    y = np.arange(ymax, ymin, dy) + 0.5 * dy
    x = np.arange(xmin, xmax, dx) + 0.5 * dx
    coords = {"layer": layer, "y": y, "x": x}

    # Discretization data
    idomain = xr.DataArray(np.ones(shape), coords=coords, dims=dims)
    bottom = xr.DataArray([-200.0, -300.0, -450.0], {"layer": layer}, ("layer",))

    # Constant head
    head = xr.full_like(idomain, np.nan).sel(layer=[1, 2])
    head[...] = np.nan
    head[..., 0] = 0.0

    # Drainage
    elevation = xr.full_like(idomain.sel(layer=1), np.nan)
    conductance = xr.full_like(idomain.sel(layer=1), np.nan)
    elevation[7, 1:10] = np.array([0.0, 0.0, 10.0, 20.0, 30.0, 50.0, 70.0, 90.0, 100.0])
    conductance[7, 1:10] = 1.0

    # Node properties
    icelltype = xr.DataArray([1, 0, 0], {"layer": layer}, ("layer",))
    k = xr.DataArray([1.0e-3, 1.0e-4, 2.0e-4], {"layer": layer}, ("layer",))
    k33 = xr.DataArray([2.0e-8, 2.0e-8, 2.0e-8], {"layer": layer}, ("layer",))

    # Recharge
    rch_rate = xr.full_like(idomain.sel(layer=1), 3.0e-8)

    # Well
    layer = [3, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    row = [5, 4, 6, 9, 9, 9, 9, 11, 11, 11, 11, 13, 13, 13, 13]
    column = [11, 6, 12, 8, 10, 12, 14, 8, 10, 12, 14, 8, 10, 12, 14]
    rate = [
        -5.0,
        -5.0,
        -5.0,
        -5.0,
        -5.0,
        -5.0,
        -5.0,
        -5.0,
        -5.0,
        -5.0,
        -5.0,
        -5.0,
        -5.0,
        -5.0,
        -5.0,
    ]

    # Create and fill the groundwater model.
    gwf_model = imod.mf6.GroundwaterFlowModel()
    gwf_model["dis"] = imod.mf6.StructuredDiscretization(
        top=200.0, bottom=bottom, idomain=idomain
    )
    gwf_model["chd"] = imod.mf6.ConstantHead(
        head, print_input=True, print_flows=True, save_flows=True
    )
    gwf_model["drn"] = imod.mf6.Drainage(
        elevation=elevation,
        conductance=conductance,
        print_input=True,
        print_flows=True,
        save_flows=True,
    )
    gwf_model["ic"] = imod.mf6.InitialConditions(head=0.0)
    gwf_model["npf"] = imod.mf6.NodePropertyFlow(
        icelltype=icelltype,
        k=k,
        k33=k33,
        variable_vertical_conductance=True,
        dewatered=True,
        perched=True,
        save_flows=True,
    )
    gwf_model["oc"] = imod.mf6.OutputControl(save_head=True, save_budget=True)
    gwf_model["rch"] = imod.mf6.Recharge(rch_rate)
    gwf_model["wel"] = imod.mf6.Well(
        layer=layer,
        row=row,
        column=column,
        rate=rate,
        print_input=True,
        print_flows=True,
        save_flows=True,
    )

    # Attach it to a simulation
    simulation = imod.mf6.Modflow6Simulation("ex01-twri")
    simulation["GWF_1"] = gwf_model
    # Define solver settings
    simulation["solver"] = imod.mf6.Solution(
        print_option="summary",
        csv_output=False,
        no_ptc=True,
        outer_hclose=1.0e-4,
        outer_maximum=500,
        under_relaxation=None,
        inner_hclose=1.0e-4,
        inner_rclose=0.001,
        inner_maximum=100,
        linear_acceleration="cg",
        scaling_method=None,
        reordering_method=None,
        relaxation_factor=0.97,
    )
    # Collect time discretization
    simulation.time_discretization(times=["2000-01-01", "2000-01-02"])
    return simulation


def test_dis_render(twri_model, tmp_path):
    simulation = twri_model
    dis = simulation["GWF_1"]["dis"]
    actual = dis.render(tmp_path, "dis")
    path = tmp_path.as_posix()
    expected = textwrap.dedent(
        f"""\
            begin options
              xorigin 0.0
              yorigin 0.0
            end options
            
            begin dimensions
              nlay 3
              nrow 15
              ncol 15
            end dimensions
            
            begin griddata
              delr
                constant 5000.0
              delc
                constant 5000.0
              top
                constant 200.0
              botm layered
                constant -200.0
                constant -300.0
                constant -450.0
              idomain
                open/close {path}/dis/idomain.bin (binary)
            end griddata"""
    )
    assert actual == expected
    dis.write(tmp_path, "dis")
    assert (tmp_path / "dis.dis").is_file()
    assert (tmp_path / "dis").is_dir()
    assert (tmp_path / "dis" / "idomain.bin").is_file()


def test_chd_render(twri_model, tmp_path):
    simulation = twri_model
    globaltimes = simulation["time_discretization"]["time"].values
    chd = simulation["GWF_1"]["chd"]
    actual = chd.render(tmp_path, "chd", globaltimes)
    path = tmp_path.as_posix()
    expected = textwrap.dedent(
        f"""\
            begin options
              print_input
              print_flows
              save_flows
            end options
            
            begin dimensions
              maxbound 30
            end dimensions
            
            begin period 1
              open/close {path}/chd/chd.bin (binary)
            end period"""
    )
    assert actual == expected
    chd.write(tmp_path, "chd", globaltimes)
    assert (tmp_path / "chd.chd").is_file()
    assert (tmp_path / "chd").is_dir()
    assert (tmp_path / "chd" / "chd.bin").is_file()


def test_drn_render(twri_model, tmp_path):
    simulation = twri_model
    globaltimes = simulation["time_discretization"]["time"].values
    drn = simulation["GWF_1"]["drn"]
    actual = drn.render(tmp_path, "drn", globaltimes)
    path = tmp_path.as_posix()
    expected = textwrap.dedent(
        f"""\
            begin options
              print_input
              print_flows
              save_flows
            end options
            
            begin dimensions
              maxbound 9
            end dimensions
            
            begin period 1
              open/close {path}/drn/drn.bin (binary)
            end period"""
    )
    assert actual == expected
    drn.write(tmp_path, "drn", globaltimes)
    assert (tmp_path / "drn.drn").is_file()
    assert (tmp_path / "drn").is_dir()
    assert (tmp_path / "drn" / "drn.bin").is_file()


def test_ic_render(twri_model, tmp_path):
    simulation = twri_model
    ic = simulation["GWF_1"]["ic"]
    actual = ic.render(tmp_path, "ic")
    expected = textwrap.dedent(
        """\
            begin options
            end options

            begin griddata
              strt
                constant 0.0
            end griddata"""
    )
    assert actual == expected
    ic.write(tmp_path, "ic")
    assert (tmp_path / "ic.ic").is_file()


def test_npf_render(twri_model, tmp_path):
    simulation = twri_model
    npf = simulation["GWF_1"]["npf"]
    actual = npf.render(tmp_path, "npf")
    expected = textwrap.dedent(
        """\
            begin options
              save_flows
              variablecv dewatered
              perched
            end options

            begin griddata
              icelltype layered
                constant 1
                constant 0
                constant 0
              k layered
                constant 0.001
                constant 0.0001
                constant 0.0002
              k33 layered
                constant 2e-08
                constant 2e-08
                constant 2e-08
            end griddata"""
    )
    assert actual == expected
    npf.write(tmp_path, "npf")
    assert (tmp_path / "npf.npf").is_file()


def test_oc_render(twri_model, tmp_path):
    simulation = twri_model
    globaltimes = simulation["time_discretization"]["time"].values
    oc = simulation["GWF_1"]["oc"]
    path = tmp_path.as_posix()
    actual = oc.render(tmp_path, "oc", globaltimes)
    expected = textwrap.dedent(
        f"""\
            begin options
              budget fileout {path}/{tmp_path.stem}.cbb
              head fileout {path}/{tmp_path.stem}.hds
            end options
            
            begin period 1
              save head all
              save budget all
            end period"""
    )
    assert actual == expected
    oc.write(tmp_path, "oc", globaltimes)
    assert (tmp_path / "oc.oc").is_file()


def test_rch_render(twri_model, tmp_path):
    simulation = twri_model
    globaltimes = simulation["time_discretization"]["time"].values
    rch = simulation["GWF_1"]["rch"]
    actual = rch.render(tmp_path, "rch", globaltimes)
    path = tmp_path.as_posix()
    expected = textwrap.dedent(
        f"""\
            begin options
            end options

            begin dimensions
              maxbound 225
            end dimensions

            begin period 1
              open/close {path}/rch/rch.bin (binary)
            end period"""
    )
    assert actual == expected
    rch.write(tmp_path, "rch", globaltimes)
    assert (tmp_path / "rch.rch").is_file()
    assert (tmp_path / "rch").is_dir()
    assert (tmp_path / "rch" / "rch.bin").is_file()


def test_wel_render(twri_model, tmp_path):
    simulation = twri_model
    globaltimes = simulation["time_discretization"]["time"].values
    wel = simulation["GWF_1"]["wel"]
    actual = wel.render(tmp_path, "wel", globaltimes)
    path = tmp_path.as_posix()
    expected = textwrap.dedent(
        f"""\
            begin options
              print_input
              print_flows
              save_flows
            end options

            begin dimensions
              maxbound 15
            end dimensions

            begin period 1
              open/close {path}/wel/wel.bin (binary)
            end period"""
    )
    assert actual == expected
    wel.write(tmp_path, "wel", globaltimes)
    assert (tmp_path / "wel.wel").is_file()
    assert (tmp_path / "wel").is_dir()
    assert (tmp_path / "wel" / "wel.bin").is_file()


def test_solver_render(twri_model, tmp_path):
    simulation = twri_model
    solver = simulation["solver"]
    actual = solver.render()
    expected = textwrap.dedent(
        """\
            begin options
              print_option summary
            end options
            
            begin nonlinear
              outer_hclose 0.0001
              outer_maximum 500
            end nonlinear
            
            begin linear
              inner_maximum 100
              inner_hclose 0.0001
              inner_rclose 0.001
              linear_acceleration cg
              relaxation_factor 0.97
            end linear"""
    )
    assert actual == expected
    solver.write(tmp_path, "solver")
    assert (tmp_path / "solver.ims").is_file()


def test_gwfmodel_render(twri_model, tmp_path):
    simulation = twri_model
    globaltimes = simulation["time_discretization"]["time"].values
    gwfmodel = simulation["GWF_1"]
    actual = gwfmodel.render(tmp_path)
    path = tmp_path.as_posix()
    expected = textwrap.dedent(
        f"""\
            begin options
            end options

            begin packages
              dis6 {path}/dis.dis
              chd6 {path}/chd.chd
              drn6 {path}/drn.drn
              ic6 {path}/ic.ic
              npf6 {path}/npf.npf
              oc6 {path}/oc.oc
              rch6 {path}/rch.rch
              wel6 {path}/wel.wel
            end packages"""
    )
    assert actual == expected
    gwfmodel.write(tmp_path / "GWF_1", globaltimes)
    assert (tmp_path / "GWF_1.nam").is_file()
    assert (tmp_path / "GWF_1").is_dir()


def test_simulation_render(twri_model):
    simulation = twri_model
    actual = simulation.render()
    expected = textwrap.dedent(
        """\
            begin options
            end options

            begin timing
              tdis6 time_discretization.tdis
            end timing

            begin models
              gwf6 GWF_1/GWF_1.nam GWF_1
            end models

            begin exchanges
            end exchanges

            begin solutiongroup 1
              ims6 solver.ims GWF_1
            end solutiongroup"""
    )
    assert actual == expected


@pytest.mark.skipif(sys.version_info < (3, 7), reason="capture_output added in 3.7")
def test_simulation_write(twri_model, tmp_path):
    simulation = twri_model
    modeldir = tmp_path / "ex01-twri"
    simulation.write(modeldir)
    with imod.util.cd(modeldir):
        p = subprocess.run("mf6", check=True, capture_output=True, text=True)
        assert p.stdout.endswith("Normal termination of simulation.\n")
        # hds file is identical to the official example, except for the
        # time units, which are days here and seconds in the official one
        head = imod.mf6.open_hds("GWF_1/GWF_1.hds", "GWF_1/dis.dis.grb")
        assert head.dims == ("time", "layer", "y", "x")
        assert head.shape == (1, 3, 15, 15)
        meanhead_layer = head.groupby("layer").mean(dim=xr.ALL_DIMS)
        mean_answer = np.array([59.79181509, 30.44132373, 24.88576811])
        assert np.allclose(meanhead_layer, mean_answer)
