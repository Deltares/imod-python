import pathlib
import subprocess
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


def test_timedis_render(twri_model):
    simulation = twri_model
    actual = simulation["time_discretization"].render()
    expected = textwrap.dedent(
        """\
            begin options
              time_units days
            end options

            begin dimensions
              nper 1
            end dimensions

            begin perioddata
              1.0 1 1.0
            end perioddata"""
    )
    assert actual == expected


def test_gwfmodel_render(twri_model, tmp_path):
    simulation = twri_model
    actual = simulation["GWF_1"].render(tmp_path)
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
        print(head)
        assert head.dims == ("time", "layer", "y", "x")
        assert head.shape == (1, 3, 15, 15)
        meanhead_layer = head.groupby("layer").mean()
        mean_answer = np.array([59.79181509, 30.44132373, 24.88576811])
        assert np.allclose(meanhead_layer, mean_answer)
