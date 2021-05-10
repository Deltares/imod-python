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
def uzf_model():
    # Initiate model
    gwf_model = imod.mf6.GroundwaterFlowModel()

    # Create discretication
    shape = nlay, nrow, ncol = 7, 9, 9
    nper = 48
    time = pd.date_range("2018-01-01", periods=nper, freq="H")

    dx = 1000.0
    dy = -1000.0
    dz = np.array([0.5, 0.5, 0.5, 2, 10, 30, 100])
    xmin = 0.0
    xmax = dx * ncol
    ymin = 0.0
    ymax = abs(dy) * nrow
    dims = ("layer", "y", "x")

    layer = np.arange(1, nlay + 1)
    y = np.arange(ymax, ymin, dy) + 0.5 * dy
    x = np.arange(xmin, xmax, dx) + 0.5 * dx
    coords = {"layer": layer, "y": y, "x": x}

    idomain = xr.DataArray(np.ones(shape), coords=coords, dims=dims)
    idomain[:, slice(0, 3), slice(0, 3)] = 0

    top = 0.0
    bottom = xr.DataArray(
        np.cumsum(layer * -1 * dz), coords={"layer": layer}, dims="layer"
    )

    gwf_model["dis"] = imod.mf6.StructuredDiscretization(
        idomain=idomain, top=top, bottom=bottom
    )

    # Create constant head
    head = xr.full_like(idomain, np.nan)
    head[..., 0] = -2.0
    head[..., -1] = -2.0
    head = head.where(idomain == 1)
    head = head.expand_dims(time=time)

    gwf_model["chd"] = imod.mf6.ConstantHead(
        head, print_input=True, print_flows=True, save_flows=True
    )

    # Create nodeproperty flow
    icelltype = xr.full_like(bottom, 0).astype(np.int32)
    k = 10.0
    k33 = 1.0
    gwf_model["npf"] = imod.mf6.NodePropertyFlow(
        icelltype=icelltype,
        k=k,
        k33=k33,
        variable_vertical_conductance=True,
        dewatered=False,
        perched=False,
        save_flows=True,
    )

    # Create unsaturated zone
    uzf_units = idomain.sel(layer=slice(1, 4)).astype(np.int16)
    window = 3

    for i, r in enumerate(range(int(ncol / window))):
        start = r * window
        end = start + window + 1
        uzf_units[..., slice(start, end)] = i + 1

    uzf_units = uzf_units * idomain.astype(np.int16)
    uzf_units = uzf_units.where(uzf_units != 0)

    # Create data unsaturated zone
    uds = {}

    ones_shape = uzf_units.where(np.isnan(uzf_units), 1.0)
    ones_shape_time = (
        xr.DataArray(np.ones(time.shape), coords={"time": time}, dims=("time",))
        * ones_shape
    )

    uds["kv_sat"] = uzf_units * 1.5
    uds["theta_sat"] = uzf_units * 0.1 + 0.1
    uds["theta_res"] = uzf_units * 0.05
    uds["theta_init"] = uzf_units * 0.08
    uds["epsilon"] = ones_shape * 7.0
    uds["surface_depression_depth"] = ones_shape * top + 0.1

    uds["infiltration_rate"] = ones_shape_time * 0.003
    uds["et_pot"] = (
        xr.DataArray(
            (np.sin(np.linspace(0, 1, num=nper) * 2 * np.pi) + 1) * 0.5 * 0.003,
            coords={"time": time},
            dims=("time",),
        )
        * ones_shape_time
    )
    uds["extinction_depth"] = ones_shape_time * -10.0

    uds[
        "simulate_groundwater_seepage"
    ] = False  # Model doesn't converge if set to True....

    gwf_model["uzf"] = imod.mf6.UnsaturatedZoneFlow(**uds)

    # Create initial conditions
    shd = -2.0

    gwf_model["ic"] = imod.mf6.InitialConditions(head=shd)

    # Storage
    Ss = xr.full_like(idomain, 1e-5)
    Sy = xr.full_like(idomain, 0.1)
    iconvert = xr.full_like(idomain, 0).astype(np.int8)

    gwf_model["sto"] = imod.mf6.SpecificStorage(Ss, Sy, True, iconvert)

    # Set output control
    gwf_model["oc"] = imod.mf6.OutputControl(save_head="all", save_budget="all")

    # Attach it to a simulation
    simulation = imod.mf6.Modflow6Simulation("test")
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
    simulation.time_discretization(times=time)

    return simulation


@pytest.mark.skipif(sys.version_info < (3, 7), reason="capture_output added in 3.7")
def test_simulation_write(uzf_model, tmp_path):
    simulation = uzf_model
    modeldir = tmp_path / "uzf_model"
    simulation.write(modeldir)
    with imod.util.cd(modeldir):
        p = subprocess.run("mf6", check=True, capture_output=True, text=True)
        assert p.stdout.endswith("Normal termination of simulation.\n")
        head = imod.mf6.open_hds("GWF_1/GWF_1.hds", "GWF_1/dis.dis.grb")
        assert head.dims == ("time", "layer", "y", "x")
        assert head.shape == (47, 7, 9, 9)
        meanhead = head.mean().values
        mean_answer = 17.32092902
        assert np.allclose(meanhead, mean_answer)
