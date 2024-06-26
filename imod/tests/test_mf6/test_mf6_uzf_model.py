import sys

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import imod


@pytest.fixture(scope="module")
def uzf_model():
    # Initiate model
    gwf_model = imod.mf6.GroundwaterFlowModel(newton=True)

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

    like = xr.DataArray(np.ones(shape), coords=coords, dims=dims)
    idomain = like.astype(np.int32)
    idomain[:, slice(0, 3), slice(0, 3)] = 0

    top = 0.0
    bottom = xr.DataArray(
        np.cumsum(layer * -1 * dz), coords={"layer": layer}, dims="layer"
    )

    gwf_model["dis"] = imod.mf6.StructuredDiscretization(
        idomain=idomain, top=top, bottom=bottom
    )

    # Create constant head
    head = xr.full_like(like, np.nan)
    head[..., 0] = -2.0
    head[..., -1] = -2.0
    head = head.where(idomain == 1)
    head = head.expand_dims(time=time)

    gwf_model["chd"] = imod.mf6.ConstantHead(
        head, print_input=True, print_flows=True, save_flows=True
    )

    # Create nodeproperty flow
    icelltype = xr.full_like(idomain, 0)
    k = 10.0
    k33 = 1.0
    gwf_model["npf"] = imod.mf6.NodePropertyFlow(
        icelltype=icelltype,
        k=k,
        k33=k33,
        variable_vertical_conductance=False,  # cant be true when newton is active
        dewatered=False,
        perched=False,
        save_flows=True,
    )

    # Create unsaturated zone
    uzf_units = idomain.sel(
        layer=slice(1, 4)
    ).copy()  # Copy because sel returns a view.

    window = 3

    for i, r in enumerate(range(int(ncol / window))):
        start = r * window
        end = start + window + 1
        uzf_units[..., slice(start, end)] = i + 1

    uzf_units = uzf_units.where(idomain != 0)

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
    uds["surface_depression_depth"] = xr.where(
        ones_shape.layer == 1, ones_shape * top + 0.1, ones_shape * 0.0
    )

    uds["infiltration_rate"] = (ones_shape_time * 0.003).where(
        ones_shape_time.layer == 1
    )
    uds["et_pot"] = (
        xr.DataArray(
            (np.sin(np.linspace(0, 1, num=nper) * 2 * np.pi) + 1) * 0.5 * 0.003,
            coords={"time": time},
            dims=("time",),
        )
        * ones_shape_time
    ).where(ones_shape_time.layer == 1)
    uds["extinction_depth"] = (ones_shape_time * -10.0).where(
        ones_shape_time.layer == 1
    )

    uds["simulate_groundwater_seepage"] = True
    uds["save_flows"] = True
    uds["budget_fileout"] = "GWF_1/uzf.cbc"

    gwf_model["uzf"] = imod.mf6.UnsaturatedZoneFlow(**uds)

    # Create initial conditions
    shd = -2.0

    gwf_model["ic"] = imod.mf6.InitialConditions(start=shd)

    # Storage
    gwf_model["sto"] = imod.mf6.SpecificStorage(1e-5, 0.1, True, 0)

    # Set output control
    gwf_model["oc"] = imod.mf6.OutputControl(save_head="all", save_budget="all")

    # Attach it to a simulation
    simulation = imod.mf6.Modflow6Simulation("test")
    simulation["GWF_1"] = gwf_model
    # Define solver settings
    simulation["solver"] = imod.mf6.SolutionPresetComplex(modelnames=["GWF_1"])
    # Collect time discretization
    simulation.create_time_discretization(additional_times=time)
    return simulation


@pytest.mark.skipif(sys.version_info < (3, 7), reason="capture_output added in 3.7")
def test_simulation_write(uzf_model, tmp_path):
    simulation = uzf_model
    modeldir = tmp_path / "uzf_model"
    simulation.write(modeldir)
    with imod.util.cd(modeldir):
        simulation.run()
        head = imod.mf6.open_hds("GWF_1/GWF_1.hds", "GWF_1/dis.dis.grb")
        assert head.dims == ("time", "layer", "y", "x")
        assert head.shape == (47, 7, 9, 9)
        meanhead = head.mean().values
        mean_answer = -1.54998241
        assert np.allclose(meanhead, mean_answer)
        budget_mf6 = head = imod.mf6.open_cbc("GWF_1/GWF_1.cbc", "GWF_1/dis.dis.grb")
        budgets_uzf = imod.mf6.open_cbc("GWF_1/uzf.cbc", "GWF_1/dis.dis.grb")
        assert np.allclose(
            budget_mf6["uzf-gwrch_uzf"], -budgets_uzf["gwf_gwf_1"], equal_nan=True
        )
