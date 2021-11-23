import numpy as np
import pytest
import xarray as xr

import imod


@pytest.fixture(scope="session")
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
    gwf_model["oc"] = imod.mf6.OutputControl(save_head="all", save_budget="all")
    gwf_model["rch"] = imod.mf6.Recharge(rch_rate)
    gwf_model["wel"] = imod.mf6.WellDisStructured(
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
        outer_dvclose=1.0e-4,
        outer_maximum=500,
        under_relaxation=None,
        inner_dvclose=1.0e-4,
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


@pytest.mark.usefixtures("twri_model")
@pytest.fixture(scope="session")
def twri_result(tmpdir_factory):
    # Using a tmpdir_factory is the canonical way of sharing a tempory pytest
    # directory between different testing modules.
    modeldir = tmpdir_factory.mktemp("ex01-twri")
    simulation = make_twri_model()
    simulation.write(modeldir)
    simulation.run()
    return modeldir
