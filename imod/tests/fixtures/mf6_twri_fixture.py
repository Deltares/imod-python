import numpy as np
import pandas as pd
import pytest
import xarray as xr

import imod
from imod.typing.grid import zeros_like


def make_twri_model():
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
    like = xr.DataArray(np.ones(shape), coords=coords, dims=dims)
    idomain = like.astype(np.int8)
    bottom = xr.DataArray([-200.0, -300.0, -450.0], {"layer": layer}, ("layer",))

    # Constant head
    head = xr.full_like(like, np.nan).sel(layer=[1, 2])
    head[..., 0] = 0.0

    # Drainage
    elevation = xr.full_like(like.sel(layer=1), np.nan)
    conductance = xr.full_like(like.sel(layer=1), np.nan)
    elevation[7, 1:10] = np.array([0.0, 0.0, 10.0, 20.0, 30.0, 50.0, 70.0, 90.0, 100.0])
    conductance[7, 1:10] = 1.0

    # Node properties
    icelltype = xr.DataArray([1, 0, 0], {"layer": layer}, ("layer",))
    k = xr.DataArray([1.0e-3, 1.0e-4, 2.0e-4], {"layer": layer}, ("layer",))
    k33 = xr.DataArray([2.0e-8, 2.0e-8, 2.0e-8], {"layer": layer}, ("layer",))

    # Recharge
    rch_rate = xr.full_like(like.sel(layer=1), 3.0e-8)

    # Well
    wells_x = [
        52500.0,
        27500.0,
        57500.0,
        37500.0,
        47500.0,
        57500.0,
        67500.0,
        37500.0,
        47500.0,
        57500.0,
        67500.0,
        37500.0,
        47500.0,
        57500.0,
        67500.0,
    ]
    wells_y = [
        52500.0,
        57500.0,
        47500.0,
        32500.0,
        32500.0,
        32500.0,
        32500.0,
        22500.0,
        22500.0,
        22500.0,
        22500.0,
        12500.0,
        12500.0,
        12500.0,
        12500.0,
    ]
    screen_top = [
        -300.0,
        -200.0,
        -200.0,
        200.0,
        200.0,
        200.0,
        200.0,
        200.0,
        200.0,
        200.0,
        200.0,
        200.0,
        200.0,
        200.0,
        200.0,
    ]
    screen_bottom = [
        -450.0,
        -300.0,
        -300.0,
        -200.0,
        -200.0,
        -200.0,
        -200.0,
        -200.0,
        -200.0,
        -200.0,
        -200.0,
        -200.0,
        -200.0,
        -200.0,
        -200.0,
    ]
    rate_wel = [
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
    gwf_model["wel"] = imod.mf6.Well(
        x=wells_x,
        y=wells_y,
        screen_top=screen_top,
        screen_bottom=screen_bottom,
        rate=rate_wel,
        minimum_k=1e-19,
    )
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
    gwf_model["ic"] = imod.mf6.InitialConditions(start=0.0)
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

    gwf_model["sto"] = imod.mf6.SpecificStorage(
        specific_storage=1.0e-15,
        specific_yield=0.15,
        convertible=0,
        transient=False,
    )

    # Attach it to a simulation
    simulation = imod.mf6.Modflow6Simulation("ex01-twri")
    simulation["GWF_1"] = gwf_model
    # Define solver settings
    simulation["solver"] = imod.mf6.Solution(
        modelnames=["GWF_1"],
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
    simulation.create_time_discretization(additional_times=["2000-01-01", "2000-01-02"])
    return simulation


@pytest.fixture(scope="function")
def twri_model():
    """Returns steady-state confined model."""
    return make_twri_model()


@pytest.fixture(scope="function")
def transient_twri_model():
    """Returns transient confined model."""
    simulation = make_twri_model()
    gwf_model = simulation["GWF_1"]
    like = gwf_model["dis"]["idomain"].astype(float)
    gwf_model["sto"] = imod.mf6.SpecificStorage(
        specific_storage=xr.full_like(like, 1.0e-15),
        specific_yield=xr.full_like(like, 0.15),
        convertible=0,
        transient=True,
        save_flows=True,
    )
    simulation.create_time_discretization(
        additional_times=pd.date_range("2000-01-01", " 2000-01-31")
    )
    return simulation


@pytest.fixture(scope="function")
def transient_unconfined_twri_model():
    """Returns transient unconfined model, also saves specific discharges."""
    simulation = make_twri_model()
    gwf_model = simulation["GWF_1"]
    like = gwf_model["dis"]["idomain"].astype(float)
    # Force storage to unconfined
    gwf_model["sto"] = imod.mf6.SpecificStorage(
        specific_storage=xr.full_like(like, 1.0e-15),
        specific_yield=xr.full_like(like, 0.15),
        convertible=1,
        transient=True,
        save_flows=True,
    )
    # Force npf to unconfined
    layer = np.array([1, 2, 3])
    icelltype = xr.DataArray([1, 1, 1], {"layer": layer}, ("layer",))
    gwf_model["npf"]["icelltype"] = icelltype
    # Store save cell saturation
    gwf_model["npf"]["save_saturation"] = True
    # Write specific discharges
    gwf_model["npf"]["save_specific_discharge"] = True
    simulation.create_time_discretization(
        additional_times=pd.date_range("2000-01-01", " 2000-01-31")
    )
    return simulation


@pytest.mark.usefixtures("twri_model")
@pytest.fixture(scope="function")
def twri_result(tmpdir_factory):
    # Using a tmpdir_factory is the canonical way of sharing a tempory pytest
    # directory between different testing modules.
    modeldir = tmpdir_factory.mktemp("ex01-twri")
    simulation = make_twri_model()
    simulation.write(modeldir)
    simulation.run()
    return modeldir


@pytest.mark.usefixtures("transient_twri_model")
@pytest.fixture(scope="function")
def transient_twri_result(tmpdir_factory, transient_twri_model):
    # Using a tmpdir_factory is the canonical way of sharing a tempory pytest
    # directory between different testing modules.
    modeldir = tmpdir_factory.mktemp("ex01-twri-transient")
    simulation = transient_twri_model
    simulation.write(modeldir)
    simulation.run()
    return modeldir


@pytest.mark.usefixtures("transient_unconfined_twri_model")
@pytest.fixture(scope="function")
def transient_unconfined_twri_result(tmpdir_factory, transient_unconfined_twri_model):
    # Using a tmpdir_factory is the canonical way of sharing a tempory pytest
    # directory between different testing modules.
    modeldir = tmpdir_factory.mktemp("ex01-twri-transient-unconfined")
    simulation = transient_unconfined_twri_model
    simulation.write(modeldir)
    simulation.run()
    return modeldir


@pytest.mark.usefixtures("transient_twri_model")
@pytest.fixture(scope="function")
def split_transient_twri_model(transient_twri_model):
    active = transient_twri_model["GWF_1"].domain.sel(layer=1)
    number_partitions = 3
    split_location = np.linspace(active.y.min(), active.y.max(), number_partitions + 1)

    coords = active.coords
    submodel_labels = zeros_like(active)
    for id in np.arange(1, number_partitions):
        submodel_labels.loc[
            (coords["y"] > split_location[id]) & (coords["y"] <= split_location[id + 1])
        ] = id

    split_simulation = transient_twri_model.split(submodel_labels)

    return split_simulation
