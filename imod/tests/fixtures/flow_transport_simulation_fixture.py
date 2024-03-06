# %%

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import imod


def create_transport_model(flow_model, species_name, dispersivity, retardation, decay):
    """
    Function to create a transport model, as we intend to create four similar
    transport models.

    Parameters
    ----------
    flow_model: GroundwaterFlowModel
    species_name: str
    dispersivity: float
    retardation: float
    decay: float

    Returns
    -------
    transportmodel: GroundwaterTransportModel
    """

    rhobulk = 1150.0
    porosity = 0.25

    transport_model = imod.mf6.GroundwaterTransportModel()
    transport_model["ssm"] = imod.mf6.SourceSinkMixing.from_flow_model(
        flow_model, species_name, save_flows=True
    )
    transport_model["adv"] = imod.mf6.AdvectionUpstream()
    transport_model["dsp"] = imod.mf6.Dispersion(
        diffusion_coefficient=0.0,
        longitudinal_horizontal=dispersivity,
        transversal_horizontal1=0.0,
        xt3d_off=False,
        xt3d_rhs=False,
    )

    # Compute the sorption coefficient based on the desired retardation factor
    # and the bulk density. Because of this, the exact value of bulk density
    # does not matter for the solution.
    if retardation != 1.0:
        sorption = "linear"
        kd = (retardation - 1.0) * porosity / rhobulk
    else:
        sorption = None
        kd = 1.0

    transport_model["mst"] = imod.mf6.MobileStorageTransfer(
        porosity=porosity,
        decay=decay,
        decay_sorbed=decay,
        bulk_density=rhobulk,
        distcoef=kd,
        first_order_decay=True,
        sorption=sorption,
    )

    transport_model["ic"] = imod.mf6.InitialConditions(start=0.0)
    transport_model["oc"] = imod.mf6.OutputControl(
        save_concentration="all", save_budget="last"
    )
    transport_model["dis"] = flow_model["dis"]
    return transport_model


# %%
@pytest.fixture(scope="function")
def flow_transport_simulation():
    """
    This fixture is a variation on the model also present in
    examples/mf6/example_1d_transport.py. To make that model more useful for
    testing eg partitioning or regridding, some boundary conditions were added
    (2 wells, one extractor and one injector which injects with a nonzero
    concentration) as well as a recharge zone with a nonzero concentration.
    """
    nlay = 1
    nrow = 2
    ncol = 101
    dx = 10.0
    xmin = 0.0
    xmax = dx * ncol
    layer = [1]
    y = [1.5, 0.5]
    x = np.arange(xmin, xmax, dx) + 0.5 * dx

    grid_dims = ("layer", "y", "x")
    grid_coords = {"layer": layer, "y": y, "x": x, "dx": dx, "dy": 1.}
    grid_shape = (nlay, nrow, ncol)
    grid = xr.DataArray(
        np.ones(grid_shape, dtype=int), coords=grid_coords, dims=grid_dims
    )
    bottom = xr.full_like(grid, -1.0, dtype=float)

    gwf_model = imod.mf6.GroundwaterFlowModel()
    gwf_model["ic"] = imod.mf6.InitialConditions(0.0)

    # %%
    # Create the input for a constant head boundary and its associated concentration.
    constant_head = xr.full_like(grid, np.nan, dtype=float)
    constant_head[..., 0] = 60.0
    constant_head[..., 100] = 0.0

    constant_conc = xr.full_like(grid, np.nan, dtype=float)
    constant_conc[..., 0] = 1.0
    constant_conc[..., 100] = 0.0
    constant_conc = constant_conc.expand_dims(
        species=["species_a", "species_b", "species_c", "species_d"]
    )

    gwf_model["chd"] = imod.mf6.ConstantHead(constant_head, constant_conc)

    # %%
    # Add other flow packages.

    gwf_model["npf"] = imod.mf6.NodePropertyFlow(
        icelltype=1,
        k=xr.full_like(grid, 1.0, dtype=float),
    )
    gwf_model["dis"] = imod.mf6.StructuredDiscretization(
        top=0.0,
        bottom=bottom,
        idomain=grid,
    )
    gwf_model["oc"] = imod.mf6.OutputControl(save_head="all", save_budget="all")
    gwf_model["sto"] = imod.mf6.SpecificStorage(
        specific_storage=1.0e-5,
        specific_yield=0.15,
        transient=False,
        convertible=0,
    )
    recharge_conc = xr.full_like(grid, np.nan, dtype=float)
    recharge_conc[..., 20:60] = 0.001
    recharge_conc = recharge_conc.expand_dims(
        species=["species_a", "species_b", "species_c", "species_d"]
    )
    recharge_rate = xr.full_like(grid, np.nan, dtype=float)
    recharge_rate[..., 20:60] = 0.0001
    gwf_model["rch"] = imod.mf6.Recharge(recharge_rate, recharge_conc, "AUX")
    # %%
    # Create the simulation.
    injection_concentration = xr.DataArray(
        [[0.2, 0.23], [0.5, 0.2], [0.2, 0.23], [0.5, 0.2]],
        coords={
            "species": ["species_a", "species_b", "species_c", "species_d"],
            "index": [0, 1],
        },
        dims=("species", "index"),
    )

    gwf_model["well"] = imod.mf6.Well(
        x=[20.0, 580.0],
        y=[0.6, 1.2],
        concentration_boundary_type="Aux",
        screen_top=[0.0, 0.0],
        screen_bottom=[-1.0, -1.0],
        rate=[0.1, -0.2],
        minimum_k=0.0001,
        concentration=injection_concentration,
    )

    simulation = imod.mf6.Modflow6Simulation("1d_tpt_benchmark")
    simulation["flow"] = gwf_model

    # %%
    # Add four transport simulations, and setup the solver flow and transport.

    simulation["tpt_a"] = create_transport_model(gwf_model, "species_a", 0.0, 1.0, 0.0)
    simulation["tpt_b"] = create_transport_model(gwf_model, "species_b", 10.0, 1.0, 0.0)
    simulation["tpt_c"] = create_transport_model(gwf_model, "species_c", 10.0, 5.0, 0.0)
    simulation["tpt_d"] = create_transport_model(
        gwf_model, "species_d", 10.0, 5.0, 0.002
    )

    simulation["solver"] = imod.mf6.Solution(
        modelnames=["flow"],
        print_option="summary",
        csv_output=False,
        no_ptc=True,
        outer_dvclose=1.0e-4,
        outer_maximum=500,
        under_relaxation=None,
        inner_dvclose=1.0e-4,
        inner_rclose=0.001,
        inner_maximum=100,
        linear_acceleration="bicgstab",
        scaling_method=None,
        reordering_method=None,
        relaxation_factor=0.97,
    )
    simulation["transport_solver"] = imod.mf6.Solution(
        modelnames=["tpt_a", "tpt_b", "tpt_c", "tpt_d"],
        print_option="summary",
        csv_output=False,
        no_ptc=True,
        outer_dvclose=1.0e-6,
        outer_maximum=500,
        under_relaxation=None,
        inner_dvclose=1.0e-6,
        inner_rclose=0.0001,
        inner_maximum=200,
        linear_acceleration="bicgstab",
        scaling_method=None,
        reordering_method=None,
        relaxation_factor=0.9,
    )

    duration = pd.to_timedelta("2000d")
    start = pd.to_datetime("2000-01-01")
    simulation.create_time_discretization(additional_times=[start, start + duration])
    simulation["time_discretization"]["n_timesteps"] = 100

    return simulation
    # %%
