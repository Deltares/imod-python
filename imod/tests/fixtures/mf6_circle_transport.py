import imod
import numpy as np
import xarray as xr
import xugrid as xu
import pandas as pd

def make_circle_transport_model():
    porosity = 0.10
    max_concentration = 35.0
    min_concentration = 0.0
    max_density = 1025.0
    min_density = 1000.0
    k_value = 10.0

    grid_triangles = imod.data.circle()

    grid = grid_triangles.tesselate_centroidal_voronoi()

    nface = grid.n_face
    nlayer = 15

    layer = np.arange(nlayer, dtype=int) + 1

    idomain = xu.UgridDataArray(
        xr.DataArray(
            np.ones((nlayer, nface), dtype=np.int32),
            coords={"layer": layer},
            dims=["layer", grid.face_dimension],
        ),
        grid=grid,
    )
    icelltype = xu.full_like(idomain, 0)
    k = xu.full_like(idomain, k_value, dtype=float)
    k33 = k.copy()

    top = 0.0
    bottom = xr.DataArray(top - (layer * 10.0), dims=["layer"])
    # RCH
    rch_rate = xu.full_like(idomain.sel(layer=1), 0.001, dtype=float)
    rch_concentration = xu.full_like(rch_rate, min_concentration)
    rch_concentration = rch_concentration.expand_dims(species=["salinity"])
    # GHB
    ghb_location = xu.zeros_like(idomain.sel(layer=1), dtype=bool).ugrid.binary_dilation(
        border_value=True
    )
    constant_head = xu.full_like(idomain, 0.0, dtype=float).where(ghb_location)
    conductance = (idomain * grid.area * k_value).where(
        ghb_location
    )
    constant_concentration = xu.full_like(constant_head, max_concentration).where(
        ghb_location
    )
    constant_concentration = constant_concentration.expand_dims(species=["salinity"])

    gwf_model = imod.mf6.GroundwaterFlowModel()
    gwf_model["disv"] = imod.mf6.VerticesDiscretization(
        top=top, bottom=bottom, idomain=idomain
    )
    gwf_model["ghb"] = imod.mf6.GeneralHeadBoundary(
        constant_head,
        conductance=conductance,
        concentration=constant_concentration,
        print_input=True,
        print_flows=True,
        save_flows=True,
    )
    gwf_model["ic"] = imod.mf6.InitialConditions(start=0.0)
    gwf_model["npf"] = imod.mf6.NodePropertyFlow(
        icelltype=icelltype,
        k=k,
        k33=k33,
        save_flows=True,
    )
    gwf_model["sto"] = imod.mf6.SpecificStorage(
        specific_storage=1.0e-5,
        specific_yield=0.15,
        transient=False,
        convertible=0,
    )
    gwf_model["oc"] = imod.mf6.OutputControl(save_head="last", save_budget="last")
    gwf_model["rch"] = imod.mf6.Recharge(
        rch_rate, concentration=rch_concentration, print_flows=True, save_flows=True
    )

    simulation = imod.mf6.Modflow6Simulation("circle")
    simulation["flow"] = gwf_model
    simulation["flow_solver"] = imod.mf6.Solution(
        modelnames=["flow"],
        print_option="summary",
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

    simtimes = pd.date_range(start="2000-01-01", end="2030-01-01", freq="As")
    simulation.create_time_discretization(additional_times=simtimes)
    simulation["time_discretization"]["n_timesteps"] = 10

    slope = (max_density - min_density) / (max_concentration - min_concentration)
    gwf_model["buoyancy"] = imod.mf6.Buoyancy(
        reference_density=min_density,
        modelname=["transport"],
        reference_concentration=[min_concentration],
        density_concentration_slope=[slope],
        species=["salinity"],
    )

    transport_model = imod.mf6.GroundwaterTransportModel()
    transport_model["ssm"] = imod.mf6.SourceSinkMixing.from_flow_model(
        gwf_model, "salinity"
    )
    transport_model["disv"] = gwf_model["disv"]

    al = 0.001

    transport_model["dsp"] = imod.mf6.Dispersion(
        diffusion_coefficient=1e-4,
        longitudinal_horizontal=al,
        transversal_horizontal1=al * 0.1,
        transversal_vertical=al * 0.01,
        xt3d_off=False,
        xt3d_rhs=False,
    )
    transport_model["adv"] = imod.mf6.AdvectionUpstream()
    transport_model["mst"] = imod.mf6.MobileStorageTransfer(porosity)
    transport_model["ic"] = imod.mf6.InitialConditions(start=max_concentration)
    transport_model["oc"] = imod.mf6.OutputControl(
        save_concentration="last", save_budget="last"
    )

    simulation["transport"] = transport_model
    simulation["transport_solver"] = imod.mf6.Solution(
        modelnames=["transport"],
        print_option="summary",
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
    return simulation

def circle_transport_model():
    """
    Create a circle transport model with a RCH and GHB boundary. This model
    simulates groundwater flow and salt transport in a circular domain
    """
    return make_circle_transport_model()
