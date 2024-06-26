"""
Example models
==============

This source file contains functions that create a simulation that can be used in
examples that are not focused on building a simulation, but on doing something
with it (such as regridding).

"""

import numpy as np
import xarray as xr
import xugrid as xu

import imod


def create_twri_simulation() -> imod.mf6.Modflow6Simulation:
    """There is a separate example contained in :doc:`/examples/mf6/ex01_twri`
    that you should look at if you are interested in the model building. The
    TWRI model has 3 layers and contains wells, a drain and recharge.
    Geometrically it is rectangular with prescribed head on some of the
    boundaries. Conductivity is highly anisotropic but constant in each layer.
    Simulation is steady state.
    """

    nlay = 3
    nrow = 15
    ncol = 15
    shape = (nlay, nrow, ncol)

    dx = 5000.0
    dy = 5000.0
    xmin = 0.0
    xmax = dx * ncol
    ymin = 0.0
    ymax = dy * nrow
    dims = ("layer", "y", "x")

    layer = np.array([1, 2, 3])
    y = np.arange(ymax, ymin, -dy) - 0.5 * dy
    x = np.arange(xmin, xmax, dx) + 0.5 * dx
    coords = {"layer": layer, "y": y, "x": x, "dx": dx, "dy": -dy}

    # %%
    # Create DataArrays
    # -----------------
    #
    # Now that we have the grid coordinates setup, we can start defining model
    # parameters. The model is characterized by:
    #
    # * a constant head boundary on the left
    # * a single drain in the center left of the model
    # * uniform recharge on the top layer
    # * a number of wells scattered throughout the model.

    idomain = xr.DataArray(np.ones(shape, dtype=int), coords=coords, dims=dims)
    bottom = xr.DataArray([-200.0, -300.0, -450.0], {"layer": layer}, ("layer",))

    # Constant head
    constant_head = xr.full_like(idomain, np.nan, dtype=float).sel(layer=[1, 2])
    constant_head[..., 0] = 0.0

    # Drainage
    elevation = xr.full_like(idomain.sel(layer=1), np.nan, dtype=float)
    conductance = xr.full_like(idomain.sel(layer=1), np.nan, dtype=float)
    elevation[7, 1:10] = np.array([0.0, 0.0, 10.0, 20.0, 30.0, 50.0, 70.0, 90.0, 100.0])
    conductance[7, 1:10] = 1.0

    # Recharge
    rch_rate = xr.full_like(idomain.sel(layer=1), 3.0e-8, dtype=float)

    # Well
    screen_layer = [2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # we set the screen top and bottoms such that each well falls in one layer and is long enough not to be filtered out
    perforation_length = 50
    delta_z = 0.1

    screen_bottom = bottom[screen_layer] + delta_z
    screen_top = screen_bottom + delta_z + perforation_length

    # we compute the x and y cooordinates of the wells based on the row and column indices presented in the original twri model
    well_y = (
        ymax
        - np.array(
            [
                5.0,
                4.0,
                6.0,
                9.0,
                9.0,
                9.0,
                9.0,
                11.0,
                11.0,
                11.0,
                11.0,
                13.0,
                13.0,
                13.0,
                13.0,
            ]
        )
        * abs(dy)
        + dy / 2
    )
    well_x = (
        np.array(
            [
                11.0,
                6.0,
                12.0,
                8.0,
                10.0,
                12.0,
                14.0,
                8.0,
                10.0,
                12.0,
                14.0,
                8.0,
                10.0,
                12.0,
                14.0,
            ]
        )
        * dx
        - dx / 2
    )
    well_rate = [-5.0] * 15

    # Node properties
    icelltype = xr.DataArray([1, 0, 0], {"layer": layer}, ("layer",))
    k = xr.DataArray([1.0e-3, 1.0e-4, 2.0e-4], {"layer": layer}, ("layer",))
    k33 = xr.DataArray([2.0e-8, 2.0e-8, 2.0e-8], {"layer": layer}, ("layer",))

    # %%
    # Build the model
    # ---------------
    #
    # The first step is to define an empty model, the parameters and boundary
    # conditions are added in the form of the familiar MODFLOW packages.

    gwf_model = imod.mf6.GroundwaterFlowModel()
    gwf_model["dis"] = imod.mf6.StructuredDiscretization(
        top=200.0, bottom=bottom, idomain=idomain
    )
    gwf_model["chd"] = imod.mf6.ConstantHead(
        constant_head, print_input=True, print_flows=True, save_flows=True
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
    gwf_model["sto"] = imod.mf6.SpecificStorage(
        specific_storage=1.0e-5,
        specific_yield=0.15,
        transient=False,
        convertible=0,
    )
    gwf_model["oc"] = imod.mf6.OutputControl(save_head="all", save_budget="all")
    gwf_model["rch"] = imod.mf6.Recharge(rch_rate)
    gwf_model["wel"] = imod.mf6.Well(
        x=well_x,
        y=well_y,
        screen_top=screen_top,
        screen_bottom=screen_bottom,
        rate=well_rate,
        minimum_k=0.0001,
    )

    # %%
    # Attach it to a simulation
    # ---------------

    simulation = imod.mf6.Modflow6Simulation("ex01-twri")
    simulation["GWF_1"] = gwf_model
    # Define solver settings
    simulation["solver"] = imod.mf6.Solution(
        modelnames=["GWF_1"],
        print_option="summary",
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
    simulation.create_time_discretization(
        additional_times=["2000-01-01", "2000-01-02", "2000-01-03", "2000-01-04"]
    )
    return simulation


def create_circle_simulation():
    """
    There is a separate example contained in :doc:`/examples/mf6/circle`
    that you should look at if you are interested in the model building. The
    circle model uses an unstructured grid. It has 2 layers of constant
    thickness. In conductivity, it is isotropic, and constant in space.
    Boundary conditions include recharge and constant head. It is a transient
    simulation.
    """

    # %%
    # Create a mesh
    # -------------

    grid = imod.data.circle()

    nface = grid.n_face
    nlayer = 2

    idomain = xu.UgridDataArray(
        xr.DataArray(
            np.ones((nlayer, nface), dtype=np.int32),
            coords={"layer": [1, 2]},
            dims=["layer", grid.face_dimension],
        ),
        grid=grid,
    )

    # Create model and packages
    icelltype = xu.full_like(idomain, 0)
    k = xu.full_like(idomain, 1.0, dtype=float)
    k33 = k.copy()
    rch_rate = xu.full_like(idomain.sel(layer=1), 0.001, dtype=float)
    bottom = idomain * xr.DataArray([5.0, 0.0], dims=["layer"])

    chd_location = xu.zeros_like(
        idomain.sel(layer=2), dtype=bool
    ).ugrid.binary_dilation(border_value=True)
    constant_head = xu.full_like(idomain.sel(layer=2), 1.0, dtype=float).where(
        chd_location
    )

    gwf_model = imod.mf6.GroundwaterFlowModel()
    gwf_model["disv"] = imod.mf6.VerticesDiscretization(
        top=10.0, bottom=bottom, idomain=idomain
    )
    gwf_model["chd"] = imod.mf6.ConstantHead(
        constant_head, print_input=True, print_flows=True, save_flows=True
    )
    gwf_model["ic"] = imod.mf6.InitialConditions(start=0.0)
    gwf_model["npf"] = imod.mf6.NodePropertyFlow(
        icelltype=icelltype, k=k, k33=k33, save_flows=True, save_specific_discharge=True
    )
    gwf_model["sto"] = imod.mf6.SpecificStorage(
        specific_storage=1.0e-5,
        specific_yield=0.15,
        transient=False,
        convertible=0,
    )
    gwf_model["oc"] = imod.mf6.OutputControl(save_head="all", save_budget="all")
    gwf_model["rch"] = imod.mf6.Recharge(rch_rate)

    # Create simulation
    simulation = imod.mf6.Modflow6Simulation("circle")
    simulation["GWF_1"] = gwf_model
    simulation["solver"] = imod.mf6.Solution(
        modelnames=["GWF_1"],
        print_option="summary",
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
    simulation.create_time_discretization(additional_times=["2000-01-01", "2000-01-02"])

    return simulation
