"""
TWRI
====

This example has been converted from the `MODFLOW6 Example problems`_.  See the
`description`_ and the `notebook`_ which uses `FloPy`_ to setup the model.

This example is a modified version of the original MODFLOW example
("`Techniques of Water-Resources Investigation`_" (TWRI)) described in
(`McDonald & Harbaugh, 1988`_) and duplicated in (`Harbaugh & McDonald, 1996`_).
This problem is also is distributed with MODFLOW-2005 (`Harbaugh, 2005`_). The
problem has been modified from a quasi-3D problem, where confining beds are not
explicitly simulated, to an equivalent three-dimensional problem.

In overview, we'll set the following steps:

    * Create a structured grid for a rectangular geometry.
    * Create the xarray DataArrays containg the MODFLOW6 parameters.
    * Feed these arrays into the imod mf6 classes.
    * Write to modflow6 files.
    * Run the model.
    * Open the results back into DataArrays.
    * Visualize the results.

"""
# %%
# We'll start with the usual imports. As this is an simple (synthetic)
# structured model, we can make due with few packages.

import numpy as np
import xarray as xr
import imod
from imod.mf6.regridding_utils import RegridderType
import matplotlib.pyplot as plt

def create_twri_simulation() -> imod.mf6.Modflow6Simulation:

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
    coords = {"layer": layer, "y": y, "x": x, "dx" :dx, "dy": dy}

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
    screen_layer =[2, 1, 1, 0, 0, 0, 0, 0,0, 0,0, 0, 0, 0,0]
    screen_bottom =bottom[screen_layer] + 0.1
    screen_top = screen_bottom + 10
    well_y = np.array([5., 4, 6, 9, 9, 9, 9, 11, 11, 11, 11, 13, 13, 13, 13])*abs(dy)
    well_x = np.array([11., 6, 12, 8, 10, 12, 14, 8, 10, 12, 14, 8, 10, 12, 14])*dx
    well_rate = [-5.0] * 15

    # Node properties
    icelltype = xr.DataArray([1, 0, 0], {"layer": layer}, ("layer",))
    k = xr.DataArray([1.0e-3, 1.0e-4, 2.0e-4], {"layer": layer}, ("layer",))
    k33 = xr.DataArray([2.0e-8, 2.0e-8, 2.0e-8], {"layer": layer}, ("layer",))

    # %%
    # Write the model
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
    gwf_model["wel"] = imod.mf6.Well(screen_top=screen_top, screen_bottom=screen_bottom, y=well_y, x=well_x, rate=well_rate )


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
    simulation.create_time_discretization(
        additional_times=["2000-01-01", "2000-01-02", "2000-01-03", "2000-01-04"]
    )
    return simulation

# %%
# now we create the twri simulation itself. It yields a simulation of a flow problem, with a grid of 3 layers and 15 cells in both x and y directions.
# To better illustrate the regridding, we replace the K field with a normal random K field. The original k-field is a constant per layer. 
simulation = create_twri_simulation()
idomain = simulation["GWF_1"]["dis"]["idomain"]
heterogeneous_k = xr.zeros_like(idomain , dtype= np.double)
heterogeneous_k.values = np.random.lognormal(-2, 2, heterogeneous_k.shape)
simulation["GWF_1"]["npf"]["k"] = heterogeneous_k

# %%
# Let's plot the k-field. This is going to be the input for the regridder, and the regridded output should somewhat resemble it. 
fig, ax = plt.subplots()
heterogeneous_k.sel(layer=1).plot(y="y", yincrease=False, ax=ax)

# %%
# now we create a new grid for this simulation. It has 3 layers,  45 rows and 20 columns.
# The length of the domain is slightly different from the input grid. That was 15*5000 = 75000 long in x and y
# but the new grid is 75015 long in x and  75020 long in y 

nlay = 3
nrow = 45
ncol = 20
shape = (nlay, nrow, ncol)

dx = 3751.0
dy = -1667.0
xmin = 0.0
xmax = dx * ncol
ymin = 0.0
ymax = abs(dy) * nrow
dims = ("layer", "y", "x")

layer = np.array([1, 2, 3])
y = np.arange(ymax, ymin, dy) + 0.5 * dy
x = np.arange(xmin, xmax, dx) + 0.5 * dx
coords = {"layer": layer, "y": y, "x": x, "dx":dx, "dy":dy}
new_idomain = xr.DataArray(np.ones(shape, dtype=int), coords=coords, dims=dims)

# %%
# a first way to regrid the twri model is to regrid the whole simulation object. This is the most straightforward method, 
# and it uses default regridding methods for each input field. To see which ones are used, look at the _regrid_method 
# class attribute of the relevant package. For example the _regrid_method attribute  of the NodePropertyFlow package 
# specifies that field "k" uses an OVERLAP regridder in combination with the averaging function "geometric_mean".
new_simulation = simulation.regrid_like("regridded_twri", target_grid=new_idomain)

# %%
# Let's plot the k-field. This is the regridded output, and it should should somewhat resemble the original k-field plotted earlier.  
regridded_k_1 = new_simulation["GWF_1"]["npf"]["k"]
fig, ax = plt.subplots()

regridded_k_1.sel(layer=1).plot(y="y", yincrease=False, ax=ax)

# %%
# a second way to regrid  twri  is to regrid the groundwater flow model.

model = simulation["GWF_1"]
new_model = model.regrid_like(new_idomain)

regridded_k_2 = new_model["npf"]["k"]
fig, ax = plt.subplots()
regridded_k_2.sel(layer=1).plot(y="y", yincrease=False, ax=ax)


# %%
# finally, we can regrid package per package. This allows us to choose the regridding method as well.
# in this example we'll regrid the npf package manually and the rest of the packages using default methods.

regridder_types ={     "k":( RegridderType.CENTROIDLOCATOR, None)}
npf_regridded = model["npf"].regrid_like(target_grid = new_idomain, regridder_types = regridder_types )
new_model["npf"] = npf_regridded


regridded_k_3 = new_model["npf"]["k"]
fig, ax = plt.subplots()
regridded_k_3.sel(layer=1).plot(y="y", yincrease=False, ax=ax)
pass