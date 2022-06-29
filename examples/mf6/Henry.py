"""
Henry
======

This example illustrates how to setup a variable density groundwater
flow and transport model using the ``imod`` package and associated packages.
We will simulate a variable transport

In overview, we'll set the following steps:

    * Create a suitable 2d mesh
    * Create a simulation for flow and transport with variable density
    * Write to modflow6 files.
    * Run the model.
    * Open the results back into UgridDataArrays.
    * Visualize the results.

We are simulating the Henry problem- although not the original one but the one outlined in the modflow 6 manual
(jupyter notebooks example 51). This is the modified Henry problem with half the inflow rate.
The domain is a 2d rectangle, vertically oriented, of 2m long and 1m high
Fresh water flows in over the left boundary. The right boundary is in direct contact with sea water of
a density of 1025 kg/m3. There, a general head boundary is in place. It has an external head consistent with
hydrostatic pressure distribution of sea water. Water that flows in the model over this boundary has
the salinity of sea water, and water that leaves the flow model over this boundary has the simulated concentration in the
domain.
The result should be a saltwater intrusion cone and a mixing zone.
"""
# %%
# We'll start with the usual imports. As this is an simple (synthetic)
# structured model, we can make due with few packages.

from datetime import datetime, timedelta

import numpy as np
import xarray as xr

import imod


# %%
# helper function for creating iterable given starting date, step size and number of steps
def daterange(start_time, number_steps, step_size):
    current_start = start_time
    yield start_time
    for _ in range(number_steps):
        current_start = current_start + timedelta(step_size)
        yield current_start


# %%
# define grid layout. We are modelling a 2d rectangular domain, oriented vertically.
# 2m long and 1m deep.
nlay = 40
nrow = 1
ncol = 80
shape = (nlay, nrow, ncol)


xmax = 2
xmin = 0.0
dx = (xmax - xmin) / ncol
dims = ("layer", "y", "x")

layer = np.arange(1, 41, 1)
y = np.array([0.5])
x = np.arange(xmin, xmax, dx) + 0.5 * dx

layer_thickness = 1 / nlay
col_width = 1 / ncol
dy = -1
coords = {"layer": layer, "y": y, "x": x, "dy": dy, "dx": dx}

idomain = xr.DataArray(np.ones(shape, dtype=int), coords=coords, dims=dims)


top = xr.full_like(idomain.sel(layer=1), 1.0, dtype=np.floating)
bottom_level_of_top_layer = 1.0 - layer_thickness
bottom = xr.DataArray(
    np.arange(bottom_level_of_top_layer, -layer_thickness, -layer_thickness),
    {"layer": layer},
    ("layer",),
)

depth_cell_centers = 1 - (
    np.arange(bottom_level_of_top_layer, -layer_thickness, -layer_thickness)
    + layer_thickness / 2
)

# %%
# define parameters for fluid concentration and density.
max_concentration = 35.0
min_concentration = 0.0
max_density = 1025.0
min_density = 1000.0


# %%
# Now make the flow model. We'll start with the non-boundary condition packages
gwf_model = imod.mf6.GroundwaterFlowModel()
gwf_model["dis"] = imod.mf6.StructuredDiscretization(
    top=top, bottom=bottom, idomain=idomain
)

gwf_model["npf"] = imod.mf6.NodePropertyFlow(
    icelltype=0,
    k=864.0,
    k33=864.0,
)
gwf_model["sto"] = imod.mf6.SpecificStorage(
    specific_storage=1.0e-4,
    specific_yield=0.15,
    transient=False,
    convertible=0,
)

gwf_model["ic"] = imod.mf6.InitialConditions(head=0.0)
gwf_model["oc"] = imod.mf6.OutputControl(save_head="last", save_budget="last")


# %%
# Now let's make the constant head boundary condition.
constant_head = xr.full_like(idomain, np.nan, dtype=float)

head_cell_centers = depth_cell_centers * (max_density - min_density) / min_density
head_cell_centers = head_cell_centers[np.newaxis]  # now its a 2d column vector
constant_head[
    ..., ncol - 1
] = (
    head_cell_centers.T
)  # transpose the 2d vector so that it becomes a column vector, now it fits the layout of constant_head
conc = np.full_like(head_cell_centers.T, max_concentration)
k = 864 * layer_thickness * (0.5 / col_width)
conductance = np.full_like(head_cell_centers.T, k)
bc_conductance = xr.full_like(idomain, np.nan, dtype=float)
bc_conductance[..., ncol - 1] = conductance

inflow_concentration = xr.full_like(idomain, np.nan, dtype=float)
inflow_concentration[..., ncol - 1] = conc
inflow_concentration = inflow_concentration.expand_dims(species=["salinity"])

gwf_model["right_boundary"] = imod.mf6.GeneralHeadBoundary(
    constant_head,
    conductance=bc_conductance,
    concentration=inflow_concentration,
    concentration_boundary_type="AUX",
    print_input=True,
    print_flows=True,
    save_flows=True,
)

# %%
# Now let's make the constant flux condition.
flux = np.full_like(layer, (5.7024 / nlay) / 2, dtype=np.floating)

flux_concentration = xr.DataArray(
    data=layer.copy(),
    dims=["cell"],
    coords=dict(cell=(range(0, nlay))),
)
flux_concentration[...] = min_concentration
flux_concentration = flux_concentration.expand_dims(species=["salinity"])

wellrows = np.full_like(layer, 1, dtype=np.int32)
wellcolumns = np.full_like(layer, 1, dtype=np.int32)
gwf_model["left_boundary"] = imod.mf6.WellDisStructured(
    layer=layer,
    row=wellrows,
    column=wellcolumns,
    rate=flux,
    concentration=flux_concentration,
    concentration_boundary_type="AUX",
)

# %%
# since we are solving a variable density problem, we need to add the buoyancy package.
gwf_model["buoyancy"] = imod.mf6.Buoyancy(
    denseref=min_density, densityfile="density_out.dat"
)
slope = (max_density - min_density) / (max_concentration - min_concentration)
gwf_model["buoyancy"].add_species_dependency(
    slope, min_concentration, "transport", "salinity"
)

# %%
# now let's make the transport model. It containss the standard packages of storage, dispersion and advection
# as well as initial condiations and output control.
porosity = 0.35

tpt_model = imod.mf6.GroundwaterTransportModel(gwf_model, "salinity")
tpt_model["advection"] = imod.mf6.AdvectionTVD()
tpt_model["Dispersion"] = imod.mf6.Dispersion(
    diffusion_coefficient=0.57024,
    longitudinal_horizontal=0.1,
    transversal_horizontal1=0.01,
    xt3d_off=False,
    xt3d_rhs=False,
)


tpt_model["storage"] = imod.mf6.MobileStorage(
    porosity=porosity,
)

tpt_model["ic"] = imod.mf6.InitialConditions(start=max_concentration)
tpt_model["oc"] = imod.mf6.OutputControl(save_concentration="last", save_budget="last")
tpt_model.take_discretization_from_model(gwf_model)

# %%
# now let's define a simulation using the flow and transport models.

# Attach it to a simulation
simulation = imod.mf6.Modflow6Simulation("henry")

simulation["flow"] = gwf_model
simulation["transport"] = tpt_model
# Define solver settings
simulation["solver"] = imod.mf6.Solution(
    print_option="summary",
    csv_output=False,
    no_ptc=True,
    outer_dvclose=1.0e-6,
    outer_maximum=500,
    under_relaxation=None,
    inner_dvclose=1.0e-6,
    inner_rclose=1.0e-5,
    rclose_option="STRICT",
    inner_maximum=100,
    linear_acceleration="bicgstab",
    scaling_method=None,
    reordering_method=None,
    relaxation_factor=0.97,
)
# Collect time discretization
simtimes = list(daterange(datetime(2000, 1, 1, 0, 0, 0), 500, 0.001))
nrtimes = len(simtimes) - 2
simulation.create_time_discretization(additional_times=simtimes)


# %%
# We'll create a new directory in which we will write and run the model.
with imod.util.temporary_directory() as modeldir:
    simulation.write(modeldir, binary=False)

    # %%
    # Run the model
    # -------------
    #
    # .. note::
    #
    #   The following lines assume the ``mf6`` executable is available on your PATH.
    #   :ref:`The Modflow 6 examples introduction <mf6-introduction>` shortly
    #   describes how to add it to yours.

    simulation.run()

    # %%
    # Open the results
    # ----------------
    #
    # We'll open the  density file.

    density = imod.mf6.open_hds(
        modeldir / "density_out.dat",
        modeldir / "flow/dis.dis.grb",
    )

    # %%
    # Visualize the results (to get the plot right, invert the coordinate axis of layer)
    # ---------------------
    layer2 = list(np.arange(40, 0, -1))

    density = density.assign_coords(layer=layer2)

    density.isel(y=0, time=nrtimes).plot.contourf()

    i = 0
