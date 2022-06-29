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
"""
# %%
# We'll start with the usual imports. As this is an simple (synthetic)
# structured model, we can make due with few packages.

from datetime import date, datetime, timedelta
import pathlib
import numpy as np
import xarray as xr

import imod


# %%
# helper function for creating iterable given starting date and number of days
def daterange(date1, numberSteps, step=1):
    for n in range(numberSteps):
        yield date1 + timedelta(n*step)

def daterange_expanding(date1, numberSteps, firstStep,maxStep, factor):
    currentStart = date1
    currentStep = firstStep
    yield date1
    for _ in range(numberSteps):
        currentStart= currentStart+ timedelta(currentStep)
        currentStep = min(currentStep*factor, maxStep)
        yield currentStart

nlay = 40
nrow = 1
ncol = 80
shape = (nlay, nrow, ncol)


xmax = 2
xmin = 0.0
dx = ( xmax - xmin)/ncol
dims = ("layer", "y", "x")

layer = np.arange(1, 41, 1)
y = np.array([0.5])
x = np.arange(xmin, xmax, dx) + 0.5 * dx

max_concentration = 35.0
min_concentration =  0.0
max_density = 1025.0
min_density = 1000.0

layer_thickness = 1/nlay

# %%
# notice we are adding also a dy coordinate. This is used to determine the width of
# the cells in the y direction, because it has only 1 element.
dy = -1
coords = {"layer": layer, "y": y, "x": x, "dy": dy, "dx": dx}

idomain = xr.DataArray(np.ones(shape, dtype=int), coords=coords, dims=dims)

top = xr.full_like(idomain.sel(layer=1), 1.0,  dtype=np.floating)
bottom_level_of_top_layer = 1.-layer_thickness
bottom = xr.DataArray(np.arange(bottom_level_of_top_layer,-layer_thickness,-layer_thickness), {"layer": layer}, ("layer",))

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
# Now let's make the boundary conditions. We have a constant head on the right and
# prescribed flow on the right.
constant_head = xr.full_like(idomain, np.nan, dtype=float)
inflow_concentration =  xr.full_like(idomain, np.nan, dtype=float)

depth_cell_centers = 1 - (np.arange(bottom_level_of_top_layer,-layer_thickness,-layer_thickness) + layer_thickness/2)
head_cellcentres = depth_cell_centers*(max_density - min_density) / min_density


head_cellcentres = head_cellcentres[np.newaxis]  #create 1d vector with desired values. add an axis to make it a 2d row vector with 1 column
constant_head[..., ncol-1] = head_cellcentres.T                     #transpose the 2d vector so that it becomes a column vector, now it fits the layout of constant_head
conc = np.full_like(head_cellcentres.T, max_concentration)
inflow_concentration[..., ncol-1] = conc
inflow_concentration = inflow_concentration.expand_dims(species=["salinity"])

gwf_model["right_boundary"] = imod.mf6.ConstantHead(
    constant_head, concentration=inflow_concentration, concentration_boundary_type="AUX", print_input=True, print_flows=True, save_flows=True
)

flux =  np.full_like(layer, 5.7024/nlay, dtype=np.floating)

flux_concentration = xr.DataArray(
        data=layer.copy(),
        dims=["cell"],
        coords=dict(
            cell=(range(0, nlay))
        ),
    )
flux_concentration[...]=min_concentration
flux_concentration=flux_concentration.expand_dims(species=["salinity"])

wellrows = np.full_like(layer, 1, dtype=np.int32)
wellcolumns = np.full_like(layer, 1, dtype=np.int32)
gwf_model["left_boundary"] = imod.mf6.WellDisStructured(layer=layer, row=wellrows, column=wellcolumns, rate=flux, concentration= flux_concentration, concentration_boundary_type="AUX" )

gwf_model["buoyancy"] = imod.mf6.Buoyancy(
    denseref=min_density, densityfile="density_out.dat"
)
slope = (max_density - min_density)/( max_concentration - min_concentration)
gwf_model["buoyancy"].add_species_dependency(slope, min_concentration, "transport", "salinity")


porosity = 0.35

tpt_model = imod.mf6.GroundwaterTransportModel(gwf_model, "salinity")
tpt_model["advection"] = imod.mf6.AdvectionTVD()
tpt_model["Dispersion"] = imod.mf6.Dispersion(
    diffusion_coefficient= 0.0 ,
    longitudinal_horizontal=0.1,
    transversal_horizontal1=0.01,
    xt3d_off=False,
    xt3d_rhs=False,
)


tpt_model["storage"] = imod.mf6.MobileStorage(
    porosity=porosity,
)

tpt_model["ic"] = imod.mf6.InitialConditions(start=max_concentration)
tpt_model["oc"] = imod.mf6.OutputControl(
    save_concentration="last", save_budget="last"
)
tpt_model.take_discretization_from_model(gwf_model)



# Attach it to a simulation
simulation = imod.mf6.Modflow6Simulation("henry")

simulation["flow"] = gwf_model
simulation["transport"]= tpt_model
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
    rclose_option='STRICT',
    inner_maximum=100,
    linear_acceleration="bicgstab",
    scaling_method=None,
    reordering_method=None,
    relaxation_factor=0.9,
)
# Collect time discretization
#simtimes = daterange(datetime(2000, 1, 1,0,0,0), 5000, 0.001 )
simtimes = list(daterange_expanding(datetime(2000, 1, 1,0,0,0), 500, 0.001,0.001, 1.0 ))
nrtimes = len(simtimes) -2
simulation.create_time_discretization(additional_times=simtimes)



# %%
# We'll create a new directory in which we will write and run the model.
modeldir = pathlib.Path('C:\\Users\\slooten\\AppData\\Local\\Temp\\tmp9z4kdut2')
with imod.util.temporary_directory() as someDir:
    simulation.write(modeldir,binary=False)

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
# We'll open the heads (.hds) file.

    head = imod.mf6.open_hds(
        modeldir / "flow/flow.hds",
        modeldir / "flow/dis.dis.grb",
    )
    concentration = imod.mf6.open_hds(
        modeldir / "transport/transport.ucn",
        modeldir / "flow/dis.dis.grb",
    )
    density = imod.mf6.open_hds(
        modeldir / "density_out.dat",
        modeldir / "flow/dis.dis.grb",
    )

    cbc = imod.mf6.open_cbc(
        modeldir / "flow/flow.cbc",
        modeldir / "flow/dis.dis.grb",
    )

# %%
# Visualize the results (to get the plot right, invert the coordinate axis of layer)
# ---------------------
    layer2 = list(np.arange(40, 0, -1))

    concentration = concentration.assign_coords(layer=layer2)

    concentration.isel(y=0, time=10).plot.contourf()
    concentration.isel(y=0, time=nrtimes).plot.contourf()

    i=0

