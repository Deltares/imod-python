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

dx = 2/80
xmin = 0.0
xmax = 2
dims = ("layer", "y", "x")

layer = np.arange(1, 41, 1)
y = np.array([0.5])
x = np.arange(xmin, xmax, dx) + 0.5 * dx

# %%
# notice we are adding also a dy coordinate. This is used to determine the width of
# the cells in the y direction, because it has only 1 element.
dy = 1
coords = {"layer": layer, "y": y, "x": x, "dy": dy, "dx": dx}

idomain = xr.DataArray(np.ones(shape, dtype=int), coords=coords, dims=dims)

top = xr.full_like(idomain.sel(layer=1), 1.0,  dtype=np.floating)
bottom_level_of_top_layer = 1.-1./nlay
bottom = xr.DataArray(np.arange(bottom_level_of_top_layer,0,-bottom_level_of_top_layer/40), {"layer": layer}, ("layer",))

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
    transient=True,
    convertible=0,
)

gwf_model["ic"] = imod.mf6.InitialConditions(head=0.0)
gwf_model["oc"] = imod.mf6.OutputControl(save_head="last", save_budget="last")


# %%
# Now let's make the boundary conditions. We have a constant head on the right and
# prescribed flow on the right.
constant_head = xr.full_like(idomain, np.nan, dtype=float)
inflow_concentration =  xr.full_like(idomain, np.nan, dtype=float)

heads =np.arange(0.025, 0, -0.025/nlay)[np.newaxis]  #create 1d vector with desired values. add an axis to make it a 2d row vector with 1 column
constant_head[..., ncol-1] = heads.T                     #transpose the 2d vector so that it becomes a column vector, now it fits the layout of constant_head
conc = np.full_like(heads.T, 1)
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
flux_concentration[...]=0
flux_concentration=flux_concentration.expand_dims(species=["salinity"])

wellrows = np.full_like(layer, 1, dtype=np.int32)
wellcolumns = np.full_like(layer, 1, dtype=np.int32)
gwf_model["left_boundary"] = imod.mf6.WellDisStructured(layer=layer, row=wellrows, column=wellcolumns, rate=flux, concentration= flux_concentration, concentration_boundary_type="AUX" )

gwf_model["buoyancy"] = imod.mf6.Buoyancy(
    hhformulation_rhs=True, denseref=1000, densityfile="density_out.dat"
)
gwf_model["buoyancy"].add_species_dependency(25, 0, "transport", "salinity")


porosity = 0.35

tpt_model = imod.mf6.GroundwaterTransportModel(gwf_model, "salinity")
tpt_model["advection"] = imod.mf6.AdvectionTVD()
tpt_model["Dispersion"] = imod.mf6.Dispersion(
    diffusion_coefficient=0.57024,
    longitudinal_horizontal=0.0,
    transversal_horizontal1=0.0,
    xt3d_off=False,
    xt3d_rhs=False,
)


tpt_model["storage"] = imod.mf6.MobileStorage(
    porosity=porosity,
)

tpt_model["ic"] = imod.mf6.InitialConditions(start=0.001)
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
    inner_dvclose=1.0e-5,
    inner_rclose=0.001,
    inner_maximum=100,
    linear_acceleration="bicgstab",
    scaling_method=None,
    reordering_method=None,
    relaxation_factor=0.97,
)
# Collect time discretization
#simtimes = daterange(datetime(2000, 1, 1,0,0,0), 5000, 0.001 )
simtimes = list(daterange_expanding(datetime(2000, 1, 1,0,0,0), 500, 0.001,1, 1.02 ))
nrtimes = len(simtimes) -2
simulation.create_time_discretization(additional_times=simtimes)



# %%
# We'll create a new directory in which we will write and run the model.

with imod.util.temporary_directory() as modeldir:
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
# Visualize the results
# ---------------------


    density.isel(y=0, time=nrtimes).plot.contourf()

    i=0

