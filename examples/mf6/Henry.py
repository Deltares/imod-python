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

import numpy as np
import xarray as xr

import imod
nlay = 40
nrow = 1
ncol = 80
shape = (nlay, nrow, ncol)

dx = 2/80
dy = 1
xmin = 0.0
xmax = dx * ncol
ymin = 0.0
ymax = 1
dims = ("layer", "y", "x")

layer = np.arange(0, 40, 1)
y = np.arange(ymin, ymax,  dy) + 0.5 * dy
x = np.arange(xmin, xmax, dx) + 0.5 * dx

# %%
# notice we are adding also a dy coordinate. This is used to determine the width of
# the cells in the y direction, because it has only 1 element.
dy = 1
coords = {"layer": layer, "y": y, "x": x, "dy": dy}

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
    icelltype=idomain,
    k=864.0,
    k33=864.0,
)
gwf_model["sto"] = imod.mf6.SpecificStorage(
    specific_storage=1.0e-5,
    specific_yield=0.15,
    transient=False,
    convertible=0,
)

gwf_model["ic"] = imod.mf6.InitialConditions(head=0.0)
gwf_model["oc"] = imod.mf6.OutputControl(save_head="all", save_budget="all")


# %%
# Now let's make the boundary conditions. We have a constant head on the right and
# prescribed flow on the right.
constant_head = xr.full_like(idomain, np.nan, dtype=float).isel(x=[x[-1]])
heads =np.arange(0, 1.025, 1.025/nlay)[np.newaxis]  #create 1d vector with desired values. add an axis to make it a 2d row vector with 1 column
constant_head[..., 0] = heads.T                     #transpose the 2d vector so that it becomes a column vector, now it fits the layout of constant_head
gwf_model["right_boundary"] = imod.mf6.ConstantHead(
    constant_head, print_input=True, print_flows=True, save_flows=True
)

flux =  np.full_like(layer, 5.7024/nlay, dtype=np.floating)

wellrows = np.full_like(layer, 1, dtype=np.int32)
wellcolumns = np.full_like(layer, 1, dtype=np.int32)
gwf_model["left_boundary"] = imod.mf6.WellDisStructured(layer=layer, row=wellrows, column=wellcolumns, rate=flux )

# Attach it to a simulation
simulation = imod.mf6.Modflow6Simulation("henry")
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
simulation.create_time_discretization(
    additional_times=["2000-01-01", "2000-01-02", "2000-01-03", "2000-01-04"]
)

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
        modeldir / "GWF_1/GWF_1.hds",
        modeldir / "GWF_1/dis.dis.grb",
    )

# %%
# Visualize the results
# ---------------------

    head.isel(layer=0, time=0).plot.contourf()

