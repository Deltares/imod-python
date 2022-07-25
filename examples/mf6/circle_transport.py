"""
Circle (transport)
==================

This example illustrates how to setup a very simple unstructured groundwater
transport model using the ``imod`` package and associated packages.

In overview, we'll set the following steps:

    * Setting up the flow model, just like in the circle.py example
    * set up the transport model
    * Run the simulation.
    * Visualize the results.
"""

# %%
# We'll start with the following imports:

from datetime import date, timedelta

import matplotlib.pyplot as plt
import meshzoo
import numpy as np
import xarray as xr
import xugrid as xu

import imod


# helper function for creating iterable with all dates between two dates
def daterange(date1, date2):
    for n in range(int((date2 - date1).days) + 1):
        yield date1 + timedelta(n)


# %%
# Create a mesh
# -------------
#
# as explained in circle.py we first generate a grid and a hydraulic conductivity array

nodes, triangles = meshzoo.disk(6, 6)
nodes *= 1000.0
grid = xu.Ugrid2d(*nodes.T, -1, triangles)

nface = len(triangles)
nlayer = 2

idomain = xu.UgridDataArray(
    xr.DataArray(
        np.ones((nlayer, nface), dtype=np.int32),
        coords={"layer": [1, 2]},
        dims=["layer", grid.face_dimension],
    ),
    grid=grid,
)
icelltype = xu.full_like(idomain, 0)
k = xu.full_like(idomain, 1.0, dtype=float)
k33 = k.copy()

bottom = idomain * xr.DataArray([5.0, 0.0], dims=["layer"])

# %%
# Create arrays for recharge process
# ---------------------
# we need a recharge rate for the fluid and a recharge rate for the solute.
# The fluid recharge rate is volumetric and per unit area, so the unit is Length/time
# The solute recharge rate is the concentration of solute in the recharge, and has concentration units.
rch_rate = xu.full_like(idomain.sel(layer=1), 0.001, dtype=float)
rch_concentration = xu.full_like(rch_rate, 1.0)
rch_concentration = rch_concentration.expand_dims(species=["salinity"])


# %%
# unlike a recharge boundary, with a prescribed head boundary you don't know a priori whether
# water will flow in over the boundary or leave across the boundary. If water flows into the model
# over the boundary, it carries a prescribed solute concentration. if it leaves, it leaves with the
# concentration that was computed for it.
#
# In this example we set the prescribed head value to 1 and the external concentration to 1 as well.
# the boundary only operates on the top layer.
chd_location = xu.zeros_like(idomain.sel(layer=1), dtype=bool).ugrid.binary_dilation(
    border_value=True
)
constant_head = xu.full_like(idomain, 1.0, dtype=float).where(chd_location)
constant_concentration = xu.full_like(constant_head, 1)
constant_concentration = constant_concentration.expand_dims(species=["salinity"])


# %%
# Write the flow  model
# ---------------
#
# see the circle.py example for more comments.

gwf_model = imod.mf6.GroundwaterFlowModel()
gwf_model["disv"] = imod.mf6.VerticesDiscretization(
    top=10.0, bottom=bottom, idomain=idomain
)
gwf_model["chd"] = imod.mf6.ConstantHead(
    constant_head,
    concentration=constant_concentration,
    print_input=True,
    print_flows=True,
    save_flows=True,
)
gwf_model["ic"] = imod.mf6.InitialConditions(head=0.0)
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
gwf_model["oc"] = imod.mf6.OutputControl(save_head="all", save_budget="all")
gwf_model["rch"] = imod.mf6.Recharge(
    rch_rate, concentration=rch_concentration, print_flows=True, save_flows=True
)

simulation = imod.mf6.Modflow6Simulation("circle")
simulation["GWF_1"] = gwf_model
simulation["flow_solver"] = imod.mf6.Solution(
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
    linear_acceleration="bicgstab",
    scaling_method=None,
    reordering_method=None,
    relaxation_factor=0.97,
)
simtimes = daterange(date(2000, 1, 1), date(2000, 10, 10))
simulation.create_time_discretization(additional_times=simtimes)


# %%
# Write the transport  model
# ---------------
#
# The transport model needs as input the flow field inside the domain computed by the flow model.
# It also needs the fluxes over the boundary. It uses the same discretization as the flow model.
# Create a transport model for salinity, give it the flow model, and tell it to use the same discretization.
transport_model = imod.mf6.model.GroundwaterTransportModel(gwf_model, "salinity")
transport_model["disv"] = gwf_model["disv"]

# Now we define some transport packages for simulating the physical processes of  advection, molecular
# diffusion and mechanical dispersion.
# This example is transient, and the volume available for storage is the porosity, in this case 0.3
transport_model["dsp"] = imod.mf6.Dispersion(
    1e-4, 1.0, 10.0, 1.0, 2.0, 3.0, False, False
)
transport_model["adv"] = imod.mf6.AdvectionUpstream()
transport_model["mst"] = imod.mf6.MobileStorageTransfer(0.3)

# Now we define initial conditions (0) and output options for the transport simulation
transport_model["ic"] = imod.mf6.InitialConditions(start=0.0)
transport_model["oc"] = imod.mf6.OutputControl(
    save_concentration="all", save_budget="last"
)

# assign the transport model to the simulation
simulation["GWT_1"] = transport_model
simulation["transport_solver"] = imod.mf6.Solution(
    modelnames=["GWT_1"],
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

# %%
# We'll create a new directory in which we will write and run the model.

modeldir = imod.util.temporary_directory()
simulation.write(modeldir, binary=False)

# %%
# Run the model
# -------------
#
# .. note::
#
#   The following lines assume the ``mf6`` executable is available on your
#   PATH. The examples introduction shortly describes how to add it to yours.

simulation.run()

# %%
# Open the results
# ----------------

# open the concentration results
sim_concentration = imod.mf6.out.open_conc(
    modeldir / "GWT_1/GWT_1.ucn",
    modeldir / "GWF_1/disv.disv.grb",
)


# %%
# Visualize the results
# ---------------------
#
# We can quickly and easily visualize the output with the plotting functions
# provided by xarray and xugrid:

fig, ax = plt.subplots()
sim_concentration.isel(time=33, layer=0).ugrid.plot(ax=ax)
ax.set_aspect(1)

# %%
# we observe the initial water (without solute) slowly
# being flushed out by water coming in from the recharge with a concentration of 1.
