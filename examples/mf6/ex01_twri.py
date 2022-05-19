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

# %%
# Create grid coordinates
# -----------------------
#
# The first steps consist of setting up the grid -- first the number of layer,
# rows, and columns. Cell sizes are constant throughout the model.

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
well_layer = [3, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
well_row = [5, 4, 6, 9, 9, 9, 9, 11, 11, 11, 11, 13, 13, 13, 13]
well_column = [11, 6, 12, 8, 10, 12, 14, 8, 10, 12, 14, 8, 10, 12, 14]
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
gwf_model["ic"] = imod.mf6.InitialConditions(head=0.0)
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
gwf_model["wel"] = imod.mf6.WellDisStructured(
    layer=well_layer,
    row=well_row,
    column=well_column,
    rate=well_rate,
    print_input=True,
    print_flows=True,
    save_flows=True,
)

# Attach it to a simulation
simulation = imod.mf6.Modflow6Simulation("ex01-twri")
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

modeldir = imod.util.temporary_directory()
simulation.write(modeldir)

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

# %%
# .. _MODFLOW6 example problems: https://github.com/MODFLOW-USGS/modflow6-examples
# .. _description: https://modflow6-examples.readthedocs.io/en/master/_examples/ex-gwf-twri.html
# .. _notebook: https://github.com/MODFLOW-USGS/modflow6-examples/tree/master/notebooks/ex-gwf-twri.ipynb
# .. _Techniques of Water-Resources Investigation: https://pubs.usgs.gov/twri/twri7-c1/
# .. _McDonald & Harbaugh, 1988: https://pubs.er.usgs.gov/publication/twri06A1
# .. _Harbaugh & McDonald, 1996: https://pubs.er.usgs.gov/publication/ofr96485
# .. _Harbaugh, 2005: https://pubs.er.usgs.gov/publication/tm6A16
# .. _FloPy: https://github.com/modflowpy/flopy

# %%
