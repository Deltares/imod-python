"""
Circle
======

This example illustrates how to setup a very simple unstructured groundwater
model using the ``imod`` package and associated packages.

In overview, we'll set the following steps:

    * Create a triangular mesh for a disk geometry.
    * Create the xugrid UgridDataArrays containg the MODFLOW6 parameters.
    * Feed these arrays into the imod mf6 classes.
    * Write to modflow6 files.
    * Run the model.
    * Open the results back into UgridDataArrays.
    * Visualize the results.
"""

# sphinx_gallery_thumbnail_number = -1

# %%
# We'll start with the following imports:

import matplotlib.pyplot as plt
import meshzoo
import numpy as np
import xarray as xr
import xugrid as xu

import imod

# %%
# Create a mesh
# -------------
#
# The first steps consists of generating a mesh. In this example, our geometry
# is so simple that we don't need complex mesh generators. Instead we will use
# ``meshzoo``, which can generate a number of meshes for simple domains such as
# triangles, rectangles, regular polygons, or disks.
#
# We can use the nodes and the cell node connectivity directly to initialize a
# xugrid Ugrid2d object, and we'll create a quick plot to take a look at the
# grid.

nodes, triangles = meshzoo.disk(6, 6)
nodes *= 1000.0
grid = xu.Ugrid2d(*nodes.T, -1, triangles)

fig, ax = plt.subplots()
xu.plot.line(grid, ax=ax)
ax.set_aspect(1)

# %%
# Create UgridDataArray
# ---------------------
#
# Now that we have defined the grid, we can start defining the model parameter
# data.
#
# Our goal here is to define a steady-state model with:
#
# * Uniform conductivities of 1.0 m/d;
# * Two layers of 5.0 m thick;
# * Uniform recharge of 0.001 m/d on the top layer;
# * Constant heads of 1.0 m along the exterior edges of the mesh.
#
# From these boundary conditions, we would expect circular mounding of the
# groundwater; with small flows in the center and larger flows as the recharge
# accumulates while the groundwater flows towards the exterior boundary.

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
k = xu.full_like(idomain, 1.0)
k33 = k.copy()
rch_rate = xu.full_like(idomain.sel(layer=1), 0.001, dtype=float)
bottom = idomain * xr.DataArray([5.0, 0.0], dims=["layer"])

# %%
# All the data above have been constants over the grid. For the constant head
# boundary, we'd like to only set values on the external border. We can
# `py:method:xugrid.UgridDataset.binary_dilation` to easily find these cells:

chd_location = xu.zeros_like(idomain.sel(layer=2), dtype=bool).ugrid.binary_dilation(
    border_value=True
)
constant_head = xu.full_like(idomain.sel(layer=2), 1.0).where(chd_location)

fig, ax = plt.subplots()
constant_head.ugrid.plot(ax=ax)
xu.plot.line(grid, ax=ax, color="black")
ax.set_aspect(1)

# %%
# Write the model
# ---------------
#
# The first step is to define an empty model, the parameters and boundary
# conditions are added in the form of the familiar MODFLOW packages.

gwf_model = imod.mf6.GroundwaterFlowModel()
gwf_model["disv"] = imod.mf6.VerticesDiscretization(
    top=10.0, bottom=bottom, idomain=idomain
)
gwf_model["chd"] = imod.mf6.ConstantHead(
    constant_head, print_input=True, print_flows=True, save_flows=True
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
gwf_model["rch"] = imod.mf6.Recharge(rch_rate)

simulation = imod.mf6.Modflow6Simulation("circle")
simulation["GWF_1"] = gwf_model
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
simulation.time_discretization(times=["2000-01-01", "2000-01-02"])

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
# First, we'll open the heads (.hds) file.

head = imod.mf6.out.open_hds(
    modeldir / "GWF_1/GWF_1.hds",
    modeldir / "GWF_1/disv.disv.grb",
)
head

# %%
# For a DISV MODFLOW6 model, the heads are returned as a UgridDataArray.  While
# all layers are timesteps are available, they are only loaded into memory as
# needed.
#
# We may also open the cell-by-cell flows (.cbc) file.

cbc = imod.mf6.open_cbc(
    modeldir / "GWF_1/GWF_1.cbc",
    modeldir / "GWF_1/disv.disv.grb",
)
print(cbc.keys())

# %%
# The flows are returned as a dictionary of UgridDataArrays. This dictionary
# contains all entries that are stored in the CBC file, but like for the heads
# file the data are only loaded into memory when needed.
#
# The horizontal flows are stored on the edges of the UgridDataArray topology.
# The other flows are generally stored on the faces; this includes the
# flow-lower-face.
#
# We'll create a dataset for the horizontal flows for further analysis.

ds = xu.UgridDataset(grid=grid)
ds["u"] = cbc["flow-horizontal-face-x"]
ds["v"] = cbc["flow-horizontal-face-y"]

# %%
# Visualize the results
# ---------------------
#
# We can quickly and easily visualize the output with the plotting functions
# provided by xarray and xugrid:

fig, ax = plt.subplots()
head.isel(time=0, layer=0).ugrid.plot(ax=ax)
ds.isel(time=0, layer=0).plot.quiver(
    x="mesh2d_edge_x", y="mesh2d_edge_y", u="u", v="v", color="white"
)
ax.set_aspect(1)

# %%
# As would be expected from our model input, we observe circular groundwater
# mounding and increasing flows as we move from the center to the exterior.
