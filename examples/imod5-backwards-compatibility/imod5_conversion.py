"""
This is an example of how to convert an existing iMOD5 model into an
unstructured Modflow 6 model. For this we'll use the ``convert_to_disv``
function, which is still an experimental feature. In this example, we
have to work around the following issues in the converter:

1. It expects no layer to be assigned for the rch and riv package.
    In the example iMOD5 model, a layer is assigned to the rch and riv package.
2. The BarycentricInterpolator, used to compute starting heads,
    introduced nans at the edges, whereas OverlapRegridder used to find the active
    cells, suffers from no such thing.
3. CHDs are separated into different systems for each layer
4. Due to a bug with broadcasting n_times too many wells are generated:
    94 times & 94 indices.
5. The data in the hfb packages causes xu.snap_to_grid to throw errors.

"""

# %%
# Imports
# -------
import numpy as np
import xugrid as xu

import imod

# %%
# Read data
# ---------
#
# For this example we'll get our data from the data shipped with iMOD Python.
# To read your own iMOD5 model, you can call
# `imod.formats.prj.open_projectfile_data <https://deltares.gitlab.io/imod/imod-python/api/generated/io/imod.formats.prj.open_projectfile_data.html>`_
temp_dir = imod.util.temporary_directory()

data_prj, repeat_stress = imod.data.imod5_projectfile_data(temp_dir)

data_prj

# %%
# Cleanup
# -------
#
# Remove layers
# TODO: If layer assigned, do not assign at depth?

data_prj["rch"]["rate"] = data_prj["rch"]["rate"].sel(layer=1)
data_prj["drn-1"]["conductance"] = data_prj["drn-1"]["conductance"].sel(layer=1)
data_prj["drn-1"]["elevation"] = data_prj["drn-1"]["elevation"].sel(layer=1)

# For some reason the data in the hfb packages in this example cause
# xu.snap_to_grid to fail. So remove for now.
for i in range(1, 27):
    data_prj.pop(f"hfb-{i}")

# %%
# Target grid
# -----------
#
# The rch rate is defined on a coarse grid,
# so we use this grid to make a lightweight example.

dummy = data_prj["rch"]["rate"]
dummy.load()

target = xu.UgridDataArray.from_structured(dummy)
triangular_grid = target.grid.triangulate()
voronoi_grid = triangular_grid.tesselate_centroidal_voronoi()

voronoi_grid.plot()

# %%
# Convert
# -------
#
# We can convert the iMOD5 model to a Modflow6 model on the unstructured grid
# with the following function:

mf6_model = imod.prj.convert_to_disv(data_prj, voronoi_grid)

mf6_model

# %%
# Cleanup
# -------
#
# Clean starting head. Due to regridding, there is an empty edge at the
# bottom grid in starting head.

edge = np.isnan(mf6_model["shd"]["start"].sel(layer=1))
edge = edge & (mf6_model["disv"]["idomain"] == 1)
shd_mean_layer = mf6_model["shd"]["start"].mean(dim="mesh2d_nFaces")
mf6_model["shd"]["start"] = mf6_model["shd"]["start"].where(~edge, shd_mean_layer)

# For some reason, all wells were broadcasted n_time times to index,
# resulting in 94 duplicate wells in a single cells
# Select 1, to reduce this.
# Furthermore rates we constant in time, but not along index

for pkgname in ["wel-1", "wel-2"]:
    rates = mf6_model[pkgname].dataset.isel(time=[1])["rate"].values
    mf6_model[pkgname].dataset = mf6_model[pkgname].dataset.sel(index=[1], drop=False)
    # Assign varying rates through to time to dataset
    mf6_model[pkgname].dataset["rate"].values = rates.T

# %%
# Assign to simulation
# --------------------
#
# A Modflow 6 model is not a complete simulation, we still have to define a
# Modflow6Simulation and have to include some extra information

mf6_sim = imod.mf6.Modflow6Simulation(name="mf6sim")
mf6_sim["gwf1"] = mf6_model

# %%
# Set solver
mf6_sim["ims"] = imod.mf6.SolutionPresetModerate(modelnames=["gwf1"])

# %%
# Create time discretization, we'll only have to specify the end time. iMOD
# Python will take the other time steps from the stress packages.

endtime = np.datetime64("2013-04-01T00:00:00.000000000")

mf6_sim.create_time_discretization(additional_times=[endtime])


# %%
# Write modflow 6 data

modeldir = temp_dir / "mf6"
mf6_sim.write(directory=modeldir)

# %%
# Run Modflow 6 simulation

mf6_sim.run()

# %%
# Read results from Modflow 6 simulation

hds = imod.mf6.open_hds(
    modeldir / "gwf1" / "gwf1.hds", modeldir / "gwf1" / "disv.disv.grb"
)

hds.load()

# %%
# Visualize

hds.isel(time=-1, layer=4).ugrid.plot()

# %%
