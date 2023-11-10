# %% Import packages
import matplotlib.pyplot as plt
import xarray as xr
from example_models import create_hondsrug_simulation

import imod
from imod.mf6.partitioned_simulation_postprocessing import merge_heads

# %%
# Obtain the simulation, write it, run it, and plot some heads.
# There is a separate example contained in
# `hondsrug <https://deltares.gitlab.io/imod/imod-python/examples/mf6/hondsrug.html#sphx-glr-examples-mf6-hondsrug-py>`_
# that you should look at if you are interested in the model building
gwf_simulation = create_hondsrug_simulation()


# %%
# Write the model and run it (before partitioning, so we can compare if the results are similar)
modeldir = imod.util.temporary_directory()
original_modeldir = modeldir / "original"

gwf_simulation.write(original_modeldir, False, False)

gwf_simulation.run()

# %%
# plot the simulation results of the unpartitioned model
hds_original = imod.mf6.open_hds(
    original_modeldir / "GWF_1" / "GWF_1.hds",
    original_modeldir / "GWF_1" / "dis.dis.grb",
)

fig, ax = plt.subplots()

hds_original.sel(layer=3).isel(time=6).plot(ax=ax)
ax.set_title("hondsrug original ")
# %%
# Now we partition the Hondsrug model
idomain = gwf_simulation["GWF_1"].domain

submodel_labels = xr.zeros_like(idomain.isel(layer=0))
submodel_labels[100:, :] = 1
submodel_labels[:, 250:] = 2
submodel_labels[100:, 250:] = 3

# %%
# plot the partitioning array. It shows how the model will be partitioned.
fig, ax = plt.subplots()
submodel_labels.plot(ax=ax)
ax.set_title("hondsrug partitioning geometry")

split_simulation = gwf_simulation.split(submodel_labels)

# %%
# Now we  write and run the partitioned model
split_simulation.write(modeldir, False, False)
split_simulation.run()


# %%
# Load and plot the simulation results. Also plot the differences with the original model
hds_split = merge_heads(modeldir, split_simulation)
fig, ax = plt.subplots()
hds_split.sel(layer=3).isel(time=6).plot(ax=ax)
ax.set_title("hondsrug partitioned ")

diff = hds_split - hds_original
diff_for_plot = diff.max(dim=["time", "layer"])
fig, ax = plt.subplots()
diff_for_plot.plot(ax=ax)
ax.set_title("hondsrug diff ")
