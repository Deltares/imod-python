"""
Partitioning a regional model
=============================

This example shows how a model can be partitioned into submodels. This will
allow parallelization when solving the model. The example used is the Hondsrug
model. It is partitioned into 3 rectangular parts. In the example we first run
the original, unpartitioned model. Then we partition the model and run the
resulting simulation. Finally we merge the head output of the submodels into a
head array for the whole grid. we print both the heads obtained without
partitioning, and the merged heads of the partitioned simulation, ' for
comparison.
"""

# %% Import packages
import matplotlib.pyplot as plt

import imod
from imod.mf6.multimodel.partition_generator import get_label_array

# %%
# Obtain the simulation, write it, run it, and plot some heads.
# There is a separate example contained in
# :doc:`hondsrug </examples/mf6/hondsrug>`
# that you should look at if you are interested in the model building
tmpdir = imod.util.temporary_directory()

gwf_simulation = imod.data.hondsrug_simulation(tmpdir / "hondsrug_saved")

# %%
# The model was written before the xt3d_option and rhs_option arguments were
# added to iMOD Python. Set missing options to False.
gwf_simulation["GWF"]["npf"].set_xt3d_option(is_xt3d_used=False, is_rhs=False)

# %%
# Write the model and run it (before partitioning, so we can compare if the
# results are similar).
original_modeldir = tmpdir / "original"

gwf_simulation.write(original_modeldir)
gwf_simulation.run()

# %%
# Plot the simulation results of the unpartitioned model.
hds_original = gwf_simulation.open_head()

fig, ax = plt.subplots()

hds_original.sel(layer=3).isel(time=6).plot(ax=ax)
ax.set_title("hondsrug original ")
# %%
# Now we partition the Hondsrug model
idomain = gwf_simulation["GWF"].domain
number_partitions = 16
submodel_labels = get_label_array(gwf_simulation, number_partitions)

# %%
# plot the partitioning array. It shows how the model will be partitioned.
fig, ax = plt.subplots()
submodel_labels.plot(ax=ax)
ax.set_title("hondsrug partitioning geometry")

split_simulation = gwf_simulation.split(submodel_labels)

# %%
# Now we  write and run the partitioned model
split_modeldir = tmpdir / "split"

split_simulation.write(split_modeldir)
split_simulation.run()


# %%
# Load and plot the simulation results. Also plot the differences with the original model
hds_split = split_simulation.open_head()["head"]
fig, ax = plt.subplots()
hds_split.sel(layer=3).isel(time=6).plot(ax=ax)
ax.set_title("hondsrug partitioned ")

diff = hds_split - hds_original
diff_for_plot = diff.max(dim=["time", "layer"])
fig, ax = plt.subplots()
diff_for_plot.plot(ax=ax)
ax.set_title("hondsrug diff ")

# %%
