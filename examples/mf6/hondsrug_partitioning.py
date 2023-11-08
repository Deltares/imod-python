# %% Import packages
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from example_models import create_hondsrug_simulation

import imod
from imod.typing import GridDataArray

from imod.mf6.partitioned_simulation_postprocessing import (
    merge_balances,
    merge_heads,
)

# %%
# Obtain the simulation, write it, run it, and plot some heads.
# There is a separate example contained in
# `hondsrug <https://deltares.gitlab.io/imod/imod-python/examples/mf6/hondsrug.html#sphx-glr-examples-mf6-hondsrug-py>`_
# that you should look at if you are interested in the model building
gwf_simulation = create_hondsrug_simulation()

modeldir = imod.util.temporary_directory()
original_modeldir = modeldir / "original"

gwf_simulation.write(original_modeldir, False, False)

gwf_simulation.run()
hds_original = imod.mf6.open_hds(
    original_modeldir / "GWF_1" / "GWF_1.hds",
    original_modeldir / "GWF_1" / "dis.dis.grb",
)

fig, ax = plt.subplots()
hds_original.sel(layer=3).isel(time=6).plot(ax=ax)
ax.set_title("hondsrug original ")
# %%
# Create the target grid we will regrid to.
idomain = gwf_simulation["GWF_1"]["dis"]["idomain"]


# create label array
submodel_labels = xr.zeros_like(idomain.isel(layer=0))
submodel_labels[100:, :] = 1
submodel_labels[:, 250:] = 2
submodel_labels[100:, 250:] = 3

split_simulation = gwf_simulation.split(submodel_labels)
split_simulation.write(modeldir, False, False)
split_simulation.run()

hds_split = merge_heads(modeldir, split_simulation)

fig, ax = plt.subplots()
hds_split.sel(layer=3).isel(time=6).plot(ax=ax)
ax.set_title("hondsrug partitioned ")

diff = hds_split - hds_original
fig, ax = plt.subplots()
diff.sel(layer=3).isel(time=6).plot(ax=ax)
ax.set_title("hondsrug diff ")

diff_for_plot = diff.max(dim=["time", "layer"])
fig, ax = plt.subplots()
diff_for_plot.plot(ax=ax)
ax.set_title("hondsrug diff ")

maxdiff = diff.max()
print("_---------------------_")
print(maxdiff)
pass
