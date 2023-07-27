"""
Regridding a regional model
==============

This example shows a simplified script for building a groundwater model in the
northeast of the Netherlands. A primary feature of this area is an ice-pushed
ridge called the Hondsrug. This examples demonstrates modifying external data
for use in a MODFLOW6 model.

In overview, the model features:

    * Thirteen layers: seven aquifers and six aquitards;
    * A dense ditch network in the east;
    * Pipe drainage for agriculture;
    * Precipitation and evapotranspiration.

"""

# sphinx_gallery_thumbnail_number = -1

# %% Import packages
# We'll start with the usual imports, and an import from scipy.

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from example_models import create_hondsrug_simulation

import imod

# obtain the simulation, write it, run it, and plot some heads
gwf_simulation = create_hondsrug_simulation()

original_modeldir = imod.util.temporary_directory() / "original"
gwf_simulation.write(original_modeldir, False, False)
gwf_simulation.run()
hds_original = imod.mf6.open_hds(
    original_modeldir / "GWF_1" / "GWF_1.hds",
    original_modeldir / "GWF_1" / "dis.dis.grb",
)

fig, ax = plt.subplots()
hds_original.sel(layer=3).isel(time=3).plot(ax=ax)


# create the grid we will regrid to
idomain = gwf_simulation["GWF_1"]["dis"]["idomain"]

nlay = len(idomain.coords["layer"].values)
nrow = 72
ncol = 116
shape = (nlay, nrow, ncol)

xmin = idomain.coords["x"].min().values[()]
xmax = idomain.coords["x"].max().values[()]
ymin = idomain.coords["y"].min().values[()]
ymax = idomain.coords["y"].max().values[()]

delta_x = (xmax - xmin) / ncol
delta_y = (ymax - ymin) / nrow
dims = ("layer", "y", "x")
new_x = np.arange(xmin, xmax, delta_x)
new_y = np.arange(ymax, ymin, -delta_y)
new_layer = idomain.coords["layer"].values
coords = {"layer": new_layer, "y": new_y, "x": new_x, "dx": delta_x, "dy": delta_y}
target_grid = xr.DataArray(np.ones(shape, dtype=int), coords=coords, dims=dims)


# regrid the simulation
regridded_simulation = gwf_simulation.regrid_like(
    "hondsrug-regridded", target_grid, validate=False
)

regridded_modeldir = original_modeldir / ".." / "regridded"
regridded_simulation.write(regridded_modeldir, False, False)

regridded_simulation.run()


hds_regridded = imod.mf6.open_hds(
    regridded_modeldir / "GWF_1" / "GWF_1.hds",
    regridded_modeldir / "GWF_1" / "dis.dis.grb",
)
# %%
# Results visualization
# =====================
fig, ax = plt.subplots()
hds_regridded.sel(layer=3).isel(time=3).plot(ax=ax)

# %%
# compare heads 
# =====================
last_head_original = hds_original.isel(time=6)
last_head_reridded = hds_regridded.isel(time=6)

# convert to 1d numoy array
last_head_original_as_1d = last_head_original.values.ravel()
last_head_regridded_as_1d = last_head_reridded.values.ravel()

#get rid of the Nan's 
original_filter =  ~np.isnan(last_head_original_as_1d)
last_head_original_as_1d = last_head_original_as_1d[original_filter]
regridded_filter= ~np.isnan(last_head_regridded_as_1d)
last_head_regridded_as_1d= last_head_regridded_as_1d[regridded_filter]

#plot histograms side by side
fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
axs[0].hist(last_head_original_as_1d, bins=25)
axs[1].hist(last_head_regridded_as_1d, bins=25)

#print some distribution parameters
mean_orig = last_head_original_as_1d.mean()
max_orig = last_head_original_as_1d.max()
min_orig = last_head_original_as_1d.min()
var_orig = last_head_original_as_1d.var()

mean_regridded = last_head_regridded_as_1d.mean()
max_regridded = last_head_regridded_as_1d.max()
min_regridded = last_head_regridded_as_1d.min()
var_regridded = last_head_regridded_as_1d.var()

print("At the last timestep the head distribution has the following summary statistics:")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print(f"mean (original) = {mean_orig}          mean(regridded) = {mean_regridded}")
print(f"max (original) = {max_orig}            max(regridded) = {max_regridded}")
print(f"min (original) = {min_orig}            min(regridded) = {min_regridded}")
print(f"variance (original) = {var_orig}       variance(regridded) = {var_regridded}")

pass
# %%
