"""
Regridding a regional model
==============

This example shows how a model can be regridded using default regridding
methods. The example used in the Hondsrug model. It is regridded to a coarser
grid. The simulated heads before and after the regridding are shown for
comparison, and some statistics are plotted for the head distributions at the
end of the simulation. Histograms of the input arrays before and after
regridding are shown next. 
"""

# sphinx_gallery_thumbnail_number = 3

# %% Import packages
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from example_models import create_hondsrug_simulation

import imod
from imod.typing.grid import GridDataArray

# %% 
# We define some helper functions here. These functions help plotting and printing summary statistics.
def convert_to_filtered_1d(grid: GridDataArray) -> np.ndarray:
    """This function receives an xarray DataArray and converts it to an 1d numpy
    array. All NaN's are filtered out."""
    grid_as_1d = grid.values.ravel()
    filter = ~np.isnan(grid_as_1d)
    grid_as_1d = grid_as_1d[filter]
    return grid_as_1d


def plot_histograms_side_by_side(
    array_original: GridDataArray, array_regridded: GridDataArray, title: str
):
    """This function creates a plot of normalized histograms of the 2 input
    DataArray. It plots a title above each histogram."""
    array_original_as_1d = convert_to_filtered_1d(array_original)
    array_regridded_as_1d = convert_to_filtered_1d(array_regridded)
    _, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
    axs[0].hist(array_original_as_1d, bins=25, density=True)
    axs[1].hist(array_regridded_as_1d, bins=25, density=True)
    axs[0].title.set_text(title + " (original)")
    axs[1].title.set_text(title + " (regridded)")


def write_summary_statistics(
    array_original: GridDataArray, array_regridded: GridDataArray, title: str
):
    """This function writes some statistics of the original and regridded arrays
    to screen"""

    original = convert_to_filtered_1d(array_original)
    regridded = convert_to_filtered_1d(array_regridded)
    print(f"\nsummary statistics {title}")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(
        f"mean (original) {original.mean()}                    mean (regridded) {regridded.mean()}"
    )
    print(
        f"min (original) {original.min()}                      min (regridded) {regridded.min()}"
    )
    print(
        f"max (original) {original.max()}                      max (regridded) {regridded.max()}"
    )
    print(
        f"variance (original) {original.var()}                 variance (regridded) {regridded.var()}"
    )


# %% 
# Obtain the simulation, write it, run it, and plot some heads.
# There is a separate example contained in 
# `hondsrug <https://deltares.gitlab.io/imod/imod-python/examples/mf6/hondsrug.html#sphx-glr-examples-mf6-hondsrug-py>`_ 
# that you should look at if you are interested in the model building
gwf_simulation = create_hondsrug_simulation()

original_modeldir = imod.util.temporary_directory() / "original"
gwf_simulation.write(original_modeldir, False, False)
gwf_simulation.run()
hds_original = imod.mf6.open_hds(
    original_modeldir / "GWF_1" / "GWF_1.hds",
    original_modeldir / "GWF_1" / "dis.dis.grb",
)

fig, ax = plt.subplots()
hds_original.sel(layer=3).isel(time=6).plot(ax=ax)

# %%
# Create the target grid we will regrid to. 
idomain = gwf_simulation["GWF_1"]["dis"]["idomain"]

nlay = len(idomain.coords["layer"].values)
nrow = 100
ncol = 250
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


# %% 
Regridding
==========

Regrid the simulation with the ``regrid_like`` method. Write, run, and plot the results. 
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
hds_regridded.sel(layer=3).isel(time=6).plot(ax=ax)

write_summary_statistics(hds_original.isel(time=6), hds_regridded.isel(time=6), "head")

# %%
# Comparison with histograms
# ==========================
#
# In the next segment we will compare the input and output of the models on different grids. 
# We advice to always check how your input is regridded. In this example we upscaled grid, 
# many input parameters are regridded with a ``mean`` method. This means that their input 
# range is reduced, which can be seen in tailings in the histograms becoming shorter.

plot_histograms_side_by_side(
    hds_original.isel(time=6), hds_regridded.isel(time=6), "head"
)

# %% 
# Compare constant head arrays.
plot_histograms_side_by_side(
    gwf_simulation["GWF_1"]["chd"].dataset["head"],
    regridded_simulation["GWF_1"]["chd"].dataset["head"],
    "chd head",
)

# %%
# Compare horizontal hydraulic conductivities.
plot_histograms_side_by_side(
    gwf_simulation["GWF_1"]["npf"].dataset["k"],
    regridded_simulation["GWF_1"]["npf"].dataset["k"],
    "npf k",
)
# %%
# Compare vertical hydraulic conductivities.
plot_histograms_side_by_side(
    gwf_simulation["GWF_1"]["npf"].dataset["k33"],
    regridded_simulation["GWF_1"]["npf"].dataset["k33"],
    "npf k33",
)
# compare initial conditions
plot_histograms_side_by_side(
    gwf_simulation["GWF_1"]["ic"].dataset["start"],
    regridded_simulation["GWF_1"]["ic"].dataset["start"],
    "ic start",
)

# compare river stages
plot_histograms_side_by_side(
    gwf_simulation["GWF_1"]["riv"].dataset["stage"],
    regridded_simulation["GWF_1"]["riv"].dataset["stage"],
    "riv stage",
)

# compare bottom elevations
plot_histograms_side_by_side(
    gwf_simulation["GWF_1"]["riv"].dataset["bottom_elevation"],
    regridded_simulation["GWF_1"]["riv"].dataset["bottom_elevation"],
    "riv bottom elevation",
)

# compare riverbed conductance
plot_histograms_side_by_side(
    gwf_simulation["GWF_1"]["riv"].dataset["conductance"],
    regridded_simulation["GWF_1"]["riv"].dataset["conductance"],
    "riv conductance",
)

# compare recharge
plot_histograms_side_by_side(
    gwf_simulation["GWF_1"]["rch"].dataset["rate"],
    regridded_simulation["GWF_1"]["rch"].dataset["rate"],
    "rch rate",
)

# compare drainage
plot_histograms_side_by_side(
    gwf_simulation["GWF_1"]["drn-pipe"].dataset["elevation"],
    regridded_simulation["GWF_1"]["drn-pipe"].dataset["elevation"],
    "drn-pipe elevation",
)

# compute conductance
plot_histograms_side_by_side(
    gwf_simulation["GWF_1"]["drn-pipe"].dataset["conductance"],
    regridded_simulation["GWF_1"]["drn-pipe"].dataset["conductance"],
    "drn-pipe conductance",
)


pass
# %%
