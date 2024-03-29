"""
Regridding a regional model
===========================

This example shows how a model can be regridded using default regridding
methods. The example used is the Hondsrug model. It is regridded to a coarser
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

import imod
from imod.typing import GridDataArray

# %%
# Obtain the simulation, write it, run it, and plot some heads.
# There is a separate example contained in :doc:`/examples/mf6/hondsrug`
# that you should look at if you are interested in the model building
tmpdir = imod.util.temporary_directory()

gwf_simulation = imod.data.hondsrug_simulation(tmpdir / "hondsrug_saved")

# %%
# The model was written before the xt3d_option and rhs_option arguments were
# added to iMOD Python. Set missing options to False.
gwf_simulation["GWF"]["npf"].set_xt3d_option(is_xt3d_used=False, is_rhs=False)

# %%
# Write simulation
original_modeldir = tmpdir / "original"
gwf_simulation.write(original_modeldir)

# %%
# Run and open heads
gwf_simulation.run()
hds_original = gwf_simulation.open_head()

fig, ax = plt.subplots()
hds_original.sel(layer=3).isel(time=6).plot(ax=ax)

# %%
# Create the target grid we will regrid to.
idomain = gwf_simulation["GWF"]["dis"]["idomain"]

nlay = len(idomain.coords["layer"].values)
nrow = 50
ncol = 125
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
# Regridding
# ==========

# Regrid the simulation with the ``regrid_like`` method. Write, run, and plot the results.
regridded_simulation = gwf_simulation.regrid_like(
    "hondsrug-regridded", target_grid, validate=False
)

regridded_modeldir = tmpdir / "regridded"
regridded_simulation.write(regridded_modeldir, validate=False)

regridded_simulation.run()

hds_regridded = regridded_simulation.open_head()

# %%
# Results visualization
# =====================
fig, ax = plt.subplots()
hds_regridded.sel(layer=3).isel(time=6).plot(ax=ax)


def plot_histograms_side_by_side(
    array_original: GridDataArray, array_regridded: GridDataArray, title: str
):
    """This function creates a plot of normalized histograms of the 2 input
    DataArray. It plots a title above each histogram."""
    _, (ax0, ax1) = plt.subplots(1, 2, sharey=True, tight_layout=True)
    array_original.plot.hist(ax=ax0, bins=25, density=True)
    array_regridded.plot.hist(ax=ax1, bins=25, density=True)
    ax0.title.set_text(f"{title} (original)")
    ax1.title.set_text(f"{title} (regridded)")


def write_summary_statistics(
    array_original: GridDataArray, array_regridded: GridDataArray, title: str
):
    def convert_to_filtered_1d(grid: GridDataArray) -> np.ndarray:
        """This function receives an xarray DataArray and converts it to an 1d numpy
        array. All NaN's are filtered out."""
        grid_as_1d = grid.values.ravel()
        filter = ~np.isnan(grid_as_1d)
        grid_as_1d = grid_as_1d[filter]
        return grid_as_1d

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
    gwf_simulation["GWF"]["chd"].dataset["head"],
    regridded_simulation["GWF"]["chd"].dataset["head"],
    "chd head",
)

# %%
# Compare horizontal hydraulic conductivities.
plot_histograms_side_by_side(
    gwf_simulation["GWF"]["npf"].dataset["k"],
    regridded_simulation["GWF"]["npf"].dataset["k"],
    "npf k",
)
# %%
# Compare vertical hydraulic conductivities.
plot_histograms_side_by_side(
    gwf_simulation["GWF"]["npf"].dataset["k33"],
    regridded_simulation["GWF"]["npf"].dataset["k33"],
    "npf k33",
)
# %%
# Compare starting heads.
plot_histograms_side_by_side(
    gwf_simulation["GWF"]["ic"].dataset["start"],
    regridded_simulation["GWF"]["ic"].dataset["start"],
    "ic start",
)

# %%
# Compare river stages.
plot_histograms_side_by_side(
    gwf_simulation["GWF"]["riv"].dataset["stage"],
    regridded_simulation["GWF"]["riv"].dataset["stage"],
    "riv stage",
)

# %%
# Compare river bottom elevations.
plot_histograms_side_by_side(
    gwf_simulation["GWF"]["riv"].dataset["bottom_elevation"],
    regridded_simulation["GWF"]["riv"].dataset["bottom_elevation"],
    "riv bottom elevation",
)

# %%
# Compare riverbed conductance.
plot_histograms_side_by_side(
    gwf_simulation["GWF"]["riv"].dataset["conductance"],
    regridded_simulation["GWF"]["riv"].dataset["conductance"],
    "riv conductance",
)

# %%
# Compare recharge rates.
plot_histograms_side_by_side(
    gwf_simulation["GWF"]["rch"].dataset["rate"],
    regridded_simulation["GWF"]["rch"].dataset["rate"],
    "rch rate",
)

# %%
# Compare drainage elevations.
plot_histograms_side_by_side(
    gwf_simulation["GWF"]["drn-pipe"].dataset["elevation"],
    regridded_simulation["GWF"]["drn-pipe"].dataset["elevation"],
    "drn-pipe elevation",
)

# %%
# Compare drain conductances.
plot_histograms_side_by_side(
    gwf_simulation["GWF"]["drn-pipe"].dataset["conductance"],
    regridded_simulation["GWF"]["drn-pipe"].dataset["conductance"],
    "drn-pipe conductance",
)
# %%
