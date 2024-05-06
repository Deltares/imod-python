"""
Defining the topsystem
======================

iMOD Python has multiple features to help you define the topsystem of the
groundwater system. With "topsystem" we mean all forcings which act on the top
of the groundwater system. These are usually either meteorological
(precipitation & evapotranspiration) or hydrological (rivers, ditches, lakes,
sea) in nature. This data is always provided as planar grids (x, y) without any
vertical dimension. This user guide will show you how to allocate these forcings
across model layers to grid cells and how to distribute conductances for
Robin-like boundary conditions over model layers.

"""

# %%
# Example data
# ------------
#
# Let's load the data first. We have a layer model containing a basic
# hydrogeological schemitization of our model, so the tops and bottoms of model
# layers, the hydraulic conductivity (k), and which cells are active (idomain=1)
# or vertical passthrough (idomain=-1).

import imod

layer_model = imod.data.hondsrug_layermodel()

# %%
# Make layer model more interesting for this example by subdividing layers
# into n_new layers.
import numpy as np
import xarray as xr

n_new = 4
n_max_old = 5

new_ds_ls = []
for i in range(n_max_old):
    sub_iter = np.arange(n_new) + 1
    layer_coord = sub_iter + i * (n_max_old - 1)
    distribution_factors = 1 / n_new * sub_iter
    da_distribution = xr.DataArray(
        distribution_factors, coords={"layer": layer_coord}, dims=("layer",)
    )
    layer_model_sel = layer_model.sel(layer=i + 1, drop=True)
    D = layer_model_sel["top"] - layer_model_sel["bottom"]

    new_ds = xr.Dataset()
    new_ds["k"] = xr.ones_like(da_distribution) * layer_model_sel["k"]
    new_ds["idomain"] = xr.ones_like(da_distribution) * layer_model_sel["idomain"]
    # Put da_distribution in front of equation to enforce dims as (layer, y, x)
    new_ds["top"] = (da_distribution - 1 / n_new) * -D + layer_model_sel["top"]
    new_ds["bottom"] = da_distribution * -D + layer_model_sel["top"]

    new_ds_ls.append(new_ds)


new_layer_model = xr.concat(new_ds_ls, dim="layer")


# %%
# Furthermore we have planar grid of river, containing a river stage, bed
# elevation and conductance.
planar_river = imod.data.hondsrug_river().max(dim="layer")

planar_river


# %% 
# Let's adapt the river stages to make them closer to the top, this makes for
# more interesting data in this example to show.

planar_river["stage"] = (
    new_layer_model["top"].sel(layer=1) - planar_river["stage"]
) / 2 + planar_river["stage"]

# %%
# Let's plot the top elevation of the model on a map. You can see we have a
# ridge roughly the centre of the model, sided by two low-lying areas.
import numpy as np

imod.visualize.plot_map(
    new_layer_model["top"].sel(layer=1), "viridis", np.linspace(1, 20, 11)
)

# %%
# Let's plot the river stages on a map. You can see most rivers are located
# in the low-lying areas.
imod.visualize.plot_map(planar_river["stage"], "viridis", np.linspace(-1, 19, 9))

# %%
# Allocate river cells
# --------------------
#
# Let's allocate river cells across model layers to cells.
from imod.prepare import ALLOCATION_OPTION, allocate_riv_cells

riv_allocated, _ = allocate_riv_cells(
    allocation_option=ALLOCATION_OPTION.at_elevation,
    active=new_layer_model["idomain"] == 1,
    top=new_layer_model["top"],
    bottom=new_layer_model["bottom"],
    stage=planar_river["stage"],
    bottom_elevation=planar_river["bottom"],
)

# %%
#
# Let's take a look at what we just produced. Since we are dealing with
# information with depth, it is simplest to make a crosssection plot. For that,
# we first have to select a crosssection.
import geopandas as gpd
from shapely.geometry import LineString

geometry = LineString([[238000, 562200], [242000, 559800]])

# Define overlay
overlays = [
    {"gdf": gpd.GeoDataFrame(geometry=[geometry]), "edgecolor": "black", "linewidth": 3}
]
# Plot
imod.visualize.plot_map(
    planar_river["stage"], "viridis", np.linspace(-1, 19, 9), overlays
)

# %%
#
# Select a cross section. The plot also requires top and bottom information
# which we will add first as coordinates before selecting.
riv_allocated.coords["top"] = new_layer_model["top"]
riv_allocated.coords["bottom"] = new_layer_model["bottom"]

xsection_allocated = imod.select.cross_section_linestring(riv_allocated, geometry)

xsection_allocated
# %%
#
# Now that we have selected our data we can plot it.
imod.visualize.cross_section(xsection_allocated, "viridis", [0, 1])

# %%
#
# Let's plot the locations of our river stages and bottom elevations.
import matplotlib.pyplot as plt

stage_line = imod.select.cross_section_linestring(planar_river["stage"], geometry)
stage_bottom = imod.select.cross_section_linestring(planar_river["bottom"], geometry)

fig, ax = plt.subplots()

imod.visualize.cross_section(xsection_allocated, "viridis", [0, 1], fig=fig, ax=ax)
x_line = stage_line["s"] + stage_line["ds"] / 2
ax.scatter(x_line, stage_line.values, marker=7, c="k")
ax.scatter(x_line, stage_bottom.values, marker=6, c="k")


# %%
# The above plot might look a bit off. Let's plot the layer numbers, so that we
# can identify where model layers are located.
import xarray as xr

layer_grid = new_layer_model.layer * xr.ones_like(new_layer_model["top"])
layer_grid.coords["top"] = new_layer_model["top"]
layer_grid.coords["bottom"] = new_layer_model["bottom"]
xsection_layer_nr = imod.select.cross_section_linestring(layer_grid, geometry)

imod.visualize.cross_section(xsection_layer_nr, "viridis", np.arange(21))


# %% 
# Overview allocation options
# ---------------------------
#
# There are multiple options available to allocate rivers. For a full
# description of all options, see the documentation of
# :func:`imod.prepare.ALLOCATION_OPTION`. We can print all possible options as
# follows:

for option in ALLOCATION_OPTION:
    print(option.name)

# %% 
# Let's make plots for each option to visualize the effect of each choice.

# Create grid for plots
fig, axes = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(11, 11))
axes = np.ravel(axes)

# The top left plot shows the elevation of the river bottom (upward triangle)
# and river stage (downward triangle).
ax = axes[0]
imod.visualize.cross_section(
    xsection_layer_nr,
    "viridis",
    np.arange(21),
    kwargs_colorbar = dict(plot_colorbar=False),
    fig=fig,
    ax=ax,
)
ax.scatter(x_line, stage_line.values, s=32.0, marker=7, c="k", linewidths=0)
ax.scatter(x_line, stage_bottom.values, s=32.0, marker=6, c="k", linewidths=0)
ax.set_title("stage and bottom elevation")

# Loop over allocation options, and plot the allocated cells as a polygon,
# using the "aquitard" feature of the cross_section plot.
for i, option in enumerate(ALLOCATION_OPTION, start=1):
    riv_allocated, _ = allocate_riv_cells(
        allocation_option=option,
        active=new_layer_model["idomain"] == 1,
        top=new_layer_model["top"],
        bottom=new_layer_model["bottom"],
        stage=planar_river["stage"],
        bottom_elevation=planar_river["bottom"],
    )
    riv_allocated.coords["top"] = new_layer_model["top"]
    riv_allocated.coords["bottom"] = new_layer_model["bottom"]

    xsection_allocated = imod.select.cross_section_linestring(riv_allocated, geometry)
    ax = axes[i]
    if (i % 2) == 0:
        kwargs_colorbar = dict(plot_colorbar=False)
    else:
        kwargs_colorbar = dict(plot_colorbar=True)
    kwargs_aquitards = {"hatch": "/", "edgecolor": "k"}
    imod.visualize.cross_section(
        xsection_layer_nr,
        "viridis",
        np.arange(21),
        aquitards=xsection_allocated,
        kwargs_aquitards=kwargs_aquitards,
        kwargs_colorbar=kwargs_colorbar,
        fig=fig,
        ax=ax,
    )
    ax.set_title(f"option: {option.name}")

# Enforce tight layout to remove whitespace inbetween plots.
plt.tight_layout()


# %%
# Distribute conductance
# ----------------------
#
# Next, we'll take a look at distributing conductances, as there are multiple
# ways to distribute conductances over layers. For example, it is possible to
# distribute conductances equally across layers, weighted by layer thickness, or
# by transmissivity.

from imod.prepare import DISTRIBUTING_OPTION, distribute_riv_conductance

# %%
# Here's a map of how the conductances are distributed in our dataset.

imod.visualize.plot_map(planar_river["conductance"], "viridis", np.logspace(-2, 3, 6), overlays)


# %%
# First compute the allocated river cells for stage to river bottom elevation
# again. This time we'll use the ``stage_to_riv_bot`` option.

riv_allocated, _ = allocate_riv_cells(
    allocation_option=ALLOCATION_OPTION.stage_to_riv_bot,
    active=new_layer_model["idomain"] == 1,
    top=new_layer_model["top"],
    bottom=new_layer_model["bottom"],
    stage=planar_river["stage"],
    bottom_elevation=planar_river["bottom"],
)

riv_allocated

# %%
# Distribute river conductance over model layers. There are multiple options
# available, which are fully described in
# :func:`imod.prepare.DISTRIBUTING_OPTION`. We can print all possible options as
# follows:

for option in DISTRIBUTING_OPTION:
    print(option.name)

# %% 
# To reduce duplicate code, we are going to store all input data in this
# dictionary which we can provide further as keyword arguments.

distributing_data = dict(
    allocated=riv_allocated,
    conductance=planar_river["conductance"],
    top=new_layer_model["top"],
    bottom=new_layer_model["bottom"],
    k=new_layer_model["k"],
    stage=planar_river["stage"],
    bottom_elevation=planar_river["bottom"]
    )

# %%
# Let's keep things simple first and distribute conductances across layers
# equally.

riv_conductance = distribute_riv_conductance(
    distributing_option=DISTRIBUTING_OPTION.by_layer_thickness, **distributing_data

)
riv_conductance.coords["top"] = new_layer_model["top"]
riv_conductance.coords["bottom"] = new_layer_model["bottom"]

# %%
# Lets repeat the earlier process to produce a nice cross-section plot.

# Select the conductance over the cross section again.
xsection_distributed = imod.select.cross_section_linestring(riv_conductance, geometry)

fig, ax = plt.subplots()

# Plot grey background of active cells
is_active = ~np.isnan(xsection_distributed.coords["top"])
imod.visualize.cross_section(is_active, "Greys", [0,1,2], kwargs_colorbar = dict(plot_colorbar=False), fig=fig, ax=ax)
# Plot conductances
imod.visualize.cross_section(xsection_distributed, "viridis", np.logspace(-2, 3, 6), fig=fig, ax=ax)

# %%
# Let's compare the results of all possible options visually. On the top left
# we'll plot the hydraulic conductivity, as we haven't looked at that yet. The
# other plots show the effects of different settings. Again, distributing
# options are described in more detail in :func:`imod.prepare.DISTRIBUTING_OPTION`

fig, axes = plt.subplots(4, 2, figsize=[11, 15], sharex=True, sharey=True)
axes = np.ravel(axes)

k = distributing_data["k"].copy()
k.coords["top"] = new_layer_model["top"]
k.coords["bottom"] = new_layer_model["bottom"]

xsection_k = imod.select.cross_section_linestring(k, geometry)

ax = axes[0]
imod.visualize.cross_section(xsection_k, "viridis", np.logspace(-2, 3, 11), kwargs_colorbar = dict(plot_colorbar=False), fig=fig, ax=ax)
ax.set_title("hydraulic conductivity")

for i, option in enumerate(DISTRIBUTING_OPTION, start=1):
    ax = axes[i]
    riv_conductance = distribute_riv_conductance(
        distributing_option=option, **distributing_data

    )
    riv_conductance.coords["top"] = new_layer_model["top"]
    riv_conductance.coords["bottom"] = new_layer_model["bottom"]

    xsection_distributed = imod.select.cross_section_linestring(riv_conductance, geometry)

    if (i % 2) == 0:
        kwargs_colorbar = dict(plot_colorbar=False)
    else:
        kwargs_colorbar = dict(plot_colorbar=True)

    # Plot grey background of active cells
    is_active = ~np.isnan(xsection_distributed.coords["top"])
    imod.visualize.cross_section(is_active, "Greys", [0,1,2], kwargs_colorbar = dict(plot_colorbar=False), fig=fig, ax=ax)
    # Plot conductances
    imod.visualize.cross_section(xsection_distributed, "viridis", np.logspace(-2, 3, 11), kwargs_colorbar = kwargs_colorbar, fig=fig, ax=ax)

    ax.set_title(f"option: {option.name}")

plt.tight_layout()

# %%
