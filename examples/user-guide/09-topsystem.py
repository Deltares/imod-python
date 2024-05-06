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
# Let's adapt the river stages to make them closer to the top.

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

# geometry = LineString([[246500,560500],[249900, 563900]])
# geometry = LineString([[238000,559100],[239300, 563900], [242000,563000]])
# geometry = LineString([[239300, 563500], [242000,563000]])
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
# There are multiple options available to allocate rivers. Let's make plots
# for each option to visualize the effect of each choice.

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
    ax.set_title(option.name)

# Enforce tight layout to remove whitespace inbetween plots.
plt.tight_layout()

# %%
