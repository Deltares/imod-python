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
# Make layer model more interesting for this example by displacing a few 
import numpy as np
import xarray as xr

n_new = 4
n_max_old = 5

new_ds_ls = []
for i in range(1, n_max_old+1):
    new_layer = (np.arange(n_new)+1)
    da_new_layer = xr.DataArray(new_layer, coords={"layer": new_layer}, dims=("layer",))
    layer_model_sel = layer_model.sel(layer=i, drop=True)
    D = layer_model_sel["top"] - layer_model_sel["bottom"]
    
    new_ds = xr.Dataset()
    new_ds["k"] = xr.ones_like(da_new_layer) * layer_model_sel["k"]
    new_ds["idomain"] = xr.ones_like(da_new_layer) * layer_model_sel["k"]
    new_ds["top"] = layer_model_sel["top"] - D/(da_new_layer-1)
    new_ds["bottom"] = layer_model_sel["top"] - D/(da_new_layer)

    new_ds_ls.append(new_ds)




# %% 
# Let's plot the top elevation of the model on a map. You can see we have a
# ridge roughly the centre of the model, sided by two low-lying areas.
import numpy as np

imod.visualize.plot_map(layer_model["top"].sel(layer=1), "viridis", np.linspace(1, 20, 11))

# %% 
# Furthermore we have planar grid of river, containing a river stage, bed
# elevation and conductance.
planar_river = imod.data.hondsrug_river().mean(dim="layer")

planar_river

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
    allocation_option = ALLOCATION_OPTION.at_elevation, 
    active = layer_model["idomain"] == 1,
    top = layer_model["top"],
    bottom = layer_model["bottom"],
    stage = planar_river["stage"],
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
geometry = LineString([[239300, 563500], [242000,563000]])

# Define overlay
overlays = [{"gdf": gpd.GeoDataFrame(geometry=[geometry]), "edgecolor": "black",  "linewidth":3}]
# Plot
imod.visualize.plot_map(planar_river["stage"], "viridis", np.linspace(-1, 19, 9), overlays)

# %%
#
# Select a cross section. The plot also requires top and bottom information
# which we will add first as coordinates before selecting.
riv_allocated.coords["top"] = layer_model["top"]
riv_allocated.coords["bottom"] = layer_model["bottom"]

xsection = imod.select.cross_section_linestring(riv_allocated, geometry)

xsection
# %%
#
# Now that we have selected our data we can plot it.
imod.visualize.cross_section(xsection, "viridis", [0, 1])

# %%
# 
# Let's zoom in to the relevant parts
fig, ax = imod.visualize.cross_section(xsection, "viridis", [0, 1])

ax.set_ylim(-20.0, 5)

# %%
#
# Let's plot the locations of our river stages and bottom elevations.
import matplotlib.pyplot as plt

stage_line = imod.select.cross_section_linestring(planar_river["stage"], geometry)
stage_bottom = imod.select.cross_section_linestring(planar_river["bottom"], geometry)

fig, ax = plt.subplots()

imod.visualize.cross_section(xsection, "viridis", [0, 1], fig=fig, ax=ax)
x_line = stage_line["s"] + stage_line["ds"]/2
ax.scatter(x_line, stage_line.values, marker=7, c="k")
ax.scatter(x_line, stage_bottom.values, marker=6, c="k")

ax.set_xlim(1000, 2000)
ax.set_ylim(-15.0, 5)



# %%
import xarray as xr

geometry = LineString([[246500,560500],[249900, 563900]])

layer_grid = layer_model.layer * xr.ones_like(layer_model["top"])
layer_grid.coords["top"] = layer_model["top"]
layer_grid.coords["bottom"] = layer_model["bottom"]
xsection = imod.select.cross_section_linestring(layer_grid, geometry)

imod.visualize.cross_section(xsection, "viridis", np.arange(14))

plt.show()

# %%
