"""
Assign Wells to Model Layers
============================

iMOD Python provides two grid-agnostic well classes. Use
:class:`imod.mf6.Well` when the physical top and bottom of a well screen are
known. Use :class:`imod.mf6.LayeredWell` when the model layer for every well
record is already known.

This example converts both representations on the same two-layer grid to show
how their rates are handled.
"""

# %%
# Create a small grid with two layers. Both layers are five metres thick, but
# the second layer has three times the horizontal hydraulic conductivity.

import numpy as np
import xarray as xr

import imod

coords = {"layer": [1, 2], "y": [1.5, 0.5], "x": [0.5, 1.5]}
idomain = xr.DataArray(
    np.ones((2, 2, 2), dtype=int),
    coords=coords,
    dims=("layer", "y", "x"),
)
top = xr.DataArray(
    np.full((2, 2), 10.0),
    coords={"y": coords["y"], "x": coords["x"]},
    dims=("y", "x"),
)
bottom = xr.DataArray(
    np.stack([np.full((2, 2), 5.0), np.full((2, 2), 0.0)]),
    coords=coords,
    dims=("layer", "y", "x"),
)
k = xr.DataArray(
    np.stack([np.full((2, 2), 1.0), np.full((2, 2), 3.0)]),
    coords=coords,
    dims=("layer", "y", "x"),
)

# %%
# ``Well`` receives one total extraction rate for a screen from elevation 10
# to 0. The screen intersects both layers. Their transmissivities are
# ``1 * 5 = 5`` and ``3 * 5 = 15``, so the total rate of -40 is split in a
# 1:3 ratio.

screen_based = imod.mf6.Well(
    x=[0.5],
    y=[0.5],
    screen_top=[10.0],
    screen_bottom=[0.0],
    rate=[-40.0],
)
screen_based_mf6 = screen_based.to_mf6_pkg(idomain, top, bottom, k)

screen_based_mf6["cellid"]

# %%
# The first layer receives -10 and the more transmissive second layer receives
# -30. The sum remains equal to the input rate.

screen_based_mf6["rate"]

np.testing.assert_allclose(screen_based_mf6["rate"].values, [-10.0, -30.0])

# %%
# ``LayeredWell`` receives one record per known model layer. These rates are
# already allocated by the user, so hydraulic conductivity does not change
# their distribution during conversion.

layer_based = imod.mf6.LayeredWell(
    x=[0.5, 0.5],
    y=[0.5, 0.5],
    layer=[1, 2],
    rate=[-20.0, -20.0],
)
layer_based_mf6 = layer_based.to_mf6_pkg(idomain, top, bottom, k)

layer_based_mf6["cellid"]

# %%
# Both supplied layer rates remain -20, even though the layers have different
# hydraulic conductivities.

layer_based_mf6["rate"]

np.testing.assert_allclose(layer_based_mf6["rate"].values, [-20.0, -20.0])

# %%
# In short: choose ``Well`` for screen elevations and automatic allocation;
# choose ``LayeredWell`` for explicit model layers and pre-allocated rates.
