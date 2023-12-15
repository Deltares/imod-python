"""
Flow velocities and streamlines
===============================

In this section we will plot flow velocities and streamlines for some model results.

"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# %%
# We'll start with the usual imports
import imod

# sphinx_gallery_thumbnail_number = -1

# %%
# Load and unpack the data
ds_fluxes = imod.data.fluxes()
ds_fluxes = ds_fluxes.isel(time=-1)

ds_fluxes

lower = ds_fluxes["bdgflf"]
right = ds_fluxes["bdgfrf"]
front = ds_fluxes["bdgfff"]
heads = ds_fluxes["head"]

# %%
# Calculating flow velocity
# -------------------------
#
# The imod-python function imod.evaluate.flow_velocity() computes flow
# velocities in m/d based on the budget results (bdgflf - flow lower face,
# bdgfrf - flow right face and bdgfff - flow front face). To apply this
# function, we first need to define a top_bot array.

top_bot = xr.full_like(lower, 1.0)
top_bot["top"] = top_bot["z"] - 0.5 * top_bot["dz"]
top_bot["bot"] = top_bot["z"] + 0.5 * top_bot["dz"]
top_bot

# %%
# Next we'll calculate the velocities and plot a cross-section of the vertical
# velocity at for y = 450050

fig, ax = plt.subplots()
vx, vy, vz = imod.evaluate.flow_velocity(
    front=front, lower=lower, right=right, top_bot=top_bot, porosity=0.3
)
vz.sel(y=450050.0, method="nearest").plot(cmap="RdYlBu", yincrease=False)

plt.title("Vz")

# %%
# Quiver plot
# -----------
#
# It is also possible to make quiver plots for a cross section defined by two
# pairs of coordinates. We will first define arrays indicating the start and end
# location of the cross section to be evaluated.

start = np.array([97132.710, 457177.928])
end = np.array([103736.517, 457215.557])

# %%
# Using the function ``imod.evaluate.quiver_line()`` considering the starting and ending points

u, v = imod.evaluate.quiver_line(right, front, lower, start, end)

# %%
# Adding top and bottom information to the previously obtained arrays
u["top"] = u["z"] + 0.5 * u["dz"]
u["bottom"] = u["z"] - 0.5 * u["dz"]

v["top"] = v["z"] + 0.5 * v["dz"]
v["bottom"] = v["z"] - 0.5 * v["dz"]

# %%
# Defining a cross section that shows the heads for the same location where the start and end points
# were defined, to use as a background image for the plot

cross_section = imod.select.cross_section_line(heads, start, end)
cross_section["top"] = cross_section["z"] - 0.5 * cross_section["dz"]
cross_section["bottom"] = cross_section["z"] + 0.5 * cross_section["dz"]

# %%
# Ploting the cross section
colors = "magma"
levels = np.arange(-8, 0.0, 0.5)
skip = (slice(None, None, 2), slice(None, None, 2))
fig, ax = plt.subplots()
fig, ax = imod.visualize.cross_section(
    cross_section, colors=colors, levels=levels, ax=ax, fig=fig
)
imod.visualize.quiver(u[skip], v[skip], ax, kwargs_quiver={"color": "k"})

# %%
# Streamline function
# -------------------
#
# This function shows the streamlines for a line cross section through a 3D flow field.
# We will use the previously created arrays that indicate the start and end location of the cross section and apply them in the function imod.evaluate.streamfunction_line()

streamfunction = imod.evaluate.streamfunction_line(right, front, start, end)

# %%
# The previous array contains the streamfunction projected on the cross-section defined
# by provided linestring, with new dimension “s” along the cross-section.
# The cellsizes along “s” are given in the “ds” coordinate.
# The streamline function can be plotted using imod.visualize.streamfunction(),
# but first we need to define the 'top' and 'bottom' of the layers in the array
# Note: By default the steamlines are plotted in white.

streamfunction["bottom"] = streamfunction["z"] + streamfunction["dz"]
streamfunction["top"] = streamfunction["z"]

fig, ax = plt.subplots()
fig, ax = imod.visualize.cross_section(
    cross_section, colors=colors, levels=levels, ax=ax, fig=fig
)
imod.visualize.streamfunction(streamfunction, ax=ax, n_streamlines=10)

# %%
