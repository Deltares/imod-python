"""
Model Creation
==============
In this example, we'll create an iMODFLOW model
from scratch with complex boundary conditions
and horizontal barriers.

There are two surface water systems:
the outer two edges of the grid feature ditches
with a rising stage, whereas the central ditch
has a periodic boundary conditions with a
summer and winter stage.

The model will be written as a projectfile
with a set of IDFs containing all the grid information
and a .tim file containing the time discretization.
"""

# %%
# We'll start with the usual imports, supplied
# with geopandas and shapely to specify vector data
# for the `hfb` package.

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import LineString

import imod
import imod.flow as flow

# %%
# Discretization
# --------------
#
# We'll start off by creating a model discretization.
# The model consists of a 3 by 9 by 9 three-dimensional grid.
#
# We'll specify the grid first.
shape = nlay, nrow, ncol = 3, 9, 9

dx = 100.0
dy = -100.0
dz = np.array([5, 30, 100])
xmin = 0.0
xmax = dx * ncol
ymin = 0.0
ymax = abs(dy) * nrow
dims = ("layer", "y", "x")

# %%
# Next, we'll create the coordinates
# which set the grid dimensions.

layer = np.arange(1, nlay + 1)
y = np.arange(ymax, ymin, dy) + 0.5 * dy
x = np.arange(xmin, xmax, dx) + 0.5 * dx
coords = {"layer": layer, "y": y, "x": x}

# %%
# The vertical grid discretization (tops and bottoms) are set with a 1D DataArray.

surface = 0.0
interfaces = np.insert((surface - np.cumsum(dz)), 0, surface)

bottom = xr.DataArray(interfaces[1:], coords={"layer": layer}, dims="layer")
top = xr.DataArray(interfaces[:-1], coords={"layer": layer}, dims="layer")

# %%
# We'll have to create a time discretization as well.
# Create 1 year of monthly timesteps
times = pd.date_range(start="1/1/2018", end="12/1/2018", freq="MS")

# %%
# We'll create our first 3 dimensional grid here,
# the `ibound` grid, which sets where active cells are `(ibound = 1.0)`
ibound = xr.DataArray(np.ones(shape), coords=coords, dims=dims)
ibound

# %%
# Hydrogeology
# ------------
#
# We'll create a very simple hydrogeology,
# by specifying kh, kva, and sto as constants

kh = 10.0
kva = 1.0
sto = 0.001

# %%
# Initial conditions
# ------------------
#
# We do not put much effort in the creation of the initial conditions
# in this example, instead we copy the ibounds.
# This is a 3D grid filled with the value 1, and we can use it as a
# inital condition as well.
starting_head = ibound.copy()

# %%
# Boundary conditions
# -------------------
#
# We will put some more effort in creating some complex
# boundary conditions. We'll create both two outer ditches
# with a rising stage, as well as a central ditch with periodic
# (summer-winter) stage.
#
# Rising outer ditches
# ~~~~~~~~~~~~~~~~~~~~
#
# First, we'll create rising trend,
# by creating a 1D array ones with the same size as the time dimension,
# and computing the cumulative sum over it.
trend = np.ones(times[:-1].shape)
trend = np.cumsum(trend)
trend_da = xr.DataArray(trend, coords={"time": times[:-1]}, dims=["time"])

trend_da

# %%
# Next, we assign values only to edges of model x domain.

is_x_edge = starting_head.x.isin([x[0], x[-1]])
head_edge = starting_head.where(is_x_edge)
head_edge

# %%
# Now let's multiply our
# 1D DataArray with dimension ``time``, with the static 3D grid
# with dimension ``layer, y, x``,
# which xarray automatically broadcasts to a 4D array,
# with dimensions ``time, layer, y, x``
# This finishes our

head_edge_rising = trend_da * head_edge
head_edge_rising

# %%
# Periodic central ditch
# ~~~~~~~~~~~~~~~~~~~~~~
#
# We'll take only the central column of the grid with
# (``where``), the rest will be set to ``np.nan``,
# and from this we'll select only the upper layer,
# as the ditch will be located only in the upper layer.

is_x_central = starting_head.x == x[4]
head_central = starting_head.where(is_x_central).sel(layer=1)

# %%
# Create period times, we let these times start before
# the model starts.
# This is necessary because `iMODFLOW` only forward fills periods.
# Otherwise, in this case there wouldn't be a
# periodic boundary condition until april.
#
# We will do this by selecting the months april and october,
# and then subtracting a year
period_times = times[[3, 9]] - np.timedelta64(365, "D")

# %%
# We are creating a summer and winter level.
periods_da = xr.DataArray([4, 10], coords={"time": period_times}, dims=["time"])
head_periodic = periods_da * head_central

head_periodic

# %%
# Create dictionary to tell `iMOD`
# which period name corresponds to which date.
timemap = {
    period_times[0]: "summer",
    period_times[1]: "winter",
}

# %%
# Wells
# ~~~~~
#
# Wells are specified as a pandas dataframe.
# We create a diagonal line of wells through the domain.
#
# Because we can.

wel_df = pd.DataFrame()
wel_df["id_name"] = np.arange(len(x)).astype(str)
wel_df["x"] = x
wel_df["y"] = y
wel_df["rate"] = dx * dy * -1 * 0.5
wel_df["time"] = np.tile(times[:-1], 2)[: len(x)]
wel_df["layer"] = 2

wel_df

# %%
# Horizontal Flow Barrier
# -----------------------
#
# Create barriers between ditches in layer 1 and 2 (but not 3).

line1 = LineString([(x[2], ymin), (x[2], ymax)])
line2 = LineString([(x[7], ymin), (x[7], ymax)])

# %%
# We'll have to repeat each line for each layer
lines = np.array([line1, line2, line1, line2], dtype="object")
hfb_layers = np.array([1, 1, 2, 2])

# %%
# We can specify names for our own bookkeeping
id_name = ["left_upper", "right_upper", "left_lower", "right_lower"]

# The hfb has to specified as a geopandas `GeoDataFrame
# <https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoDataFrame.html>`_

hfb_gdf = gpd.GeoDataFrame(
    geometry=lines, data=dict(id_name=id_name, layer=hfb_layers, resistance=100.0)
)

hfb_gdf

# %%
# Build
# -----
#
# Finally, we build the model.

m = flow.ImodflowModel("my_first_imodflow_model")
m["pcg"] = flow.PreconditionedConjugateGradientSolver()

m["bnd"] = flow.Boundary(ibound)
m["top"] = flow.Top(top)
m["bottom"] = flow.Bottom(bottom)

m["khv"] = flow.HorizontalHydraulicConductivity(kh)
m["kva"] = flow.VerticalAnisotropy(kva)
m["sto"] = flow.StorageCoefficient(sto)

m["shd"] = flow.StartingHead(starting_head)

m["chd"] = flow.ConstantHead(head=head_edge_rising)

# %%
# Create periodic boundary condition
# and specify it as a periodic stress package.
m["ghb"] = flow.GeneralHeadBoundary(head=head_periodic, conductance=10.0)
m["ghb"].periodic_stress(timemap)

# %%
# We can specify a second stress package as follows:
m["ghb2"] = flow.GeneralHeadBoundary(head=head_periodic + 10.0, conductance=1.0)
# You also need to specify periodic stresses for second system.
m["ghb2"].periodic_stress(timemap)

m["wel"] = flow.Well(**wel_df)

m["hfb"] = flow.HorizontalFlowBarrier(**hfb_gdf)

# imod-python wants you to provide the model endtime to your time_discretization!
m.time_discretization(times[-1])

# %%
# Now we write the model
# Writes both .IDFs as well as projectfile, an inifile,
# and a .tim file that contains the time discretization.

modeldir = imod.util.temporary_directory()
m.write(directory=modeldir)

# %%
# Run
# ---
#
# You can run the model using the comand prompt and the iMOD executables.
# This is part of the iMOD v5 release, which can be downloaded here:
# https://oss.deltares.nl/web/imod/download-imod5 .
# iMOD only works on Windows.
#
# To run your model, open up a command prompt
# and run the following commands:
#
# .. code-block:: batch
#
#    cd c:\path\to\modeldir
#    c:\path\to\imod\folder\iMOD_v5_3_X64R.EXE my_first_imodflow_model.ini
#    c:\path\to\imod\folder\iMODFLOW_V5_3_METASWAP_SVN1977_X64R.EXE my_first_imodflow_model.nam
#
