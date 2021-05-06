"""
Create an iMODFLOW model with all the bells and whistles.
"""


import imod.flow as flow

import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd

import pathlib
from shapely.geometry import LineString

####################
# Create test data #
####################
# Path management
name = "my_first_imodflow_model"
wdir = pathlib.Path("./tests/")

# Discretization
shape = nlay, nrow, ncol = 3, 9, 9

dx = 100.0
dy = -100.0
dz = np.array([5, 30, 100])
xmin = 0.0
xmax = dx * ncol
ymin = 0.0
ymax = abs(dy) * nrow
dims = ("layer", "y", "x")

layer = np.arange(1, nlay + 1)
y = np.arange(ymax, ymin, dy) + 0.5 * dy
x = np.arange(xmin, xmax, dx) + 0.5 * dx
coords = {"layer": layer, "y": y, "x": x}

ibound = xr.DataArray(np.ones(shape), coords=coords, dims=dims)

surface = 0.0
interfaces = np.insert((surface - np.cumsum(dz)), 0, surface)

bottom = xr.DataArray(interfaces[1:], coords={"layer": layer}, dims="layer")
top = xr.DataArray(interfaces[:-1], coords={"layer": layer}, dims="layer")

## Create 1 year of monthly timesteps
times = pd.date_range(start="1/1/2018", end="12/1/2018", freq="MS")

# Layer properties
kh = 10.0
kva = 1.0
sto = 0.001

# Initial conditions
starting_head = ibound.copy()

# Boundary conditions
## Create rising trend.
trend = np.ones(times[:-1].shape)
trend = np.cumsum(trend)
trend_da = xr.DataArray(trend, coords={"time": times[:-1]}, dims=["time"])

## Assign values only to edges of model x domain.
###      x
###   1000001
### y 1000001
###   1000001
head_edge = starting_head.where(starting_head.x.isin([x[0], x[-1]]))
## Calculate outer product with rising trend.
head = trend_da * head_edge

## Create a DataArray for the periodic boundary condition
## Assign values only to the center along x dimension
###      x
###   0001000
### y 0001000
###   0001000
head_central = starting_head.where(starting_head.x == x[4]).sel(layer=1)
## Create period times, we let these times start before the model starts
## This is because iMOD only forward fills periods
## Otherwise, in this case there wouldn't be a periodic boundary condition until april
period_times = times[[3, 9]] - np.timedelta64(365, "D")
## We are creating a summer and winter level.
periods_da = xr.DataArray([4, 10], coords={"time": period_times}, dims=["time"])
head_periodic = periods_da * head_central

## Create dictionary to tell iMOD which period name corresponds to which date.
timemap = {
    period_times[0]: "summer",
    period_times[1]: "winter",
}

# Wells
wel_df = pd.DataFrame()
wel_df["id_name"] = np.arange(len(x)).astype(str)
wel_df["x"] = x
wel_df["y"] = y
wel_df["rate"] = dx * dy * -1 * 0.5
wel_df["time"] = np.tile(times[:-1], 2)[: len(x)]
wel_df["layer"] = 2

# Horizontal Flow Barrier
## Create barriers between ditches in layer 1 and 2 (but not 3).
###      x
###   0100010
### y 0100010
###   0100010
line1 = LineString([(x[2], ymin), (x[2], ymax])
line2 = LineString([(x[7], ymin), (x[7], ymax])

lines = np.array([line1, line2, line1, line2], dtype="object")
hfb_layers = np.array([1, 1, 2, 2])
id_name = ["left_upper", "right_upper", "left_lower", "right_lower"]

hfb_gdf = gpd.GeoDataFrame(
    geometry=lines, data=dict(id_name=id_name, layer=hfb_layers, resistance=100.0)
)

###############
# Build model #
###############
m = flow.ImodflowModel(name)
m["pcg"] = flow.PreconditionedConjugateGradientSolver()

m["bnd"] = flow.Boundary(ibound)
m["top"] = flow.Top(top)
m["bottom"] = flow.Bottom(bottom)

m["khv"] = flow.HorizontalHydraulicConductivity(kh)
m["kva"] = flow.VerticalAnisotropy(kva)
m["sto"] = flow.StorageCoefficient(sto)

m["shd"] = flow.StartingHead(starting_head)

## Create multiple systems, times will be made congruent.
m["chd"] = flow.ConstantHead(head=10.0)
m["chd2"] = flow.ConstantHead(head=head)

## Create periodic boundary condition
m["ghb"] = flow.GeneralHeadBoundary(head=head_periodic, conductance=10.0)
m["ghb"].periodic_stress(timemap)

## Create second periodic boundary condition
m["ghb2"] = flow.GeneralHeadBoundary(head=head_periodic + 10.0, conductance=1.0)
### You also need to specify periodic stresses for second system.
m["ghb2"].periodic_stress(timemap)

m["wel"] = flow.Well(**wel_df)

m["hfb"] = flow.HorizontalFlowBarrier(**hfb_gdf)

## imod-python wants you to provide the model endtime to your time_discretization!
m.time_discretization(times[-1])

## Writes both .IDFs as well as projectfile, an inifile,
## and a .tim file that contains the time discretization.
m.write(directory=wdir)
