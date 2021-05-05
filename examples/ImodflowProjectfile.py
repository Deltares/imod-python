"""
Create a iMODFLOW model with all the bells and whistles.

The boundary condition data itself does not make too much sense
(CHD everywhere? Overlapping GHB and CHD?)
But it should be enough to give you a quickstart of 
how to create an imododflow model with projectfile in imod-python.
"""


import imod.flow as flow

import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd

import pathlib
from shapely.geometry import LineString

# Path management
name = "my_first_imodflow_model"
wdir = pathlib.Path("./tests/")

# Discretization
shape = nlay, nrow, ncol = 3, 9, 9

dx = 10.0
dy = -10.0
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

## Create monthly timesteps
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

## Assign values only to edges of model domain.
head = starting_head.where(starting_head.x.isin([x[0], x[-1]]))
## Calculate outer product with rising trend.
head = xr.DataArray(trend, coords={"time": times[:-1]}, dims=["time"]) * head

## Create a DataArray for the periodic boundary condition
head_periodic = head.isel(time=[3, 9])
head_periodic = head_periodic.assign_coords(
    time=head_periodic.time - np.timedelta64(365, "D")
)

# Wells
wel_df = pd.DataFrame()
wel_df["id_name"] = np.arange(len(x)).astype(str)
wel_df["x"] = x
wel_df["y"] = y
wel_df["rate"] = dx * dy * -1 * 0.5
wel_df["time"] = np.tile(times[:-1], 2)[: len(x)]
wel_df["layer"] = 2

# Horizontal Flow Barrier
line1 = LineString([(x[1], y[1]), (x[1], y[-2])])
line2 = LineString([(x[4], y[1]), (x[4], y[-2])])

lines = np.array([line1, line2, line1, line2], dtype="object")
hfb_layers = np.array([3, 3, 4, 4])

hfb_gdf = gpd.GeoDataFrame(
    geometry=lines, data=dict(layer=hfb_layers, resistance=100.0)
)

# Build model
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
timemap = {
    head_periodic.time.values[0]: "summer",
    head_periodic.time.values[1]: "winter",
}
m["ghb"].periodic_stress(timemap)

## Create second periodic boundary condition
m["ghb2"] = flow.GeneralHeadBoundary(head=head_periodic + 10.0, conductance=10.0)
### You also need to specify periodic stresses for second system.
m["ghb2"].periodic_stress(timemap)

m["wel"] = flow.Well(**wel_df)

m["hfb"] = flow.HorizontalFlowBarrier(**hfb_gdf)

## imod-python wants you to provide the model endtime to your time_discretization!
m.time_discretization(times[-1])

## Writes both .IDFs as well as projectfile, an inifile,
## and a .tim file that contains the time discretization.
m.write(directory=wdir)
