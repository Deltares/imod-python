import pathlib
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import imod

## %% start of simulation
nlay = 190
nrow = 1
ncol = 100
shape = (nlay, nrow, ncol)

dx = 0.06
dy = -1.0
xmin = 0.0
xmax = dx * ncol
ymin = 0.0
ymax = abs(dy) * nrow
dims = ("layer", "y", "x")

layer = np.arange(1, nlay+1)
y = np.arange(ymax, ymin, dy) + 0.5 * dy
x = np.arange(xmin, xmax, dx) + 0.5 * dx
coords = {"layer": layer, "y": y, "x": x, "dx": dx, "dy": dy}

# Discretization data
idomain = xr.DataArray(np.ones(shape), coords=coords, dims=dims)
bottom = xr.DataArray([-0.03,-0.06,-0.09,-0.12,-0.15,-0.18,-0.21,-0.24,-0.27,-0.30,-0.33,-0.36,-0.39,-0.42,-0.45,-0.48,-0.51,-0.54,-0.57,-0.60,-0.63,-0.66,-0.69,-0.72,-0.75,-0.78,-0.81,-0.84,-0.87,-0.90,-0.93,-0.96,-0.99,-1.02,-1.05,-1.08,-1.11,-1.14,-1.17,-1.2,-1.23,-1.26,-1.29,-1.32,-1.35,-1.38,-1.41,-1.44,-1.47,-1.5,-1.53,-1.56,-1.59,-1.62,-1.65,-1.68,-1.71,-1.74,-1.77,-1.8,-1.83,-1.86,-1.89,-1.92,-1.95,-1.98,-2.01,-2.04,-2.07,-2.1,-2.13,-2.16,-2.19,-2.22,-2.25,-2.28,-2.31,-2.34,-2.37,-2.4,-2.43,-2.46,-2.49,-2.52,-2.55,-2.58,-2.61,-2.64,-2.67,-2.7,-2.73,-2.76,-2.79,-2.82,-2.85,-2.88,-2.91,-2.94,-2.97,-3.,-3.03,-3.06,-3.09,-3.12,-3.15,-3.18,-3.21,-3.24,-3.27,-3.3,-3.33,-3.36,-3.39,-3.42,-3.45,-3.48,-3.51,-3.54,-3.57,-3.6,-3.63,-3.66,-3.69,-3.72,-3.75,-3.78,-3.81,-3.84,-3.87,-3.9,-3.93,-3.96,-3.99,-4.02,-4.05,-4.08,-4.11,-4.14,-4.17,-4.2,-4.23,-4.26,-4.29,-4.32,-4.35,-4.38,-4.41,-4.44,-4.47,-4.5,-4.53,-4.56,-4.59,-4.62,-4.65,-4.68,-4.71,-4.74,-4.77,-4.8,-4.83,-4.86,-4.89,-4.92,-4.95,-4.98,-5.01,-5.04,-5.07,-5.1,-5.13,-5.16,-5.19,-5.22,-5.25,-5.28,-5.31,-5.34,-5.37,-5.4,-5.43,-5.46,-5.49,-5.52,-5.55,-5.58,-5.61,-5.64,-5.67,-5.7], {"layer": layer}, ("layer",))

# Constant head
head = xr.full_like(idomain, 1.0)
head = 1.0  # check if this works

# Node properties
icelltype = xr.full_like(idomain, 0.0)
k = xr.full_like(idomain, 1.0)
k33 = xr.full_like(idomain, 1.0)
icelltype = 0.0  # check if this works
k = 1.0  # check if this works
k33 = 1.0  # check if this works

# Create and fill the groundwater model.
gwf_model = imod.mf6.GroundwaterFlowModel()
gwf_model["dis"] = imod.mf6.StructuredDiscretization(
    top=0.0, bottom=bottom, idomain=idomain
)
gwf_model["chd"] = imod.mf6.ConstantHead(
    head, print_input=True, print_flows=True, save_flows=False
)
gwf_model["ic"] = imod.mf6.InitialConditions(head=1.0)
gwf_model["npf"] = imod.mf6.NodePropertyFlow(
    icelltype=icelltype,
    k=k,
    k33=k33,
    variable_vertical_conductance=False,
    dewatered=False,
    perched=False,
    save_flows=True,
)
gwf_model["oc"] = imod.mf6.OutputControl(save_head=True, save_budget=True)

# Attach it to a simulation
simulation = imod.mf6.Modflow6Simulation("mfsim")
simulation["pbc"] = gwf_model
# Define solver settings
simulation["solver"] = imod.mf6.Solution(
    outer_hclose=1.0e-5,
    outer_maximum=50,
    inner_hclose=1.0e-6,
    inner_rclose=1.0e-5,
    inner_maximum=300,
    linear_acceleration="cg",
    print_option="all",
)
# Collect time discretization
simulation.time_discretization(times=["2000-01-01", "2000-01-02"])
#%% end of simulation


print(simulation.render())

modeldir = pathlib.Path(r"d:\repo\imod\mf6-dist\examples\ex32-periodicbc-imod")
simulation.write(modeldir)

#%% start of run model
with imod.util.cd(modeldir):
    p = subprocess.run("mf6", check=True, capture_output=True, text=True)
    assert p.stdout.endswith("Normal termination of simulation.\n")
#%% end of run model
