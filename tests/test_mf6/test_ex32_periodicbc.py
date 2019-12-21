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
# TODO split initial head scalar 1, and constant head (values below)
constant_head = xr.DataArray([0.0314107583763, 0.0941083112213, 0.156434461572, 0.2181432366, 0.27899109997, 0.338737912978, 0.397147882257, 0.453990490355, 0.509041405475, 0.562083366817, 0.612907042001, 0.661311853209, 0.707106768773, 0.750111057092, 0.790154999895, 0.827080562039, 0.860742015208, 0.891006513031, 0.917754615366, 0.940880759678, 0.960293677645, 0.975916755352, 0.987688335652, 0.995561961497, 0.999506559285, 0.999506561491, 0.995561968105, 0.987688346637, 0.97591677067, 0.960293697235, 0.940880783464, 0.917754643253, 0.89100654491, 0.860742050953, 0.827080601509, 0.790155042933, 0.75011110353, 0.707106818426, 0.661311905882, 0.612907097486, 0.562083424895, 0.509041465917, 0.453990552921, 0.397147946702, 0.338737979046, 0.278991167402, 0.218143305128, 0.156434530928, 0.0941083811297, 0.0314108285617, -0.0314106881909, -0.0941082413128, -0.156434392217, -0.218143168071, -0.278991032538, -0.338737846909, -0.397147817812, -0.453990427788, -0.509041345034, -0.56208330874, -0.612906986516, -0.661311800536, -0.70710671912, -0.750111010655, -0.790154956856, -0.827080522569, -0.860741979463, -0.891006481151, -0.917754587478, -0.940880735891, -0.960293658054, -0.975916740034, -0.987688324667, -0.995561954889, -0.999506557079, -0.999506563696, -0.995561974714, -0.987688357622, -0.975916785988, -0.960293716826, -0.94088080725, -0.917754671141, -0.891006576789, -0.860742086698, -0.827080640978, -0.790155085971, -0.750111149967, -0.707106868079, -0.661311958555, -0.612907152971, -0.562083482973, -0.509041526358, -0.453990615488, -0.397148011147, -0.338738045115, -0.278991234834, -0.218143373657, -0.156434600284, -0.0941084510381, -0.0314108987471], {"x": x}, ("x",))
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
