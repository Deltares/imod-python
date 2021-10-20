import pathlib
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import imod

# %% start of simulation
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

layer = np.arange(1, nlay + 1)
y = np.arange(ymax, ymin, dy) + 0.5 * dy
x = np.arange(xmin, xmax, dx) + 0.5 * dx
coords = {"layer": layer, "y": y, "x": x, "dx": dx, "dy": dy}

# Discretization data
idomain = xr.DataArray(np.ones(shape), coords=coords, dims=dims)
bottom = xr.DataArray(np.arange(-0.03, -5.71, step=-0.03), {"layer": layer}, ("layer",))

# layer 1, row 1, all columns
constant_head_values = np.array(
    [
        0.0314107583763,
        0.0941083112213,
        0.156434461572,
        0.2181432366,
        0.27899109997,
        0.338737912978,
        0.397147882257,
        0.453990490355,
        0.509041405475,
        0.562083366817,
        0.612907042001,
        0.661311853209,
        0.707106768773,
        0.750111057092,
        0.790154999895,
        0.827080562039,
        0.860742015208,
        0.891006513031,
        0.917754615366,
        0.940880759678,
        0.960293677645,
        0.975916755352,
        0.987688335652,
        0.995561961497,
        0.999506559285,
        0.999506561491,
        0.995561968105,
        0.987688346637,
        0.97591677067,
        0.960293697235,
        0.940880783464,
        0.917754643253,
        0.89100654491,
        0.860742050953,
        0.827080601509,
        0.790155042933,
        0.75011110353,
        0.707106818426,
        0.661311905882,
        0.612907097486,
        0.562083424895,
        0.509041465917,
        0.453990552921,
        0.397147946702,
        0.338737979046,
        0.278991167402,
        0.218143305128,
        0.156434530928,
        0.0941083811297,
        0.0314108285617,
        -0.0314106881909,
        -0.0941082413128,
        -0.156434392217,
        -0.218143168071,
        -0.278991032538,
        -0.338737846909,
        -0.397147817812,
        -0.453990427788,
        -0.509041345034,
        -0.56208330874,
        -0.612906986516,
        -0.661311800536,
        -0.70710671912,
        -0.750111010655,
        -0.790154956856,
        -0.827080522569,
        -0.860741979463,
        -0.891006481151,
        -0.917754587478,
        -0.940880735891,
        -0.960293658054,
        -0.975916740034,
        -0.987688324667,
        -0.995561954889,
        -0.999506557079,
        -0.999506563696,
        -0.995561974714,
        -0.987688357622,
        -0.975916785988,
        -0.960293716826,
        -0.94088080725,
        -0.917754671141,
        -0.891006576789,
        -0.860742086698,
        -0.827080640978,
        -0.790155085971,
        -0.750111149967,
        -0.707106868079,
        -0.661311958555,
        -0.612907152971,
        -0.562083482973,
        -0.509041526358,
        -0.453990615488,
        -0.397148011147,
        -0.338738045115,
        -0.278991234834,
        -0.218143373657,
        -0.156434600284,
        -0.0941084510381,
        -0.0314108987471,
    ]
)
constant_head = xr.full_like(idomain, np.nan)
constant_head.values[0, 0, :] = constant_head_values

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
    constant_head, print_input=True, print_flows=True, save_flows=False
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
gwf_model["oc"] = imod.mf6.OutputControl(save_head="all", save_budget="all")

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
# %% end of simulation


# print(simulation.render())

# modeldir = pathlib.Path(r"d:\repo\imod\mf6-dist\examples\ex32-periodicbc-imod")
# simulation.write(modeldir)

# %% start of run model
# with imod.util.cd(modeldir):
#     p = subprocess.run("mf6", check=True, capture_output=True, text=True)
#     assert p.stdout.endswith("Normal termination of simulation.\n")
# %% end of run model
