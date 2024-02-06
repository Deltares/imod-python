import numpy as np
import pandas as pd
import xarray as xr

import imod
from imod.typing.grid import nan_like, zeros_like
from imod.mf6.multimodel.partition_generator import get_label_array
# Model parameters
nlay = 1  # Number of layers
nrow = 31  # Number of rows
ncol = 46  # Number of columns
delr = 10.0  # Column width ($m$)
delc = 10.0  # Row width ($m$)
delz = 10.0  # Layer thickness ($m$)
shape = (nlay, nrow, ncol)
top = 10.0
dims = ("layer", "y", "x")

y = np.arange(delr * nrow, 0, -delr)
x = np.arange(0, delc * ncol, delc)
coords = {"layer": [1], "y": y, "x": x, "dx": delc, "dy": -delr}
idomain = xr.DataArray(np.ones(shape, dtype=int), coords=coords, dims=dims)

bottom = xr.DataArray([0.0], {"layer": [1]}, ("layer",))
gwf_model = imod.mf6.GroundwaterFlowModel()
gwf_model["dis"] = imod.mf6.StructuredDiscretization(
    top=10.0, bottom=bottom, idomain=idomain
)

gwf_model["sto"] = imod.mf6.SpecificStorage(
    specific_storage=0.0,
    specific_yield=0.0,
    transient=False,
    convertible=0,
)
gwf_model["npf"] = imod.mf6.NodePropertyFlow(
    icelltype=idomain,
    k=1.0,
    save_flows=True,
)
Lx = 460
v = 1.0 / 3.0
prsity = 0.3
q = v * prsity
h1 = q * Lx
chd_field = nan_like(idomain)
chd_field.values[0, :, 0] = h1
chd_field.values[0, :, -1] = 0.1
chd_concentration = nan_like(idomain)
chd_concentration.values[0, :, 0] = 0.0
chd_concentration.values[0, :, -1] = 0.0
chd_concentration = chd_concentration.expand_dims(species=["Au"])


gwf_model["chd"] = imod.mf6.ConstantHead(
    chd_field,
    concentration=chd_concentration,
    print_input=True,
    print_flows=True,
    save_flows=True,
)
injection_concentration = xr.DataArray(
    [[1000.0]],
    coords={
        "species": ["Au"],
        "index": [0],
    },
    dims=("species", "index"),
)
gwf_model["wel"] = imod.mf6.Well(
    x=[150.0],
    y=[150.0],
    screen_top=[10.0],
    screen_bottom=[0.0],
    rate=[1.0],
    concentration=injection_concentration,
    concentration_boundary_type="aux",
)
gwf_model["oc"] = imod.mf6.OutputControl(save_head="all", save_budget="all")
gwf_model["ic"] = imod.mf6.InitialConditions(start=10.0)
simulation = imod.mf6.Modflow6Simulation("ex01-twri")


tpt_model = imod.mf6.GroundwaterTransportModel()
tpt_model["ssm"] = imod.mf6.SourceSinkMixing.from_flow_model(
    gwf_model, species="Au", save_flows=True
)
tpt_model["adv"] = imod.mf6.AdvectionUpstream()
tpt_model["dsp"] = imod.mf6.Dispersion(
    diffusion_coefficient=0.0,
    longitudinal_horizontal=10.0,
    transversal_horizontal1=3.0,
    xt3d_off=False,
    xt3d_rhs=False,
)
tpt_model["mst"] = imod.mf6.MobileStorageTransfer(
    porosity=0.3,
)

tpt_model["ic"] = imod.mf6.InitialConditions(start=0.0)
tpt_model["oc"] = imod.mf6.OutputControl(save_concentration="all", save_budget="last")
tpt_model["dis"] = gwf_model["dis"]

simulation["GWF_1"] = gwf_model
simulation["TPT_1"] = tpt_model


simulation["flow_solver"] = imod.mf6.Solution(
    modelnames=["GWF_1"],
    print_option="summary",
    csv_output=False,
    no_ptc=True,
    outer_dvclose=1.0e-4,
    outer_maximum=500,
    under_relaxation=None,
    inner_dvclose=1.0e-4,
    inner_rclose=0.001,
    inner_maximum=100,
    linear_acceleration="cg",
    scaling_method=None,
    reordering_method=None,
    relaxation_factor=0.97,
)
simulation["transport_solver"] = imod.mf6.Solution(
    modelnames=["TPT_1"],
    print_option="summary",
    csv_output=False,
    no_ptc=True,
    outer_dvclose=1.0e-4,
    outer_maximum=500,
    under_relaxation=None,
    inner_dvclose=1.0e-4,
    inner_rclose=0.001,
    inner_maximum=100,
    linear_acceleration="bicgstab",
    scaling_method=None,
    reordering_method=None,
    relaxation_factor=0.97,
)
# Collect time discretization

duration = pd.to_timedelta("365d")
start = pd.to_datetime("2002-01-01")
simulation.create_time_discretization(additional_times=[start, start + duration])
simulation["time_discretization"]["n_timesteps"] = 365



label_array = get_label_array(simulation, 4)
modeldir = imod.util.temporary_directory()
simulation.write(modeldir, binary=False)
split_simulation = simulation.split(label_array)


simulation.run()
hds = simulation.open_head()
conc = simulation.open_concentration()

split_modeldir = modeldir /"split"
split_simulation.write(modeldir, binary=False)
split_simulation.run()
split_hds =  split_simulation.open_head()["head"]
split_conc =  split_simulation.open_head()["concentration"]

print(conc.sel(time=365.0, layer=1).values)
a = conc.sel(time=365.0, layer=1)
print(a.max().values)
perlen = 365  # Simulation time ($days$)
