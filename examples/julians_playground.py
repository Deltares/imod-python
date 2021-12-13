# %%
# Define input and output dirs
from pathlib import Path

import numpy as np
import xarray as xr

import imod
from imod import couplers, mf6, msw

output_dir = Path("D:/checkouts/imod_python_hooghoudt/Hooghoudt_sprinkling_playground")

# %%
# Parse netcdf files
grid_dataset = xr.open_dataset(
    "D:/checkouts/imod_python_hooghoudt/idf_and_netcdf/netcdf/metaswap_grids.nc"
)

meteo_array = xr.open_dataset(
    "D:/checkouts/imod_python_hooghoudt/idf_and_netcdf/netcdf/cap-mst_l1.nc"
)["cap-mst"]

surface_elevation_array = xr.open_dataset(
    "D:/checkouts/imod_python_hooghoudt/idf_and_netcdf/netcdf/cap-sfl_l1.nc"
)["cap-sfl"]

soil_physical_unit_array = xr.open_dataset(
    "D:/checkouts/imod_python_hooghoudt/idf_and_netcdf/netcdf/cap-slt_l1.nc"
)["cap-slt"]

active_array = xr.open_dataset(
    "D:/checkouts/imod_python_hooghoudt/idf_and_netcdf/netcdf/active.nc"
)["active"]

artifial_recharge_array = xr.open_dataset(
    "D:/checkouts/imod_python_hooghoudt/idf_and_netcdf/netcdf/cap-spr_l1.nc"
)["cap-spr"]

artifial_recharge_layer_array = array = xr.open_dataset(
    "D:/checkouts/imod_python_hooghoudt/idf_and_netcdf/netcdf/cap-spr_layer_l1.nc"
)["cap-spr_layer"]

print(meteo_array)
print(meteo_array["dx"])
exit()

# %% Create `GridData` and write to file
grid_data = msw.GridData(
    grid_dataset["area"],
    grid_dataset["landuse"],
    grid_dataset["rootzone_depth"],
    surface_elevation_array,
    soil_physical_unit_array,
    active_array,
)

# %%
# Parse netcdf files
downward_resistance_array = xr.open_dataset(
    "D:/checkouts/imod_python_hooghoudt/idf_and_netcdf/netcdf/downward_resistance.nc"
)["downward_resistance"]

upward_resistance_array = xr.open_dataset(
    "D:/checkouts/imod_python_hooghoudt/idf_and_netcdf/netcdf/upward_resistance.nc"
)["upward_resistance"]

bottom_resistance_array = xr.open_dataset(
    "D:/checkouts/imod_python_hooghoudt/idf_and_netcdf/netcdf/bottom_resistance.nc"
)["bottom_resistance"]

extra_storage_coefficient_array = xr.open_dataset(
    "D:/checkouts/imod_python_hooghoudt/idf_and_netcdf/netcdf/extra_storage_coefficient.nc"
)["extra_storage_coefficient"]

# %% Create `Infiltration` and write to file
infiltration = msw.Infiltration(
    grid_dataset["qinf"],
    downward_resistance_array,
    upward_resistance_array,
    bottom_resistance_array,
    extra_storage_coefficient_array,
    active_array,
)
# %%
msw_model = msw.MetaSwapModel()
msw_model["grid_data"] = grid_data
msw_model["infiltration"] = infiltration

# %%
recharge_array = xr.open_dataset(
    "D:/checkouts/imod_python_hooghoudt/idf_and_netcdf/netcdf/recharge.nc"
)["recharge"]


# %%
# Create grid coordinates
# -----------------------
#
# The first steps consist of setting up the grid -- first the number of layer,
# rows, and columns. Cell sizes are constant throughout the model.

nlay = 3
nrow = 9
ncol = 9
shape = (nlay, nrow, ncol)

dx = 5000.0
dy = -5000.0
xmin = 0.0
xmax = dx * ncol
ymin = 0.0
ymax = abs(dy) * nrow
dims = ("layer", "y", "x")

layer = np.array([1, 2, 3])
y = np.arange(ymax, ymin, dy) + 0.5 * dy
x = np.arange(xmin, xmax, dx) + 0.5 * dx
coords = {"layer": layer, "y": y, "x": x}

# %%
# Create DataArrays
# -----------------
#
# Now that we have the grid coordinates setup, we can start defining model
# parameters. The model is characterized by:
#
# * a constant head boundary on the left
# * a single drain in the center left of the model
# * uniform recharge on the top layer
# * a number of wells scattered throughout the model.

idomain = xr.DataArray(np.ones(shape), coords=coords, dims=dims)
bottom = xr.DataArray([-200.0, -300.0, -450.0], {"layer": layer}, ("layer",))

# Constant head
constant_head = xr.full_like(idomain, np.nan).sel(layer=[1, 2])
constant_head[..., 0] = 0.0

# Drainage
elevation = xr.full_like(idomain.sel(layer=1), np.nan)
conductance = xr.full_like(idomain.sel(layer=1), np.nan)
elevation[:] = 10.0
conductance[:] = 1.0


# Well
well_layer = [3, 2, 1]
well_row = [1, 1, 4]
well_column = [1, 1, 3]
well_rate = [-5.0] * 3

# Node properties
icelltype = xr.DataArray([1, 0, 0], {"layer": layer}, ("layer",))
k = xr.DataArray([1.0e-3, 1.0e-4, 2.0e-4], {"layer": layer}, ("layer",))
k33 = xr.DataArray([2.0e-8, 2.0e-8, 2.0e-8], {"layer": layer}, ("layer",))

# %%
# Write the modflow model
# ---------------
#
# The first step is to define an empty model, the parameters and boundary
# conditions are added in the form of the familiar MODFLOW packages.

gwf_model = imod.mf6.GroundwaterFlowModel()
gwf_model["dis"] = imod.mf6.StructuredDiscretization(
    top=200.0, bottom=bottom, idomain=idomain
)

gwf_model["chd"] = imod.mf6.ConstantHead(
    constant_head, print_input=True, print_flows=True, save_flows=True
)
gwf_model["drn"] = imod.mf6.Drainage(
    elevation=elevation,
    conductance=conductance,
    print_input=True,
    print_flows=True,
    save_flows=True,
)
gwf_model["ic"] = imod.mf6.InitialConditions(head=0.0)
gwf_model["npf"] = imod.mf6.NodePropertyFlow(
    icelltype=icelltype,
    k=k,
    k33=k33,
    variable_vertical_conductance=True,
    dewatered=True,
    perched=True,
    save_flows=True,
)
gwf_model["oc"] = imod.mf6.OutputControl(save_head="all", save_budget="all")
gwf_model["rch"] = imod.mf6.Recharge(recharge_array)
gwf_model["wel"] = imod.mf6.Well(
    layer=well_layer,
    row=well_row,
    column=well_column,
    rate=well_rate,
    print_input=True,
    print_flows=True,
    save_flows=True,
)
gwf_model["sto"] = imod.mf6.SpecificStorage(
    specific_storage=1.0e-15,
    specific_yield=0.15,
    convertible=0,
    transient=False,
)

# Attach it to a mf6_simulation
mf6_simulation = imod.mf6.Modflow6Simulation("ex01-twri")
mf6_simulation["GWF_1"] = gwf_model
# Define solver settings
mf6_simulation["solver"] = imod.mf6.Solution(
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
# Collect time discretization
mf6_simulation.time_discretization(
    times=["2000-01-01", "2000-01-02", "2000-01-03", "2000-01-04"]
)


# %%
metamod = couplers.MetaMod(msw_model, mf6_simulation)
metamod.write(output_dir)
