# %%
# Define input and output dirs
from pathlib import Path

import xarray as xr

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
gwf_model = mf6.GroundwaterFlowModel()

gwf_model["recharge"] = mf6.Recharge(recharge_array)

# Well
layer = [3, 2, 1]
row = [1, 1, 4]
column = [1, 1, 3]
rate = [-5.0] * 3
gwf_model["wel"] = mf6.Well(layer=layer, row=row, column=column, rate=rate)

simulation = mf6.Modflow6Simulation("ex01-twri")
simulation["GWF_1"] = gwf_model
# Define solver settings
simulation["solver"] = mf6.Solution(
    print_option="summary",
    csv_output=False,
    no_ptc=True,
    outer_maximum=500,
    under_relaxation=None,
    inner_hclose=1.0e-4,
    inner_rclose=0.001,
    inner_maximum=100,
    linear_acceleration="cg",
    scaling_method=None,
    reordering_method=None,
    relaxation_factor=0.97,
)
# Collect time discretization
simulation.time_discretization(times=["2000-01-01", "2000-01-02"])

# %%
metamod = couplers.MetaMod(msw_model, gwf_model)
metamod.write(output_dir)
