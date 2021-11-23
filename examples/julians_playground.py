# %%
# Define input and output dirs
from pathlib import Path

import xarray as xr

from imod import msw

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
grid_data.write(output_dir)

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
infiltration.write(output_dir)

# %%
