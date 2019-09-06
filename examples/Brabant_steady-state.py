import os

import numpy as np
import pandas as pd
import scipy.ndimage.morphology
import xarray as xr

import imod


os.chdir("/projects/imodx/Brabant_stationary")


# Read the griddata
kd = imod.rasterio.read("data/kdc/tx*.tif", pattern="tx{layer}")
c = imod.rasterio.read("data/kdc/CL*.tif", pattern="cl{layer}")
idomain = imod.rasterio.read("data/boundary/ibound.tif")
top = imod.rasterio.read("data/topbot/RL*.tif", pattern="RL{layer}")
bot = imod.rasterio.read("data/topbot/TH*.tif", pattern="TH{layer}")
recharge = imod.rasterio.read("data/recharge/RP1.tif")

# Read the list data
def dataframe_to_dataset(df, like, k, i, j, **kwargs):
    ds = xr.Dataset()
    k = df[k].values.astype(np.int32)
    i = df[i].values.astype(np.int32)
    j = df[j].values.astype(np.int32)
    for key, value in kwargs.items():
        da = xr.full_like(like, np.nan)
        da.values[k, i, j] = df[value].values.astype(np.float64)
        ds[key] = da.dropna("layer", how="all")
    return ds


# Do data correction
# Get rid of nodata
top = top.where(top > -9990.0)
bot = bot.where(bot > -9990.0)

# Compute horizontal conductivity
thickness = top - bot
kh = kd / thickness
kh = kh.fillna(1.0e-6)

# Compute vertical conductivity
# Offset the tops and bottoms to calculate the resistance of the layers.
c_top = bot.isel(layer=slice(None, -1))
c_bot = top.isel(layer=slice(1, None))
c_bot["layer"].values -= 1
c_thickness = c_top - c_bot  # m

# Create 37 layered model input
bot_even = top.isel(layer=slice(1, None))
bot_even["layer"].values = bot_even["layer"].values * 2 - 2
bot["layer"].values = bot["layer"].values * 2 - 1
bot = xr.concat([bot, bot_even], dim="layer").sortby("layer")
filler_bot = xr.full_like(bot, 1.0e-3).cumsum("layer") * -1.0
bot = bot.combine_first(filler_bot)

# Create a boolean array
is_active = idomain == 1
kv = c_thickness / c  # m/d

kh["layer"].values = 2 * kh["layer"].values - 1
kv["layer"].values *= 2

k = xr.full_like(bot, np.nan).combine_first(kh).fillna(1.0e-6)
k33 = xr.full_like(bot, np.nan).combine_first(kv).fillna(1.0e6)

# Initial condition and constant head
startingheads = imod.rasterio.read("data/startingheads/HH*.tif", pattern="HH{layer}")
startingheads = xr.where(startingheads > 1000.0, 0.0, startingheads)
evenheads = (startingheads.isel(layer=slice(None, -1))).copy()
startingheads["layer"].values = startingheads["layer"].values * 2 - 1
evenheads["layer"].values *= 2
startingheads = xr.concat([startingheads, evenheads], dim="layer").sortby("layer")

eroded = idomain.copy()
eroded.values = scipy.ndimage.morphology.binary_erosion(eroded.values, np.ones((3, 3)))
is_boundary = (eroded == 0) & is_active
# Startingheads array already has layer dimension
constanthead = startingheads.where(is_boundary)
# Expand idomain into three dimensions
idomain = xr.full_like(k, idomain, dtype=np.int32)

wells = pd.read_csv("data/wells/sq_list.csv")
well_layer = wells["ilay"].values * 2 - 1
dict_xy = imod.select.points_indices(
    da=k, x=wells["xcoordinate"].values, y=wells["ycoordinate"].values
)
well_row = dict_ij["y"]
well_column = dict_ij["x"]
well_active = idomain.values[well_layer - 1, well_row, well_column] == 1

well_layer = well_layer[well_active]
well_row = well_row[well_active] + 1
well_column = well_column[well_active] + 1
well_rate = wells["q_assigned"].values[well_active]


df = pd.read_csv("data/mf2005/riv_data.csv")
river = dataframe_to_dataset(
    df=df,
    like=kd,
    k="k",
    i="i",
    j="j",
    conductance="cond0",
    stage="stage0",
    bottom="rbot0",
)
river = river.where(is_active)

# GHB has up to 27 occurences in a single cell!
# df.groupby(["k", "i", "j"]).size().max()
# on average 1.69 elements...

df = pd.read_csv("data/mf2005/ghb_data.csv")
generalheadboundary = []
for i in range(4):  # captures > 90% of boundary elements
    duplicated = df.duplicated(subset=["k", "i", "j"], keep="first")
    ds = dataframe_to_dataset(
        df=df[~duplicated],
        like=kd,
        k="k",
        i="i",
        j="j",
        conductance="cond0",
        head="bhead0",
    )
    ds = ds.where(is_active)
    ds["layer"] = 2 * ds["layer"].values - 1
    generalheadboundary.append(ds)
    df = df[duplicated]

df = pd.read_csv("data/mf2005/drn_data.csv")
drainage = []
for i in range(4):
    duplicated = df.duplicated(subset=["k", "i", "j"], keep="first")
    ds = dataframe_to_dataset(
        df=df[~duplicated],
        like=kd,
        k="k",
        i="i",
        j="j",
        conductance="cond0",
        elevation="elev0",
    )
    ds = ds.where(is_active)
    ds["layer"] = 2 * ds["layer"].values - 1
    drainage.append(ds)
    df = df[duplicated]

# Recharge
recharge = recharge.assign_coords(layer=1).where(is_active)

# Initialize and fill the model
gwf_model = imod.mf6.GroundwaterFlowModel()
gwf_model["dis"] = imod.mf6.StructuredDiscretization(
    top=top.sel(layer=1).fillna(0.0), bottom=bot, idomain=idomain
)
gwf_model["npf"] = imod.mf6.NodePropertyFlow(k=k, k22=k, k33=k33, icelltype=0)
gwf_model["ic"] = imod.mf6.InitialConditions(head=startingheads)

for i, ds in enumerate(drainage):
    gwf_model[f"drn-{i+1}"] = imod.mf6.Drainage(
        elevation=ds["elevation"], conductance=ds["conductance"]
    )

for i, ds in enumerate(generalheadboundary):
    gwf_model[f"ghb-{i+1}"] = imod.mf6.GeneralHeadBoundary(
        head=ds["head"], conductance=ds["conductance"]
    )
gwf_model["riv"] = imod.mf6.River(
    stage=river["stage"],
    conductance=river["conductance"],
    bottom_elevation=river["bottom"],
)
gwf_model["wel"] = imod.mf6.Well(
    layer=well_layer, row=well_row, column=well_column, rate=well_rate
)
gwf_model["rch"] = imod.mf6.Recharge(rate=recharge, print_input=True)
gwf_model["chd"] = imod.mf6.ConstantHead(head=constanthead)
gwf_model["oc"] = imod.mf6.OutputControl(save_head=True, save_budget=False)

# Attach it to a simulation
simulation = imod.mf6.Modflow6Simulation("Brabant-steady")
simulation["gwf_1"] = gwf_model
simulation["ims"] = imod.mf6.SolutionPresetSimple(
    print_option="summary", csv_output=False, no_ptc=True
)

# Determine time discretization
simulation.time_discretization(times=["2000-01-01", "2000-01-02"])

# Write!
simulation.write("Brabant-steady")

print("Done!")
