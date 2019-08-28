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
def list_to_dense(path, like, k, i, j, **kwargs):
    df = pd.read_csv(path)
    ds = xr.Dataset()
    k = df[k].values.astype(np.int32)
    i = df[i].values.astype(np.int32)
    j = df[j].values.astype(np.int32)
    layers_present = np.unique(k) + 1
    for key, value in kwargs.items(): 
        da = xr.full_like(like.sel(layer=layers_present), np.nan)
        da.values[k, i, j] = df[value].values.astype(np.float64)
        ds[key] = da
    return ds

# Do data correction
# Get rid of nodata
top = top.where(top > -9990.0)
bot = bot.where(bot > -9990.0)

# Create a boolean array
is_active = (idomain == 1)

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
kv = c_thickness / c  # m/d
kv = kv.fillna(1.0e-6)

kh["layer"].values = 2 * kh["layer"].values - 1
kv["layer"].values *= 2
k = xr.concat([kh, kv], dim="layer").sortby("layer")

bot_even = top.isel(layer=slice(1, None))
bot_even["layer"].values = bot_even["layer"].values * 2 - 2
bot["layer"].values = bot["layer"].values * 2 - 1
bot = xr.concat([bot, bot_even], dim="layer").sortby("layer")
filler_bot = xr.full_like(bot, 1.0e-3).cumsum("layer") * -1.0
bot = bot.combine_first(filler_bot)

# Initial condition and constant head
startingheads = imod.rasterio.read("data/startingheads/HH*.tif", pattern="HH{layer}")
startingheads = xr.where(startingheads > 1000.0, 0.0, startingheads)
evenheads = (startingheads.isel(layer=slice(None, -1))).copy()
startingheads["layer"].values = startingheads["layer"].values * 2 - 1
evenheads["layer"].values *= 2
startingheads = xr.concat([startingheads, evenheads], dim="layer").sortby("layer")

eroded = idomain.copy()
eroded.values = scipy.ndimage.morphology.binary_erosion(
        eroded.values, np.ones((3, 3))
)
is_boundary = (eroded == 0) & is_active
# Startingheads array already has layer dimension
constanthead = startingheads.where(is_boundary)
# Expand idomain into three dimensions
idomain = xr.full_like(k, idomain)

wells = pd.read_csv("data/wells/sq_list.csv")
rivers = list_to_dense(
        path="data/mf2005/riv_data.csv",
        like=startingheads,
        k="k",
        i="i",
        j="j",
        conductance="cond0",
        stage="stage0",
        bottom="rbot0",
)
generalheadboundary = list_to_dense(
        path="data/mf2005/ghb_data.csv",
        like=startingheads,
        k="k",
        i="i",
        j="j",
        conductance="cond0",
        head="bhead0",
)
drainage = list_to_dense(
        path="data/mf2005/drn_data.csv",
        like=startingheads,
        k="k",
        i="i",
        j="j",
        conductance="cond0",
        elevation="elev0",
)

# Remove nodata parts
rivers = rivers.where(is_active)
generalheadboundary = generalheadboundary.where(is_active)
drainage = drainage.where(is_active)

# Recharge
recharge = recharge.assign_coords(layer=1).where(is_active)

# Initialize and fill the model
gwf_model = imod.mf6.GroundwaterFlowModel()
gwf_model["dis"] = imod.mf6.StructuredDiscretization(
    top=top.sel(layer=1).fillna(0.0),
    bottom=bot,
    idomain=idomain,
)
gwf_model["npf"] = imod.mf6.NodePropertyFlow(
    k=k,
    k22=k,
    k33=k,
    icelltype=0,
)
gwf_model["ic"] = imod.mf6.InitialConditions(head=startingheads)
gwf_model["drn"] = imod.mf6.Drainage(
    elevation=drainage["elevation"],
    conductance=drainage["conductance"],
)
gwf_model["ghb"] = imod.mf6.GeneralHeadBoundary(
    head=generalheadboundary["head"],
    conductance=generalheadboundary["conductance"],
)
gwf_model["rch"] = imod.mf6.Recharge(rate=recharge)
gwf_model["chd"] = imod.mf6.ConstantHead(head=constanthead)
gwf_model["oc"] = imod.mf6.OutputControl(save_head=True, save_budget=False)

# Attach it to a simulation
simulation = imod.mf6.Modflow6Simulation("Brabant-steady")
simulation["gwf_1"] = gwf_model
simulation["ims"] = imod.mf6.SolutionPresetSimple(print_option=True, csv_output=False, no_ptc=True)
#simulation["solver"] = imod.mf6.Solution(
#    print_option=False,
#    csv_output=False,
#    no_ptc=True,
#    outer_hclose=1.0e-4,
#    outer_maximum=500,
#    under_relaxation=None,
#    inner_hclose=1.0e-4,
#    inner_rclose=0.001,
#    inner_maximum=100,
#    linear_acceleration="cg",
#    scaling_method=None,
#    reordering_method=None,
#    relaxation_factor=0.97,
#)
# Determine time discretization
simulation.time_discretization(starttime="2000-01-01", endtime="2000-01-02")

# Write!
simulation.write("Brabant-steady")

print("Done!")
