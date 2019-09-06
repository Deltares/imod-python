import numpy as np
import xarray as xr

import imod

ibound = imod.idf.open("../model_v3.3.0/dbase/bnd/*.idf")
is_active = ibound != 0

# wvp_top_l1 = imod.idf.open("../lagenmodel/ahn_f250_m.idf").assign_coords(layer=1)
# wvp_top_l2 = imod.idf.open("../lagenmodel/bot_sdl1_m.idf").assign_coords(layer=2)
# wvp_top_l3 = imod.idf.open("../lagenmodel/bot_sdl2_m.idf").assign_coords(layer=3)
# wvp_top_l4 = imod.idf.open("../lagenmodel/bot_sdl3_m.idf").assign_coords(layer=4)
# wvp_top_l5 = imod.idf.open("../lagenmodel/bot_sdl4_m.idf").assign_coords(layer=5)
# wvp_top_l6 = imod.idf.open("../lagenmodel/bot_sdl5_m.idf").assign_coords(layer=6)
wvp_top_l1 = imod.idf.open("../lagenmodel/ahn_f250_m.idf").assign_coords(layer=1)
wvp_top_l2 = imod.idf.open("../lagenmodel/top_sdl1_m.idf").assign_coords(layer=2)
wvp_top_l3 = imod.idf.open("../lagenmodel/top_sdl2_m.idf").assign_coords(layer=3)
wvp_top_l4 = imod.idf.open("../lagenmodel/top_sdl3_m.idf").assign_coords(layer=4)
wvp_top_l5 = imod.idf.open("../lagenmodel/top_sdl4_m.idf").assign_coords(layer=5)
wvp_top_l6 = imod.idf.open("../lagenmodel/top_sdl5_m.idf").assign_coords(layer=6)
wvp_top_l7 = imod.idf.open("../lagenmodel/top_sdl6_m.idf").assign_coords(layer=7)
wvp_bot_l1 = imod.idf.open("../lagenmodel/top_sdl1_m.idf").assign_coords(layer=1)
wvp_bot_l2 = imod.idf.open("../lagenmodel/top_sdl2_m.idf").assign_coords(layer=2)
wvp_bot_l3 = imod.idf.open("../lagenmodel/top_sdl3_m.idf").assign_coords(layer=3)
wvp_bot_l4 = imod.idf.open("../lagenmodel/top_sdl4_m.idf").assign_coords(layer=4)
wvp_bot_l5 = imod.idf.open("../lagenmodel/top_sdl5_m.idf").assign_coords(layer=5)
wvp_bot_l6 = imod.idf.open("../lagenmodel/top_sdl6_m.idf").assign_coords(layer=6)
wvp_bot_l7 = imod.idf.open(
    "../lagenmodel/top_geohydrologische_basis.idf"
).assign_coords(layer=7)

wvp_top = xr.concat(
    [
        wvp_top_l1,
        wvp_top_l2,
        wvp_top_l3,
        wvp_top_l4,
        wvp_top_l5,
        wvp_top_l6,
        wvp_top_l7,
    ],
    dim="layer",
)
wvp_bot = xr.concat(
    [
        wvp_bot_l1,
        wvp_bot_l2,
        wvp_bot_l3,
        wvp_bot_l4,
        wvp_bot_l5,
        wvp_bot_l6,
        wvp_bot_l7,
    ],
    dim="layer",
)
is_active = is_active & wvp_bot.notnull()
idomain = (
    is_active.where(wvp_top > wvp_bot, other=-1)
    .where(is_active)
    .fillna(0.0)
    .astype(np.int32)
)
is_active = idomain == 1

wvp_top = wvp_top.where(wvp_top > wvp_bot, other=wvp_bot)
wvp_thickness = wvp_top - wvp_bot
assert wvp_thickness.min() >= 0.0

sdl_top = imod.idf.open("../lagenmodel/TOP_SDL*_M.idf", pattern="{name}_sdl{layer}_m")
sdl_bot = imod.idf.open("../lagenmodel/BOT_SDL*_M.idf", pattern="{name}_sdl{layer}_m")
sdl_thickness = sdl_top - sdl_bot
assert sdl_thickness.min() >= 0.0

kd = imod.idf.open("../model_v3.3.0/dbase/kd/*.idf")
c = imod.idf.open("../model_v3.3.0/dbase/c/*.idf")

# replace negative c value...
c = c.where(~(c < 0.0), other=10.0)
assert c.min() >= 0.0
assert kd.min() >= 0.0

# Add a centimer to avoid dividin by zero ...
kh = (kd / (wvp_thickness + 0.01)).where(is_active)
kv_tmp = ((sdl_thickness + 0.01) / c).where(is_active)
kv = kh.copy()
kv[:-1, ...].values = kv_tmp.values


# drainage
drn_buis_cond = imod.idf.open(
    "../model_v3.3.0/dbase/drn/cond_b_250.idf", pattern="{name}"
)
drn_buis_elev = imod.idf.open(
    "../model_v3.3.0/dbase/drn/BODH_B_250_HYDT_CORRECT2.IDF", pattern="{name}"
)
is_buis = drn_buis_cond.notnull() & drn_buis_elev.notnull()
drn_buis_cond = drn_buis_cond.where(is_buis).where(is_active)
drn_buis_elev = drn_buis_elev.where(is_buis).where(is_active)

drn_greppel_cond = imod.idf.open(
    "../model_v3.3.0/dbase/drn/COND_BRP2012_MVGREP_250M.IDF", pattern="{name}"
)
drn_greppel_elev = imod.idf.open(
    "../model_v3.3.0/dbase/drn/DIEPTEtovMV_BRP2012_MVGREP_250M.IDF", pattern="{name}"
)
is_greppel = drn_greppel_cond.notnull() & drn_greppel_elev.notnull()
drn_greppel_cond = drn_greppel_cond.where(is_greppel).where(is_active)
drn_greppel_elev = drn_greppel_elev.where(is_greppel).where(is_active)

drn_overland_cond = imod.idf.open(
    "../model_v3.3.0/dbase/drn/COND_SOF_250_3.2.X_Mask.IDF", pattern="{name}"
)
drn_overland_elev = imod.idf.open(
    "../model_v3.3.0/dbase/drn/BODH_SOF_250_3.2.X_mask_corr.IDF", pattern="{name}"
)
is_overland = drn_overland_cond.notnull() & drn_overland_elev.notnull()
drn_overland_cond = drn_overland_cond.where(is_overland).where(is_active)
drn_overland_elev = drn_overland_elev.where(is_overland).where(is_active)

# general head boundary
ghb_cond = imod.idf.open("../model_v3.3.0/dbase/ghb/ghb_cond_l2.idf", pattern="{name}")
ghb_head = imod.idf.open("../model_v3.3.0/dbase/ghb/ghb_stage_l2.idf", pattern="{name}")
is_ghb = ghb_cond.notnull() & ghb_head.notnull()
ghb_head = ghb_head.where(is_ghb).where(is_active)
ghb_cond = ghb_cond.where(is_ghb).where(is_active)

# recharge
rch_rate = imod.idf.open(
    "../model_v3.3.0/dbase/rch/NL9_GWA_MMD_19980101-20070101.IDF", pattern="{name}"
).where(is_active)

# rivers
# main waterways
main_cond_l1 = imod.idf.open(
    "../model_v3.3.0/dbase/riv/COND_HL1_250.IDF", pattern="{name}"
).assign_coords(layer=1)
main_stage_l1 = imod.idf.open(
    "../model_v3.3.0/dbase/riv/PEILH_hws_stat_1996_2006.idf", pattern="{name}"
).assign_coords(layer=1)
main_bot_l1 = imod.idf.open(
    "../model_v3.3.0/dbase/riv/both_w_l1.idf", pattern="{name}"
).assign_coords(layer=1)
is_main_l1 = main_cond_l1.notnull() & main_stage_l1.notnull() & main_bot_l1.notnull()
main_cond_l1 = main_cond_l1.where(is_main_l1).where(is_active)
main_stage_l1 = main_stage_l1.where(is_main_l1).where(is_active)
main_bot_l1 = main_bot_l1.where(is_main_l1).where(is_active)

main_cond_l2 = imod.idf.open(
    "../model_v3.3.0/dbase/riv/COND_HL2_250.IDF", pattern="{name}"
).assign_coords(layer=2)
main_stage_l2 = imod.idf.open(
    "../model_v3.3.0/dbase/riv/PEILH_hws_stat_1996_2006.idf", pattern="{name}"
).assign_coords(layer=2)
main_bot_l2 = imod.idf.open(
    "../model_v3.3.0/dbase/riv/both_w_l2.idf", pattern="{name}"
).assign_coords(layer=2)
is_main_l2 = main_cond_l2.notnull() & main_stage_l2.notnull() & main_bot_l2.notnull()
main_cond_l2 = main_cond_l2.where(is_main_l2).where(is_active)
main_stage_l2 = main_stage_l2.where(is_main_l2).where(is_active)
main_bot_l2 = main_bot_l2.where(is_main_l2).where(is_active)

main_cond = xr.concat([main_cond_l1, main_cond_l2], dim="layer")
main_stage = xr.concat([main_stage_l1, main_stage_l2], dim="layer")
main_bot = xr.concat([main_bot_l1, main_bot_l2], dim="layer")

# primary waterways
primary_cond = imod.idf.open(
    "../model_v3.3.0/dbase/riv/COND_P_L0.IDF", pattern="{name}"
).assign_coords(layer=1)
primary_stage = imod.idf.open(
    "../model_v3.3.0/dbase/riv/PEIL_P1_stat_250.IDF", pattern="{name}"
).assign_coords(layer=1)
primary_bot = imod.idf.open(
    "../model_v3.3.0/dbase/riv/BODH_P1W_250.IDF", pattern="{name}"
).assign_coords(layer=1)
is_primary = primary_cond.notnull() & primary_stage.notnull() & primary_bot.notnull()
primary_cond = primary_cond.where(is_primary).where(is_active)
primary_stage = primary_stage.where(is_primary).where(is_active)
primary_bot = primary_bot.where(is_primary).where(is_active)

# secondary waterways
secondary_cond = imod.idf.open(
    "../model_v3.3.0/dbase/riv/COND_S_L0.IDF", pattern="{name}"
).assign_coords(layer=1)
secondary_stage = imod.idf.open(
    "../model_v3.3.0/dbase/riv/PEIL_S1_stat_250.IDF", pattern="{name}"
).assign_coords(layer=1)
secondary_bot = imod.idf.open(
    "../model_v3.3.0/dbase/riv/BODH_S1W_250.IDF", pattern="{name}"
).assign_coords(layer=1)
is_secondary = (
    secondary_cond.notnull() & secondary_stage.notnull() & secondary_bot.notnull()
)
secondary_cond = secondary_cond.where(is_secondary).where(is_active)
secondary_stage = secondary_stage.where(is_secondary).where(is_active)
secondary_bot = secondary_bot.where(is_secondary).where(is_active)

# tertiary waterways, bottom is stage, only drainage!
tertiary_cond = imod.idf.open(
    "../model_v3.3.0/dbase/riv/COND_T_L0.IDF", pattern="{name}"
).assign_coords(layer=1)
tertiary_stage = imod.idf.open(
    "../model_v3.3.0/dbase/riv/PEIL_T1_stat_250.IDF", pattern="{name}"
).assign_coords(layer=1)
tertiary_bot = imod.idf.open(
    "../model_v3.3.0/dbase/riv/PEIL_T1_stat_250.IDF", pattern="{name}"
).assign_coords(layer=1)
is_tertiary = (
    tertiary_cond.notnull() & tertiary_stage.notnull() & tertiary_bot.notnull()
)
tertiary_cond = tertiary_cond.where(is_tertiary).where(is_active)
tertiary_stage = tertiary_stage.where(is_tertiary).where(is_active)
tertiary_bot = tertiary_bot.where(is_tertiary).where(is_active)

# boils
boils_cond = imod.idf.open(
    "../model_v3.3.0/dbase/riv/COND_WEL.IDF", pattern="{name}"
).assign_coords(layer=2)
boils_stage = imod.idf.open(
    "../model_v3.3.0/dbase/riv/PEIL_stat_WEL.IDF", pattern="{name}"
).assign_coords(layer=2)
boils_bot = imod.idf.open(
    "../model_v3.3.0/dbase/riv/BODH_WEL.IDF", pattern="{name}"
).assign_coords(layer=2)
is_boils = boils_cond.notnull() & boils_stage.notnull() & boils_bot.notnull()
boils_cond = boils_cond.where(is_boils).where(is_active)
boils_stage = boils_stage.where(is_boils).where(is_active)
boils_bot = boils_bot.where(is_boils).where(is_active)


# starting head
shd = imod.idf.open("../model_v3.3.0/dbase/starting_heads/*.IDF").where(is_active)

# constant head
constant_head = shd.where(ibound == -1).where(is_active)

# Groundwater model
lhm = imod.mf6.GroundwaterFlowModel()

lhm["dis"] = imod.mf6.StructuredDiscretization(
    top=wvp_top.sel(layer=1), bottom=wvp_bot, idomain=idomain
)
lhm["npf"] = imod.mf6.NodePropertyFlow(icelltype=0, k=kh, k33=kv)
lhm["ic"] = imod.mf6.InitialConditions(head=shd)
lhm["chd-edges"] = imod.mf6.ConstantHead(constant_head)
lhm["ghb"] = imod.mf6.GeneralHeadBoundary(head=ghb_head, conductance=ghb_cond)
lhm["drn-overland"] = imod.mf6.Drainage(
    elevation=drn_overland_elev, conductance=drn_overland_cond
)
lhm["drn-greppel"] = imod.mf6.Drainage(
    elevation=drn_greppel_elev, conductance=drn_greppel_cond
)
lhm["drn-buis"] = imod.mf6.Drainage(elevation=drn_buis_elev, conductance=drn_buis_cond)
lhm["riv-main"] = imod.mf6.River(
    conductance=main_cond, stage=main_stage, bottom_elevation=main_bot
)
lhm["riv-primary"] = imod.mf6.River(
    conductance=primary_cond, stage=primary_stage, bottom_elevation=primary_bot
)
lhm["riv-secondary"] = imod.mf6.River(
    conductance=secondary_cond, stage=secondary_stage, bottom_elevation=secondary_bot
)
lhm["riv-tertiary"] = imod.mf6.River(
    conductance=tertiary_cond, stage=tertiary_stage, bottom_elevation=tertiary_bot
)
lhm["riv-boils"] = imod.mf6.River(
    conductance=boils_cond, stage=boils_stage, bottom_elevation=boils_bot
)

simulation = imod.mf6.Modflow6Simulation("LHM-3.3.0-stationary")
simulation["lhm"] = lhm
simulation["ims"] = imod.mf6.Solution(
    print_option="summary",
    csv_output=False,
    no_ptc=True,
    outer_hclose=0.001,
    outer_maximum=150,
    under_relaxation=None,
    inner_hclose=0.001,
    inner_rclose=100.0,
    inner_maximum=30,
    linear_acceleration="cg",
    scaling_method=None,
    reordering_method=None,
    relaxation_factor=0.98,
)
simulation.time_discretization(times=["2000-01-01", "2000-01-02"])


simulation.write("test-LHM")

imod.idf.save("idomain/idomain", idomain)
imod.idf.save("top/top", wvp_top)
imod.idf.save("bot/bot", wvp_bot)
