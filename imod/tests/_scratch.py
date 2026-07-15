# %%
import imod
import xarray as xr
from imod.util.dims import drop_layer_dim_cap_data


# %%
SUBUNIT = 0

df_points = imod.ipf.read(r"c:\Users\engelen\projects_wdir\imod-python\imod5_converter\NHI_sprint\Peelvenen\BASIS7\METASWAP\grid\Sprinkling\BEREGEN_LOC.IPF")
art_grid = imod.idf.open(r"c:\Users\engelen\projects_wdir\imod-python\imod5_converter\NHI_sprint\Peelvenen\BASIS7\METASWAP\grid\Sprinkling\BEREGENINGS_LOCATIES.IDF").compute().astype(int)

imod5_data = {"cap": {"artificial_recharge": art_grid, "artificial_recharge_layer": df_points}}

well = imod.mf6.LayeredWell.from_imod5_cap_data(
    imod5_data, target_dis=None, regridder_types=None, regrid_cache=None
)
# %%
prj_data, period_data = imod.formats.prj.open_projectfile_data(r"c:\Users\engelen\projects_wdir\imod-python\imod5_converter\NHI_sprint\Peelvenen\prjfiles\Peelvenen_relative_paths_dis_npf.PRJ")

# %%
dis_pkg = imod.mf6.StructuredDiscretization.from_imod5_data(prj_data, validate=False)
dis_pkg["idomain"] = dis_pkg["idomain"].clip(min=0)
npf_pkg = imod.mf6.NodePropertyFlow.from_imod5_data(prj_data, dis_pkg.dataset["idomain"])

prj_data = drop_layer_dim_cap_data(prj_data)
griddata, msw_active = imod.msw.GridData.from_imod5_data(prj_data, dis_pkg)
# %%
# Convert to args of Sprinkling._render()
mf6_well = well.to_mf6_pkg(dis_pkg["idomain"], dis_pkg["top"], dis_pkg["bottom"], npf_pkg["k"])
isactive_1d, svat = griddata.generate_isactive_svat_arrays()

# %%
# In from_imod5_cap_data
arl_points = df_points.iloc[:, :5]
arl_points.columns = ["x_p", "y_p", "layer_p", "id2grid_p", "capacity"]
# Enforce dtypes
arl_points = arl_points.astype({"x_p": float, "y_p": float, "layer_p": int, "id2grid_p": int, "capacity": float})
arl_points["id"] = arl_points.index.astype(str)
arl_points = arl_points.set_index("id")

points_ds = arl_points.to_xarray()
art_grid = art_grid.rename("id_msw")
dataset = xr.merge([art_grid, points_ds])

# %%
# Flatten id_msw grid → (y, x, id_msw) table, drop cells with no well
# TODO: Verify with Peter that only the first subunit (landuse: agriculture) is
#   relevant for the SVAT mapping.
grids = xr.merge([art_grid, svat.sel(subunit=SUBUNIT, drop=True)])

art_df = (
    grids
    .to_dataframe()
    .reset_index()
    .query("(id_msw > 0) & (svat > 0)")
)
# Drop unnecessary columns. We preserve the x, y coords as they might prove
# useful for debugging.
art_df = art_df.drop(["dx", "dy"], axis=1)

# Join: each SVAT cell gets the matching well row(s) from arl_points
msw_merged_df = art_df.merge(
    arl_points.reset_index(),   # brings id back as a column
    left_on="id_msw",
    right_on="id2grid_p",
    how="inner",
    validate="many_to_one"
)

# %%
# Promote id to dim to join datasets
mf6_well_ds = mf6_well.dataset.set_coords("id").swap_dims({"ncellid": "id"})
# Convert the cellid DataArray to a broad table for easier manipulation.
mf6_cellid_df = mf6_well_ds["cellid"].to_dataset("dim_cellid").to_dataframe()
# Select only the cellid columns we need and reset index to promote id to column
# for merging
dim_cellid = ["layer", "row", "column"]
mf6_cellid_df = mf6_cellid_df.loc[:, dim_cellid].reset_index()
# Merge again to confine to wells actually used in the modflow6 model.
# This drops points that are not in art_grid.
msw_mf6_merged_df = msw_merged_df.merge(
    mf6_cellid_df,
    on="id",
    how="inner",
    validate="many_to_one"
)

# %% 
# Derive the SVAT subunit for each well from the SVAT grid based on the
# well's row/col location. 
indexer = msw_mf6_merged_df.loc[:, dim_cellid].to_numpy() - 1
# We need to align the SVAT grid with the dis_pkg grid as the SVAT grid might be
# smaller.
idomain_flat = dis_pkg.dataset["idomain"].isel(layer=0, drop=True)
_, svat_aligned = xr.align(idomain_flat, svat, join="left")
svat_groundwater = svat_aligned.data[SUBUNIT, indexer[:, 1], indexer[:, 2]]

msw_mf6_merged_df["svat_groundwater"] = svat_groundwater.astype(int)

# %%
# Deal with edge case: wells that are outside art_grid, but in model domain
outside_art = ~mf6_cellid_df["id"].isin(msw_mf6_merged_df["id"].unique())
indexer_outside = mf6_cellid_df.loc[outside_art, dim_cellid].to_numpy() - 1
svat_outside = svat_aligned.data[SUBUNIT, indexer_outside[:, 1], indexer_outside[:, 2]]

# %%
# Select columns that need to be written to scap_svat.inp
dataframe = msw_mf6_merged_df[["svat", "layer", "svat_groundwater"]]
dataframe["svat"] = dataframe["svat"].astype(int)
capacity = msw_mf6_merged_df["capacity"]
is_gw_extraction = msw_mf6_merged_df["layer"] > 0 
dataframe["max_abstraction_groundwater"] = capacity.where(is_gw_extraction, 0.0)
dataframe["max_abstraction_surfacewater"] = capacity.where(~is_gw_extraction, 0.0)

# Cases to catch:
# 1. Well is outside the model domain → surfacewater
# 2. Well is in layer 0 → surfacewater


