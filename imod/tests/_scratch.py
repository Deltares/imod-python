# %%
import pandas as pd
import xarray as xr

import imod
from imod.util.dims import drop_layer_dim_cap_data


def get_indexer(df: pd.DataFrame, columns: list[str]):
    """
    Get the indexer for a dataframe of wells to select the SVAT subunit for each
    well based on its row/col location in the model grid.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the wells with columns "subunit", "row",
        and "column" (1-based).

    Returns
    -------
    np.ndarray
        Indexer array for selecting SVAT subunits from svat_da.
    """
    if not set("row", "column").issubset(columns):
        raise ValueError("columns must contain 'row' and 'column'")
    df.loc[:, ["row", "column"]] -= 1  # Convert to 0-based indexing for xarray

    indexer = df.loc[:, columns].to_numpy()
    return indexer.T


def double_length_df_subunit(
    df: pd.DataFrame, subunit_col: str = "subunit"
) -> pd.DataFrame:
    """
    Duplicate the rows of a DataFrame for each subunit (0 and 1).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to duplicate.
    subunit_col : str, optional
        Name of the column to assign subunit values, by default "subunit".

    Returns
    -------
    pd.DataFrame
        DataFrame with duplicated rows for each subunit.
    """
    return pd.concat(
        [df.assign(**{subunit_col: 0}), df.assign(**{subunit_col: 1})],
        ignore_index=True,
    )


# %%
df_points = imod.ipf.read(
    r"c:\Users\engelen\projects_wdir\imod-python\imod5_converter\NHI_sprint\Peelvenen\BASIS7\METASWAP\grid\Sprinkling\BEREGEN_LOC.IPF"
)
art_grid = (
    imod.idf.open(
        r"c:\Users\engelen\projects_wdir\imod-python\imod5_converter\NHI_sprint\Peelvenen\BASIS7\METASWAP\grid\Sprinkling\BEREGENINGS_LOCATIES.IDF"
    )
    .compute()
    .astype(int)
)

imod5_data = {
    "cap": {"artificial_recharge": art_grid, "artificial_recharge_layer": df_points}
}

well = imod.mf6.LayeredWell.from_imod5_cap_data(
    imod5_data, target_dis=None, regridder_types=None, regrid_cache=None
)
# %%
# Setup to get example args for _render() of SprinklingPoints
prj_data, period_data = imod.formats.prj.open_projectfile_data(
    r"c:\Users\engelen\projects_wdir\imod-python\imod5_converter\NHI_sprint\Peelvenen\prjfiles\Peelvenen_relative_paths_dis_npf.PRJ"
)

dis_pkg = imod.mf6.StructuredDiscretization.from_imod5_data(prj_data, validate=False)
dis_pkg["idomain"] = dis_pkg["idomain"].clip(min=0)
npf_pkg = imod.mf6.NodePropertyFlow.from_imod5_data(
    prj_data, dis_pkg.dataset["idomain"]
)

prj_data = drop_layer_dim_cap_data(prj_data)
griddata, msw_active = imod.msw.GridData.from_imod5_data(prj_data, dis_pkg)
# Convert to args of Sprinkling._render()
mf6_well = well.to_mf6_pkg(
    dis_pkg["idomain"], dis_pkg["top"], dis_pkg["bottom"], npf_pkg["k"]
)
isactive_1d, svat = griddata.generate_isactive_svat_arrays()

# %%
# In from_imod5_cap_data
arl_points = df_points.iloc[:, :5]
arl_points.columns = ["x_p", "y_p", "layer_p", "id_sprinkling_p", "capacity"]
# Enforce dtypes
arl_points = arl_points.astype(
    {
        "x_p": float,
        "y_p": float,
        "layer_p": int,
        "id_sprinkling_p": int,
        "capacity": float,
    }
)
arl_points["id"] = arl_points.index.astype(str)
arl_points = arl_points.set_index("id")

points_ds = arl_points.to_xarray()

# in def __init__
art_grid = art_grid.rename("id_sprinkling")
dataset = xr.merge([art_grid, points_ds])

# %%
# In render()
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
points_mf6_merged_df = arl_points.reset_index().merge(
    mf6_cellid_df, on="id", how="right", validate="many_to_one"
)
# %%
# Flatten id_sprinkling grid → (y, x, id_sprinkling) table, drop cells with no well
grids = xr.merge([art_grid, svat])
art_df = grids.to_dataframe().reset_index().query("(id_sprinkling > 0) & (svat > 0)")
# Drop unnecessary columns. We preserve the x, y coords as they might prove
# useful for debugging.
art_df = art_df.drop(["dx", "dy"], axis=1)

# Join: each SVAT cell gets the matching well row(s) from arl_points
msw_mf6_merged_df = art_df.merge(
    points_mf6_merged_df,  # brings id back as a column
    left_on="id_sprinkling",
    right_on="id_sprinkling_p",
    how="inner",
    validate="many_to_one",
)

# %%
# Derive the SVAT subunit for each well from the SVAT grid based on the
# well's row/col location.
indexer = get_indexer(msw_mf6_merged_df, columns=["subunit", "row", "column"])
# We need to align the SVAT grid with the dis_pkg grid as the SVAT grid might be
# smaller.
idomain_flat = dis_pkg.dataset["idomain"].isel(layer=0, drop=True)
_, svat_aligned = xr.align(idomain_flat, svat, join="left")
svat_groundwater = svat_aligned.data[*indexer]

msw_mf6_merged_df["svat_groundwater"] = svat_groundwater.astype(int)

# %%
# Select columns that need to be written to scap_svat.inp
dataframe = msw_mf6_merged_df[["svat", "layer", "svat_groundwater"]]
dataframe["svat"] = dataframe["svat"].astype(int)
capacity = msw_mf6_merged_df["capacity"]
# Set wells with layer > 0 to groundwater abstraction, and wells with layer = 0
# to surfacewater abstraction.
is_gw_extraction = msw_mf6_merged_df["layer"] > 0
dataframe["max_abstraction_groundwater"] = capacity.where(is_gw_extraction, 0.0)
dataframe["max_abstraction_surfacewater"] = capacity.where(~is_gw_extraction, 0.0)

# %%
#
# Deal with edge case: wells that are outside art_grid, but in model domain.
# These will be assigned to surfacewater abstraction. We can identify these by
# checking which wells in mf6_cellid_df are not in msw_mf6_merged_df.
is_outside_art = ~mf6_cellid_df["id"].isin(msw_mf6_merged_df["id"].unique())
outside_df = points_mf6_merged_df.loc[is_outside_art, ["layer", "capacity"]]
outside_df = double_length_df_subunit(outside_df)
# Select the SVAT subunit for these wells based on their row/col location.
df_cellid_outside = mf6_cellid_df.loc[is_outside_art]
df_cellid_outside = double_length_df_subunit(df_cellid_outside)
indexer_outside = get_indexer(df_cellid_outside, columns=["subunit", "row", "column"])
svat_outside = svat_aligned.data[
    indexer_outside[0], indexer_outside[1], indexer_outside[2]
]
outside_df["svat_groundwater"] = svat_outside.astype(int)
outside_df["svat"] = svat_outside.astype(int)
# Set capacity to surfacewater abstraction, and set groundwater abstraction to 0.
outside_df = outside_df.rename(columns={"capacity": "max_abstraction_surfacewater"})
outside_df["max_abstraction_groundwater"] = 0.0
# drop subunit column as it is no longer needed
outside_df = outside_df.drop(columns=["subunit"])
# drop wells that are outside the active metaswap model domain (svat = 0)
outside_df = outside_df.query("svat > 0").reset_index(drop=True)

# %%
#
# Combine
dataframe_out = pd.concat([dataframe, outside_df], axis=0, ignore_index=True)
dataframe_out = dataframe_out.sort_values(by=["svat"]).reset_index(drop=True)
# %%
#
# TODO: Verify if iMOD5 SVAT grid the same as the one derived in this script.
