from typing import TextIO

import numpy as np
import pandas as pd
import xarray as xr

from imod.common.interfaces.iregridpackage import IRegridPackage
from imod.mf6.dis import StructuredDiscretization
from imod.mf6.mf6_wel_adapter import Mf6Wel
from imod.msw.fixed_format import VariableMetaData
from imod.msw.pkgbase import MetaSwapPackage
from imod.msw.regrid.regrid_schemes import (
    SprinklingPointsRegridMethod,
    SprinklingRegridMethod,
)
from imod.msw.utilities.common import concat_imod5
from imod.msw.utilities.imod5_converter import (
    get_cell_area_from_imod5_data,
)
from imod.typing import GridDataDict, Imod5DataDict, IntArray
from imod.typing.grid import zeros_like


def _ravel_per_subunit(da: xr.DataArray) -> np.ndarray:
    # per defined well element, all subunits
    array_out = da.to_numpy().ravel()
    # per defined well element, per defined subunits
    return array_out[np.isfinite(array_out)]


def _sprinkling_data_from_imod5_ipf(cap_data: GridDataDict) -> GridDataDict:
    raise NotImplementedError(
        "Assigning sprinkling wells with an IPF file is not supported, please specify them as IDF."
    )


def _sprinkling_data_from_imod5_grid(cap_data: GridDataDict) -> GridDataDict:
    # Convert units from mm/d to m3/d
    msw_area = get_cell_area_from_imod5_data(cap_data)
    capacity_mmd = cap_data["artificial_recharge_capacity"]
    capacity_m3d = capacity_mmd * 1e-3 * msw_area.sel(subunit=0, drop=True)

    artificial_rch_type = cap_data["artificial_recharge"]
    from_groundwater = artificial_rch_type == 1
    from_surfacewater = artificial_rch_type == 2
    is_active = artificial_rch_type != 0

    zero_where_active = zeros_like(artificial_rch_type).where(is_active)

    # Add zero where active, to have active cells set to 0.0.
    max_abstraction_groundwater_rural = zero_where_active.where(
        ~from_groundwater, capacity_m3d
    )
    max_abstraction_surfacewater_rural = zero_where_active.where(
        ~from_surfacewater, capacity_m3d
    )

    # No sprinkling for urban environments
    max_abstraction_urban = zero_where_active

    data = {}
    data["max_abstraction_groundwater"] = concat_imod5(
        max_abstraction_groundwater_rural, max_abstraction_urban
    )
    data["max_abstraction_surfacewater"] = concat_imod5(
        max_abstraction_surfacewater_rural, max_abstraction_urban
    )
    return data


def _extract_indexer_for_svat(df: pd.DataFrame, columns: list[str]):
    """
    Get the indexer for a dataframe of wells to select the SVAT subunit for each
    well based on its row/col location in the model grid.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the wells with columns "subunit", "row",
        and "column". "row" and "column" are 1-based indices.
    columns : list[str]
        List of column names to use for indexing. Must include "row" and "column".

    Returns
    -------
    np.ndarray
        Indexer array for selecting SVAT subunits from svat_da.
    """
    if not {"row", "column"}.issubset(columns):
        raise ValueError("columns must contain 'row' and 'column'")
    df.loc[:, ["row", "column"]] -= 1  # Convert to 0-based indexing for xarray

    indexer = df.loc[:, columns].to_numpy()
    return indexer.T


def _replicate_dataframe_by_subunit(
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
    subunit_nrs = [0, 1]
    df_ls = [df.assign(**{subunit_col: subunit_nr}) for subunit_nr in subunit_nrs]
    return pd.concat(df_ls, ignore_index=True)


def _get_mf6_cellid_dataframe(mf6_well: Mf6Wel) -> pd.DataFrame:
    """
    Get cellids from the Mf6Wel objects dataset and convert to a dataframe for
    easy merging with sprinkling data.
    """
    # Promote id to dim to join datasets
    mf6_well_ds = mf6_well.dataset.set_coords("id").swap_dims({"ncellid": "id"})
    # Convert the cellid DataArray to a broad table for easier manipulation.
    mf6_cellid_df = mf6_well_ds["cellid"].to_dataset("dim_cellid").to_dataframe()
    # Select only the cellid columns we need and reset index to promote id to column
    # for merging
    dim_cellid = ["layer", "row", "column"]
    mf6_cellid_df = mf6_cellid_df.loc[:, dim_cellid].reset_index()
    return mf6_cellid_df


def _make_sprinkling_well_points_dataframe(
    sprinkling_dataset: xr.Dataset, mf6_cellid_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create a dataframe of sprinkling well points from the sprinkling dataset and
    merge it with the mf6_cellid_df to get the row/col of each well.
    """
    # Get point data from sprinkling dataset and convert to dataframe for easy merging
    points_keys = [
        key for key, da in sprinkling_dataset.data_vars.items() if "id" in da.dims
    ]
    sprinkling_points_df = (
        sprinkling_dataset[points_keys].drop_vars(["dx", "dy"]).to_dataframe()
    )
    # Merge again to confine to wells actually used in the modflow6 model.
    # This drops points that are outside model domain.
    return sprinkling_points_df.reset_index().merge(
        mf6_cellid_df, on="id", how="right", validate="many_to_one"
    )


def _merge_sprinkling_points_with_grids(
    points_df: pd.DataFrame, svat: xr.DataArray, sprinkling_id_grid: xr.DataArray
) -> pd.DataFrame:
    """
    Merge sprinkling points with SVAT grids.
    """

    # TODO: Rename "id_msw" and "id2grid_p" to something clearer like "id_sprinkling"
    # Flatten id_msw grid → (y, x, id_msw) table, drop cells with no well
    grids = xr.merge([sprinkling_id_grid, svat])
    art_df = grids.to_dataframe().reset_index().query("(id_msw > 0) & (svat > 0)")
    # Drop unnecessary columns. We preserve the x, y coords as they might
    # prove useful for debugging.
    art_df = art_df.drop(["dx", "dy"], axis=1)

    # Join: each SVAT cell gets the matching well row(s) from arl_points
    return art_df.merge(
        points_df,  # brings id back as a column
        left_on="id_msw",
        right_on="id2grid_p",
        how="inner",
        validate="many_to_one",
    )


def align_svat_with_dis(
    svat: xr.DataArray, dis_pkg: StructuredDiscretization
) -> xr.DataArray:
    """
    Align the SVAT grid with the dis_pkg grid as the SVAT grid might be smaller.
    """
    idomain_flat = dis_pkg.dataset["idomain"].isel(layer=0, drop=True)
    # Assign to _dummy instead of _ to avoid MyPy 2.3 crashing on the next line.
    # See https://github.com/python/mypy/issues/21824
    _dummy, svat_aligned = xr.align(idomain_flat, svat, join="left")
    return svat_aligned


def _get_svat_groundwater_for_wells(
    msw_mf6_sprinkling_df: pd.DataFrame, svat_aligned: xr.DataArray
) -> np.ndarray:
    """
    Get the SVAT subunit for each well from the SVAT grid based on the
    well's row/col location.
    """
    indexer = _extract_indexer_for_svat(
        msw_mf6_sprinkling_df, columns=["subunit", "row", "column"]
    )
    svat_groundwater = svat_aligned.data[*indexer]
    return svat_groundwater.astype(int)


def _get_wells_outside_art_grid_dataframe(
    mf6_cellid_df: pd.DataFrame,
    points_df: pd.DataFrame,
    msw_mf6_sprinkling_df: pd.DataFrame,
    svat_aligned: xr.DataArray,
) -> pd.DataFrame:
    """
    Deal with edge case: wells that are outside art_grid, but in model domain.
    These will be assigned to surfacewater abstraction. We can identify these by
    checking which wells in mf6_cellid_df are not in msw_mf6_merged_df.
    """
    is_outside_art = ~mf6_cellid_df["id"].isin(msw_mf6_sprinkling_df["id"].unique())
    outside_df = points_df.loc[is_outside_art, ["layer", "capacity"]]
    outside_df = _replicate_dataframe_by_subunit(outside_df)
    # Select the SVAT subunit for these wells based on their row/col location.
    cellid_outside_df = mf6_cellid_df.loc[is_outside_art]
    cellid_outside_df = _replicate_dataframe_by_subunit(cellid_outside_df)
    indexer_outside = _extract_indexer_for_svat(
        cellid_outside_df, columns=["subunit", "row", "column"]
    )
    svat_outside = svat_aligned.data[*indexer_outside]
    outside_df["svat_groundwater"] = svat_outside.astype(int)
    outside_df["svat"] = svat_outside.astype(int)
    # Set capacity to surfacewater abstraction, and set groundwater abstraction to 0.
    outside_df = outside_df.rename(columns={"capacity": "max_abstraction_surfacewater"})
    outside_df["max_abstraction_groundwater"] = 0.0
    # drop subunit column as it is no longer needed
    outside_df = outside_df.drop(columns=["subunit"])
    # drop wells that are outside the active metaswap model domain (svat = 0)
    return outside_df.query("svat > 0").reset_index(drop=True)


class Sprinkling(MetaSwapPackage, IRegridPackage):
    """
    This contains the sprinkling capacities of links between SVAT units and
    groundwater/surface water locations. Input is provided as grids for the
    maximum abstraction of groundwater and surfacewater to SVAT units. To
    specify the sprinkling capacity as points, see
    :class:`imod.msw.SprinklingPoints`.

    This class is responsible for the file `scap_svat.inp`

    Parameters
    ----------
    max_abstraction_groundwater: array of floats (xr.DataArray)
        Describes the maximum abstraction of groundwater to SVAT units in m3 per
        day. This array must not have a subunit coordinate.
    max_abstraction_surfacewater: array of floats (xr.DataArray)
        Describes the maximum abstraction of surfacewater to SVAT units in m3
        per day. This array must not have a subunit coordinate.
    """

    _file_name = "scap_svat.inp"
    _metadata_dict = {
        "svat": VariableMetaData(10, 1, 99999999, int),
        "max_abstraction_groundwater_mm_d": VariableMetaData(8, None, None, str),
        "max_abstraction_surfacewater_mm_d": VariableMetaData(8, None, None, str),
        "max_abstraction_groundwater": VariableMetaData(8, 0.0, 1e9, float),
        "max_abstraction_surfacewater": VariableMetaData(8, 0.0, 1e9, float),
        "svat_groundwater": VariableMetaData(10, 1, 99999999, int),
        "layer": VariableMetaData(6, 1, 9999, int),
        "trajectory": VariableMetaData(10, None, None, str),
    }

    _with_subunit = (
        "max_abstraction_groundwater",
        "max_abstraction_surfacewater",
    )
    _without_subunit = ()

    _to_fill = (
        "max_abstraction_groundwater_mm_d",
        "max_abstraction_surfacewater_mm_d",
        "trajectory",
    )

    _regrid_method = SprinklingRegridMethod()

    def __init__(
        self,
        max_abstraction_groundwater: xr.DataArray,
        max_abstraction_surfacewater: xr.DataArray,
    ):
        super().__init__()
        self.dataset["max_abstraction_groundwater"] = max_abstraction_groundwater
        self.dataset["max_abstraction_surfacewater"] = max_abstraction_surfacewater

        self._pkgcheck()

    def _render(
        self,
        file: TextIO,
        index: IntArray,
        svat: xr.DataArray,
        mf6_dis: StructuredDiscretization,
        mf6_well: Mf6Wel,
    ):
        if not isinstance(mf6_well, Mf6Wel):
            raise TypeError(rf"well not of type 'Mf6Wel', got '{type(mf6_well)}'")

        well_cellid = mf6_well["cellid"]

        well_layer = well_cellid.sel(dim_cellid="layer").data
        well_row = well_cellid.sel(dim_cellid="row").data - 1
        well_column = well_cellid.sel(dim_cellid="column").data - 1

        max_rate = (
            self.dataset["max_abstraction_groundwater"]
            + self.dataset["max_abstraction_surfacewater"]
        )
        max_rate_per_svat = max_rate.where(svat > 0)
        well_layer_per_svat = xr.full_like(max_rate_per_svat, np.nan)
        well_layer_per_svat.values[:, well_row, well_column] = well_layer

        is_active_per_svat = (max_rate_per_svat > 0) & well_layer_per_svat.notnull()

        layer_active = well_layer_per_svat.where(is_active_per_svat)
        layer_source = _ravel_per_subunit(layer_active).astype(dtype=np.int32)
        svat_active = svat.where(is_active_per_svat)
        svat_source_target = _ravel_per_subunit(svat_active).astype(dtype=np.int32)

        data_dict: dict[str, str | np.ndarray] = {
            "svat": svat_source_target,
            "layer": layer_source,
            "svat_groundwater": svat_source_target,
        }

        for var in self._with_subunit:
            data_with_well = self.dataset[var].where(is_active_per_svat)
            data_dict[var] = _ravel_per_subunit(data_with_well)

        for var in self._to_fill:
            data_dict[var] = ""

        dataframe = pd.DataFrame(
            data=data_dict, columns=list(self._metadata_dict.keys())
        )

        self._check_range(dataframe)

        return self._write_dataframe_fixed_width(file, dataframe)

    @classmethod
    def from_imod5_data(cls, imod5_data: Imod5DataDict) -> "Sprinkling":
        """
        Import sprinkling data from imod5 data. Abstraction data for sprinkling
        is defined in iMOD5 either with grids (IDF) or points (IPF) combined
        with a grid. Depending on the type, the method does different conversions:

        - grids (IDF)
            The ``"artifical_recharge_layer"`` variable was defined as grid
            (IDF), this grid defines in which layer a groundwater abstraction
            well should be placed. The ``"artificial_recharge"`` grid contains
            types which point to the type of abstraction:

                * 0: no abstraction
                * 1: groundwater abstraction
                * 2: surfacewater abstraction

            The ``"artificial_recharge_capacity"`` grid/constant defines the
            capacity of each groundwater or surfacewater abstraction. This is an
            ``1:1`` mapping: Each grid cell maps to a separate well.

        - points with grid (IPF & IDF)
            The ``"artifical_recharge_layer"`` variable was defined as point
            data (IPF), this table contains wellids with an abstraction capacity
            and layer. The ``"artificial_recharge"`` grid contains a mapping of
            grid cells to wellids in the point data. The
            ``"artificial_recharge_capacity"`` is ignored as the abstraction
            capacity is already defined in the point data. This is an ``n:1``
            mapping: multiple grid cells can map to one well.

        Parameters
        ----------
        imod5_data: dict[str, dict[str, GridDataArray]]
            dictionary containing the arrays mentioned in the project file as
            xarray datasets, under the key of the package type to which it
            belongs, as returned by
            :func:`imod.formats.prj.open_projectfile_data`.

        Returns
        -------
        Sprinkling package
        """
        cap_data = imod5_data["cap"]
        if isinstance(cap_data["artificial_recharge_layer"], pd.DataFrame):
            data = _sprinkling_data_from_imod5_ipf(cap_data)
        else:
            data = _sprinkling_data_from_imod5_grid(cap_data)

        return cls(**data)


class SprinklingPoints(MetaSwapPackage, IRegridPackage):
    """
    This contains the sprinkling capacities of links between SVAT units and
    groundwater/surface water locations. This class is capable of handling point
    data (IPF) for sprinkling wells, which is a mapping of grid cells to well
    locations. To specify the sprinkling capacity as grid, see
    :class:`imod.msw.Sprinkling`.

    This class is responsible for the file `scap_svat.inp`

    Parameters
    ----------
    art_grid: xr.DataArray
        Grid of the artificial recharge types, with subunit coordinate.
    x_p: np.ndarray | list[float]
        x-coordinates of the artificial recharge locations.
    y_p: np.ndarray | list[float]
        y-coordinates of the artificial recharge locations.
    layer_p: np.ndarray | list[int]
        layer indices of the artificial recharge locations.
    id2grid_p: np.ndarray | list[int]
        mapping of the artificial recharge locations to the grid cells.
    capacity_p: np.ndarray | list[float]
        abstraction capacities of the artificial recharge locations.

    """

    _file_name = "scap_svat.inp"
    _metadata_dict = {
        "svat": VariableMetaData(10, 1, 99999999, int),
        "max_abstraction_groundwater_mm_d": VariableMetaData(8, None, None, str),
        "max_abstraction_surfacewater_mm_d": VariableMetaData(8, None, None, str),
        "max_abstraction_groundwater": VariableMetaData(8, 0.0, 1e9, float),
        "max_abstraction_surfacewater": VariableMetaData(8, 0.0, 1e9, float),
        "svat_groundwater": VariableMetaData(10, 1, 99999999, int),
        "layer": VariableMetaData(6, 1, 9999, int),
        "trajectory": VariableMetaData(10, None, None, str),
    }

    _with_subunit = (
        "max_abstraction_groundwater",
        "max_abstraction_surfacewater",
    )
    _without_subunit = ()

    _to_fill = (
        "max_abstraction_groundwater_mm_d",
        "max_abstraction_surfacewater_mm_d",
        "trajectory",
    )

    _regrid_method = SprinklingPointsRegridMethod()

    def __init__(
        self,
        art_grid: xr.DataArray,
        x_p: np.ndarray | list[float],
        y_p: np.ndarray | list[float],
        layer_p: np.ndarray | list[int],
        id2grid_p: np.ndarray | list[int],
        capacity_p: np.ndarray | list[float],
    ):
        super().__init__()
        # Replicate well ids as they were also created in
        # imod.mf6.LayeredWell.from_imod5_cap_data()
        id_index = pd.Index(range(len(x_p)), name="id").astype(str)
        points_ds = xr.Dataset(
            {
                "x_p": (("id",), x_p),
                "y_p": (("id",), y_p),
                "layer_p": (("id",), layer_p),
                "id2grid_p": (("id",), id2grid_p),
                "capacity_p": (("id",), capacity_p),
            },
            coords={"id": id_index},
        )
        art_grid = art_grid.rename("id_msw")
        self.dataset = xr.merge([art_grid, points_ds])

    @classmethod
    def from_imod5_data(cls, imod5_data: Imod5DataDict) -> "SprinklingPoints":
        cap_data = imod5_data["cap"]
        art_grid = cap_data["artificial_recharge"]
        df_points = cap_data["artificial_recharge_layer"]

        arl_points = df_points.iloc[:, :5]
        arl_points.columns = ["x_p", "y_p", "layer_p", "id2grid_p", "capacity_p"]
        # Enforce dtypes
        arl_points = arl_points.astype(
            {
                "x_p": float,
                "y_p": float,
                "layer_p": int,
                "id2grid_p": int,
                "capacity_p": float,
            }
        )

        return cls(
            art_grid=art_grid,
            x_p=arl_points["x_p"].to_numpy(),
            y_p=arl_points["y_p"].to_numpy(),
            layer_p=arl_points["layer_p"].to_numpy(),
            id2grid_p=arl_points["id2grid_p"].to_numpy(),
            capacity_p=arl_points["capacity_p"].to_numpy(),
        )

    def _render(self, file, index, svat, mf6_dis, mf6_well):
        """
        Render the sprinkling points to the scap_svat.inp file.

        This method first merges the sprinkling points with the mf6_well cellids
        to get the row/col of each well, then merges the sprinkling points with
        the svat and id_msw grid. It then selects the columns that need to be
        written to scap_svat.inp and sets wells with layer > 0 to groundwater
        abstraction, and wells with layer = 0 to surfacewater abstraction.
        Finally, it deals with edge cases for wells that are outside art_grid
        but in the model domain, and writes the dataframe to the file.
        """
        # Merge the sprinkling points with the mf6_well cellids to get the
        # row/col of each well.
        mf6_cellid_df = _get_mf6_cellid_dataframe(mf6_well)
        points_df = _make_sprinkling_well_points_dataframe(self.dataset, mf6_cellid_df)
        # Merge the sprinkling points with the svat and id_msw grid
        msw_mf6_sprinkling_df = _merge_sprinkling_points_with_grids(
            points_df, svat, self.dataset["id_msw"]
        )
        svat_aligned = align_svat_with_dis(svat, mf6_dis)
        msw_mf6_sprinkling_df["svat_groundwater"] = _get_svat_groundwater_for_wells(
            msw_mf6_sprinkling_df, svat_aligned
        )

        # Select columns that need to be written to scap_svat.inp
        dataframe = msw_mf6_sprinkling_df[["svat", "layer", "svat_groundwater"]]
        dataframe["svat"] = dataframe["svat"].astype(int)
        capacity = msw_mf6_sprinkling_df["capacity"]
        # Set wells with layer > 0 to groundwater abstraction, and wells with layer = 0
        # to surfacewater abstraction.
        is_gw_extraction = msw_mf6_sprinkling_df["layer"] > 0
        dataframe["max_abstraction_groundwater"] = capacity.where(is_gw_extraction, 0.0)
        dataframe["max_abstraction_surfacewater"] = capacity.where(
            ~is_gw_extraction, 0.0
        )

        # TODO: Make sure wells in svats are all present in the dataframe. If
        #   these svats are 0 in the art_grid, they should get a 0.0 capacity.

        # Deal with edge case: wells that are outside art_grid, but in model domain.
        # These will be assigned to surfacewater abstraction.
        outside_df = _get_wells_outside_art_grid_dataframe(
            mf6_cellid_df, points_df, msw_mf6_sprinkling_df, svat_aligned
        )

        dataframe_out = pd.concat([dataframe, outside_df], axis=0, ignore_index=True)
        dataframe_out = dataframe_out.sort_values(by=["svat"]).reset_index(drop=True)

        for var in self._to_fill:
            dataframe_out[var] = ""

        self._check_range(dataframe_out)

        return self._write_dataframe_fixed_width(file, dataframe_out)
