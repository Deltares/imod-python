"""Cleanup utilities"""

from enum import Enum
from typing import TYPE_CHECKING, Optional

import pandas as pd
import xarray as xr

from imod.mf6.utilities.mask import mask_arrays
from imod.prepare.hfb import clip_line_gdf_by_grid
from imod.prepare.wells import locate_wells, validate_well_columnnames
from imod.schemata import scalar_None
from imod.typing import GridDataArray
from imod.util.imports import MissingOptionalModule

if TYPE_CHECKING:
    import geopandas as gpd
else:
    try:
        import geopandas as gpd
    except ImportError:
        gpd = MissingOptionalModule("geopandas")


class AlignLevelsMode(Enum):
    TOPDOWN = 0
    BOTTOMUP = 1


def align_nodata(grids: dict[str, xr.DataArray]) -> dict[str, xr.DataArray]:
    return mask_arrays(grids)


def align_interface_levels(
    top: GridDataArray,
    bottom: GridDataArray,
    method: AlignLevelsMode = AlignLevelsMode.TOPDOWN,
) -> tuple[GridDataArray, GridDataArray]:
    to_align = top < bottom

    match method:
        case AlignLevelsMode.BOTTOMUP:
            return top.where(~to_align, bottom), bottom
        case AlignLevelsMode.TOPDOWN:
            return top, bottom.where(~to_align, top)
        case _:
            raise TypeError(f"Unmatched case for method, got {method}")


def _cleanup_robin_boundary(
    idomain: GridDataArray, grids: dict[str, GridDataArray]
) -> dict[str, GridDataArray]:
    """Cleanup robin boundary condition (i.e. bc with conductance)"""
    active = idomain > 0
    # Deactivate conductance cells outside active domain; this nodata
    # inconsistency will be aligned in the final call to align_nodata
    conductance = grids["conductance"].where(active)
    concentration = grids["concentration"]
    # Make conductance cells with erronous values inactive
    grids["conductance"] = conductance.where(conductance > 0.0)
    # Clip negative concentration cells to 0.0
    if (concentration is not None) and not scalar_None(concentration):
        grids["concentration"] = concentration.clip(min=0.0)
    else:
        grids.pop("concentration")

    # Align nodata
    return align_nodata(grids)


def cleanup_riv(
    idomain: GridDataArray,
    bottom: GridDataArray,
    stage: GridDataArray,
    conductance: GridDataArray,
    bottom_elevation: GridDataArray,
    concentration: Optional[GridDataArray] = None,
) -> dict[str, GridDataArray]:
    """
    Clean up river data, fixes some common mistakes causing ValidationErrors by
    doing the following:

    - Cells where conductance <= 0 are deactivated.
    - Cells where concentration < 0 are set to 0.0.
    - Cells outside active domain (idomain==1) are removed.
    - Align NoData: If one variable has an inactive cell in one cell, ensure
      this cell is deactivated for all variables.
    - River bottom elevations below model bottom of a layer are set to model
      bottom of that layer.
    - River bottom elevations which exceed river stage are lowered to river
      stage.

    Parameters
    ----------
    idomain: xarray.DataArray | xugrid.UgridDataArray
        MODFLOW 6 model domain. idomain==1 is considered active domain.
    bottom: xarray.DataArray | xugrid.UgridDataArray
        Grid with model bottoms
    stage: xarray.DataArray | xugrid.UgridDataArray
        Grid with river stages
    conductance: xarray.DataArray | xugrid.UgridDataArray
        Grid with conductances
    bottom_elevation: xarray.DataArray | xugrid.UgridDataArray
        Grid with river bottom elevations
    concentration: xarray.DataArray | xugrid.UgridDataArray, optional
        Optional grid with concentrations

    Returns
    -------
    dict[str, xarray.DataArray | xugrid.UgridDataArray]
        Dict of cleaned up grids. Has keys: "stage", "conductance",
        "bottom_elevation", "concentration".
    """
    # Output dict
    output_dict = {
        "stage": stage,
        "conductance": conductance,
        "bottom_elevation": bottom_elevation,
        "concentration": concentration,
    }
    output_dict = _cleanup_robin_boundary(idomain, output_dict)
    if (output_dict["stage"] < bottom).any():
        raise ValueError(
            "River stage below bottom of model layer, cannot fix this. "
            "Probably rivers are assigned to the wrong layer, you can reallocate "
            "river data to model layers with: "
            "``imod.prepare.topsystem.allocate_riv_cells``."
        )
    # Ensure bottom elevation above model bottom
    output_dict["bottom_elevation"], _ = align_interface_levels(
        output_dict["bottom_elevation"], bottom, AlignLevelsMode.BOTTOMUP
    )
    # Ensure stage above bottom_elevation
    output_dict["stage"], output_dict["bottom_elevation"] = align_interface_levels(
        output_dict["stage"], output_dict["bottom_elevation"], AlignLevelsMode.TOPDOWN
    )
    return output_dict


def cleanup_drn(
    idomain: GridDataArray,
    elevation: GridDataArray,
    conductance: GridDataArray,
    concentration: Optional[GridDataArray] = None,
) -> dict[str, GridDataArray]:
    """
    Clean up drain data, fixes some common mistakes causing ValidationErrors by
    doing the following:

    - Cells where conductance <= 0 are deactivated.
    - Cells where concentration < 0 are set to 0.0.
    - Cells outside active domain (idomain==1) are removed.
    - Align NoData: If one variable has an inactive cell in one cell, ensure
      this cell is deactivated for all variables.

    Parameters
    ----------
    idomain: xarray.DataArray | xugrid.UgridDataArray
        MODFLOW 6 model domain. idomain==1 is considered active domain.
    elevation: xarray.DataArray | xugrid.UgridDataArray
        Grid with drain elevations
    conductance: xarray.DataArray | xugrid.UgridDataArray
        Grid with conductances
    concentration: xarray.DataArray | xugrid.UgridDataArray, optional
        Optional grid with concentrations

    Returns
    -------
    dict[str, xarray.DataArray | xugrid.UgridDataArray]
        Dict of cleaned up grids. Has keys: "elevation", "conductance",
        "concentration".
    """
    # Output dict
    output_dict = {
        "elevation": elevation,
        "conductance": conductance,
        "concentration": concentration,
    }
    return _cleanup_robin_boundary(idomain, output_dict)


def cleanup_ghb(
    idomain: GridDataArray,
    head: GridDataArray,
    conductance: GridDataArray,
    concentration: Optional[GridDataArray] = None,
) -> dict[str, GridDataArray]:
    """
    Clean up general head boundary data, fixes some common mistakes causing
    ValidationErrors by doing the following:

    - Cells where conductance <= 0 are deactivated.
    - Cells where concentration < 0 are set to 0.0.
    - Cells outside active domain (idomain==1) are removed.
    - Align NoData: If one variable has an inactive cell in one cell, ensure
      this cell is deactivated for all variables.

    Parameters
    ----------
    idomain: xarray.DataArray | xugrid.UgridDataArray
        MODFLOW 6 model domain. idomain==1 is considered active domain.
    head: xarray.DataArray | xugrid.UgridDataArray
        Grid with heads
    conductance: xarray.DataArray | xugrid.UgridDataArray
        Grid with conductances
    concentration: xarray.DataArray | xugrid.UgridDataArray, optional
        Optional grid with concentrations

    Returns
    -------
    dict[str, xarray.DataArray | xugrid.UgridDataArray]
        Dict of cleaned up grids. Has keys: "head", "conductance",
        "concentration".
    """
    # Output dict
    output_dict = {
        "head": head,
        "conductance": conductance,
        "concentration": concentration,
    }
    return _cleanup_robin_boundary(idomain, output_dict)


def _locate_wells_in_bounds(
    wells: pd.DataFrame, top: GridDataArray, bottom: GridDataArray
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Locate wells in model bounds, wells outside bounds are dropped. Returned
    dataframes and series have well "id" as index.

    Returns
    -------
    wells_in_bounds: pd.DataFrame
        wells in model boundaries. Has "id" as index.
    xy_top_series: pd.Series
        model top at well xy location. Has "id" as index.
    xy_base_series: pd.Series
        model base at well xy location. Has "id" as index.
    """
    id_in_bounds, xy_top, xy_bottom, _ = locate_wells(
        wells, top, bottom, validate=False
    )
    xy_base_model = xy_bottom.isel(layer=-1, drop=True)

    # Assign id as coordinates
    xy_top = xy_top.assign_coords(id=("index", id_in_bounds))
    xy_base_model = xy_base_model.assign_coords(id=("index", id_in_bounds))
    # Create pandas dataframes/series with "id" as index.
    xy_top_series = xy_top.to_dataframe(name="top").set_index("id")["top"]
    xy_base_series = xy_base_model.to_dataframe(name="bottom").set_index("id")["bottom"]
    wells_in_bounds = wells.set_index("id").loc[id_in_bounds]
    return wells_in_bounds, xy_top_series, xy_base_series


def _clip_filter_screen_to_surface_level(
    cleaned_wells: pd.DataFrame, xy_top_series: pd.Series
) -> pd.DataFrame:
    cleaned_wells["screen_top"] = cleaned_wells["screen_top"].clip(upper=xy_top_series)
    return cleaned_wells


def _drop_wells_below_model_base(
    cleaned_wells: pd.DataFrame, xy_base_series: pd.Series
) -> pd.DataFrame:
    is_below_base = cleaned_wells["screen_top"] >= xy_base_series
    return cleaned_wells.loc[is_below_base]


def _clip_filter_bottom_to_model_base(
    cleaned_wells: pd.DataFrame, xy_base_series: pd.Series
) -> pd.DataFrame:
    cleaned_wells["screen_bottom"] = cleaned_wells["screen_bottom"].clip(
        lower=xy_base_series
    )
    return cleaned_wells


def _set_inverted_filters_to_point_filters(cleaned_wells: pd.DataFrame) -> pd.DataFrame:
    # Convert all filters where screen bottom exceeds screen top to
    # point filters
    cleaned_wells["screen_bottom"] = cleaned_wells["screen_bottom"].clip(
        upper=cleaned_wells["screen_top"]
    )
    return cleaned_wells


def _set_ultrathin_filters_to_point_filters(
    cleaned_wells: pd.DataFrame, minimum_thickness: float
) -> pd.DataFrame:
    not_ultrathin_layer = (
        cleaned_wells["screen_top"] - cleaned_wells["screen_bottom"]
    ) > minimum_thickness
    cleaned_wells["screen_bottom"] = cleaned_wells["screen_bottom"].where(
        not_ultrathin_layer, cleaned_wells["screen_top"]
    )
    return cleaned_wells


def cleanup_wel(
    wells: pd.DataFrame,
    top: GridDataArray,
    bottom: GridDataArray,
    minimum_thickness: float = 0.05,
) -> pd.DataFrame:
    """
    Clean up dataframe with wells, fixes some common mistakes in the following
    order:

    1. Wells outside grid bounds are dropped
    2. Filters above surface level are set to surface level
    3. Drop wells with filters entirely below base
    4. Clip filter screen_bottom to model base
    5. Clip filter screen_bottom to screen_top
    6. Well filters thinner than minimum thickness are made point filters

    Parameters
    ----------
    wells: pandas.Dataframe
        Dataframe with wells to be cleaned up. Requires columns ``"x", "y",
        "id", "screen_top", "screen_bottom"``
    top: xarray.DataArray | xugrid.UgridDataArray
        Grid with model top
    bottom: xarray.DataArray | xugrid.UgridDataArray
        Grid with model bottoms
    minimum_thickness: float
        Minimum thickness, filter thinner than this thickness are set to point
        filters

    Returns
    -------
    pandas.DataFrame
        Cleaned well dataframe.
    """
    validate_well_columnnames(
        wells, names={"x", "y", "id", "screen_top", "screen_bottom"}
    )

    cleaned_wells, xy_top_series, xy_base_series = _locate_wells_in_bounds(
        wells, top, bottom
    )
    cleaned_wells = _clip_filter_screen_to_surface_level(cleaned_wells, xy_top_series)
    cleaned_wells = _drop_wells_below_model_base(cleaned_wells, xy_base_series)
    cleaned_wells = _clip_filter_bottom_to_model_base(cleaned_wells, xy_base_series)
    cleaned_wells = _set_inverted_filters_to_point_filters(cleaned_wells)
    cleaned_wells = _set_ultrathin_filters_to_point_filters(
        cleaned_wells, minimum_thickness
    )
    return cleaned_wells


def cleanup_hfb(barrier: gpd.GeoDataFrame, idomain: GridDataArray) -> gpd.GeoDataFrame:
    """
    Clean up HFB data, fixes some common mistakes causing ValidationErrors by
    doing the following:

    - Drop HFB segments outside active domain (idomain==1)

    Parameters
    ----------
    barrier: geopandas.GeoDataFrame
        GeoDataFrame with HFB data
    idomain: xarray.DataArray | xugrid.UgridDataArray
        MODFLOW 6 model domain. idomain==1 is considered active domain.

    Returns
    -------
    geopandas.GeoDataFrame
        Cleaned up GeoDataFrame with HFB data.
    """
    active = idomain > 0
    # Drop HFB cells outside active domain
    clipped_barrier = clip_line_gdf_by_grid(barrier, active)
    return clipped_barrier
