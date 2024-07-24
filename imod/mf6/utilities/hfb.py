from typing import TYPE_CHECKING, Tuple

import pandas as pd

from imod.typing import GeoDataFrameType, GeoSeriesType
from imod.util.imports import MissingOptionalModule

if TYPE_CHECKING:
    import geopandas as gpd
else:
    try:
        import geopandas as gpd
    except ImportError:
        gpd = MissingOptionalModule("geopandas")

try:
    import shapely
except ImportError:
    shapely = MissingOptionalModule("shapely")


def _create_zlinestring_from_bound_df(bound: pd.DataFrame) -> GeoDataFrameType:
    """Create geodataframe with linestring geometry from dataframe with bounds."""
    # Make sure only x, y, z or x, y in columns
    columns = sorted({"x", "y", "z"} & set(bound.columns))
    index_names = list(bound.index.names)
    # Prevent multiindex to be created by avoiding list
    if bound.index.name:
        index_to_group = bound.index.name
    else:
        index_to_group = index_names
    # Each linestring has its own index, therefore groupby index.
    mapping_linestrings = [
        (g[0], shapely.LineString(g[1][columns].values)) for g in bound.groupby(index_to_group)
    ]
    index, linestrings= zip(*mapping_linestrings)

    gdf = gpd.GeoDataFrame(
        linestrings, index=index, columns=["geometry"], geometry="geometry"
    )
    gdf.index = gdf.index.set_names(index_names)
    return gdf


def _create_zpolygon_from_polygon_df(polygon_df: pd.DataFrame) -> GeoDataFrameType:
    """Create geodataframe with polygon geometry from dataframe with polygon nodes."""
    index_names = ["index", "parts"]
    polygons = [
        (g[0], shapely.Polygon(g[1].values)) for g in polygon_df.groupby(index_names)
    ]
    index_tuples, polygons_data = list(zip(*polygons))
    multi_index = pd.MultiIndex.from_tuples(index_tuples, names=index_names)
    return gpd.GeoDataFrame(
        polygons_data, columns=["geometry"], index=multi_index, geometry="geometry"
    )


def _extract_hfb_bounds_from_zpolygons(
    dataframe: GeoDataFrameType,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract hfb bounds from dataframe. Requires dataframe geometry to be of type
    shapely "Z Polygon".
    """
    if not dataframe.geometry.has_z.all():
        raise TypeError("GeoDataFrame geometry has no z, which is required.")

    coordinates = dataframe.geometry.get_coordinates(include_z=True)

    groupby_names = list(dataframe.index.names) + ["x", "y"]
    grouped = coordinates.reset_index().groupby(groupby_names)

    lower = grouped.min().reset_index(["x", "y"])
    upper = grouped.max().reset_index(["x", "y"])

    return lower, upper


def hfb_zpolygons_to_zlinestrings(dataframe: GeoDataFrameType) -> GeoDataFrameType:
    """
    Convert GeoDataFrame with zpolygons to zlinestrings.

    Paramaters
    ----------
    dataframe: GeoDataFrame
        GeoDataFrame with a Z Polygons as datatype.

    Returns
    -------
    GeoDataFrame with upper and lower bound as linestrings.
        The multi-index denotes whether linestring designates "upper" or "lower"
        bound.
    """
    lower, upper = _extract_hfb_bounds_from_zpolygons(dataframe)

    lower_gdf = _create_zlinestring_from_bound_df(lower)
    upper_gdf = _create_zlinestring_from_bound_df(upper)

    bounds_gdf = pd.concat(
        [lower_gdf, upper_gdf],
        keys=[
            "lower",
            "upper",
        ],
    )

    return bounds_gdf


def _flip_linestrings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flip linestrings, preserve linestring order.
    """
    # Add extra index ascending to sort with, with reset.
    # This new index denotes unique nodes
    df_reset = df.reset_index()
    # Set to multi-index to prepare sort
    df_multi = df_reset.set_index(["index", "parts", df_reset.index])
    # Sort, only reverse newly added index.
    df_sorted = df_multi.sort_index(
        level=[0, 1, 2], ascending=[True, True, False], axis=0
    )
    # Drop index added for sorting
    return df_sorted.reset_index(level=2, drop=True)


def hfb_zlinestrings_to_zpolygons(bounds_gdf: GeoSeriesType) -> GeoDataFrameType:
    """
    Convert zlinestrings of bounds_gdf to zpolygons

    Parameters
    ----------
    bounds_gdf: Dataframe
        Dataframe with for each polygon an upper and lower shapely.LINESTRING,
        indicated by index "upper" and "lower".
    """
    # Empty Dataframe
    if bounds_gdf.shape[0] == 0:
        return bounds_gdf

    coordinates = bounds_gdf.get_coordinates(include_z=True)
    # Sort index to ascending everywhere to be able to assign flip upper bound
    # linestrings without errors.
    coordinates = coordinates.sort_index(
        level=[0, 1, 2], ascending=[True, True, True], axis=0
    )
    # Reverse upper bound to prevent bowtie polygon from being made.
    coordinates.loc["upper"] = _flip_linestrings(coordinates.loc["upper"]).values
    # Drop index with "upper" and "lower" in it.
    coordinates = coordinates.reset_index(level=0, drop=True)

    return _create_zpolygon_from_polygon_df(coordinates)
