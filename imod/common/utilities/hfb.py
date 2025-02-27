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


def _create_zbound_gdf_from_zbound_df(bound: pd.DataFrame) -> GeoDataFrameType:
    """Create geodataframe with linestring geometry from dataframe with bounds."""
    bound = _prepare_index_names(bound)
    # Make sure only x, y, z or x, y in columns
    columns = sorted({"x", "y", "z"} & set(bound.columns))
    index_names = list(bound.index.names)
    # Prevent multiindex to be created in groupby by avoiding list
    if bound.index.name:
        index_to_group = bound.index.name
    else:
        index_to_group = index_names
    # Each linestring has its own index, therefore groupby index.
    mapping_linestrings = [
        (g[0], shapely.LineString(g[1][columns].values))
        for g in bound.groupby(index_to_group)
    ]
    index, linestrings = zip(*mapping_linestrings)

    zbound_gdf = gpd.GeoDataFrame(
        linestrings, index=index, columns=["geometry"], geometry="geometry"
    )
    zbound_gdf.index = zbound_gdf.index.set_names(index_names)
    return zbound_gdf


def _create_vertical_polygon_from_vertices_df(
    polygon_df: pd.DataFrame,
) -> GeoDataFrameType:
    """
    Create geodataframe with polygon geometry from dataframe with polygon
    vertices. The dataframe with vertices must have a multi-index with name
    ["index", "parts"]
    """
    index_names = ["index", "parts"]
    polygons = [
        (g[0], shapely.Polygon(g[1].values)) for g in polygon_df.groupby(index_names)
    ]
    index_tuples, polygons_data = list(zip(*polygons))
    multi_index = pd.MultiIndex.from_tuples(index_tuples, names=index_names)
    return gpd.GeoDataFrame(
        polygons_data, columns=["geometry"], index=multi_index, geometry="geometry"
    )


def _prepare_index_names(
    dataframe: GeoDataFrameType,
) -> GeoDataFrameType:
    """
    Prepare index names, if single index, index should be named 'index', if
    multi-index, it should be '(index, parts)'; where 'index' refers to the line
    index of the original linestrings provided by user and 'parts' to segment of
    this linestring after clipping. If the line index was not named 'index', but
    is None, this function sets it to 'index'. This is aligned with how pandas
    names an unnamed index when calling df.reset_index().
    """
    index_names = dataframe.index.names

    match index_names:
        case ["index"] | ["index", "parts"]:
            return dataframe
        case [None]:  # Unnamed line index
            new_index_names = ["index"]
        case [None, "parts"]:  # Unnamed line index
            new_index_names = ["index", "parts"]
        case _:
            raise IndexError(
                f"Index names should be ['index'] or ['index', 'parts']. Got {index_names}"
            )

    dataframe.index = dataframe.index.set_names(new_index_names)
    return dataframe


def _extract_zbounds_from_vertical_polygons(
    dataframe: GeoDataFrameType,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract z bounds from dataframe. Requires dataframe geometry to be of type
    shapely "Z Polygon".
    """
    dataframe = _prepare_index_names(dataframe)

    if not dataframe.geometry.has_z.all():
        raise TypeError("GeoDataFrame geometry has no z, which is required.")

    coordinates = dataframe.geometry.get_coordinates(include_z=True)

    groupby_names = list(dataframe.index.names) + ["x", "y"]
    grouped = coordinates.reset_index().groupby(groupby_names)

    lower = grouped.min().reset_index(["x", "y"])
    upper = grouped.max().reset_index(["x", "y"])

    return lower, upper


def vertical_polygons_to_zbound_linestrings(
    dataframe: GeoDataFrameType,
) -> GeoDataFrameType:
    """
    Convert GeoDataFrame with vertical polygons to linestrings with z bounds.

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
    lower, upper = _extract_zbounds_from_vertical_polygons(dataframe)

    lower_gdf = _create_zbound_gdf_from_zbound_df(lower)
    upper_gdf = _create_zbound_gdf_from_zbound_df(upper)

    zbounds_gdf = pd.concat(
        [lower_gdf, upper_gdf],
        keys=[
            "lower",
            "upper",
        ],
    )

    return zbounds_gdf


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


def clipped_zbound_linestrings_to_vertical_polygons(
    zbounds_gdf: GeoSeriesType,
) -> GeoDataFrameType:
    """
    Convert clipped zlinestrings provided with bounds_gdf to zpolygons

    Parameters
    ----------
    bounds_gdf: Dataframe
        Dataframe with for each polygon an upper and lower shapely.LINESTRING,
        indicated by index "upper" and "lower".
    """
    # Empty Dataframe
    if zbounds_gdf.shape[0] == 0:
        return zbounds_gdf

    coordinates = zbounds_gdf.get_coordinates(include_z=True)
    # Sort index to ascending everywhere to be able to assign flip upper bound
    # linestrings without errors.
    coordinates = coordinates.sort_index(
        level=[0, 1, 2], ascending=[True, True, True], axis=0
    )
    # Reverse upper bound to prevent bowtie polygon from being made.
    coordinates.loc["upper"] = _flip_linestrings(coordinates.loc["upper"]).values
    # Drop index with "upper" and "lower" in it.
    coordinates = coordinates.reset_index(level=0, drop=True)

    return _create_vertical_polygon_from_vertices_df(coordinates)
