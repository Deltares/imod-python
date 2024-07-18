"""
Module to define type aliases.
"""

from typing import TYPE_CHECKING, TypeAlias, TypeVar, Union

import numpy as np
import xarray as xr
import xugrid as xu

GridDataArray: TypeAlias = Union[xr.DataArray, xu.UgridDataArray]
GridDataset: TypeAlias = Union[xr.Dataset, xu.UgridDataset]
ScalarAsDataArray: TypeAlias = Union[xr.DataArray, xu.UgridDataArray]
ScalarAsDataset: TypeAlias = Union[xr.Dataset, xu.UgridDataset]
UnstructuredData: TypeAlias = Union[xu.UgridDataset, xu.UgridDataArray]
FloatArray: TypeAlias = np.ndarray
IntArray: TypeAlias = np.ndarray


# Types for optional dependencies.

if TYPE_CHECKING:
    import geopandas as gpd
    import shapely

    GeoDataFrameType: TypeAlias = gpd.GeoDataFrame
    GeoSeriesType: TypeAlias = gpd.GeoSeries
    PolygonType: TypeAlias = shapely.Polygon
    LineStringType: TypeAlias = shapely.LineString
else:
    GeoDataFrameType = TypeVar("GeoDataFrameType")
    GeoSeriesType = TypeVar("GeoSeriesType")
    PolygonType = TypeVar("PolygonType")
    LineStringType = TypeVar("LineStringType")
