"""
Module to define type aliases.
"""

from typing import TYPE_CHECKING, Literal, TypeAlias, TypedDict, TypeVar, Union

import numpy as np
import xarray as xr
import xugrid as xu
from numpy.typing import NDArray

GridDataArray: TypeAlias = Union[xr.DataArray, xu.UgridDataArray]
GridDataset: TypeAlias = Union[xr.Dataset, xu.UgridDataset]
GridDataDict: TypeAlias = dict[str, GridDataArray]
Imod5DataDict: TypeAlias = dict[str, GridDataDict | dict[str, list[str]]]
ScalarAsDataArray: TypeAlias = Union[xr.DataArray, xu.UgridDataArray]
ScalarAsDataset: TypeAlias = Union[xr.Dataset, xu.UgridDataset]
UnstructuredData: TypeAlias = Union[xu.UgridDataset, xu.UgridDataArray]
FloatArray: TypeAlias = NDArray[np.floating]
IntArray: TypeAlias = NDArray[np.int_]


class SelSettingsType(TypedDict, total=False):
    layer: int
    drop: bool
    missing_dims: Literal["raise", "warn", "ignore"]


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
