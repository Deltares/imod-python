"""
Module to define type aliases.
"""

from datetime import datetime
from typing import TYPE_CHECKING, Literal, TypeAlias, TypedDict, TypeVar, Union

import numpy as np
import xarray as xr
import xugrid as xu
from numpy.typing import NDArray

GridDataArray: TypeAlias = Union[xr.DataArray, xu.UgridDataArray]
GridDataset: TypeAlias = Union[xr.Dataset, xu.UgridDataset]
GridDataDict: TypeAlias = dict[str, GridDataArray]
ScalarAsDataArray: TypeAlias = Union[xr.DataArray, xu.UgridDataArray]
ScalarAsDataset: TypeAlias = Union[xr.Dataset, xu.UgridDataset]
UnstructuredData: TypeAlias = Union[xu.UgridDataset, xu.UgridDataArray]
FloatArray: TypeAlias = NDArray[np.floating]
IntArray: TypeAlias = NDArray[np.int_]
StressPeriodTimesType: TypeAlias = list[datetime] | Literal["steady-state"]


class SelSettingsType(TypedDict, total=False):
    layer: int
    drop: bool
    missing_dims: Literal["raise", "warn", "ignore"]


class DropVarsType(TypedDict, total=False):
    names: str
    errors: Literal["raise", "ignore"]


class Imod5DataDict(TypedDict, total=False):
    cap: GridDataDict
    extra: dict[str, list[str]]


GEOPANDAS_AVAILABLE = False
try:
    import geopandas as gpd

    GEOPANDAS_AVAILABLE = True
except ImportError:
    pass

SHAPELY_AVAILABLE = False
try:
    import shapely

    SHAPELY_AVAILABLE = True
except ImportError:
    pass

if TYPE_CHECKING or GEOPANDAS_AVAILABLE:
    GeoDataFrameType: TypeAlias = gpd.GeoDataFrame
    GeoSeriesType: TypeAlias = gpd.GeoSeries
else:
    GeoDataFrameType = TypeVar("GeoDataFrameType")
    GeoSeriesType = TypeVar("GeoSeriesType")

if TYPE_CHECKING or SHAPELY_AVAILABLE:
    PolygonType: TypeAlias = shapely.Polygon
    LineStringType: TypeAlias = shapely.LineString
else:
    PolygonType = TypeVar("PolygonType")
    LineStringType = TypeVar("LineStringType")
