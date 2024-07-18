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
GEOPANDAS_CONFIRMED = False
if TYPE_CHECKING:
    try:
        import geopandas as gpd
        GEOPANDAS_CONFIRMED = True
    except ImportError:
        pass 

if GEOPANDAS_CONFIRMED:
    GeoDataFrameType = gpd.GeoDataFrame
    GeoSeriesType = gpd.GeoSeries
else:
    GeoDataFrameType = TypeVar('GeoDataFrameType')
    GeoSeriesType = TypeVar("GeoSeriesType")


SHAPELY_CONFIRMED = False
if TYPE_CHECKING:
    try:
        import shapely
        SHAPELY_CONFIRMED = True
    except ImportError:
        pass 

if SHAPELY_CONFIRMED:
    PolygonType = shapely.Polygon
    LineStringType = shapely.LineString
else:
    PolygonType = TypeVar('PolygonType')
    LineStringType = TypeVar("LineStringType")
