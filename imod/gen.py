from enum import IntEnum
import io
from itertools import accumulate
from pathlib import Path
import struct
from typing import List, Optional, Union
import warnings

import geopandas as gpd
import numba
import numpy as np
import pandas as pd
from scipy.io import FortranFile, FortranFormattingError
import shapely.geometry as sg


def monkeypatch_method(cls):
    def decorator(func):
        setattr(cls, func.__name__, func)
        return func

    return decorator


@monkeypatch_method(FortranFile)
def read_char_record(self):
    first_size = self._read_size(eof_ok=True)
    string = self._fp.read(first_size).decode("utf-8")
    if len(string) != first_size:
        raise FortranFormattingError("End of file in the middle of a record")
    second_size = self._read_size(eof_ok=True)
    if first_size != second_size:
        raise IOError("Sizes do not agree in the header and footer for this record")
    return string


@monkeypatch_method(FortranFile)
def write_char_record(self, string: str):
    total_size = len(string)
    bytes_string = string.encode("ascii")
    nb = np.array([total_size], dtype=self._header_dtype)
    nb.tofile(self._fp)
    self._fp.write(bytes_string)
    nb.tofile(self._fp)


def circle(xy: np.ndarray) -> sg.Polygon:
    radius = np.sqrt((xy[1] - xy[0]) ** 2)
    return sg.Point(xy[0]).buffer(radius)


def point(xy: np.ndarray) -> sg.Polygon:
    return sg.Point(xy[0])


def rectangle(xy: np.ndarray) -> sg.Polygon:
    return sg.box(xy[0, 0], xy[0, 1], xy[1, 0], xy[1, 1])


# From the iMOD User Manual
FLOAT_TYPE = np.float64
INT_TYPE = np.int32
HEADER_TYPE = np.int32
CIRCLE = 1024
POLYGON = 1025
RECTANGLE = 1026
POINT = 1027
LINE = 1028

# From gen to geopandas:
GEN_TO_GEOM = {
    CIRCLE: circle,
    POLYGON: sg.Polygon,
    RECTANGLE: rectangle,
    POINT: point,
    LINE: sg.LineString,
}
GEN_TO_NAME = {
    CIRCLE: "circle",
    POLYGON: "polygon",
    RECTANGLE: "rectangle",
    POINT: "point",
    LINE: "line",
}

# From geopandas to gen:
# Circles, rectangles are not present in geopandas
GEOM_TO_GEN = {
    sg.Polygon: POLYGON,
    sg.Point: POINT,
    sg.LineString: LINE,
}
NAME_TO_GEN = {k: v for k, v in GEN_TO_NAME.items()}
# So a circle, rectangle is instead defined by the feature_type column:
NAME_TO_GEOM = {
    "circle": sg.Polygon,
    "rectangle": sg.Polygon,
    "polygon": sg.Polygon,
    "point": sg.Point,
    "line": sg.LineString,
}


def read(path: Union[str, Path]) -> gpd.GeoDataFrame:
    with warnings.catch_warnings(record=True):
        warnings.filterwarnings("ignore", message="Given a dtype which is not unsigned.")
        with FortranFile(path, mode="r", header_dtype=HEADER_TYPE) as f:
            f.read_reals(dtype=FLOAT_TYPE)  # Skip the bounding box
            n_feature, n_column = f.read_ints(dtype=INT_TYPE)
            if n_column > 0:
                widths = f.read_ints(dtype=INT_TYPE)
                indices = [0] + list(accumulate(widths))
                string = f.read_char_record()
                names = [string[i:j].strip() for i, j in zip(indices[:-1], indices[1:])]

            xy = []
            rows = []
            feature_type = np.empty(n_feature, dtype=INT_TYPE)
            for i in range(n_feature):
                _, ftype = f.read_ints(dtype=INT_TYPE)
                feature_type[i] = ftype
                if n_column > 0:
                    rows.append(f.read_char_record())
                f.read_reals(dtype=FLOAT_TYPE)  # skip the bounding box
                xy.append(f.read_reals(dtype=FLOAT_TYPE).reshape((-1, 2)))

    if n_column > 0:
        df = pd.read_fwf(
            io.StringIO("\n".join(rows)),
            widths=widths,
            names=names,
        )
    else:
        df = pd.DataFrame()
    df["type"] = feature_type
    df["type"] = df["type"].replace(GEN_TO_NAME)

    geometry = []
    for ftype, geom in zip(feature_type, xy):
        geometry.append(GEN_TO_GEOM[ftype](geom))

    return gpd.GeoDataFrame(df, geometry=geometry)


def polygon_to_circle(geometry: sg.Polygon) -> (np.ndarray, int):
    xy = np.array(geometry.exterior)
    center = np.mean(xy, axis=0)
    xy = np.array([center, xy[0]])
    return xy, 2


def polygon_to_rectangle(geometry: sg.Polygon) -> (np.ndarray, int):
    xy = np.array(geometry.exterior)
    if (geometry.area / geometry.minimum_rotated_rectangle.area) < 0.999:
        raise ValueError("Feature_type is rectangle, but geometry is not a rectangular")
    return xy[:2], 2


def vertices(geometry: Union[sg.Point, sg.Polygon, sg.LineString], ftype: str):
    # Start by checking whether the feature type matches the geometry
    if ftype != "":
        expected = NAME_TO_GEOM[ftype]
        if not isinstance(geometry, expected):
            raise ValueError(
                f"Feature type is {ftype}, expected {expected}. Got instead: {type(geometry)}"
            )
        ftype = NAME_TO_GEN[ftype]
    else:
        ftype = GEOM_TO_GEN[type(geometry)]

    if ftype == CIRCLE:
        xy, n_vertex = polygon_to_circle(geometry)
    elif ftype == RECTANGLE:
        xy, n_vertex = polygon_to_rectangle(geometry)

    if isinstance(geometry, sg.Polygon):
        xy = np.array(geometry.exterior)
        n_vertex = xy.shape[0]
    elif isinstance(geometry, sg.LineString):
        xy = np.array(geometry)
        n_vertex = xy.shape[0]
    elif isinstance(geometry, sg.Point):
        xy = np.array(geometry)
        n_vertex = 1
    else:
        raise TypeError(
            "Geometry type not allowed. Should be Polygon, Linestring, or Point."
            f" Got {type(geometry)} instead."
        )

    return ftype, n_vertex, xy


def write(
    path: Union[str, Path],
    geodataframe: gpd.GeoDataFrame,
    feature_type: Optional[str] = None,
) -> None:
    df = pd.DataFrame(geodataframe.drop(columns="geometry")).astype("string")
    n_feature, n_column = df.shape
    if feature_type is not None:
        types = df.pop(feature_type).str.lower()
    else:  # Create a dummy iterator
        types = ("" for i in range(n_feature))

    # Truncate column names to 11 chars, then make everything at least 11 chars
    column_names = "".join([c[:11].ljust(11) for c in df])
    # Get the widths of the columns. Make room for at least 11
    widths = []
    for column in df:
        width = max(11, df[column].str.len().max())
        df[column] = df[column].str.pad(width, side="right")
        widths.append(width)

    with warnings.catch_warnings(record=True):
        warnings.filterwarnings("ignore", message="Given a dtype which is not unsigned.")
        with FortranFile(path, mode="w", header_dtype=HEADER_TYPE) as f:
            f.write_record(geodataframe.total_bounds.astype(FLOAT_TYPE))
            f.write_record(np.array([n_feature, n_column], dtype=INT_TYPE))
            if n_column > 0:
                f.write_record(np.array(widths).astype(INT_TYPE))
                f.write_char_record(column_names)
            for geometry, (i, row), ftype in zip(
                geodataframe.geometry, df.iterrows(), types
            ):
                ftype, n_vertex, xy = vertices(geometry, ftype)
                f.write_record(np.array([n_vertex, ftype], dtype=INT_TYPE))
                if n_column > 0:
                    f.write_char_record("".join(row.values))
                f.write_record(np.array(geometry.bounds).astype(FLOAT_TYPE))
                f.write_record(xy.astype(FLOAT_TYPE))
