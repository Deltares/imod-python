from enum import IntEnum
import io
from pathlib import Path
import struct
from typing import List, Optional, Union

import geopandas as gpd
import numba
import numpy as np
import pandas as pd
import shapely.geometry as sg


NEWLINE = ord("\n")
# From the iMOD User Manual
FLOAT_FORMAT = "d"
FLOAT_TYPE = np.float64
FLOAT_SIZE = FLOAT_TYPE(0).itemsize
INT_FORMAT = "i"
INT_TYPE = np.int32
INT_SIZE = INT_TYPE(0).itemsize
HEADER_FORMAT = "i"
HEADER_TYPE = np.int32
HEADER_SIZE = HEADER_TYPE(0).itemsize


def circle(xy: np.ndarray) -> sg.Polygon:
    radius = np.sqrt((xy[1] - xy[0]) ** 2)
    return sg.Point(xy[0]).buffer(radius)


class GeometryType(IntEnum):
    CIRCLE = 1024
    POLYGON = 1025
    RECTANGLE = 1026
    POINT = 1027
    LINE = 1028


GEOMETRY_MAPPING = {
    GeometryType.CIRCLE: circle,
    GeometryType.POLYGON: sg.Polygon,
    GeometryType.RECTANGLE: sg.Polygon,
    GeometryType.POINT: sg.Point,
    GeometryType.LINE: sg.LineString,
}


def read_float(f) -> float:
    return struct.unpack(FLOAT_FORMAT, f.read(FLOAT_SIZE))[0]


def read_int(f) -> int:
    return struct.unpack(INT_FORMAT, f.read(INT_SIZE))[0]


@numba.njit(inline="always")
def parse_size(blob: np.ndarray, pos: int) -> (int, HEADER_TYPE):
    new_pos = pos + HEADER_SIZE
    return new_pos, blob[pos:new_pos].view(HEADER_TYPE)[0]


@numba.njit(inline="always")
def parse_int(blob: np.ndarray, pos: int) -> (int, INT_TYPE):
    new_pos = pos + INT_SIZE
    return new_pos, blob[pos:new_pos].view(INT_TYPE)[0]


@numba.njit(inline="always")
def parse_float(blob: np.ndarray, pos: int) -> (int, FLOAT_TYPE):
    new_pos = pos + FLOAT_SIZE
    return new_pos, blob[pos:new_pos].view(FLOAT_TYPE)[0]


@numba.njit(inline="always")
def skip_record(blob: np.ndarray, pos: int) -> int:
    pos, first_size = parse_size(blob, pos)
    new_pos = pos + first_size + HEADER_SIZE
    return new_pos


@numba.njit
def parse_gen_blob(
    blob: np.ndarray,
    n_feature: int,
    n_column: int,
    total_width: int,
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    # First steps: figure out how much memory to allocate
    # Depends on the total number of points stored
    bytes_text = np.empty((n_feature, total_width + 1), dtype=np.int8)
    bytes_text[:, -1] = NEWLINE
    feature_ptr = np.empty(n_feature, dtype=np.int64)
    feature_n_point = np.empty(n_feature, dtype=np.int64)
    feature_type = np.empty(n_feature, dtype=np.int32)
    fid = 0
    n_total = 0

    pos = 0
    while fid < n_feature:
        # Get npoints, itype of feature
        pos, size = parse_size(blob, pos)
        pos, n = parse_int(blob, pos)
        pos, itype = parse_int(blob, pos)
        feature_n_point[fid] = n
        feature_type[fid] = itype
        n_total += n
        pos += HEADER_SIZE
        # Store the text data
        pos, size = parse_size(blob, pos)
        bytes_text[fid, :-1] = blob[pos : pos + size]
        pos += size + HEADER_SIZE
        # Store the location of the coordinates
        pos = skip_record(blob, pos)  # bounding box
        pos, size = parse_size(blob, pos)
        feature_ptr[fid] = pos
        pos += size + HEADER_SIZE
        # Increment feature id
        fid += 1

    # We now know the total number of points, allocate the array:
    xy = np.empty((n_total, 2), dtype=np.float64)
    indices = np.empty(n_total, dtype=np.int64)
    i = 0
    for fid in range(n_feature):
        n = feature_n_point[fid]
        pos = feature_ptr[fid]
        for _ in range(n):
            pos, x = parse_float(blob, pos)
            pos, y = parse_float(blob, pos)
            xy[i, 0] = x
            xy[i, 1] = y
            indices[i] = fid
            i += 1

    return xy, indices, feature_n_point, feature_type, bytes_text


def read(path: Union[str, Path]) -> gpd.GeoDataFrame:
    with open(path, "rb") as f:
        # First record
        f.seek(HEADER_SIZE + 4 * FLOAT_SIZE + HEADER_SIZE)

        # Second record
        f.read(HEADER_SIZE)
        n_feature = read_int(f)
        n_column = read_int(f)
        f.read(HEADER_SIZE)

        if n_column > 0:
            f.read(HEADER_SIZE)
            widths = [read_int(f) for _ in range(n_column)]
            f.read(HEADER_SIZE)
            f.read(HEADER_SIZE)
            names = [f.read(w).decode("utf-8").strip() for w in widths]
            f.read(HEADER_SIZE)

        # Read everything else as a big binary blob
        blob = np.fromfile(f, dtype=np.int8)

    total_width = sum(widths)
    # TODO: use indices to do vectorized pygeos geometry construction (WIP)
    # https://github.com/pygeos/pygeos/issues/241
    # https://github.com/pygeos/pygeos/pull/322
    xy, _, feature_n_point, feature_type, bytes_text = parse_gen_blob(
        blob, n_feature, n_column, total_width
    )
    df = pd.read_fwf(
        io.StringIO(bytes_text.tobytes().decode("utf-8")),
        widths=widths,
        names=names,
    )
    df["type"] = feature_type
    df["type"] = df["type"].replace({e.value: e.name.lower() for e in GeometryType})
    # TODO: pygeos, see above
    geometry = []
    start = 0
    for ftype, n in zip(feature_type, feature_n_point):
        geometry.append(GEOMETRY_MAPPING[ftype](xy[start : start + n]))
        start += n

    return gpd.GeoDataFrame(df, geometry=geometry)
