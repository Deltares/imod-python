import io
import warnings
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.io import FortranFile, FortranFormattingError

from imod.util import MissingOptionalModule

try:
    import shapely.geometry as sg
except ImportError:
    sg = MissingOptionalModule("shapely")

try:
    import geopandas as gpd
except ImportError:
    gpd = MissingOptionalModule("geopandas")


# Unfortunately, the binary GEN files are written as Fortran Record files, so
# they cannot be read directly with e.g. numpy.fromfile (like direct access) The
# scipy FortranFile is mostly adequate, it just misses a method to read char
# records (always ascii encoded; note all ascii is valid utf-8, but not vice
# versa). Reading and writing methods are monkeypatched here.
#
# https://mail.python.org/pipermail/python-dev/2008-January/076194.html
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


# The binary GEN file has some idiosyncratic geometries:
# * A separate circle geometry
# * A seperate rectangle geometry
# In OGC (WKT) terms, these are polygons.
#
# Both are defined by two vertices:
# * Circle: center and any other outside point (from which we can infer the radius)
# * Rectangle: (left, lower), (right, upper); may also be (left, upper), (right, lower)
#
# The other shapely geometries can be generated directly from the vertices.


def from_circle(xy: np.ndarray) -> sg.Polygon:
    radius = np.sqrt(np.sum((xy[1] - xy[0]) ** 2))
    return sg.Point(xy[0]).buffer(radius)


def from_point(xy: np.ndarray) -> sg.Polygon:
    return sg.Point(xy[0])


def from_rectangle(xy: np.ndarray) -> sg.Polygon:
    return sg.box(xy[0, 0], xy[0, 1], xy[1, 0], xy[1, 1])


def to_circle(geometry: sg.Polygon) -> Tuple[np.ndarray, int]:
    xy = np.array([geometry.centroid.coords[0], geometry.exterior.coords[0]])
    return xy, 2


def to_rectangle(geometry: sg.Polygon) -> Tuple[np.ndarray, int]:
    xy = np.array(geometry.exterior)
    if (geometry.area / geometry.minimum_rotated_rectangle.area) < 0.999:
        raise ValueError("Feature_type is rectangle, but geometry is not a rectangular")
    # First and third vertex will give (left, right) and (lower, upper)
    return xy[[0, 2]], 2


def to_polygon(geometry: sg.Polygon) -> Tuple[np.ndarray, int]:
    xy = np.array(geometry.exterior)
    return xy, xy.shape[0]


def to_point(geometry: sg.Point) -> Tuple[np.ndarray, int]:
    return np.array(geometry), 1


def to_line(geometry: sg.LineString) -> Tuple[np.ndarray, int]:
    xy = np.array(geometry)
    return xy, xy.shape[0]


# From the iMOD User Manual
FLOAT_TYPE = np.float64
INT_TYPE = np.int32
HEADER_TYPE = np.int32
CIRCLE = 1024
POLYGON = 1025
RECTANGLE = 1026
POINT = 1027
LINE = 1028
MAX_NAME_WIDTH = 11

# Map integer enumerators to strings
GENTYPE_TO_NAME = {
    CIRCLE: "circle",
    POLYGON: "polygon",
    RECTANGLE: "rectangle",
    POINT: "point",
    LINE: "line",
}
NAME_TO_GENTYPE = {v: k for k, v in GENTYPE_TO_NAME.items()}
# From gen itype to shapely geometry:
GENTYPE_TO_GEOM = {
    CIRCLE: from_circle,
    POLYGON: sg.Polygon,
    RECTANGLE: from_rectangle,
    POINT: from_point,
    LINE: sg.LineString,
}
# Infer gentype on the basis of shapely type
GEOM_TO_GENTYPE = {
    sg.Polygon: POLYGON,
    sg.Point: POINT,
    sg.LineString: LINE,
}
# Checking names with actual geometry types
NAME_TO_GEOM = {
    "circle": sg.Polygon,
    "rectangle": sg.Polygon,
    "polygon": sg.Polygon,
    "point": sg.Point,
    "line": sg.LineString,
}
GENTYPE_TO_VERTICES = {
    CIRCLE: to_circle,
    RECTANGLE: to_rectangle,
    POLYGON: to_polygon,
    POINT: to_point,
    LINE: to_line,
}


def read(path: Union[str, Path]) -> gpd.GeoDataFrame:
    """
    Read a binary GEN file to a geopandas GeoDataFrame.

    Parameters
    ----------
    path: Union[str, Path]

    Returns
    -------
    geodataframe: gpd.GeoDataFrame
    """
    with warnings.catch_warnings(record=True):
        warnings.filterwarnings(
            "ignore", message="Given a dtype which is not unsigned."
        )
        with FortranFile(path, mode="r", header_dtype=HEADER_TYPE) as f:
            f.read_reals(dtype=FLOAT_TYPE)  # Skip the bounding box
            n_feature, n_column = f.read_ints(dtype=INT_TYPE)
            if n_column > 0:
                widths = f.read_ints(dtype=INT_TYPE)
                indices = range(0, (n_column + 1) * MAX_NAME_WIDTH, MAX_NAME_WIDTH)
                string = f.read_char_record()  # pylint:disable=no-member
                names = [string[i:j].strip() for i, j in zip(indices[:-1], indices[1:])]

            xy = []
            rows = []
            feature_type = np.empty(n_feature, dtype=INT_TYPE)
            for i in range(n_feature):
                _, ftype = f.read_ints(dtype=INT_TYPE)
                feature_type[i] = ftype
                if n_column > 0:
                    rows.append(f.read_char_record())  # pylint:disable=no-member
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
    df["feature_type"] = feature_type
    df["feature_type"] = df["feature_type"].replace(GENTYPE_TO_NAME)

    geometry = []
    for ftype, geom in zip(feature_type, xy):
        geometry.append(GENTYPE_TO_GEOM[ftype](geom))

    return gpd.GeoDataFrame(df, geometry=geometry)


def vertices(
    geometry: Union[sg.Point, sg.Polygon, sg.LineString], ftype: str
) -> Tuple[int, np.ndarray, int]:
    """
    Infer from geometry, or convert from string, the feature type to the GEN
    expected Enum (int).

    Convert the geometry to the GEN expected vertices, and the number of
    vertices.
    """
    if ftype != "":
        # Start by checking whether the feature type matches the geometry
        try:
            expected = NAME_TO_GEOM[ftype]
        except KeyError as e:
            raise ValueError(
                f"feature_type should be one of {set(NAME_TO_GEOM.keys())}. Got instead {ftype}"
            ) from e
        if not isinstance(geometry, expected):
            raise ValueError(
                f"Feature type is {ftype}, expected {expected}. Got instead: {type(geometry)}"
            )
        ftype: int = NAME_TO_GENTYPE[ftype]
    else:
        try:
            ftype: int = GEOM_TO_GENTYPE[type(geometry)]
        except KeyError as e:
            raise TypeError(
                "Geometry type not allowed. Should be Polygon, Linestring, or Point."
                f" Got {type(geometry)} instead."
            ) from e

    xy, n_vertex = GENTYPE_TO_VERTICES[ftype](geometry)
    return ftype, xy, n_vertex


def write(
    path: Union[str, Path],
    geodataframe: gpd.GeoDataFrame,
    feature_type: Optional[str] = None,
) -> None:
    """
    Write a GeoDataFrame to a binary GEN file.

    Note that the binary GEN file has two geometry types, circles and
    rectangles, which cannot be mapped directly from a shapely type. Points,
    lines, and polygons can be converted automatically.

    In shapely, circles and rectangles will also be represented by polygons. To
    specifically write circles and rectangles to a binary GEN file, an
    additional column of strings is required which specifies the geometry type.

    Parameters
    ----------
    path : Union[str, Path]
    geodataframe : gpd.GeoDataFrame
    feature_type : Optional[str]
        Which column to interpret as geometry type, one of: point, line, polygon, circle,
        rectangle. Default value is ``None``.

    Returns
    -------
    None
        Writes file.
    """
    df = pd.DataFrame(geodataframe.drop(columns="geometry")).astype("string")
    if feature_type is not None:
        types = df.pop(feature_type).str.lower()
    else:  # Create a dummy iterator
        types = ("" for i in range(len(df)))

    n_feature, n_column = df.shape
    # Truncate column names to 11 chars, then make everything at least 11 chars
    column_names = "".join([c[:11].ljust(11) for c in df])
    # Get the widths of the columns. Make room for at least 11
    widths = []
    for column in df:
        width = max(11, df[column].str.len().max())
        df[column] = df[column].str.pad(width, side="right")
        widths.append(width)

    with warnings.catch_warnings(record=True):
        warnings.filterwarnings(
            "ignore", message="Given a dtype which is not unsigned."
        )
        with FortranFile(path, mode="w", header_dtype=HEADER_TYPE) as f:
            f.write_record(geodataframe.total_bounds.astype(FLOAT_TYPE))
            f.write_record(np.array([n_feature, n_column], dtype=INT_TYPE))
            if n_column > 0:
                f.write_record(np.array(widths).astype(INT_TYPE))
                f.write_char_record(column_names)  # pylint:disable=no-member
            for geometry, (_, row), ftype in zip(
                geodataframe.geometry, df.iterrows(), types
            ):
                ftype, xy, n_vertex = vertices(geometry, ftype)
                f.write_record(np.array([n_vertex, ftype], dtype=INT_TYPE))
                if n_column > 0:
                    f.write_char_record("".join(row.values))  # pylint:disable=no-member
                f.write_record(np.array(geometry.bounds).astype(FLOAT_TYPE))
                f.write_record(xy.astype(FLOAT_TYPE))
