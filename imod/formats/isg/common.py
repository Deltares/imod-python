"""Base classes shared by the ISG file models."""

from dataclasses import dataclass
from functools import partial
from os import PathLike
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    ClassVar,
    Generic,
    Self,
    TypeVar,
    get_args,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, GetPydanticSchema, model_validator
from pydantic_core import core_schema

from imod.formats.isg.utils import (
    decode,
    read_records,
    record_length,
    scalar_type,
    sibling_path,
    to_datetime,
)
from imod.formats.isg.validators import to_vector
from imod.util.imports import MissingOptionalModule

if TYPE_CHECKING:
    import geopandas as gpd
    import shapely
    from matplotlib.axes import Axes
else:
    try:
        import geopandas as gpd
        import shapely
    except ImportError:
        gpd = MissingOptionalModule("geopandas")
        shapely = MissingOptionalModule("shapely")

ScalarT = TypeVar("ScalarT", bound=np.generic)
FloatT = TypeVar("FloatT", np.float32, np.float64)
TableT = TypeVar("TableT", bound="Table")

NAME = (np.bytes_, 32)


Vector = Annotated[
    NDArray[ScalarT],
    GetPydanticSchema(
        lambda source, _: core_schema.no_info_plain_validator_function(
            partial(to_vector, dtype=scalar_type(source))
        )
    ),
]
"""A read-only one-dimensional numpy array: one column of a table."""


@dataclass(frozen=True)
class References:
    """
    Marks a vector of first record numbers in the associated file ``file``; the
    number of records is stored in the vector ``count``. A first record of 0
    with a count of 0 references nothing.
    """

    file: str
    count: str
    absolute: bool = False
    """Use the absolute value of ``count``, as its sign carries a meaning."""


class IsgBaseModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, use_attribute_docstrings=True
    )


class Table(IsgBaseModel):
    """Equal length vectors (columns); record ids are one-based positions."""

    @model_validator(mode="after")
    def check_equal_length(self) -> "Table":
        lengths = {name: len(column) for name, column in self.columns().items()}
        if len(set(lengths.values())) > 1:
            raise ValueError(f"columns differ in length: {lengths}")
        return self

    def columns(self) -> dict[str, np.ndarray]:
        return {name: value for name, value in self if isinstance(value, np.ndarray)}

    def __len__(self) -> int:
        return len(next(iter(self.columns().values()), ()))

    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.columns(), index=pd.RangeIndex(1, len(self) + 1, name="id")
        )

    def take(self, rows: slice) -> Self:
        """The records in ``rows``, as views."""
        columns = {name: column[rows] for name, column in self.columns().items()}
        return self.model_copy(update=columns)

    @classmethod
    def references(cls) -> dict[str, References]:
        """Vectors that reference records in associated files."""
        return {
            name: meta
            for name, field in cls.model_fields.items()
            for meta in field.metadata
            if isinstance(meta, References)
        }


class TimeSeries(Table):
    """A table of records at a date (IDATE) and time (CTIME)."""

    levels: ClassVar[tuple[str, ...]]
    """Columns plotted by default."""

    def time(self) -> pd.DatetimeIndex:
        return to_datetime(getattr(self, "idate"), getattr(self, "ctime", None))

    def plot(
        self, ax: "Axes | None" = None, y: str | list[str] | None = None, **kwargs: Any
    ) -> "Axes":
        """Plot ``y``, by default the levels, against time."""
        y = list(self.levels) if y is None else y
        frame = self.dataframe().set_index(self.time())
        return frame.plot(ax=ax, y=y, drawstyle="steps-post", marker=".", **kwargs)


class SpatialTable(Table):
    """A table of which each record has a geometry."""

    def geometry(self) -> NDArray[np.object_]:
        """One shapely geometry per record."""
        raise NotImplementedError

    def geodataframe(self) -> "gpd.GeoDataFrame":
        return gpd.GeoDataFrame(self.dataframe(), geometry=self.geometry())

    def plot(self, ax: "Axes | None" = None, **kwargs: Any) -> "Axes":
        """Plot the geometries on a map; ``kwargs`` go to ``GeoDataFrame.plot``."""
        return self.geodataframe().plot(ax=ax, **kwargs)

    @property
    def __geo_interface__(self) -> dict[str, Any]:
        """All geometries as one multi-geometry."""
        geometries = self.geometry()
        types = set(shapely.get_type_id(geometries).tolist())
        multi = {
            0: shapely.multipoints,
            1: shapely.multilinestrings,
            3: shapely.multipolygons,
        }
        collect = multi.get(types.pop()) if len(types) == 1 else None
        return (collect or shapely.geometrycollections)(geometries).__geo_interface__


class FileModel(IsgBaseModel):
    """
    A model read from a file, e.g. ``ISP[np.float32](path)``.

    Missing fields are read from ``filepath``, with the extension replaced by
    the extension of the class: an associated file only needs the path of its
    ISG-file.
    """

    filepath: Path | None = Field(default=None, exclude=True, repr=False)

    def __init__(self, filepath: str | PathLike[str] | None = None, /, **data: Any):
        if filepath is not None:
            data["filepath"] = filepath
        super().__init__(**data)

    @classmethod
    def extension(cls) -> str:
        raise NotImplementedError

    @model_validator(mode="before")
    @classmethod
    def load(cls, data: Any) -> Any:
        if isinstance(data, str | PathLike):
            data = {"filepath": data}
        if not isinstance(data, dict) or data.get("filepath") is None:
            return data
        missing = [
            name
            for name, field in cls.model_fields.items()
            if field.is_required() and name not in data
        ]
        if not missing:
            return data
        path = sibling_path(Path(data["filepath"]), cls.extension())
        return {**cls.parse(path), **data, "filepath": path}

    @classmethod
    def parse(cls, path: Path) -> dict[str, Any]:
        """Read the file into a dictionary of field values."""
        raise NotImplementedError

    @classmethod
    def files(cls) -> list[str]:
        """Names of the fields holding associated files."""
        names = []
        for name, field in cls.model_fields.items():
            types = (field.annotation, *get_args(field.annotation))
            if any(isinstance(t, type) and issubclass(t, FileModel) for t in types):
                names.append(name)
        return names

    @classmethod
    def real(cls) -> type[np.floating]:
        """The float type the model is parameterized with."""
        args = cls.__pydantic_generic_metadata__["args"]
        if not args or isinstance(args[0], TypeVar):
            raise TypeError(
                f"parameterize {cls.__name__} with np.float32 or np.float64, "
                f"e.g. {cls.__name__}[np.float32]"
            )
        return args[0]


class BinaryFile(FileModel, Table, Generic[FloatT]):
    """
    A binary file of fixed length records, read into one vector per attribute.

    Parameterize with ``np.float32`` or ``np.float64`` for single or double
    precision files.
    """

    @classmethod
    def record(cls, real: type[np.floating]) -> np.dtype:
        """Record layout for single (float32) or double (float64) precision."""
        raise NotImplementedError

    @classmethod
    def parse(cls, path: Path) -> dict[str, Any]:
        dtype = cls.record(cls.real())
        if record_length(path) != dtype.itemsize:
            raise ValueError(
                f"{path}: record length of {record_length(path)} bytes, "
                f"expected {dtype.itemsize}"
            )
        records = read_records(path, dtype)
        return {name: decode(records[name]) for name in records.dtype.names or ()}

    @classmethod
    def precision(cls, path: str | PathLike[str]) -> type[np.floating]:
        """Float type of a file, derived from the record length in its header."""
        path = sibling_path(Path(path), cls.extension())
        reclen = record_length(path)
        for real in (np.float32, np.float64):
            if cls.record(real).itemsize == reclen:
                return real
        raise ValueError(f"{path}: unknown record length of {reclen} bytes")


class IndexFile(BinaryFile[FloatT]):
    """
    The ISD1-, ISC1-, IST1- and ISQ1-files share their layout: N, IREF, DIST
    and CNAME, referencing N records from IREF in the associated \\*2-file.
    """

    n: Vector[np.int32]
    """Number of data records in the associated \\*2-file."""
    iref: Vector[np.int32]
    """Record number within the associated \\*2-file for the first data record."""

    @classmethod
    def record(cls, real: type[np.floating]) -> np.dtype:
        return np.dtype(
            [("n", np.int32), ("iref", np.int32), ("dist", real), ("cname", NAME)]
        )

    def split(self, data: TableT) -> list[TableT]:
        """The N records from IREF in ``data`` (the \\*2-file) per record, as views."""
        rows = zip(self.n, self.iref)
        return [data.take(slice(i - 1, i - 1 + abs(n))) for n, i in rows]

    def plot(
        self, data: Any, y: str, ax: "Axes | None" = None, **kwargs: Any
    ) -> "Axes":
        """Plot ``y`` of ``data`` (the \\*2-file) per record, labelled by CNAME."""
        if ax is None:
            _, ax = plt.subplots()
        for part, name in zip(self.split(data), getattr(self, "cname")):
            ax = part.plot(ax=ax, y=y, label=name, **kwargs)
        return ax
