"""
ISG-files, iMOD User Manual 5.6.1, section 9.9.

See README.md for an overview of the files and their relations.
"""

from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Generic, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, model_validator

from imod.formats.isg.common import (
    BinaryFile,
    FileModel,
    FloatT,
    IndexFile,
    References,
    SpatialTable,
    TimeSeries,
    Vector,
    gpd,
    shapely,
)
from imod.formats.isg.enums import CalculationOption, SurfaceFlowRouting
from imod.formats.isg.utils import read_rows, record_index
from imod.formats.isg.validators import check_records, member_of
from imod.formats.isg.views import (
    CrossSection,
    CrossSection1D,
    CrossSection2D,
    CrossSection2DReferenceHeight,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

CTIME = ("ctime", (np.bytes_, 8))


# Zm, Zc, Zp in the first 4 bytes of the third ISC2 real.
class ISP(BinaryFile[FloatT], SpatialTable):
    """
    Node file, containing the coordinates of nodes for segments in the ISG file.

    The ISP-file supports single- and double precision syntax. In case single
    precision coordinates are used the ISP file is built with a record length of
    8 bytes/2 words, for a double precision ISP file this is 16 bytes/4 words.
    The first record of the ISP is reserved to store the record length (2295 and
    4343 for single- and double precision respectively). The ISEG variable in
    the ISG points to the record number that determines the first coordinate of
    the segment. Since the first record is reserved already, iMOD actually reads
    the ISEG+1 record instead. From each record two reals will be read that
    represent the x and y coordinate of the current node on the segment.

    Geometry: a Point per node.
    """

    x: Vector[FloatT]
    """X-coordinate of node (meter)"""
    y: Vector[FloatT]
    """Y-coordinate of node (meter)"""

    @classmethod
    def extension(cls) -> str:
        return ".ISP"

    @classmethod
    def record(cls, real: type[np.floating]) -> np.dtype:
        return np.dtype([("x", real), ("y", real)])

    def geometry(self) -> NDArray[np.object_]:
        return shapely.points(self.x, self.y)


class ISD1(IndexFile[FloatT]):
    """
    Calculation points: location and name, referencing their timeseries in
    the ISD2-file.

    The ISD1-file supports single- and double precision syntax. In case single
    precision the ISD1-file is built with a record length of 44 bytes/11 words,
    for double precision this is 48 bytes/13 words. The first record is solely
    reserved to store the record length (11511 and 12535 for single- and double
    precision, respectively). The ICLC variable in the ISG-file points to the
    record number that determines the first calculation point on the segment.
    Since the first record is reserved already, iMOD actually reads the ICLC+1
    record instead.

    Note: For the SFR-option, only two calculation points are allowed, one at
    the beginning of a segment and one at the end.
    """

    n: Vector[np.int32]
    """
    Number of data records in the ISD2-file that describes the timeserie of the
    calculation point.
    """
    iref: Annotated[Vector[np.int32], References("isd2", count="n")]
    """
    Record number within the ISD2-file for the first data record that describes
    the timeserie for the calculation point.
    """
    dist: Vector[FloatT]
    """
    Distance (meters) measured from the beginning of the segment (node 1) that
    located the calculation point.
    """
    cname: Vector[np.str_]
    """Name of the calculation point."""

    @classmethod
    def extension(cls) -> str:
        return ".ISD1"


class RivISD2(BinaryFile[FloatT], TimeSeries):
    """
    Timeseries of the calculation points for RIV (ASFR=0): water level,
    bottom level, resistance and infiltration factor.

    It depends on the ASFR whether this file contains information for the RIV
    package (ASFR=0) or alternatively for the SFR package (ASFR=1). The
    ISD2-file is built with a record length of 20 bytes/5 words (single
    precision) and 36 bytes/9 words (double precision) for ASFR=0. The first
    record is solely reserved to store the record length (ASFR=0: 5367/9463).
    The IREF variable in the ISD1-file points to the record number that
    determines the data for the calculation point on the segment. Since the
    first record is reserved already, iMOD actually reads the IREF+1 record
    instead.

    Implementation note: double precision files have a record length of 44
    bytes (11511), including CTIME.
    """

    levels = ("wlvl", "btml")

    asfr: Literal[SurfaceFlowRouting.RIV] = SurfaceFlowRouting.RIV
    idate: Vector[np.int32]
    """Date representation as yyyymmdd."""
    ctime: Vector[np.str_] | None = None
    """
    Time representation as hh:mm:ss. This additional record is only applicable
    for double precision ISD2-files. It is not used yet!
    """
    wlvl: Vector[FloatT]
    """Waterlevel of the river (m+MSL)"""
    btml: Vector[FloatT]
    """Bottom level of the riverbed (m+MSL)."""
    resis: Vector[FloatT]
    """Resistance of the riverbed (days)."""
    inff: Vector[FloatT]
    """Infiltration factor (-)"""

    @classmethod
    def extension(cls) -> str:
        return ".ISD2"

    @classmethod
    def record(cls, real: type[np.floating]) -> np.dtype:
        time: list[tuple[str, Any]] = [] if real is np.float32 else [CTIME]
        levels: list[tuple[str, Any]] = [
            (name, real) for name in ("wlvl", "btml", "resis", "inff")
        ]
        return np.dtype([("idate", np.int32), *time, *levels])


class SfrISD2(BinaryFile[FloatT], TimeSeries):
    """
    Timeseries of the calculation points for SFR (ASFR=1): levels, riverbed,
    segment connections and flows.

    It depends on the ASFR whether this file contains information for the RIV
    package (ASFR=0) or alternatively for the SFR package (ASFR=1). For ASFR=1,
    the record length is 44 bytes/13 words (single precision) and 64 bytes/16
    words (double precision). The first record is solely reserved to store the
    record length (ASFR=1: 12535/16631). The IREF variable in the ISD1-file
    points to the record number that determines the data for the calculation
    point on the segment. Since the first record is reserved already, iMOD
    actually reads the IREF+1 record instead.

    Implementation note: single precision files have a record length of 64
    bytes (16631) and store IUPSEG before IDOWNSEG, matching the labels in the
    ISG-file. Double precision files have a record length of 100 bytes
    (25847), with the same order (iMOD source, ``ISGREADISD``).
    """

    levels = ("wlvl", "btml")

    asfr: Literal[SurfaceFlowRouting.SFR] = SurfaceFlowRouting.SFR
    idate: Vector[np.int32]
    """Date representation as yyyymmdd."""
    ctime: Vector[np.str_]
    """Time representation as hh:mm:ss."""
    wlvl: Vector[FloatT]
    """Waterlevel of the river (m+MSL). Only used in conjunction with ICALCOPT=1."""
    btml: Vector[FloatT]
    """Bottom level of the riverbed (m+MSL). Only used in conjunction with ICALCOPT=1."""
    width: Vector[FloatT]
    """Width the riverbed (m+MSL). Only used in conjunction with ICALCOPT=≤ 2."""
    thickness: Vector[FloatT]
    """Thickness of the river bed (m)."""
    hcfact: Vector[FloatT]
    """Conductivity of the river bed (m/d)."""
    iupseg: Vector[np.int32]
    """
    An integer value of the upstream segment (-) from which water is diverted
    (or withdrawn) to supply inflow to this stream segment if this segment
    originates as a diversion from an upstream segment.

    - IUPSEG > 0 If this stream segment receives inflow as a diversion from an
      upstream segment, enter the stream number;
    - IUPSEG < 0 If the source of a stream segment is discharge from a lake, set
      IUPSEG equal to the negative value of the lake identification number;
      where the minus sign is used as a flag to tell the model that streamflow
      into this segment is derived from a lake rather than a stream segment;
    - IUPSEG = 0 If this stream segment does not receive inflow as a diversion
      from an upstream segment, then set IUPSEG = 0.
    """
    idownseg: Vector[np.int32]
    """
    An integer value of the downstream stream segment (-) that receives
    tributary inflow from the last downstream reach of this segment.

    - IDOWNSEG > 0 If this segment feeds (or discharges into) another downstream
      (tributary) segment, then enter the number of that stream;
    - IDOWNSEG = 0 If this segment does not feed (or discharge into) another
      downstream (tributary) segment, then enter a value of 0 for this variable.
      If the segment ends within the modelled grid and IDOWNSEG = 0, outflow
      from the segment is not routed anywhere and is no longer part of the
      stream network. One may wish to use this if all flow in the stream gets
      diverted into a lined canal or into a pipe;
    - IDOWNSEG < 0 If the flow out of this segment discharges into a lake,
      IDOWNSEG will be equal to the negative value of the lake identification
      number; where the minus sign is used as a flag to tell the model that
      flow enters a lake rather than a tributary stream segment.
    """
    # member_of as validator, as we chose to keep the raw numpy data
    calcopt: Annotated[Vector[np.int32], member_of(CalculationOption)]
    """
    Method determines how stream depth and width are calculated for each reach
    in a segment (-), see :class:`CalculationOption`.
    """
    divopt: Vector[np.int32]
    """
    An integer value that only is specified if IUPSEG > 0 (do not specify a
    value in this field if IUPSEG = 0 or IUPSEG < 0). DIVOPT defines the
    prioritization system for diversion, such as when insufficient water is
    available to meet all diversion stipulations, and is used in conjunction
    with the value of QFLOW (specified below) (-), see :class:`DiversionOption`.
    """
    qflow: Vector[FloatT]
    """
    Streamflow entering or leaving the upstream end of a stream segment (i.e.
    the first reach) (m3/s). iMOD will check whether you have entered a value
    larger than 100000.0, as this might indicate that the entered volume is m3/d
    instead of m3/s. Be aware of the fact that the meaning of QFLOW depends on
    the fact whether the stream is a

    - If the stream is a headwater stream (IUPSEG = 0) which is the first stream
      of a segment without any segment upstreams, QFLOW defines the total inflow
      to the first reach of the segment. The value can be any number ≥ 0.0;
    - If the stream is a tributary stream (IUPSEG = 0) with an upstream
      dewatering stream, QFLOW defines additional specified inflow to or
      withdrawal from the first reach of the segment (that is, in addition to
      the discharge from the upstream segment of which this is a tributary).
      This additional flow does not interact with the groundwater system. For
      example, a positive number might be used to represent direct outflow into
      a stream from a sewage treatment plant, whereas a negative number might be
      used to represent pumpage directly from a stream into an intake pipe for a
      municipal water treatment plant;
    - If the stream is a diversionary stream (IUPSEG ≠ 0), and the diversion is
      from another stream segment, QFLOW defines the streamflow diverted from
      the last reach of stream segment IUPSEG into the first reach of this
      segment. The diversion is computed or adjusted according to the value of
      DIVOPT;
    - If the stream is a diversionary stream (IUPSEG ≠ 0), and the diversion is
      from a lake, QFLOW defines a fixed rate of discharge diverted from the
      lake into the first reach of this stream segment (unless the lake goes
      dry) and flow from the lake is not dependent on the value of ICALC.
      However, if QFLOW = 0, then the lake outflow into the first reach of this
      segment will be calculated on the basis of lake stage relative to the top
      of the streambed for the first reach using one of the methods defined by
      ICALC.
    """
    qrunoff: Vector[FloatT]
    """
    volumetric rate (m3/s) of the diffuse overland flow runoff that enters the
    stream segment, the rate is apportioned to each reach of the segment.
    """
    pptsw: Vector[FloatT]
    """
    precipitation (mm/d) that is the volumetric rate per unit area of water
    added by precipitation directly on the stream channel (in units of length
    (millimeter) per time (day)).
    """
    etsw: Vector[FloatT]
    """
    evaporation (mm/d) that is the volumetric rate per unit area of water
    removed by evapotranspiration directly from the stream channel (in units of
    length (millimeter) per time (day)). ETSW is always defined as a positive
    value.
    """

    @classmethod
    def extension(cls) -> str:
        return ".ISD2"

    @classmethod
    def record(cls, real: type[np.floating]) -> np.dtype:
        reals: tuple[str, ...] = ("wlvl", "btml", "width", "thickness", "hcfact")
        integers: tuple[str, ...] = ("iupseg", "idownseg", "calcopt", "divopt")
        flows: tuple[str, ...] = ("qflow", "qrunoff", "pptsw", "etsw")
        return np.dtype(
            [
                ("idate", np.int32),
                CTIME,
                *[(name, real) for name in reals],
                *[(name, np.int32) for name in integers],
                *[(name, real) for name in flows],
            ]
        )


class ISC1(IndexFile[FloatT]):
    """
    Cross-sections: location and name, referencing their data in the
    ISC2-file.

    The ISC1-file supports single- and double precision syntax. In case single
    precision the ISC1-file is built with a record length of 44 bytes/11 words,
    for double precision this is 48 bytes/12 words. The first record is solely
    reserved to store the record length (11511 and 12535 for single- and double
    precision, respectively). The ICRS variable in the ISG-file points to the
    record number that determines the first calculation point on the segment.
    Since the first record is reserved already, iMOD actually reads the ICRS+1
    record instead.
    """

    n: Vector[np.int32]
    """
    The meaning of this attribute is twofold:

    - >0: Number of data records in the ISC2-file that describes the actual
      cross-section.
    - <0: The absolute number of data records in the ISC2-file that describes
      the riverbed as a collection of x,y,z points including an extra record to
      describe the dimensions (DX,DY) of the network that captured the x,y,z
      points.
    """
    iref: Annotated[Vector[np.int32], References("isc2", count="n", absolute=True)]
    """
    Record number within the ISC2-file for the first data record that describes
    the cross-section.
    """
    dist: Vector[FloatT]
    """
    Distance (meters) measured from the beginning of the segment (node 1) that
    locates the cross-section.
    """
    cname: Vector[np.str_]
    """Name of the cross-section."""

    @classmethod
    def extension(cls) -> str:
        return ".ISC1"

    def cross_sections(self, isc2: "ISC2[FloatT]") -> list[CrossSection]:
        """
        The cross-section of each record in ``isc2``, as views: 1-D if N > 0,
        2-D otherwise, with a reference height if DX < 0 and DY < 0.
        """
        sections: list[CrossSection] = []
        for n, iref in zip(self.n, self.iref):
            first = iref - 1
            kind: type[CrossSection] = CrossSection1D
            if n < 0 and isc2.a[first] < 0.0 and isc2.b[first] < 0.0:
                kind = CrossSection2DReferenceHeight
            elif n < 0:
                kind = CrossSection2D
            sections.append(kind.view(isc2, slice(first, first + abs(n))))
        return sections


class ISC2(BinaryFile[FloatT]):
    """
    Cross-section data, interpreted per cross-section by
    ``ISC1.cross_sections()``.

    The ISC2-file is built with a record length of 12 bytes/3 words for single
    precision and with 20 bytes/5 words for a double precision. The first record
    is solely reserved to store the record length (3319 and 5367 for single- and
    double precision, respectively). The IREF variable in the ISC1-file points
    to the record number that determines the data for the calculation point on
    the segment. Since the first record is reserved already, iMOD actually reads
    the IREF+1 record instead.

    The meaning of the three attributes depends on the ISC1 record that points
    to them; :meth:`ISC1.cross_sections` interprets them.

    Implementation note: double precision files have a record length of 24
    bytes (6391); Zm, Zc, Zp are packed in the first 4 bytes of the third real.
    """

    a: Vector[FloatT]
    """DISTANCE (N>0), DX (N<0, first record) or X (N<0, following records)."""
    b: Vector[FloatT]
    """BOTTOM (N>0), DY (N<0, first record) or Y (N<0, following records)."""
    c: Vector[FloatT]
    """
    MRC (N>0), HREF (N<0, first record), Z (N<0, DX > 0.0) or Zm, Zc, Zp packed
    as 2, 1 and 1 byte integers (N<0, DX < 0.0).
    """

    @classmethod
    def extension(cls) -> str:
        return ".ISC2"

    @classmethod
    def record(cls, real: type[np.floating]) -> np.dtype:
        return np.dtype([("a", real), ("b", real), ("c", real)])

    def plot(self, ax: "Axes | None" = None, y: str = "b", **kwargs: Any) -> "Axes":
        """Plot B against A: BOTTOM against DISTANCE for 1-D records."""
        return self.dataframe().set_index("a").plot(ax=ax, y=y, **kwargs)


class IST1(IndexFile[FloatT]):
    """
    Weirs/structures: location and name, referencing their timeseries in the
    IST2-file.

    The IST1-file supports single- and double precision syntax. In case single
    precision the IST1-file is built with a record length of 44 bytes/11 words,
    for double precision this is 48 bytes/12 words. The first record is solely
    reserved to store the record length (11511 and 12535 for single- and double
    precision, respectively). The ISTW variable in the ISG-file points to the
    record number that determines the first calculation point on the segment.
    Since the first record is reserved already, iMOD actually reads the ISTW+1
    record instead.
    """

    n: Vector[np.int32]
    """
    Number of data records in the IST2-file that describes the actual timeserie
    for the weir/structure.
    """
    iref: Annotated[Vector[np.int32], References("ist2", count="n")]
    """
    Record number within the IST2-file for the first data record that describes
    the weirs/structure.
    """
    dist: Vector[FloatT]
    """
    Distance (meters) measured from the beginning of the segment (node 1) that
    locates the weir/structure.
    """
    cname: Vector[np.str_]
    """Name of the weir/structure."""

    @classmethod
    def extension(cls) -> str:
        return ".IST1"


class IST2(BinaryFile[FloatT], TimeSeries):
    """
    Timeseries of the weirs/structures: upstream and downstream water levels.

    The ISC2-file is built with a record length of 12 bytes/3 words for single
    precision and with 20 bytes/5 words for a double precision. The first record
    is solely reserved to store the record length (3319 and 5367 for single- and
    double precision, respectively). The IREF variable in the IST1-file points
    to the record number that determines the data for the calculation point on
    the segment. Since the first record is reserved already, iMOD actually reads
    the IREF+1 record instead.

    Implementation note: this paragraph of the manual names the ISC2-file, but
    describes the IST2-file. Double precision files have a record length of 28
    bytes (7415), including CTIME.
    """

    levels = ("wlvl_up", "wlvl_dwn")

    idate: Vector[np.int32]
    """Date representation as yyyymmdd."""
    ctime: Vector[np.str_] | None = None
    """
    Time representation as hh:mm:ss. This additional record is only applicable
    for double precision ISD2-files. It is not used yet!
    """
    wlvl_up: Vector[FloatT]
    """Water level for the upstream side of the weir/structure (m+MSL)."""
    wlvl_dwn: Vector[FloatT]
    """Water level for the downstream side of the weir/structure (m+MSL)."""

    @classmethod
    def extension(cls) -> str:
        return ".IST2"

    @classmethod
    def record(cls, real: type[np.floating]) -> np.dtype:
        time: list[tuple[str, Any]] = [] if real is np.float32 else [CTIME]
        levels: list[tuple[str, Any]] = [("wlvl_up", real), ("wlvl_dwn", real)]
        return np.dtype([("idate", np.int32), *time, *levels])


class ISQ1(IndexFile[FloatT]):
    """
    Q-width/depth relations: location and name, referencing their tables in
    the ISQ2-file.

    The ISQ1-file supports single- and double precision syntax. In case single
    precision the ISQ1-file is built with a record length of 44 bytes/11 words,
    for double precision this is 48 bytes/12 words. The IQHR variable in the
    ISG-file points to the record number that determines the first calculation
    point on the segment. Since the first record is reserved already, iMOD
    actually reads the IQHR+1 record instead.
    """

    n: Vector[np.int32]
    """
    Number of data records in the ISQ2-file that describes the actual timeserie
    for the q-width/depth relation ship.
    """
    iref: Annotated[Vector[np.int32], References("isq2", count="n")]
    """
    Record number within the ISQ2-file for the first data record that describes
    the q-width/depth relation ship.
    """
    dist: Vector[FloatT]
    """
    Distance (meters) measured from the beginning of the segment (node 1) that
    locates the q-width/depth relation ship.
    """
    cname: Vector[np.str_]
    """Name of the q-width/depth relation ship."""

    @classmethod
    def extension(cls) -> str:
        return ".ISQ1"


class ISQ2(BinaryFile[FloatT]):
    """
    Q-width/depth tables: width and depth of the river per discharge Q.

    The ISQ2-file is built with a record length of 12 bytes/3 words for single
    precision and with 20 bytes/5 words for a double precision. The first record
    is solely reserved to store the record length (3319 and 5367 for single- and
    double precision, respectively). The IREF variable in the ISQ1-file points
    to the record number that determines the data for the calculation point on
    the segment. Since the first record is reserved already, iMOD actually reads
    the IREF+1 record instead.

    Implementation note: single precision files have a record length of 16
    bytes (4343), as four attributes are stored; 32 bytes (8439) for double
    precision.
    """

    q: Vector[FloatT]
    """Discharge in m3/d."""
    width: Vector[FloatT]
    """Width of the stream at the given discharge Q (m)."""
    depth: Vector[FloatT]
    """Depth of the stream at the given discharge Q (m)."""
    factor: Vector[FloatT]
    """Obsolete factor at the given discharge Q (-)."""

    @classmethod
    def extension(cls) -> str:
        return ".ISQ2"

    @classmethod
    def record(cls, real: type[np.floating]) -> np.dtype:
        return np.dtype([(name, real) for name in ("q", "width", "depth", "factor")])

    def plot(
        self,
        ax: "Axes | None" = None,
        y: str | list[str] = ["width", "depth"],
        **kwargs: Any,
    ) -> "Axes":
        """Plot WIDTH and DEPTH against Q."""
        return self.dataframe().set_index("q").plot(ax=ax, y=y, marker=".", **kwargs)


class ISG(FileModel, SpatialTable, Generic[FloatT]):
    """
    Segment file: the segments, referencing their records in the associated
    files, which are read with it.

    The ISG-file format is developed to capture all relevant information used by
    surface water elements in direct relation with groundwater. It stores
    stages, bottom heights, infiltration factors, resistances, and moreover, the
    actual outline of the surface water element. To store all these different
    types of information the ISG-file format consists of associated files that
    are connected by the ISG-file. This ISG-file is the one that will be
    actually read by iMOD, the other files will be opened by iMOD automatically.

    For nodes, calculations points, cross-section, structures a similar setup is
    used. A first reference is made from the \\*.ISG file to the record in the
    \\*.ISD1 file. From there another reference is made to the \\*.ISD2 file that
    contains the specific configuration parameters.

    The files associated from the ISG-file (ISP-, ISD-, ISC-, IST- and
    ISQ-files) are all binary and indexed files and cannot be edited in regular
    text editors.

    The ISG is a table of segments: for each segment add the following columns.
    Read with ``ISG.read(path)``, which detects the precision, or with
    ``ISG[np.float32](path)``.

    Geometry: a LineString per segment through its ISP nodes; ``locate()``
    places the ISD1, ISC1, IST1 and ISQ1 records along it.

    Implementation note: the manual swaps the descriptions of IQHR and NQHR;
    like the other pairs, IQHR is the first record and NQHR the number of
    records.
    """

    asfr: SurfaceFlowRouting
    """
    Usage of surface-flow routing (SFR) ability from the associated file
    \\*.ISD\\*. If ASFR=0, a default ISG file is used, if ASFR=1, a SFR
    compliant version of the ISG is used.

    Implementation note: like iMOD, a header of only NSEG means ASFR=0.
    """
    labels: tuple[str, ...]
    """Enter the name of the labels that are present in the ISD file."""
    label: Vector[np.str_]
    """
    Name of the segment, use quotes to distinguish names with empty spaces.
    Maximum size of each label is 52 characters.

    Implementation note: iMOD reads 50 characters.
    """
    iseg: Annotated[Vector[np.int32], References("isp", count="nseg")]
    """
    Record number that defines the first coordinate (node) in the associated
    ISP-file.
    """
    nseg: Vector[np.int32]
    """Number of records in the ISP-file that describes the segment by coordinates."""
    iclc: Annotated[Vector[np.int32], References("isd1", count="nclc")]
    """
    Record number that defines the first calculation points on the segment ISEG
    within the associated ISD1-file.
    """
    nclc: Vector[np.int32]
    """Number of calculation points on segment ISEG."""
    icrs: Annotated[Vector[np.int32], References("isc1", count="ncrs")]
    """
    Record number that defines the first cross-section on the segment ISEG
    within the associated ISC1-file.
    """
    ncrs: Vector[np.int32]
    """Number of cross-sections on segment ISEG."""
    istw: Annotated[Vector[np.int32], References("ist1", count="nstw")]
    """
    Record number that defines the first weir/structure on the segment ISEG
    within the associated IST1-file.
    """
    nstw: Vector[np.int32]
    """Number of weirs/structures on segment ISEG."""
    iqhr: Annotated[Vector[np.int32], References("isq1", count="nqhr")]
    """
    Record number that defines the first discharge-water level relationships on
    the segment ISEG within the associated ISQ1-file.
    """
    nqhr: Vector[np.int32]
    """Number of discharge-water level relationships."""

    isp: ISP[FloatT] = Field(repr=False)
    """The associated ISP-file: nodes."""
    isd1: ISD1[FloatT] = Field(repr=False)
    """The associated ISD1-file: calculation points."""
    isd2: Annotated[
        RivISD2[FloatT] | SfrISD2[FloatT], Field(discriminator="asfr", repr=False)
    ]
    """The associated ISD2-file: timeseries of the calculation points."""
    isc1: ISC1[FloatT] = Field(repr=False)
    """The associated ISC1-file: cross-sections."""
    isc2: ISC2[FloatT] = Field(repr=False)
    """The associated ISC2-file: cross-section data."""
    ist1: IST1[FloatT] = Field(repr=False)
    """The associated IST1-file: weirs/structures."""
    ist2: IST2[FloatT] = Field(repr=False)
    """The associated IST2-file: timeseries of the weirs/structures."""
    isq1: ISQ1[FloatT] = Field(repr=False)
    """The associated ISQ1-file: Q-width/depth relations."""
    isq2: ISQ2[FloatT] = Field(repr=False)
    """The associated ISQ2-file: Q-width/depth tables."""

    @classmethod
    def extension(cls) -> str:
        return ".ISG"

    @classmethod
    def record(cls) -> np.dtype:
        """Layout of a segment row."""
        counts = [
            "iseg",
            "nseg",
            "iclc",
            "nclc",
            "icrs",
            "ncrs",
            "istw",
            "nstw",
            "iqhr",
            "nqhr",
        ]
        integers: list[tuple[str, Any]] = [(name, np.int32) for name in counts]
        return np.dtype([("label", np.str_, 50), *integers])

    @classmethod
    def read(cls, path: str | PathLike[str]) -> "ISG[Any]":
        """Read an ISG-file; unless parameterized, with the precision of its ISP-file."""
        isg: Any = cls
        if not cls.__pydantic_generic_metadata__["args"]:
            isg = ISG[ISP.precision(path)]  # type: ignore[misc]
        return isg(path)

    @classmethod
    def parse(cls, path: Path) -> dict[str, Any]:
        (nseg, *header), *rows = read_rows(path)
        asfr, *labels = header or ["0"]
        rows = rows[: int(nseg)]
        table = np.array([tuple(row) for row in rows], dtype=cls.record())
        files: dict[str, Any] = {name: {"filepath": path} for name in cls.files()}
        files["isd2"]["asfr"] = int(asfr)
        return {
            "asfr": int(asfr),
            "labels": tuple(labels),
            **{name: table[name] for name in table.dtype.names or ()},
            **files,
        }

    @model_validator(mode="after")
    def check_references(self) -> "ISG[FloatT]":
        tables = [self, *(getattr(self, name) for name in self.files())]
        for table in tables:
            for name, ref in table.references().items():
                count = getattr(table, ref.count)
                check_records(
                    f"{table.extension().lstrip('.')}.{name.upper()}",
                    getattr(table, name),
                    np.abs(count) if ref.absolute else count,
                    len(getattr(self, ref.file)),
                )
        return self

    def geometry(self) -> NDArray[np.object_]:
        """The segments as LineStrings through their ISP nodes."""
        index = record_index(self.iseg, self.nseg)
        segment = np.repeat(np.arange(len(self)), self.nseg)
        return shapely.linestrings(
            self.isp.x[index], self.isp.y[index], indices=segment
        )

    def segment(self, name: str) -> NDArray[np.int64]:
        """Segment (ISG record) of each record of file ``name``; 0 if none."""
        first, ref = next(
            (k, v) for k, v in self.references().items() if v.file == name
        )
        count = getattr(self, ref.count)
        segment = np.zeros(len(getattr(self, name)), dtype=np.int64)
        segment[record_index(getattr(self, first), count)] = np.repeat(
            np.arange(1, len(self) + 1), count
        )
        return segment

    def position(
        self, segment: NDArray[np.int64], dist: NDArray[np.floating]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """X, y and direction (radians) at ``dist`` along ``segment``."""
        lines = np.concatenate([[None], self.geometry()])[segment]
        dist = np.asarray(dist, dtype=np.float64)
        point = shapely.line_interpolate_point(lines, dist)
        before = shapely.line_interpolate_point(lines, np.maximum(dist - 1e-3, 0.0))
        after = shapely.line_interpolate_point(lines, dist + 1e-3)
        angle = np.arctan2(
            shapely.get_y(after) - shapely.get_y(before),
            shapely.get_x(after) - shapely.get_x(before),
        )
        return shapely.get_x(point), shapely.get_y(point), angle

    def locate(self, name: str) -> "gpd.GeoDataFrame":
        """
        Records of ``name`` ("isd1", "isc1", "ist1" or "isq1") with their
        segment, as Points at DIST along that segment.
        """
        table = getattr(self, name)
        segment = self.segment(name)
        x, y, _ = self.position(segment, table.dist)
        return gpd.GeoDataFrame(
            table.dataframe().assign(segment=segment), geometry=shapely.points(x, y)
        )

    def plot(self, ax: "Axes | None" = None, **kwargs: Any) -> "Axes":
        """Plot the segments with the located ISD1, ISC1, IST1 and ISQ1 records."""
        ax = super().plot(ax=ax, **kwargs)
        markers = {"isd1": "o", "isc1": "|", "ist1": "s", "isq1": "^"}
        for name, marker in markers.items():
            located = self.locate(name)
            if len(located):
                located.plot(ax=ax, marker=marker, label=name.upper(), zorder=2)
        ax.legend()
        return ax

    def cross_sections(self) -> list[CrossSection]:
        """
        The cross-sections of ``isc1``, with 1-D cross-sections located at DIST
        along their segment.
        """
        sections = self.isc1.cross_sections(self.isc2)
        is_1d = np.flatnonzero(self.isc1.n >= 0)
        if len(is_1d):
            located = self.position(self.segment("isc1")[is_1d], self.isc1.dist[is_1d])
            for i, x, y, angle in zip(is_1d, *located):
                location = {"x": float(x), "y": float(y), "angle": float(angle)}
                sections[i] = sections[i].model_copy(update=location)
        return sections
