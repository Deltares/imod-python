"""
Cross-sections: typed views of ISC2 records, as interpreted by the ISC1 record
that points to them.
"""

from typing import TYPE_CHECKING, Any, Generic, Self

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from imod.formats.isg.common import FloatT, SpatialTable, Vector, shapely

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from imod.formats.isg.isg import ISC2

PACKED: dict[str, Any] = {
    "names": ["zm", "zc", "zp"],
    "formats": ["<i2", "i1", "i1"],
    "offsets": [0, 2, 3],
}


class CrossSection1D(SpatialTable, Generic[FloatT]):
    """
    1-D cross-section (ISC1 N > 0): the bathymetry of a stream as a
    cross-section perpendicular to the direction of the stream.

    Attributes for each record in an ISC2-file, whenever N>0.

    Geometry: a Point Z per record, perpendicular to the segment at the
    cross-section, with BOTTOM as z; ``plot()`` draws BOTTOM against DISTANCE.

    Parameters
    ----------
    x: float, optional
        X-coordinate of the cross-section at DIST along its segment (meter); set
        by ``ISG.cross_sections()``.
    y: float, optional
        Y-coordinate of the cross-section at DIST along its segment (meter); set
        by ``ISG.cross_sections()``.
    angle: float, optional
        Direction of the segment at the cross-section (radians from the
        x-axis); set by ``ISG.cross_sections()``.
    distance: array of floats
        Distance of the cross-section measured from the centre of the riverbed
        (minus to the left en positive to the right).
    bottom: array of floats
        Bottom level of the riverbed (meter), whereby zero will be assigned to
        the lowest riverbed level.
    mrc: array of floats
        Manning's roughness coefficient (-).
    """

    x: float = np.nan
    y: float = np.nan
    angle: float = np.nan
    distance: Vector[FloatT]
    bottom: Vector[FloatT]
    mrc: Vector[FloatT]

    @classmethod
    def view(cls, isc2: "ISC2[Any]", rows: slice) -> Self:
        """The ISC2 records in ``rows``, as views."""
        return cls(distance=isc2.a[rows], bottom=isc2.b[rows], mrc=isc2.c[rows])

    def coordinates(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Map coordinates of the records; positive DISTANCE lies right of the segment."""
        x = self.x + self.distance * np.sin(self.angle)
        y = self.y - self.distance * np.cos(self.angle)
        return x, y

    def geometry(self) -> NDArray[np.object_]:
        return shapely.points(*self.coordinates(), self.bottom)

    def plot(self, ax: "Axes | None" = None, **kwargs: Any) -> "Axes":
        """Plot BOTTOM against DISTANCE."""
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(self.distance, self.bottom, **kwargs)
        ax.set(xlabel="distance (m)", ylabel="bottom (m)")
        return ax


class BaseCrossSection2D(SpatialTable, Generic[FloatT]):
    """
    2-D cross-section (ISC1 N < 0): the bathymetry of a stream as a 2-D
    representation, a grid of riverbed "pixels".

    Geometry: a box of ‖DX‖ by ‖DY‖ per "pixel"; ``plot()`` colours them by
    bottom level.

    Parameters
    ----------
    dx: float
        Width DX in meters of the rectangular raster that follows. Negative if
        the bottom levels are packed with a reference height, see
        :class:`CrossSection2DReferenceHeight`; the width is ‖DX‖.
    dy: float
        Height DY in meters of the rectangular raster that follows. Negative if
        the bottom levels are packed with a reference height; the height is
        ‖DY‖.
    href: float
        Reference Height in meters, this is used whenever specified (DX < 0.0
        and DY < 0.0). A pointer value Zp causes the whether a area is inundated
        once the HREF is exceeded as well as the bottom elevation (Zm + Zc). In
        this case the attribute Z (specified below) is organized differently as
        a combination of Zm, Zc and Zp.
    x: array of floats
        X coordinate (meter) for a riverbed "pixel", these coordinates need to
        be on a rectangular network with spatial distance of ‖DX‖ measured at
        the centre of the "pixel".
    y: array of floats
        Y coordinate (meter) for a riverbed "pixel", these coordinates need to
        be on a rectangular network with spatial distance of ‖DY‖ measured at
        the centre of the "pixel".
    """

    # The manual describes DX and DY as absolute values; the files store the
    # sign, which marks packed bottom levels.

    dx: float
    dy: float
    href: float
    x: Vector[FloatT]
    y: Vector[FloatT]

    @classmethod
    def view(cls, isc2: "ISC2[Any]", rows: slice) -> Self:
        """
        The ISC2 records in ``rows``, as views: the first holds DX, DY and
        HREF, the following the "pixels".
        """
        first, pixels = rows.start, slice(rows.start + 1, rows.stop)
        return cls(
            dx=float(isc2.a[first]),
            dy=float(isc2.b[first]),
            href=float(isc2.c[first]),
            x=isc2.a[pixels],
            y=isc2.b[pixels],
            **cls.unpack(isc2.c[pixels]),
        )

    @classmethod
    def unpack(cls, c: NDArray[np.floating]) -> dict[str, NDArray[Any]]:
        """The bottom level columns stored in the third value of ``c``."""
        raise NotImplementedError

    def bottom(self) -> NDArray[np.floating]:
        """Bottom level of the riverbed (meter)."""
        raise NotImplementedError

    def dataframe(self) -> pd.DataFrame:
        return super().dataframe().assign(bottom=self.bottom())

    def geometry(self) -> NDArray[np.object_]:
        hx, hy = abs(self.dx) / 2, abs(self.dy) / 2
        return shapely.box(self.x - hx, self.y - hy, self.x + hx, self.y + hy)

    def plot(self, ax: "Axes | None" = None, **kwargs: Any) -> "Axes":
        """Plot the "pixels" coloured by bottom level."""
        return super().plot(ax=ax, **{"column": "bottom", "legend": True, **kwargs})


class CrossSection2D(BaseCrossSection2D[FloatT]):
    """
    2-D cross-section with bottom levels Z.

    Following record is valid unless DX < 0.0 and DY < 0.0.

    Parameters
    ----------
    dx, dy, href, x, y
        See :class:`BaseCrossSection2D`.
    z: array of floats
        Bottom level of the riverbed (meter).
    """

    # The manual says DX > 0.0 and DY > 0.0; iMOD reads Z as a real for any
    # other combination of signs too.

    z: Vector[FloatT]

    @classmethod
    def unpack(cls, c: NDArray[np.floating]) -> dict[str, NDArray[Any]]:
        return {"z": c}

    def bottom(self) -> NDArray[np.floating]:
        return self.z


class CrossSection2DReferenceHeight(BaseCrossSection2D[FloatT]):
    """
    2-D cross-section with a reference height HREF: packed bottom levels (Zm,
    Zc) and pointers (Zp) to inundate above HREF.

    Following record is valid whenever DX < 0.0 and DY < 0.0. Zm and Zc both
    carry the sign of the bottom level, which is Zm + Zc / 100.

    Parameters
    ----------
    dx, dy, href, x, y
        See :class:`BaseCrossSection2D`.
    zm: array of ints
        Integer value of bottom level of the riverbed (integer of meters), e.g.
        bottom level is -23.43, Zm =-23.
    zc: array of ints
        Integer value of remaining digits of bottom level of the riverbed
        (remaining centimeter), e.g. bottom level is -23.43, Zc =-43
        centimeters.
    zp: array of ints
        Integer value of area affected by HREF, e.g. areas with Zp < 0 will be
        inundated only whenever the current river stage is higher than the
        Reference Height (HREF) and the river stage is higher than the
        corresponding riverbed Zm +Zc. Areas with Zp > 0, will be inundated
        whenever the river stage is higher than the current riverbed. The
        absolute value of Zp is used as a multiplication factor for the river
        bed resistances for the attribute RESIS in the ISD2-file.
    """

    # The manual's example gives Zc=43 for -23.43. iMOD truncates both parts
    # towards zero, so Zc=-43 (ISGSAVEIT), and reads Zm + Zc / 100 (ISGREADISC),
    # in imod_isg_utl.f90.

    zm: Vector[np.int16]
    zc: Vector[np.int8]
    zp: Vector[np.int8]

    @classmethod
    def unpack(cls, c: NDArray[np.floating]) -> dict[str, NDArray[Any]]:
        """Zm, Zc and Zp, packed in the first 4 bytes of each value."""
        packed = c.view(np.dtype({**PACKED, "itemsize": c.itemsize}))  # type: ignore[call-overload]
        return {name: packed[name] for name in PACKED["names"]}

    def bottom(self) -> NDArray[np.floating]:
        """Bottom level of the riverbed (meter): Zm + Zc / 100."""
        return self.zm + self.zc / 100.0


CrossSection = (
    CrossSection1D[Any] | CrossSection2D[Any] | CrossSection2DReferenceHeight[Any]
)
