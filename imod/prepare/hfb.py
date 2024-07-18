from itertools import pairwise
from typing import TYPE_CHECKING, List, Tuple

from imod.util.imports import MissingOptionalModule

if TYPE_CHECKING:
    import shapely
else:
    try:
        import shapely
    except ImportError:
        shapely = MissingOptionalModule("shapely")


def _line_to_square_zpolygon(
    x: Tuple[float, float], y: Tuple[float, float], z: Tuple[float, float]
) -> shapely.Polygon:
    """
    Creates polygon as follows::

        xy0,z0 -- xy1,z0
           |         |
           |         |
           |         |
        xy0,z1 -- xy1,z1
    """
    return shapely.Polygon(
        (
            (x[0], y[0], z[0]),
            (x[0], y[0], z[1]),
            (x[1], y[1], z[1]),
            (x[1], y[1], z[0]),
        ),
    )


def linestring_to_square_zpolygons(
    barrier_x: List[float],
    barrier_y: List[float],
    barrier_ztop: List[float],
    barrier_zbottom: List[float],
) -> List[shapely.Polygon]:
    """
    Create square vertical polygons from linestrings, with a varying ztop and
    zbottom over the line. Note: If the lists of x and y values of length N, the
    list of z values need to have length N-1. These are shaped as follows::

        xy0,zt0 -- xy1,zt0
           |          |
           |       xy1,zt1 ---- xy2,zt1
           |          |            |
        xy0,zb0 -- xy1,zb0         |
                      |            |
                      |            |
                   xy1,zb1 ---- xy2,zb1

    Parameters
    ----------
    barrier_x: list of floats
        x-locations of barrier, length N
    barrier_y: list of floats
        y-locations of barrier, length N
    barrier_ztop: list of floats
        top of barrier, length N-1
    barrier_zbot: list of floats
        bottom of barrier, length N-1

    Returns
    -------
    List of polygons with z dimension.

    Examples
    --------

    >>> x = [-10.0, 0.0, 10.0]
    >>> y = [10.0, 0.0, -10.0]
    >>> ztop = [10.0, 20.0]
    >>> zbot = [-10.0, -20.0]
    >>> polygons = linestring_to_square_zpolygons(x, y, ztop, zbot)

    You can use these polygons to construct horizontal flow barriers:

    >>> geometry = gpd.GeoDataFrame(geometry=polygons, data={
    >>>         "resistance": [1e3, 1e3],
    >>>     },
    >>> )
    >>> hfb = imod.mf6.HorizontalFlowBarrierResistance(geometry, print_input)
    """
    n = len(barrier_x)
    expected_lengths = (n, n, n - 1, n - 1)
    actual_lengths = (
        len(barrier_x),
        len(barrier_y),
        len(barrier_ztop),
        len(barrier_zbottom),
    )
    if expected_lengths != actual_lengths:
        raise ValueError(
            "Lengths of barrier data, not properly made. For lengths: (x, y,"
            f" ztop, zbottom). Expected lengths: {expected_lengths}, received"
            f" lengths: {actual_lengths}"
        )

    x_pairs = pairwise(barrier_x)
    y_pairs = pairwise(barrier_y)
    z_pairs = zip(barrier_ztop, barrier_zbottom)
    return [
        _line_to_square_zpolygon(x, y, z) for x, y, z in zip(x_pairs, y_pairs, z_pairs)
    ]


def _line_to_trapezoid_zpolygon(
    x: Tuple[float, float],
    y: Tuple[float, float],
    zt: Tuple[float, float],
    zb: Tuple[float, float],
) -> shapely.Polygon:
    """
    Creates polygon as follows::

        xy0,zt0
           |    \
           |     \
           |      xy1,zt1
           |         |   
           |         |   
           |      xy1,zb1
           |     /   
           |    /    
        xy0,zb0  
    """
    return shapely.Polygon(
        (
            (x[0], y[0], zt[0]),
            (x[0], y[0], zb[1]),
            (x[1], y[1], zt[1]),
            (x[1], y[1], zb[0]),
        ),
    )


def linestring_to_trapezoid_zpolygons(
    barrier_x: List[float],
    barrier_y: List[float],
    barrier_ztop: List[float],
    barrier_zbottom: List[float],
) -> List[shapely.Polygon]:
    """
    Create trapezoid vertical polygons from linestrings, with a varying ztop and
    zbottom over the line. These are shaped as follows::

        xy0,zt0              xy2,zt2
           |    \          /    |
           |     \        /     |
           |      xy1,zt1       |
           |         |          |
           |         |          |
           |      xy1,zb1 -- xy2,zb2
           |     /
           |    /
        xy0,zb0

    Parameters
    ----------
    barrier_x: list of floats
        x-locations of barrier, length N
    barrier_y: list of floats
        y-locations of barrier, length N
    barrier_ztop: list of floats
        top of barrier, length N
    barrier_zbot: list of floats
        bottom of barrier, length N

    Returns
    -------
    List of polygons with z dimension.

    Examples
    --------

    >>> x = [-10.0, 0.0, 10.0]
    >>> y = [10.0, 0.0, -10.0]
    >>> ztop = [10.0, 20.0, 15.0]
    >>> zbot = [-10.0, -20.0, 0.0]
    >>> polygons = linestring_to_trapezoid_zpolygons(x, y, ztop, zbot)

    You can use these polygons to construct horizontal flow barriers:

    >>> geometry = gpd.GeoDataFrame(geometry=polygons, data={
    >>>         "resistance": [1e3, 1e3],
    >>>     },
    >>> )
    >>> hfb = imod.mf6.HorizontalFlowBarrierResistance(geometry, print_input)
    """

    n = len(barrier_x)
    expected_lengths = (n, n, n, n)
    actual_lengths = (
        len(barrier_x),
        len(barrier_y),
        len(barrier_ztop),
        len(barrier_zbottom),
    )
    if expected_lengths != actual_lengths:
        raise ValueError(
            "Lengths of barrier data, not properly made. For lengths: (x, y,"
            f" ztop, zbottom). Expected lengths: {expected_lengths}, received"
            f" lengths: {actual_lengths}"
        )

    x_pairs = pairwise(barrier_x)
    y_pairs = pairwise(barrier_y)
    zt_pairs = pairwise(barrier_ztop)
    zb_pairs = pairwise(barrier_zbottom)
    return [
        _line_to_trapezoid_zpolygon(x, y, zt, zb)
        for x, y, zt, zb in zip(x_pairs, y_pairs, zt_pairs, zb_pairs)
    ]
