"""
Miscellaneous Utilities.

The utilies imported in this file are public API, and previously placed in
imod/util.py. Therefore these should be available under the imod.util namespace.
"""

import warnings

from imod.util.context import cd, ignore_warnings
from imod.util.path import temporary_directory
from imod.util.spatial import (
    empty_2d,
    empty_2d_transient,
    empty_3d,
    empty_3d_transient,
    from_mdal_compliant_ugrid2d,
    mdal_compliant_ugrid2d,
    spatial_reference,
    to_ugrid2d,
    transform,
    ugrid2d_data,
)
from imod.util.structured import replace, values_within_range, where
from imod.util.time import to_datetime


def round_extent(extent, cellsize):
    """
    This function is to preserve the imod.util.round_extent() namespace. Please
    refer to the new location in the future: imod.prepare.spatial.roundextent.
    """
    # Import locally to avoid circular imports
    from imod.prepare.spatial import round_extent
    warnings.warn(
        "Use of `imod.util.round_extent` is deprecated, please use the new "
        "location `imod.prepare.spatial.round_extent`", 
        DeprecationWarning
        )
    return round_extent(extent, cellsize)