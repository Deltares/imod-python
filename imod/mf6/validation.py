"""
This module contains specific validation utilities for Modflow 6.
"""

from typing import cast

import numpy as np

from imod.schemata import DimsSchema, NoDataComparisonSchema, ValidationError
from imod.typing import GridDataArray

PKG_DIMS_SCHEMA = (
    DimsSchema("layer", "y", "x")
    | DimsSchema("layer", "{face_dim}")
    | DimsSchema("layer")
    | DimsSchema()
)

BOUNDARY_DIMS_SCHEMA = (
    DimsSchema("time", "layer", "y", "x")
    | DimsSchema("layer", "y", "x")
    | DimsSchema("time", "layer", "{face_dim}")
    | DimsSchema("layer", "{face_dim}")
    # Layer dim not necessary, as long as there is a layer coordinate present.
    | DimsSchema("time", "y", "x")
    | DimsSchema("y", "x")
    | DimsSchema("time", "{face_dim}")
    | DimsSchema("{face_dim}")
)

CONC_DIMS_SCHEMA = (
    DimsSchema("species", "time", "layer", "y", "x")
    | DimsSchema("species", "layer", "y", "x")
    | DimsSchema("species", "time", "layer", "{face_dim}")
    | DimsSchema("species", "layer", "{face_dim}")
    # Layer dim not necessary, as long as there is a layer coordinate present.
    | DimsSchema("species", "time", "y", "x")
    | DimsSchema("species", "y", "x")
    | DimsSchema("species", "time", "{face_dim}")
    | DimsSchema("species", "{face_dim}")
)


class DisBottomSchema(NoDataComparisonSchema):
    """
    Custom schema for the bottoms as these require some additional logic,
    because of how Modflow 6 computes cell thicknesses.
    """

    def validate(self, obj: GridDataArray, **kwargs) -> None:
        other_obj = kwargs[self.other]

        active = self.is_other_notnull(other_obj)
        bottom = obj

        # Only check for multi-layered models
        if bottom.coords["layer"].size > 1:
            # Check if zero thicknesses occur in active cells. The difference across
            # layers is a "negative thickness"
            thickness = bottom.diff(dim="layer") * -1.0
            if (thickness.where(active.isel(layer=slice(1, None))) <= 0.0).any():
                raise ValidationError("found thickness <= 0.0")

            # To compute thicknesses properly, Modflow 6 requires bottom data in the
            # layer above the active cell in question.
            overlaying_top_inactive = cast(GridDataArray, np.isnan(bottom)).shift(
                layer=1, fill_value=False
            )
            if (overlaying_top_inactive & active).any():
                raise ValidationError("inactive bottom above active cell")
