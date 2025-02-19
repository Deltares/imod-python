"""
This module contains specific validation utilities for Modflow 6.
"""

from typing import Optional, cast

import numpy as np

from imod.mf6.statusinfo import NestedStatusInfo, StatusInfo, StatusInfoBase
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


def validation_pkg_error_message(pkg_errors):
    messages = []
    for var, var_errors in pkg_errors.items():
        messages.append(f"- {var}")
        messages.extend(f"    - {error}" for error in var_errors)
    return "\n" + "\n".join(messages)


def pkg_errors_to_status_info(
    pkg_name: str,
    pkg_errors: dict[str, list[ValidationError]],
    footer_text: Optional[str],
) -> StatusInfoBase:
    pkg_status_info = NestedStatusInfo(f"{pkg_name} package")
    for var_name, var_errors in pkg_errors.items():
        var_status_info = StatusInfo(var_name)
        for var_error in var_errors:
            var_status_info.add_error(str(var_error))
        pkg_status_info.add(var_status_info)
    pkg_status_info.set_footer_text(footer_text)
    return pkg_status_info
