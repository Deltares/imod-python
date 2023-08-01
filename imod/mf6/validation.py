"""
This module contains specific validation utilities for Modflow 6.
"""
from typing import Dict, List

import numpy as np
import xarray as xr
import xugrid as xu

from imod.mf6.statusinfo import NestedStatusInfo, StatusInfo, StatusInfoBase
from imod.schemata import DimsSchema, NoDataComparisonSchema, ValidationError

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

    def validate(self, obj: xr.DataArray, **kwargs):
        other_obj = kwargs[self.other]

        active = self.is_other_notnull(other_obj)
        bottom = obj

        # Only check for multi-layered models
        if bottom.coords["layer"].size > 1:
            # UgridDataArray.where() cannot handle the layer removed by .diff.
            # FUTURE: Remove instance check if this issue is resolved:
            # https://github.com/Deltares/xugrid/issues/38
            if not isinstance(active, xu.UgridDataArray):
                # Check if zero thicknesses occur in active cells. The difference across
                # layers is a "negative thickness"
                thickness = bottom.diff(dim="layer") * -1.0
                if (thickness.where(active) <= 0.0).any():
                    raise ValidationError("found thickness <= 0.0")

            # To compute thicknesses properly, Modflow 6 requires bottom data in the
            # layer above the active cell in question.
            overlaying_top_inactive = np.isnan(bottom).shift(layer=1, fill_value=False)
            if (overlaying_top_inactive & active).any():
                raise ValidationError("inactive bottom above active cell")


def validation_pkg_error_message(pkg_errors):
    messages = []
    for var, var_errors in pkg_errors.items():
        messages.append(f"* {var}")
        messages.extend(f"\t- {error}" for error in var_errors)
    return "\n" + "\n".join(messages)


def validation_model_error_message(model_errors: NestedStatusInfo) -> str:
    messages = []
    for status_info in model_errors._NestedStatusInfo__children:
        messages.append(f"{status_info.title} : {status_info.errors} ")
    return str(messages)


def pkg_errors_to_status_info(
    pkg_name: str, pkg_errors: Dict[str, List[ValidationError]]
) -> StatusInfoBase:
    pkg_status_info = NestedStatusInfo(f"{pkg_name} package")
    for var_name, var_errors in pkg_errors.items():
        var_status_info = StatusInfo(var_name)
        for var_error in var_errors:
            var_status_info.add_error(str(var_error))
        pkg_status_info.add(var_status_info)

    return pkg_status_info
