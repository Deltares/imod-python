"""
This module contains specific validation utilities for Modflow 6.
"""

import numpy as np
import xarray as xr
import xugrid as xu

from imod.schemata import DimsSchema, NoDataComparisonSchema, ValidationError

# %% Template schemata to avoid code duplication
PKG_DIMS_SCHEMA = (
    DimsSchema("layer", "y", "x")
    | DimsSchema("layer", "{face_dim}")
    | DimsSchema("layer")
    | DimsSchema()
)

BC_DIMS_SCHEMA = (
    DimsSchema("time", "layer", "y", "x")
    | DimsSchema("layer", "y", "x")
    | DimsSchema("time", "layer", "{face_dim}")
    | DimsSchema("layer", "{face_dim}")
    # Layer dim not necessary, as long as there is a layer coordinate
    # present
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
    # Layer dim not necessary, as long as there is a layer coordinate
    # present
    | DimsSchema("species", "time", "y", "x")
    | DimsSchema("species", "y", "x")
    | DimsSchema("species", "time", "{face_dim}")
    | DimsSchema("species", "{face_dim}")
)


# %% Custom schemata for Modflow 6
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
            overlaying_top_inactive = np.isnan(
                bottom.shift(
                    layer=1, fill_value=9999.0
                )  # use fill_value to make layer 1 not nan
            )
            if (overlaying_top_inactive & active).any():
                raise ValidationError("inactive bottom above active cell")


# %% Custom utility functions
def validation_model_error_message(model_errors):
    messages = []
    for name, pkg_errors in model_errors.items():
        pkg_header = f"{name}\n" + len(name) * "-" + "\n"
        messages.append(pkg_header)
        messages.append(validation_pkg_error_message(pkg_errors))
    return "\n" + "\n".join(messages)


def validation_pkg_error_message(pkg_errors):
    messages = []
    for var, var_errors in pkg_errors.items():
        messages.append(f"* {var}")
        messages.extend(f"\t- {error}" for error in var_errors)
    return "\n" + "\n".join(messages)
