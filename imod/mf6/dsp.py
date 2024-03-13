import numpy as np

from imod.mf6.interfaces.iregridpackage import IRegridPackage
from imod.mf6.package import Package
from imod.mf6.utilities.regrid import RegridderType
from imod.mf6.validation import PKG_DIMS_SCHEMA
from imod.schemata import (
    CompatibleSettingsSchema,
    DimsSchema,
    DTypeSchema,
    IdentityNoDataSchema,
    IndexesSchema,
)
from typing import Optional, Tuple

class Dispersion(Package, IRegridPackage):
    """
    Molecular Diffusion and Dispersion.

    Parameters
    ----------
    diffusion_coefficient: xr.DataArray
        effective molecular diffusion coefficient. (DIFFC)
    longitudinal_horizontal: xr.DataArray
        longitudinal dispersivity in horizontal direction. If flow is strictly
        horizontal, then this is the longitudinal dispersivity that will be
        used. If flow is not strictly horizontal or strictly vertical, then the
        longitudinal dispersivity is a function of both ALH and ALV. If
        mechanical dispersion is represented (by specifying any dispersivity
        values) then this array is required. (ALH)
    transverse_horizontal1: xr.DataArray
        transverse dispersivity in horizontal direction. This is the transverse
        dispersivity value for the second ellipsoid axis. If flow is strictly
        horizontal and directed in the x direction (along a row for a regular
        grid), then this value controls spreading in the y direction.
        If mechanical dispersion is represented (by specifying any dispersivity
        values) then this array is required. (ATH1)
    longitudinal_vertical: xr.DataArray, optional
        longitudinal dispersivity in vertical direction. If flow is strictly
        vertical, then this is the longitudinal dispsersivity value that will be
        used. If flow is not strictly horizontal or strictly vertical, then the
        longitudinal dispersivity is a function of both ALH and ALV. If this
        value is not specified and mechanical dispersion is represented, then
        this array is set equal to ALH. (ALV)
    transverse_horizontal2: xr.DataArray, optional
        transverse dispersivity in horizontal direction. This is the transverse
        dispersivity value for the third ellipsoid axis. If flow is strictly
        horizontal and directed in the x direction (along a row for a regular
        grid), then this value controls spreading in the z direction. If this
        value is not specified and mechanical dispersion is represented, then
        this array is set equal to ATH1. (ATH2)
    tranverse_vertical: xr.DataArray, optional
        transverse dispersivity when flow is in vertical direction. If flow is
        strictly vertical and directed in the z direction, then this value
        controls spreading in the x and y directions. If this value is not
        specified and mechanical dispersion is represented, then this array is
        set equal to ATH2. (ATV)
    xt3d_off: bool, optional
        deactivate the xt3d method and use the faster and less accurate
        approximation. (XT3D_OFF)
    xt3d_rhs: bool, optional
        add xt3d terms to right-hand side, when possible. This option uses less
        memory, but may require more iterations. (XT3D_RHS)
    validate: {True, False}
        Flag to indicate whether the package should be validated upon
        initialization. This raises a ValidationError if package input is
        provided in the wrong manner. Defaults to True.
    """

    _pkg_id = "dsp"
    _template = Package._initialize_template(_pkg_id)
    _grid_data = {
        "diffusion_coefficient": np.float64,
        "longitudinal_horizontal": np.float64,
        "transversal_horizontal1": np.float64,
        "longitudinal_vertical": np.float64,
        "transversal_horizontal2": np.float64,
        "transversal_vertical": np.float64,
    }
    _keyword_map = {
        "diffusion_coefficient": "diffc",
        "longitudinal_horizontal": "alh",
        "transversal_horizontal1": "ath1",
        "longitudinal_vertical": "alv",
        "transversal_horizontal2": "ath2",
        "transversal_vertical": "atv",
    }
    _init_schemata = {
        "diffusion_coefficient": [
            DTypeSchema(np.floating),
            IndexesSchema(),
            PKG_DIMS_SCHEMA,
        ],
        "longitudinal_horizontal": [
            DTypeSchema(np.floating),
            IndexesSchema(),
            PKG_DIMS_SCHEMA,
        ],
        "transversal_horizontal1": [
            DTypeSchema(np.floating),
            IndexesSchema(),
            PKG_DIMS_SCHEMA,
        ],
        "longitudinal_vertical": [
            DTypeSchema(np.floating),
            IndexesSchema(),
            PKG_DIMS_SCHEMA,
        ],
        "transversal_horizontal2": [
            DTypeSchema(np.floating),
            IndexesSchema(),
            PKG_DIMS_SCHEMA,
        ],
        "transversal_vertical": [
            DTypeSchema(np.floating),
            IndexesSchema(),
            PKG_DIMS_SCHEMA,
        ],
        "xt3d_off": [DTypeSchema(np.bool_), DimsSchema()],
        "xt3d_rhs": [DTypeSchema(np.bool_), DimsSchema(), CompatibleSettingsSchema(other="xt3d_off", other_value=False)],
    }

    _write_schemata = {
        "diffusion_coefficient": (
            IdentityNoDataSchema(other="idomain", is_other_notnull=(">", 0)),
        ),
        "longitudinal_horizontal": (
            IdentityNoDataSchema(other="idomain", is_other_notnull=(">", 0)),
        ),
        "transversal_horizontal1": (
            IdentityNoDataSchema(other="idomain", is_other_notnull=(">", 0)),
        ),
        "longitudinal_vertical": (
            IdentityNoDataSchema(other="idomain", is_other_notnull=(">", 0)),
        ),
        "transversal_horizontal2": (
            IdentityNoDataSchema(other="idomain", is_other_notnull=(">", 0)),
        ),
        "transversal_vertical": (
            IdentityNoDataSchema(other="idomain", is_other_notnull=(">", 0)),
        ),
    }

    _regrid_method = {
        "diffusion_coefficient": (RegridderType.OVERLAP, "mean"),
        "longitudinal_horizontal": (RegridderType.OVERLAP, "mean"), 
        "transversal_horizontal1": (
            RegridderType.OVERLAP,
            "mean",
        ),
        "longitudinal_vertical": (
            RegridderType.OVERLAP,
            "mean",
        ),
        "transversal_horizontal2": (RegridderType.OVERLAP, "mean"),
        "transversal_vertical": (RegridderType.OVERLAP, "mean"),
    }


    def __init__(
        self,
        diffusion_coefficient,
        longitudinal_horizontal,
        transversal_horizontal1,
        longitudinal_vertical=None,
        transversal_horizontal2=None,
        transversal_vertical=None,
        xt3d_off=False,
        xt3d_rhs=False,
        validate: bool = True,
    ):
        dict_dataset = {
            "xt3d_off": xt3d_off,
            "xt3d_rhs": xt3d_rhs,
            "diffusion_coefficient": diffusion_coefficient,
            "longitudinal_horizontal": longitudinal_horizontal,
            "transversal_horizontal1": transversal_horizontal1,
            "longitudinal_vertical": longitudinal_vertical,
            "transversal_horizontal2": transversal_horizontal2,
            "transversal_vertical": transversal_vertical,
        }
        super().__init__(dict_dataset)
        self._validate_init_schemata(validate)

    def _validate(self, schemata, **kwargs):
        # Insert additional kwargs
        kwargs["xt3d_off"] = self["xt3d_off"]
        errors = super()._validate(schemata, **kwargs)

        return errors
    
    def get_regrid_methods(self) -> Optional[dict[str, Tuple[RegridderType, str]]]:
        return self._regrid_method    