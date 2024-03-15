from typing import Optional, Tuple

import numpy as np

from imod.mf6.interfaces.iregridpackage import IRegridPackage
from imod.mf6.package import Package
from imod.mf6.utilities.logging_decorators import init_log_decorator
from imod.mf6.utilities.regrid import RegridderType
from imod.mf6.validation import PKG_DIMS_SCHEMA
from imod.schemata import (
    AllValueSchema,
    DTypeSchema,
    IdentityNoDataSchema,
    IndexesSchema,
)


class MobileStorageTransfer(Package, IRegridPackage):
    """
    Mobile Storage.

    Parameters
    ----------
    porosity: array of floats (xr.DataArray)
        volume of interconnected voids per volume of rock (percentage).
    decay : array of floats (xr.DataArray, optional)
        is the rate coefficient for first or zero-order decay for the aqueous phase of the mobile domain.
        A negative value indicates solute production. The dimensions of decay for first-order decay is one
        over time. The dimensions of decay for zero-order decay is mass per length cubed per time. decay will
        have no effect on simulation results unless either first- or zero-order decay is specified in the
        options block.
    decay_sorbed : array of floats (xr.DataArray, optional)
        is the rate coefficient for first or zero-order decay for the sorbed phase of the mobile domain.
        A negative value indicates solute production. The dimensions of decay_sorbed for first-order decay
        is one over time. The dimensions of decay_sorbed for zero-order decay is mass of solute per mass of
        aquifer per time. If decay_sorbed is not specified and both decay and sorption are active, then the
        program will terminate with an error. decay_sorbed will have no effect on simulation results unless
        the SORPTION keyword and either first- or zero-order decay are specified in the options block.
    bulk_density : array of floats (xr.DataArray, optional)
        is the bulk density of the aquifer in mass per length cubed. bulk_density is not required unless
        the SORPTION keyword is specified.
    distcoef  : array of floats (xr.DataArray, optional)
        is the distribution coefficient for the equilibrium-controlled linear sorption isotherm in dimensions
        of length cubed per mass. distcoef is not required unless the SORPTION keyword is specified.
    sp2  : array of floats (xr.DataArray, optional)
        is the exponent for the Freundlich isotherm and the sorption capacity for the Langmuir isotherm.
    save_flows: ({True, False}, optional)
        Indicates that recharge flow terms will be written to the file specified
        with "BUDGET FILEOUT" in Output Control.
        Default is False.
    zero_order_decay: bool, optional
        Requires decay to be specified
    first_order_decay: bool, optional
        Requires decay to be specified
    sorption: ({linear, freundlich, langmuir}, optional)
        Type of sorption, if any.
    validate: {True, False}
        Flag to indicate whether the package should be validated upon
        initialization. This raises a ValidationError if package input is
        provided in the wrong manner. Defaults to True.
    """

    _grid_data = {
        "porosity": np.float64,
        "decay": np.float64,
        "decay_sorbed": np.float64,
        "bulk_density": np.float64,
        "distcoef": np.float64,
        "sp2": np.float64,
    }

    _pkg_id = "mst"
    _template = Package._initialize_template(_pkg_id)
    _keyword_map = {}
    _init_schemata = {
        "porosity": [DTypeSchema(np.floating), IndexesSchema(), PKG_DIMS_SCHEMA],
        "decay": [DTypeSchema(np.floating), IndexesSchema(), PKG_DIMS_SCHEMA],
        "decay_sorbed": [DTypeSchema(np.floating), IndexesSchema(), PKG_DIMS_SCHEMA],
        "bulk_density": [DTypeSchema(np.floating), IndexesSchema(), PKG_DIMS_SCHEMA],
        "distcoef": [DTypeSchema(np.floating), IndexesSchema(), PKG_DIMS_SCHEMA],
        "sp2": [DTypeSchema(np.floating), IndexesSchema(), PKG_DIMS_SCHEMA],
        "zero_order_decay": [DTypeSchema(np.bool_)],
        "first_order_decay": [DTypeSchema(np.bool_)],
    }

    _write_schemata = {
        "porosity": (
            AllValueSchema(">=", 0.0),
            AllValueSchema("<", 1.0),
            IdentityNoDataSchema(other="idomain", is_other_notnull=(">", 0)),
        ),
        "decay": (IdentityNoDataSchema(other="idomain", is_other_notnull=(">", 0)),),
        "decay_sorbed": (
            IdentityNoDataSchema(other="idomain", is_other_notnull=(">", 0)),
        ),
        "bulk_density": (
            IdentityNoDataSchema(other="idomain", is_other_notnull=(">", 0)),
        ),
        "distcoef": (IdentityNoDataSchema(other="idomain", is_other_notnull=(">", 0)),),
        "sp2": (IdentityNoDataSchema(other="idomain", is_other_notnull=(">", 0)),),
    }


    _regrid_method = {
        "porosity": (RegridderType.OVERLAP, "mean"),
        "decay": (RegridderType.OVERLAP, "mean"),
        "decay_sorbed": (
            RegridderType.OVERLAP,
            "mean",
        ), 
        "bulk_density": (RegridderType.OVERLAP, "mean"),
        "distcoef": (RegridderType.OVERLAP, "mean"),
        "sp2": (RegridderType.OVERLAP, "mean"),        
    }
    @init_log_decorator()
    def __init__(
        self,
        porosity,
        decay=None,
        decay_sorbed=None,
        bulk_density=None,
        distcoef=None,
        sp2=None,
        save_flows=False,
        zero_order_decay: bool = False,
        first_order_decay: bool = False,
        sorption=None,
        validate: bool = True,
    ):
        if zero_order_decay and first_order_decay:
            raise ValueError(
                "zero_order_decay and first_order_decay may not both be True"
            )
        dict_dataset = {
            "porosity": porosity,
            "decay": decay,
            "decay_sorbed": decay_sorbed,
            "bulk_density": bulk_density,
            "distcoef": distcoef,
            "sp2": sp2,
            "save_flows": save_flows,
            "sorption": sorption,
            "zero_order_decay": zero_order_decay,
            "first_order_decay": first_order_decay,
        }
        super().__init__(dict_dataset)
        self._validate_init_schemata(validate)
        
    def get_regrid_methods(self) -> Optional[dict[str, Tuple[RegridderType, str]]]:
        return self._regrid_method