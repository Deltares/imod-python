import pathlib
from typing import Any, List

import numpy as np

import imod
from imod.logging import init_log_decorator
from imod.mf6.interfaces.imaskingsettings import IMaskingSettings
from imod.mf6.interfaces.iregridpackage import IRegridPackage
from imod.mf6.package import Package
from imod.mf6.regrid.regrid_schemes import DiscretizationRegridMethod
from imod.mf6.validation import DisBottomSchema
from imod.schemata import (
    ActiveCellsConnectedSchema,
    AllValueSchema,
    AnyValueSchema,
    DimsSchema,
    DTypeSchema,
    IdentityNoDataSchema,
    IndexesSchema,
)


class StructuredDiscretization(Package, IRegridPackage, IMaskingSettings):
    """
    Discretization information for structered grids is specified using the file.
    (DIS6) Only one discretization input file (DISU6, DISV6 or DIS6) can be
    specified for a model.
    https://water.usgs.gov/water-resources/software/MODFLOW-6/mf6io_6.0.4.pdf#page=35

    Parameters
    ----------
    top: array of floats (xr.DataArray)
        is the top elevation for each cell in the top model layer.
    bottom: array of floats (xr.DataArray)
        is the bottom elevation for each cell.
    idomain: array of integers (xr.DataArray)
        Indicates the existence status of a cell. Horizontal discretization
        information will be derived from the x and y coordinates of the
        DataArray. If the idomain value for a cell is 0, the cell does not exist
        in the simulation. Input and output values will be read and written for
        the cell, but internal to the program, the cell is excluded from the
        solution. If the idomain value for a cell is 1, the cell exists in the
        simulation. if the idomain value for a cell is -1, the cell does not
        exist in the simulation. Furthermore, the first existing cell above will
        be connected to the first existing cell below. This type of cell is
        referred to as a "vertical pass through" cell.
    validate: {True, False}
        Flag to indicate whether the package should be validated upon
        initialization. This raises a ValidationError if package input is
        provided in the wrong manner. Defaults to True.
    """

    _pkg_id = "dis"
    _init_schemata = {
        "top": [
            DTypeSchema(np.floating),
            DimsSchema("y", "x") | DimsSchema(),
            IndexesSchema(),
        ],
        "bottom": [
            DTypeSchema(np.floating),
            DimsSchema("layer", "y", "x") | DimsSchema("layer"),
            IndexesSchema(),
        ],
        "idomain": [
            DTypeSchema(np.integer),
            DimsSchema("layer", "y", "x"),
            IndexesSchema(),
        ],
    }
    _write_schemata = {
        "idomain": (
            ActiveCellsConnectedSchema(is_notnull=("!=", 0)),
            AnyValueSchema(">", 0),
        ),
        "top": (
            AllValueSchema(">", "bottom", ignore=("idomain", "==", -1)),
            IdentityNoDataSchema(other="idomain", is_other_notnull=(">", 0)),
            # No need to check coords: dataset ensures they align with idomain.
        ),
        "bottom": (DisBottomSchema(other="idomain", is_other_notnull=(">", 0)),),
    }

    _grid_data = {"top": np.float64, "bottom": np.float64, "idomain": np.int32}
    _keyword_map = {"bottom": "botm"}
    _template = Package._initialize_template(_pkg_id)
    _regrid_method = DiscretizationRegridMethod()

    @property
    def skip_variables(self) -> List[str]:
        return ["bottom"]

    @init_log_decorator()
    def __init__(self, top, bottom, idomain, validate: bool = True):
        dict_dataset = {
            "idomain": idomain,
            "top": top,
            "bottom": bottom,
        }
        super().__init__(dict_dataset)
        self._validate_init_schemata(validate)

    def _delrc(self, dx):
        """
        dx means dx or dy
        """
        if isinstance(dx, (int, float)):
            return f"constant {dx}"
        elif isinstance(dx, np.ndarray):
            arrstr = str(dx)[1:-1]
            return f"internal\n    {arrstr}"
        else:
            raise ValueError(f"Unhandled type of {dx}")

    def render(self, directory, pkgname, globaltimes, binary):
        disdirectory = pathlib.Path(directory) / pkgname
        d: dict[str, Any] = {}
        x = self.dataset["idomain"].coords["x"]
        y = self.dataset["idomain"].coords["y"]
        dx, xmin, _ = imod.util.spatial.coord_reference(x)
        dy, ymin, _ = imod.util.spatial.coord_reference(y)

        d["xorigin"] = xmin
        d["yorigin"] = ymin
        d["nlay"] = self.dataset["idomain"].coords["layer"].size
        d["nrow"] = y.size
        d["ncol"] = x.size
        d["delr"] = self._delrc(np.abs(dx))
        d["delc"] = self._delrc(np.abs(dy))
        _, d["top"] = self._compose_values(
            self["top"], disdirectory, "top", binary=binary
        )
        d["botm_layered"], d["botm"] = self._compose_values(
            self["bottom"], disdirectory, "botm", binary=binary
        )
        d["idomain_layered"], d["idomain"] = self._compose_values(
            self["idomain"], disdirectory, "idomain", binary=binary
        )

        return self._template.render(d)

    def _validate(self, schemata, **kwargs):
        # Insert additional kwargs
        kwargs["bottom"] = self["bottom"]
        errors = super()._validate(schemata, **kwargs)

        return errors
