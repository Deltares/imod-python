import numpy as np
import xarray as xr

import imod
from imod.mf6.pkgbase import Package


class StructuredDiscretization(Package):
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
    """

    __slots__ = ("top", "bottom", "idomain")
    _pkg_id = "dis"
    _grid_data = {"top": np.float64, "bottom": np.float64, "idomain": np.int32}
    _keyword_map = {"bottom": "botm"}
    _template = Package._initialize_template(_pkg_id)

    def __init__(self, top, bottom, idomain):
        super(__class__, self).__init__()
        self["top"] = top
        self["bottom"] = bottom
        self["idomain"] = idomain

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

    def render(self, directory, pkgname, *args, **kwargs):
        disdirectory = directory / "dis"
        d = {}
        x = self["idomain"].coords["x"]
        y = self["idomain"].coords["y"]
        dx, xmin, _ = imod.util.coord_reference(x)
        dy, ymin, _ = imod.util.coord_reference(y)

        d["xorigin"] = xmin
        d["yorigin"] = ymin
        d["nlay"] = self["idomain"].coords["layer"].size
        d["nrow"] = y.size
        d["ncol"] = x.size
        d["delr"] = self._delrc(np.abs(dx))
        d["delc"] = self._delrc(np.abs(dy))
        _, d["top"] = self._compose_values(self["top"], disdirectory, "top")
        d["botm_layered"], d["botm"] = self._compose_values(
            self["bottom"], disdirectory, "botm"
        )
        d["idomain_layered"], d["idomain"] = self._compose_values(
            self["idomain"], disdirectory, "idomain"
        )

        return self._template.render(d)
