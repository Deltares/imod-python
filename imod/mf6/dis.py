from pathlib import Path

import numpy as np
import xarray as xr

import imod
from imod.mf6.pkgbase import Package

from .read_input import read_dis_blockfile


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

    _pkg_id = "dis"
    _grid_data = {"top": np.float64, "bottom": np.float64, "idomain": np.int32}
    _keyword_map = {"bottom": "botm"}
    _template = Package._initialize_template(_pkg_id)

    def __init__(self, top, bottom, idomain):
        super(__class__, self).__init__(locals())
        self.dataset["idomain"] = idomain
        self.dataset["top"] = top
        self.dataset["bottom"] = bottom

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

    @classmethod
    def open(cls, path: str, simroot: Path) -> "StructuredDiscretization":
        content = read_dis_blockfile(simroot / path, simroot)
        nlay = content["nlay"]

        griddata = content["griddata"]
        xmin = float(content.get("xorigin", 0.0))
        ymin = float(content.get("yorigin", 0.0))
        dx = griddata.pop("delr")
        dy = griddata.pop("delc")
        x = (xmin + np.cumsum(dx)) - 0.5 * dx
        y = ((ymin + np.cumsum(dy)) - 0.5 * dy)[::-1]

        # Top is a 2D array unlike the others
        top = xr.DataArray(
            data=griddata.pop("top"),
            coords={"y": y, "x": x},
            dims=("y", "x"),
        )

        coords = {"layer": np.arange(1, nlay + 1), "y": y, "x": x}
        dims = ("layer", "y", "x")
        inverse_keywords = {v: k for k, v in cls._keyword_map.items()}
        variables = {"top": top}
        for key, value in griddata.items():
            invkey = inverse_keywords.get(key, key)
            variables[invkey] = xr.DataArray(
                value,
                coords,
                dims,
            )

        return cls(**variables)

    def render(self, directory, pkgname, globaltimes, binary):
        disdirectory = directory / pkgname
        d = {}
        x = self.dataset["idomain"].coords["x"]
        y = self.dataset["idomain"].coords["y"]
        dx, xmin, _ = imod.util.coord_reference(x)
        dy, ymin, _ = imod.util.coord_reference(y)

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
