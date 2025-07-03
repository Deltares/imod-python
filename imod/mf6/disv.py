from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd

from imod.common.interfaces.imaskingsettings import IMaskingSettings
from imod.common.interfaces.iregridpackage import IRegridPackage
from imod.logging import init_log_decorator
from imod.mf6.package import Package
from imod.mf6.regrid.regrid_schemes import DiscretizationRegridMethod
from imod.mf6.validation import DisBottomSchema
from imod.mf6.write_context import WriteContext
from imod.schemata import (
    AllCoordsValueSchema,
    AllValueSchema,
    AnyValueSchema,
    DimsSchema,
    DTypeSchema,
    IdentityNoDataSchema,
    IndexesSchema,
)


class VerticesDiscretization(Package, IRegridPackage, IMaskingSettings):
    """
    Discretization by Vertices (DISV).

    Parameters
    ----------
    top: array of floats (xu.UgridDataArray)
        is the top elevation for each cell in the top model layer.
    bottom: array of floats (xu.UgridDataArray)
        is the bottom elevation for each cell.
    idomain: array of integers (xu.UgridDataArray)
        Indicates the existence status of a cell.

        * If 0, the cell does not exist in the simulation. Input and output
          values will be read and written for the cell, but internal to the
          program, the cell is excluded from the solution.
        * If >0, the cell exists in the simulation.
        * If <0, the cell does not exist in the simulation. Furthermore, the
          first existing cell above will be connected to the first existing cell
          below. This type of cell is referred to as a "vertical pass through"
          cell.

        This UgridDataArray needs to contain a ``"layer"`` coordinate and a face
        dimension. Horizontal discretization information will be derived from
        its face dimension.
    validate: {True, False}
        Flag to indicate whether the package should be validated upon
        initialization. This raises a ValidationError if package input is
        provided in the wrong manner. Defaults to True.
    """

    _pkg_id = "disv"

    _init_schemata = {
        "top": [
            DTypeSchema(np.floating),
            DimsSchema("{face_dim}") | DimsSchema(),
            IndexesSchema(),
        ],
        "bottom": [
            DTypeSchema(np.floating),
            DimsSchema("layer", "{face_dim}") | DimsSchema("layer"),
            IndexesSchema(),
            AllCoordsValueSchema("layer", ">", 0),
        ],
        "idomain": [
            DTypeSchema(np.integer),
            DimsSchema("layer", "{face_dim}"),
            IndexesSchema(),
            AllCoordsValueSchema("layer", ">", 0),
        ],
    }
    _write_schemata = {
        "idomain": (AnyValueSchema(">", 0),),
        "top": (
            AllValueSchema(">", "bottom", ignore=("idomain", "<=", 0)),
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

    def _get_render_dictionary(self, directory, pkgname, globaltimes, binary):
        disdirectory = directory / pkgname
        d: dict[str, Any] = {}
        grid = self.dataset.ugrid.grid
        d["xorigin"] = 0.0
        d["yorigin"] = 0.0
        d["nlay"] = self.dataset["idomain"].coords["layer"].size
        facedim = grid.face_dimension
        d["ncpl"] = self.dataset["idomain"].coords[facedim].size
        d["nvert"] = grid.node_x.size

        _, d["top"] = self._compose_values(
            self.dataset["top"], disdirectory, "top", binary=binary
        )
        d["botm_layered"], d["botm"] = self._compose_values(
            self["bottom"], disdirectory, "botm", binary=binary
        )
        d["idomain_layered"], d["idomain"] = self._compose_values(
            self["idomain"], disdirectory, "idomain", binary=binary
        )
        return d

    def _verts_dataframe(self) -> pd.DataFrame:
        grid = self.dataset.ugrid.grid
        df = pd.DataFrame(grid.node_coordinates)
        df.index += 1
        return df

    def _cell2d_dataframe(self) -> pd.DataFrame:
        XUGRID_FILL = -1
        grid = self.dataset.ugrid.grid
        df = pd.DataFrame(grid.face_coordinates)
        df.index += 1
        # modflow requires clockwise; ugrid requires ccw
        face_nodes = grid.face_node_connectivity[:, ::-1]
        df[2] = (face_nodes != XUGRID_FILL).sum(axis=1)
        for i, column in enumerate(face_nodes.T):
            # Use extension array to write empty values
            # Should be more efficient than mixed column?
            df[3 + i] = pd.arrays.IntegerArray(
                values=column + 1,
                mask=(column == XUGRID_FILL),
            )
        return df

    def _append_vertices_and_cell2d(self, filename: Path | str) -> None:
        with open(filename, "a") as f:
            f.write("\n\n")

            f.write("begin vertices\n")
            self._verts_dataframe().to_csv(
                f, header=False, sep=" ", lineterminator="\n"
            )
            f.write("end vertices\n\n")

            f.write("begin cell2d\n")
            self._cell2d_dataframe().to_csv(
                f, header=False, sep=" ", lineterminator="\n"
            )
            f.write("end cell2d\n")

        return

    def write_blockfile(self, pkgname, globaltimes, write_context: WriteContext):
        super().write_blockfile(pkgname, globaltimes, write_context)
        filename = write_context.write_directory / f"{pkgname}.{self._pkg_id}"
        self._append_vertices_and_cell2d(filename)

        return

    def _validate(self, schemata, **kwargs):
        # Insert additional kwargs
        kwargs["bottom"] = self["bottom"]
        errors = super()._validate(schemata, **kwargs)

        return errors
