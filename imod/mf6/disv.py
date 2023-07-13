import pathlib

import numpy as np
import pandas as pd

from imod.mf6.pkgbase import Package
from imod.mf6.regridding_utils import RegridderType
from imod.mf6.validation import DisBottomSchema
from imod.schemata import (
    AllValueSchema,
    AnyValueSchema,
    DimsSchema,
    DTypeSchema,
    IdentityNoDataSchema,
    IndexesSchema,
)


class VerticesDiscretization(Package):
    """
    Discretization by Vertices (DISV).

    Parameters
    ----------
    top: array of floats (xu.UgridDataArray)
    bottom: array of floats (xu.UgridDataArray)
    idomain: array of integers (xu.UgridDataArray)
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
        ],
        "idomain": [
            DTypeSchema(np.integer),
            DimsSchema("layer", "{face_dim}"),
            IndexesSchema(),
        ],
    }
    _write_schemata = {
        "idomain": (AnyValueSchema(">", 0),),
        "top": (
            AllValueSchema(">", "bottom"),
            IdentityNoDataSchema(other="idomain", is_other_notnull=(">", 0)),
            # No need to check coords: dataset ensures they align with idomain.
        ),
        "bottom": (DisBottomSchema(other="idomain", is_other_notnull=(">", 0)),),
    }

    _grid_data = {"top": np.float64, "bottom": np.float64, "idomain": np.int32}
    _keyword_map = {"bottom": "botm"}
    _template = Package._initialize_template(_pkg_id)

    _regrid_method = {
        "top": (RegridderType.OVERLAP, "mean"),
        "bottom": (RegridderType.OVERLAP, "mean"),
        "idomain": (RegridderType.OVERLAP, "mean"),
    }

    _skip_mask_arrays = ["bottom"]

    def __init__(self, top, bottom, idomain, validate: bool = True):
        super().__init__(locals())
        self.dataset["idomain"] = idomain
        self.dataset["top"] = top
        self.dataset["bottom"] = bottom
        self._validate_init_schemata(validate)

    def render(self, directory, pkgname, binary):
        disdirectory = pathlib.Path(directory.stem) / pkgname
        d = {}
        grid = self.dataset.ugrid.grid
        d["xorigin"] = grid.node_x.min()
        d["yorigin"] = grid.node_y.min()
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
        return self._template.render(d)

    def _verts_dataframe(self) -> pd.DataFrame:
        grid = self.dataset.ugrid.grid
        df = pd.DataFrame(grid.node_coordinates)
        df.index += 1
        return df

    def _cell2d_dataframe(self) -> pd.DataFrame:
        grid = self.dataset.ugrid.grid
        df = pd.DataFrame(grid.face_coordinates)
        df.index += 1
        # modflow requires clockwise; ugrid requires ccw
        face_nodes = grid.face_node_connectivity[:, ::-1]
        df[2] = (face_nodes != grid.fill_value).sum(axis=1)
        for i, column in enumerate(face_nodes.T):
            # Use extension array to write empty values
            # Should be more efficient than mixed column?
            df[3 + i] = pd.arrays.IntegerArray(
                values=column + 1,
                mask=(column == grid.fill_value),
            )
        return df

    def write_blockfile(self, directory, pkgname, globaltimes, binary):
        dir_for_render = pathlib.Path(directory.stem)
        content = self.render(dir_for_render, pkgname, binary)
        filename = directory / f"{pkgname}.{self._pkg_id}"
        with open(filename, "w") as f:
            f.write(content)
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
