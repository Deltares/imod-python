import pathlib

import numpy as np
import pandas as pd

from imod.mf6.pkgbase import Package, VariableMetaData


class VerticesDiscretization(Package):
    """
    Discretization by Vertices (DISV).

    Parameters
    ----------
    top: array of floats (xu.UgridDataArray)
    bottom: array of floats (xu.UgridDataArray)
    idomain: array of integers (xu.UgridDataArray)
    """

    _pkg_id = "disv"
    _metadata_dict = {
        "top": VariableMetaData(np.floating),
        "bottom": VariableMetaData(np.floating),
        "idomain": VariableMetaData(np.integer),
    }
    _grid_data = {"top": np.float64, "bottom": np.float64, "idomain": np.int32}
    _keyword_map = {"bottom": "botm"}
    _template = Package._initialize_template(_pkg_id)

    def __init__(self, top, bottom, idomain):
        super().__init__(locals())
        self.dataset["idomain"] = idomain
        self.dataset["top"] = top
        self.dataset["bottom"] = bottom

        self._pkgcheck()

    def render(self, directory, pkgname, binary):
        disdirectory = directory / pkgname
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
            # Should be more effcient than mixed column?
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

            self._write_table_section(
                f,
                self._verts_dataframe(),
                "vertices",
                index=True,
            )
            f.write("\n")

            self._write_table_section(f, self._cell2d_dataframe(), "cell2d", index=True)

        return
