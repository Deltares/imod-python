import pathlib

import numpy as np
import xugrid as xu

import imod
from imod.mf6.pkgbase import Package


class LayeredUnstructuredDiscretization(Package):
    """
    Currently only supports Unstructured-layered models, which are equivalent
    to structured by vertices (DISV) models but directly written to the (more
    efficient) DISU MODFLOW6 formats.

    Parameters
    ----------
    top: array of floats (xu.UgridDataArray)
    bottom: array of floats (xu.UgridDataArray)
    idomain: array of integers (xu.UgridDataArray)
    """

    _pkg_id = "disu"
    _grid_data = {
        "top": np.float64,
        "bottom": np.float64,
        "area": np.float64,
        "idomain": np.int32,
        "ia": np.int32,
        "ja": np.int32,
        "ihc": np.int32,
        "cl12": np.float64,
        "hwva": np.float64,
        "angldegx": np.float64,
    }
    _keyword_map = {"bottom": "botm"}
    _template = Package._initialize_template(_pkg_id)

    def __init__(self, top, bottom, idomain):
        super(__class__, self).__init__(locals())
        self.dataset["idomain"] = idomain
        self.dataset["top"] = top
        self.dataset["bottom"] = bottom

    def render(self, directory, pkgname, globaltimes, binary):
        disdirectory = directory / pkgname
        d = {}
        d["xorigin"] = self.dataset.grid.node_x.min()
        d["yorigin"] = self.dataset.grid.node_y.min()
        nlayer = self.dataset["idomain"].coords["layer"].size
        facedim = self.dataset.grid.face_dimension
        nface = self.dataset["idomain"].coords[facedim].size
        d["nodes"] = nlayer * nface
        d["nvert"] = self.dataset.grid.node_x.size

        d["top"] = self._compose_values(
            self.dataset["top"], disdirectory, "top", binary=binary
        )
        d["bot"] = self._compose_values(
            self["bottom"], disdirectory, "bot", binary=binary
        )

        area = ...
        d["area"] = self._compose_values(area, disdirectory, "area", binary=binary)
        d["idomain"] = self._compose_values(
            self["idomain"], disdirectory, "idomain", binary=binary
        )

        return self._template.render(d)

    def write_blockfile(self, directory, pkgname, *args):
        dir_for_render = pathlib.Path(directory.stem)
        content = self.render(dir_for_render, pkgname, *args)
        filename = directory / f"{pkgname}.{self._pkg_id}"
        with open(filename, "w") as f:
            f.write(content)
        # append text arrays
        # Create dataframe of vertices
        # Create dataframe of centroids, ncvert, icvert
