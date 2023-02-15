import numpy as np
import xugrid as xu

from imod.wq.pkgbase import Package


def create_hfb_array(notnull, resistance, layer, ibound, grid):
    nlayer = ibound["ibound_layer"].size
    idomain2d = ibound.values.reshape((nlayer, -1))
    no_layer_dim = notnull.ndim == 1
    edge_faces = grid.edge_face_connectivity

    # Fill in the indices
    # For every edge, find the connected faces.
    if no_layer_dim:
        edge = np.argwhere(notnull).transpose()[0]
        layer = layer - 1
        cell2d = edge_faces[edge]
        valid = ((cell2d != grid.fill_value) & (idomain2d[layer, cell2d] > 0)).all(
            axis=1
        )
    else:
        layer, edge = np.argwhere(notnull).transpose()
        layer2d = np.repeat(layer, 2).reshape((-1, 2))
        cell2d = edge_faces[edge]
        valid = ((cell2d != grid.fill_value) & (idomain2d[layer2d, cell2d] > 0)).all(
            axis=1
        )
        layer = layer[valid]

    # Skip the exterior edges (marked with a fill value).
    cell2d = cell2d[valid]
    c = resistance[notnull][valid]

    # Define the numpy structured array dtype
    field_spec = [
        ("layer_1", np.int32),
        ("row_1", np.int32),
        ("column_1", np.int32),
        ("layer_2", np.int32),
        ("row_2", np.int32),
        ("column_2", np.int32),
        ("resistance", np.float64),
    ]
    dtype = np.dtype(field_spec)
    shape = (ibound["y"].size, ibound["x"].size)
    row_1, column_1 = np.unravel_index(cell2d[:, 0], shape)
    row_2, column_2 = np.unravel_index(cell2d[:, 1], shape)
    # Set the indices
    recarr = np.empty(len(cell2d), dtype=dtype)
    recarr["layer_1"] = layer + 1
    recarr["row_1"] = row_1 + 1
    recarr["column_1"] = column_1 + 1
    recarr["row_2"] = row_2 + 1
    recarr["column_2"] = column_2 + 1
    recarr["resistance"] = c
    return recarr


class HorizontalFlowBarrier(Package):
    """
    Horizontal Flow Barrier package.

    Parameters
    ----------
    resistance: xu.UgridDataArray
    ibound: xr.DataArray
    """

    _pkg_id = "hfb6"

    _template = "[hfb6]\n    hfbfile = {hfbfile}\n"

    def __init__(
        self,
        resistance,
        ibound,
    ):
        self.dataset = xu.UgridDataset()
        self["resistance"] = resistance
        self["ibound"] = ibound.rename({"layer": "ibound_layer"})

    def _render(self, directory, *args, **kwargs):
        d = {"hfbfile": (directory / "horizontal_flow_barrier.hfb").as_posix()}
        return self._template.format(**d)

    def save(self, directory):
        directory.mkdir(exist_ok=True)  # otherwise handled by idf.save
        path = (directory / "horizontal_flow_barrier.hfb").as_posix()

        hfb_array = create_hfb_array(
            notnull=self["resistance"].notnull().values,
            resistance=self["resistance"].values,
            layer=self["resistance"].coords["layer"].values,
            ibound=self["ibound"],
            grid=self.dataset.ugrid.grid,
        )
        nhfbnp = len(hfb_array)
        header = f"0 0 {nhfbnp} 0"
        fmt = ("%i",) * 5 + ("%.18G",)

        with open(path, "w") as f:
            np.savetxt(fname=f, X=hfb_array, fmt=fmt, header=header)

    def _pkgcheck(self, ibound=None):
        pass
