import numba as nb
import numpy as np
import xarray as xr
from scipy import sparse

import imod
from imod.mf6.pkgbase import Package

IntArray = np.ndarray


@nb.njit(inline="always")
def _number(k, i, j, nrow, ncolumn):
    return k * (nrow * ncolumn) + i * ncolumn + j


# @nb.njit
def _structured_connectivity(idomain):
    nlayer, nrow, ncolumn = idomain.shape
    # Pre-allocate: structured connectivity implies maximum of 8 neighbors
    nconnection = idomain.size * 8
    ii = np.empty(nconnection, dtype=np.int32)
    jj = np.empty(nconnection, dtype=np.int32)

    connection = 0
    for k in range(nlayer):
        for i in range(nrow):
            for j in range(ncolumn):
                # Skip inactive or pass-through cells
                if idomain[k, i, j] <= 0:
                    continue

                if j < ncolumn - 1:
                    if idomain[k, i, j + 1] > 0:
                        ii[connection] = _number(k, i, j, nrow, ncolumn)
                        jj[connection] = _number(k, i, j + 1, nrow, ncolumn)
                        connection += 1

                if i < nrow - 1:
                    if idomain[k, i + 1, j] > 0:
                        ii[connection] = _number(k, i, j, nrow, ncolumn)
                        jj[connection] = _number(k, i + 1, j, nrow, ncolumn)
                        connection += 1

                if k < nlayer - 1:
                    kk = k
                    while kk < nlayer - 1:
                        kk += 1
                        below = idomain[kk, i, j]
                        if below > 0:
                            ii[connection] = _number(k, i, j, nrow, ncolumn)
                            jj[connection] = _number(kk, i, j, nrow, ncolumn)
                            connection += 1
                            break
                        elif below == 0:
                            break

    return ii[:connection], jj[:connection]


class UnstructuredDiscretization(Package):
    """
    Unstructured Discretization (DISU).

    Parameters
    ----------
    xorigin: float
    yorigin: float
    top: array of floats (xr.DataArray)
    bottom: array of floats (xr.DataArray)
    area: array of floats (xr.DataArray)
    iac: array of integers
    ja: array of integers
    ihc: array of integers
    cl12: array of floats
    hwva: array of floats
    """

    _pkg_id = "disu"
    _grid_data = {
        "top": np.float64,
        "bottom": np.float64,
        "area": np.float64,
        "iac": np.int32,
        "ja": np.int32,
        "ihc": np.int32,
        "cl12": np.float64,
        "hwva": np.float64,
    }
    _keyword_map = {"bottom": "bot"}
    _template = Package._initialize_template(_pkg_id)

    def __init__(
        self,
        xorigin,
        yorigin,
        top,
        bottom,
        area,
        iac,
        ja,
        ihc,
        cl12,
        hwva,
    ):
        super().__init__(locals())
        self.dataset["xorigin"] = xorigin
        self.dataset["yorigin"] = yorigin
        self.dataset["top"] = top
        self.dataset["bottom"] = bottom
        self.dataset["area"] = area
        self.dataset["iac"] = iac
        self.dataset["ja"] = ja
        self.dataset["ihc"] = ihc
        self.dataset["cl12"] = cl12
        self.dataset["hwva"] = hwva

    def render(self, directory, pkgname, globaltimes, binary):
        disdirectory = directory / pkgname
        d = {}
        d["xorigin"] = float(self.dataset["xorigin"])
        d["yorigin"] = float(self.dataset["yorigin"])

        # Dimensions
        d["nodes"] = self.dataset["top"].size
        d["nja"] = int(self.dataset["iac"].sum())

        # Grid data
        d["top"] = self._compose_values(
            self.dataset["top"], disdirectory, "top", binary=binary
        )[1][0]
        d["bot"] = self._compose_values(
            self["bottom"], disdirectory, "bot", binary=binary
        )[1][0]
        d["area"] = self._compose_values(
            self["area"], disdirectory, "area", binary=binary
        )[1][0]

        # Connection data
        d["iac"] = self._compose_values(
            self["iac"], disdirectory, "iac", binary=binary
        )[1][0]
        d["ja"] = self._compose_values(self["ja"], disdirectory, "ja", binary=binary)[
            1
        ][0]
        d["ihc"] = self._compose_values(
            self["ihc"], disdirectory, "ihc", binary=binary
        )[1][0]
        d["cl12"] = self._compose_values(
            self["cl12"], disdirectory, "cl12", binary=binary
        )[1][0]
        d["hwva"] = self._compose_values(
            self["hwva"], disdirectory, "hwva", binary=binary
        )[1][0]

        return self._template.render(d)

    @staticmethod
    def from_structured(
        top,
        bottom,
        idomain,
    ):
        x = idomain.coords["x"]
        y = idomain.coords["y"]
        layer = idomain.coords["layer"]
        active = idomain.values > 0

        ncolumn = x.size
        nrow = y.size
        nlayer = layer.size
        size = idomain.size

        dx, xmin, _ = imod.util.coord_reference(x)
        dy, ymin, _ = imod.util.coord_reference(y)

        # MODFLOW6 expects the ja values to contain the cell number first
        # while the row should be otherwise sorted ascending.
        # scipy.spare.csr_matrix will sort the values ascending, but
        # would not put the cell number first. To ensure this, we use
        # the values as well as i and j; we sort on the zeros (thereby ensuring
        # it results as a first value per column), but the actual value
        # is the (negative) cell number (in v).
        ii, jj = _structured_connectivity(idomain)
        ii += 1
        jj += 1
        nodes = np.arange(1, size + 1, dtype=np.int32)[active.ravel()]
        zeros = np.zeros_like(nodes)
        i = np.concatenate([nodes, ii, jj])
        j = np.concatenate([zeros, jj, ii])
        v = np.concatenate([-nodes, jj, ii])
        csr = sparse.csr_matrix((v, (i, j)), shape=(size + 1, size + 1))
        # The first column can be identified by its negative (node) number.
        # This entry does not require data in ihc, cl12, hwva.
        is_node = csr.data < 0

        # Constructing the CSR matrix will have sorted all the values are
        # required by MODFLOW6. However, we're using the original structured
        # numbering, which includes inactive cells.
        # For MODFLOW6, we use the reduced numbering, excluding all inactive
        # cells. This means getting rid of empty rows (iac), generating (via
        # cumsum) new numbers, and extracting them in the right order.
        nnz = csr.getnnz(axis=1)
        iac = nnz[nnz > 0]
        ja_index = np.abs(csr.data) - 1  # Get rid of negative values temporarily.
        ja = active.ravel().cumsum()[ja_index]

        # From CSR back to COO form
        # connectivity for every cell: n -> m
        n = np.repeat(np.arange(size + 1), nnz) - 1
        m = csr.indices - 1
        # Ignore the values that do not represent n -> m connections
        n[is_node] = 0
        m[is_node] = 0

        # Based on the row and column number differences we can derive the type
        # of connection (unless the model is a single row or single column!).
        diff = np.abs(n - m)
        is_vertical = (diff > 0) & (diff % (nrow * ncolumn) == 0)  # one or more layers
        is_x = diff == 1
        is_y = diff == ncolumn
        is_horizontal = is_x | is_y

        # We need the indexes twice. Store for re-use.
        # As the input is structured, we need only look at cell n, not m.
        # (n = row, m = column off the connectivity matrix.)
        index_x = n[is_x]
        index_y = n[is_y]
        index_v = n[is_vertical]

        # Create flat arrays for easy indexing.
        cellheight = top.values.ravel() - bottom.values.ravel()
        dyy, dxx = np.meshgrid(
            np.ones(ncolumn) * np.abs(dx),
            np.ones(nrow) * np.abs(dy),
            indexing="ij",
        )
        dyy = np.repeat(dyy, nlayer).ravel()
        dxx = np.repeat(dxx, nlayer).ravel()
        area = dyy * dxx

        # Allocate connectivity geometry arrays, all size nja.
        ihc = is_horizontal.astype(np.int32)
        cl12 = np.zeros_like(ihc, dtype=np.float64)
        hwva = np.zeros_like(ihc, dtype=np.float64)
        # Fill.
        cl12[is_x] = 0.5 * dxx[index_x]  # cell center to vertical face
        cl12[is_y] = 0.5 * dyy[index_y]  # cell center to vertical face
        cl12[is_vertical] = 0.5 * cellheight[index_v]  # cell center to horizontal face
        hwva[is_x] = dyy[index_x]  # width
        hwva[is_y] = dxx[index_y]  # width
        hwva[is_vertical] = area[index_v]  # area

        # Set "node" and "nja" as the dimension in accordance with MODFLOW6.
        # Should probably be updated if we could move to UGRID3D...
        return UnstructuredDiscretization(
            xorigin=xmin,
            yorigin=ymin,
            top=xr.DataArray(top.values[active], dims=["node"]),
            bottom=xr.DataArray(bottom.values[active], dims=["node"]),
            area=xr.DataArray(area[active.ravel()], dims=["node"]),
            iac=xr.DataArray(iac, dims=["node"]),
            ja=xr.DataArray(ja, dims=["nja"]),
            ihc=xr.DataArray(ihc, dims=["nja"]),
            cl12=xr.DataArray(cl12, dims=["nja"]),
            hwva=xr.DataArray(hwva, dims=["nja"]),
        )
