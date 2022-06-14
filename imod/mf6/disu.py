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


@nb.njit
def _structured_connectivity(idomain: IntArray):
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


class LowLevelUnstructuredDiscretization(Package):
    """
    Unstructured Discretization (DISU).

    Parameters
    ----------
    xorigin: float
    yorigin: float
    top: xr.DataArray of floats
    bot: xr.DataArray of floats
    area: xr.DataArray of floats
    iac: xr.DataArray of integers
    ja: xr.DataArray of integers
    ihc: xr.DataArray of integers
    cl12: xr.DataArray of floats
    hwva: xr.DataArray of floats
    """

    _pkg_id = "disu"
    _grid_data = {
        "top": np.float64,
        "bot": np.float64,
        "area": np.float64,
        "iac": np.int32,
        "ja": np.int32,
        "ihc": np.int32,
        "cl12": np.float64,
        "hwva": np.float64,
        "idomain": np.int32,
    }
    _keyword_map = {}
    _template = Package._initialize_template(_pkg_id)

    def __init__(
        self,
        xorigin,
        yorigin,
        top,
        bot,
        area,
        iac,
        ja,
        ihc,
        cl12,
        hwva,
        angledegx,
        idomain=None,
    ):
        super().__init__(locals())
        self.dataset["xorigin"] = xorigin
        self.dataset["yorigin"] = yorigin
        self.dataset["top"] = top
        self.dataset["bot"] = bot
        self.dataset["area"] = area
        self.dataset["iac"] = iac
        self.dataset["ja"] = ja
        self.dataset["ihc"] = ihc
        self.dataset["cl12"] = cl12
        self.dataset["hwva"] = hwva
        self.dataset["angledegx"] = angledegx
        if idomain is not None:
            self.dataset["idomain"] = idomain

    def render(self, directory, pkgname, globaltimes, binary):
        disdirectory = directory / pkgname
        d = {}
        d["xorigin"] = float(self.dataset["xorigin"])
        d["yorigin"] = float(self.dataset["yorigin"])

        # Dimensions
        d["nodes"] = self.dataset["top"].size
        d["nja"] = int(self.dataset["iac"].sum())

        # Grid data
        for varname in self._grid_data:
            if varname in self.dataset:
                key = self._keyword_map.get(varname, varname)
                d[varname] = self._compose_values(
                    self.dataset[varname], disdirectory, key, binary=binary
                )[1][0]

        return self._template.render(d)

    @staticmethod
    def from_dis(
        top,
        bottom,
        idomain,
        reduce_nodes=False,
    ):
        """
        Parameters
        ----------
        reduce_nodes: bool, optional. Default: False.
            Reduces the node numbering, discards cells when idomain <= 0.

        Returns
        -------
        disu: LowLevelUnstructuredDiscretization
        cell_ids: ndarray of integers.
            Only provided if ``reduce_nodes`` is ``True``.
        """
        x = idomain.coords["x"]
        y = idomain.coords["y"]
        layer = idomain.coords["layer"]
        active = idomain.values.ravel() > 0

        ncolumn = x.size
        nrow = y.size
        nlayer = layer.size
        size = idomain.size

        dx, xmin, _ = imod.util.coord_reference(x)
        dy, ymin, _ = imod.util.coord_reference(y)

        # MODFLOW6 expects the ja values to contain the cell number first while
        # the row should be otherwise sorted ascending. scipy.spare.csr_matrix
        # will sort the values ascending, but would not put the cell number
        # first. To ensure this, we use the values as well as i and j; we sort
        # on the zeros (thereby ensuring it results as a first value per
        # column), but the actual value is the (negative) cell number (in v).
        ii, jj = _structured_connectivity(idomain.values)
        ii += 1
        jj += 1
        nodes = np.arange(1, size + 1, dtype=np.int32)
        if reduce_nodes:
            nodes = nodes[active.ravel()]

        zeros = np.zeros_like(nodes)
        i = np.concatenate([nodes, ii, jj])
        j = np.concatenate([zeros, jj, ii])
        v = np.concatenate([-nodes, jj, ii])
        csr = sparse.csr_matrix((v, (i, j)), shape=(size + 1, size + 1))
        # The first column can be identified by its negative (node) number.
        # This entry does not require data in ihc, cl12, hwva.
        is_node = csr.data < 0

        nnz = csr.getnnz(axis=1)
        if reduce_nodes:
            # Constructing the CSR matrix will have sorted all the values are
            # required by MODFLOW6. However, we're using the original structured
            # numbering, which includes inactive cells.
            # For MODFLOW6, we use the reduced numbering if reduce_nodes is True,
            # excluding all inactive cells. This means getting rid of empty rows
            # (iac), generating (via cumsum) new numbers, and extracting them in
            # the right order.
            iac = nnz[nnz > 0]
            ja_index = np.abs(csr.data) - 1  # Get rid of negative values temporarily.
            ja = active.cumsum()[ja_index]
        else:
            # In this case, inactive cells are included as well. They have no
            # connections to other cells and form empty rows (0 in iac), but
            # are still included. There is no need to update the cell numbers
            # in this case.
            iac = nnz[1:]  # Cell 0 does not exist.
            ja = csr.data

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
        angledegx = np.zeros_like(ihc, dtype=np.float64)
        # Fill.
        cl12[is_x] = 0.5 * dxx[index_x]  # cell center to vertical face
        cl12[is_y] = 0.5 * dyy[index_y]  # cell center to vertical face
        cl12[is_vertical] = 0.5 * cellheight[index_v]  # cell center to horizontal face
        hwva[is_x] = dyy[index_x]  # width
        hwva[is_y] = dxx[index_y]  # width
        hwva[is_vertical] = area[index_v]  # area
        angledegx[is_y] = 90.0  # angle between connection normal and x-axis.

        # Set "node" and "nja" as the dimension in accordance with MODFLOW6.
        # Should probably be updated if we could move to UGRID3D...
        if reduce_nodes:
            # If we reduce nodes, we should only take active cells from top,
            # bottom, area. There is no need to include an idomain: all defined
            # cells are active.
            disu = LowLevelUnstructuredDiscretization(
                xorigin=xmin,
                yorigin=ymin,
                top=xr.DataArray(top.values.ravel()[active], dims=["node"]),
                bot=xr.DataArray(bottom.values.ravel()[active], dims=["node"]),
                area=xr.DataArray(area[active], dims=["node"]),
                iac=xr.DataArray(iac, dims=["node"]),
                ja=xr.DataArray(ja, dims=["nja"]),
                ihc=xr.DataArray(ihc, dims=["nja"]),
                cl12=xr.DataArray(cl12, dims=["nja"]),
                hwva=xr.DataArray(hwva, dims=["nja"]),
                angledegx=xr.DataArray(angledegx, dims=["nja"]),
            )
            cell_ids = np.cumsum(active) - 1
            cell_ids[~active] = -1
            return disu, cell_ids
        else:
            return LowLevelUnstructuredDiscretization(
                xorigin=xmin,
                yorigin=ymin,
                top=xr.DataArray(top.values.ravel(), dims=["node"]),
                bot=xr.DataArray(bottom.values.ravel(), dims=["node"]),
                area=xr.DataArray(area, dims=["node"]),
                iac=xr.DataArray(iac, dims=["node"]),
                ja=xr.DataArray(ja, dims=["nja"]),
                ihc=xr.DataArray(ihc, dims=["nja"]),
                cl12=xr.DataArray(cl12, dims=["nja"]),
                hwva=xr.DataArray(hwva, dims=["nja"]),
                angledegx=xr.DataArray(angledegx, dims=["nja"]),
                idomain=xr.DataArray(active.astype(np.int32), dims=["node"]),
            )
