import numba
import numpy as np
import xarray as xr

import imod


def coord_union(*args, decreasing):
    x = np.unique(np.concatenate(args))
    if decreasing:
        return x[::-1]
    else:
        return x


def overlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def is_touching(da1, da2, dim):
    _, xmin1, xmax1 = imod.util.coord_reference(da1[dim])
    _, xmin2, xmax2 = imod.util.coord_reference(da2[dim])
    x_overlap = overlap((xmin1, xmax1), (xmin2, xmax2))
    is_touching = False
    if x_overlap == 0.0:
        if (xmin1 == xmax2) or (xmin2 == xmax1):
            is_touching = True
        else:
            raise ValueError(f"Dimension {dim} neither touches nor overlaps")
    return is_touching


def clip(x, xmin, xmax):
    return x[(x > xmin) & (x < xmax)]


def touching_union_coords(da1, da2, dim, decreasing):
    dx1, xmin1, xmax1 = imod.util.coord_reference(da1[dim])
    dx2, xmin2, xmax2 = imod.util.coord_reference(da2[dim])
    xmin = max(xmin1, xmin2)
    xmax = min(xmax1, xmax2)
    # Should be edges?
    edges1 = np.full(dx1.size + 1, xmin1)
    edges2 = np.full(dx2.size + 1, xmin2)
    edges1[1:] += np.cumsum(dx1)
    edges2[1:] += np.cumsum(dx2)
    edges1 = clip(edges1, xmin, xmax)
    edges2 = clip(edges2, xmin, xmax)
    x_edges = coord_union([xmin], edges1, edges2, [xmax], decreasing=decreasing)
    dx = np.diff(x_edges)
    x = np.cumsum(dx) - 0.5 * dx
    if xmin2 == xmax1:
        col1 = da1[dim].size - 1
        col2 = 0
    elif xmin1 == xmax2:
        col1 = 0
        col2 = da2[dim].size - 1
    else:
        raise ValueError("dim doesn't touch?")
    return x, col1, col2, edges1, edges2


def overlapping_union_coords(da1, da2, dim, decreasing):
    dx1, xmin1, xmax1 = imod.util.coord_reference(da1[dim])
    dx2, xmin2, xmax2 = imod.util.coord_reference(da2[dim])
    xmin = max(xmin1 - dx1, xmin2 - dx1)
    xmax = min(xmax2 + dx1, xmax2 + dx1)
    edges1 = np.full(dx1.size + 1, xmin1)
    edges2 = np.full(dx2.size + 1, xmin2)
    edges1[1:] += np.cumsum(dx1)
    edges2[1:] += np.cumsum(dx2)
    edges1 = clip(edges1, xmin, xmax)
    edges2 = clip(edges2, xmin, xmax)
    x_edges = coord_union([xmin], edges1, edges2, [xmax], decreasing=decreasing)
    dx = np.diff(x_edges)
    x = np.cumsum(dx) - 0.5 * dx
    return x


@numba.njit
def find_exchange(index1, index2):
    cells1 = []
    cells2 = []
    nrow, ncol = index1.shape
    # Do a horizontal pass
    for i in range(nrow):
        for j in range(ncol):
            v1 = index1[i, j]
            v2 = index2[i, j]
            if (v1 != -1) and (v2 != -1):
                raise ValueError("Overlapping cells!")
            if v1 == -1:
                if i < (nrow - 1):
                    if index1[i + 1, j] != -1:
                        # Edge detected
                        if index2[i, j] != -1:
                            # other model detected
                            cells1.append(index1[i + 1, j])
                            cells2.append(index2[i, j])
                if j < (ncol - 1):
                    if index1[i, j + 1] != -1:
                        # Edge detected
                        if index2[i, j] != -1:
                            cells1.append(index1[i, j + 1])
                            cells2.append(index2[i, j])
            else:  # v1 != -1
                if i < (nrow - 1):
                    if index1[i + 1, j] == -1:
                        # Edge detected
                        if index2[i + 1, j] != -1:
                            # other model detected
                            cells1.append(index1[i, j])
                            cells2.append(index2[i + 1, j])
                if j < (ncol - 1):
                    if index1[i, j + 1] == -1:
                        # Edge detected
                        if index2[i, j + 1] != -1:
                            cells1.append(index1[i, j])
                            cells2.append(index2[i, j + 1])

    return np.array(cells1), np.array(cells2)


def bounding_box_union(da1, da2):
    touching_x = is_touching(da1, da2, "x")
    touching_y = is_touching(da1, da2, "y")
    if all(touching_x, touching_y):
        raise ValueError("Only corners of models touch")
    elif touching_y:
        # Then, create union for x
        # x is midpoints, x1 and x2 are vertex edges
        x, row1, row2, x1, x2 = touching_union_coords(da1, da2, "x", False)
        # Compute indices
        ix1 = np.searchsorted(x, x1)
        ix2 = np.searchsorted(x, x2)
        iy1 = np.full_like(ix1, row1)
        iy2 = np.full_like(ix2, row2)
    elif touching_x:
        # Then, create union for y
        y, col1, col2, y1, y2 = touching_union_coords(da1, da2, "y", True)
        iy1 = np.searchsorted(y, y1)
        iy2 = np.searchsorted(y, y2)
        ix1 = np.full_like(iy1, col1)
        ix2 = np.full_like(iy2, col2)
    else:  # overlap situation
        x = overlapping_union_coords(da1, da2, "x", decreasing=False)
        y = overlapping_union_coords(da1, da2, "y", decreasing=True)
        nrow = y.size
        ncol = x.size
        dims = ("y", "x")
        coords = {"y": y, "x": x}
        like = xr.DataArray(np.empty(dtype=np.int), coords, dims)

        nearest_regridder = imod.prepare_regridder(method="nearest")
        idomain1 = da1.sel(layer=1)
        idomain2 = da2.sel(layer=1)
        orig_index1 = xr.full_like(idomain1, np.arange(like1.size).reshape(like1.shape))
        orig_index2 = xr.full_like(idomain2, np.arange(like2.size).reshape(like2.shape))
        orig_index1 = orig_index1.where(idomain1 != 0, other=-1)
        orig_index2 = orig_index2.where(idomain2 != 0, other=-1)
        index1 = nearest_regridder.regrid(orig_index1, like)
        index2 = nearest_regridder.regrid(orig_index2, like)
