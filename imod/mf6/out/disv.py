import os
import struct
from typing import Any, BinaryIO, Dict, List, Tuple

import dask
import numba
import numpy as np
import scipy.sparse
import xarray as xr
import xugrid as xu

from . import cbc
from .common import FilePath, FloatArray, IntArray, _to_nan


def _ugrid_iavert_javert(
    iavert: IntArray, javert: IntArray
) -> Tuple[IntArray, IntArray]:
    # The node numbers of MODFLOW loop around: the first equals the last
    # We have to remove these for the UGRID conventions, which do not loop around.
    n = np.diff(iavert) - 1
    # This also takes care of 0-based indexing:
    ia = np.concatenate(([0], np.cumsum(n)))
    keep = np.ones_like(javert, dtype=bool)
    # -2: -1 for 1- to 0-based indexing, -1 to get rid of closing node.
    keep[iavert[1:] - 2] = False
    return ia, javert[keep] - 1


def read_grb(f: BinaryIO, ntxt: int, lentxt: int) -> Dict[str, Any]:
    # we don't need any information from the the text lines that follow,
    # they are definitions that aim to make the file more portable,
    # so let's skip straight to the binary data
    f.seek(ntxt * lentxt, 1)

    ncells = struct.unpack("i", f.read(4))[0]
    nlayer = struct.unpack("i", f.read(4))[0]
    ncells_per_layer = struct.unpack("i", f.read(4))[0]
    nvert = struct.unpack("i", f.read(4))[0]
    njavert = struct.unpack("i", f.read(4))[0]
    nja = struct.unpack("i", f.read(4))[0]
    if ncells != (nlayer * ncells_per_layer):
        raise ValueError(f"Invalid file {ncells} {nlayer} {ncells_per_layer}")
    _ = struct.unpack("d", f.read(8))[0]  # xorigin
    _ = struct.unpack("d", f.read(8))[0]  # yorigin
    f.seek(8, 1)  # skip angrot
    top_np = np.fromfile(f, np.float64, ncells_per_layer)
    bottom_np = np.reshape(
        np.fromfile(f, np.float64, ncells), (nlayer, ncells_per_layer)
    )
    vertices = np.reshape(np.fromfile(f, np.float64, nvert * 2), (nvert, 2))
    _ = np.fromfile(f, np.float64, ncells_per_layer)  # cellx
    _ = np.fromfile(f, np.float64, ncells_per_layer)  # celly
    # Python is 0-based; MODFLOW6 is Fortran 1-based
    iavert = np.fromfile(f, np.int32, ncells_per_layer + 1)
    javert = np.fromfile(f, np.int32, njavert)
    ia = np.fromfile(f, np.int32, ncells + 1)
    ja = np.fromfile(f, np.int32, nja)
    idomain_np = np.reshape(
        np.fromfile(f, np.int32, ncells), (nlayer, ncells_per_layer)
    )
    icelltype_np = np.reshape(
        np.fromfile(f, np.int32, ncells), (nlayer, ncells_per_layer)
    )

    iavert, javert = _ugrid_iavert_javert(iavert, javert)
    face_nodes = scipy.sparse.csr_matrix((javert, javert, iavert))
    grid = xu.Ugrid2d(vertices[:, 0], vertices[:, 1], -1, face_nodes)
    facedim = grid.face_dimension

    top = xu.UgridDataArray(xr.DataArray(top_np, dims=[facedim], name="top"), grid)
    coords = {"layer": np.arange(1, nlayer + 1)}
    dims = ("layer", facedim)
    bottom = xr.DataArray(bottom_np, coords, dims, name="bottom")
    idomain = xr.DataArray(idomain_np, coords, dims, name="idomain")
    icelltype = xr.DataArray(icelltype_np, coords, dims, name="icelltype")

    return {
        "distype": "disv",
        "grid": grid,
        "top": xu.UgridDataArray(top, grid),
        "bottom": xu.UgridDataArray(bottom, grid),
        "coords": coords,
        "ncells": ncells,
        "nlayer": nlayer,
        "ncells_per_layer": ncells_per_layer,
        "nja": nja,
        "ia": ia,
        "ja": ja,
        "idomain": xu.UgridDataArray(idomain, grid),
        "icelltype": xu.UgridDataArray(icelltype, grid),
    }


def read_times(
    path: FilePath, ntime: int, nlayer: int, ncells_per_layer: int
) -> FloatArray:
    """
    Reads all total simulation times.
    """
    times = np.empty(ntime, dtype=np.float64)

    # Compute how much to skip to the next timestamp
    start_of_header = 16
    rest_of_header = 28
    data_single_layer = ncells_per_layer * 8
    header = 52
    nskip = (
        rest_of_header
        + data_single_layer
        + (nlayer - 1) * (header + data_single_layer)
        + start_of_header
    )

    with open(path, "rb") as f:
        f.seek(start_of_header)
        for i in range(ntime):
            times[i] = struct.unpack("d", f.read(8))[0]  # total simulation time
            f.seek(nskip, 1)
    return times


def read_hds_timestep(
    path: FilePath, nlayer: int, ncells_per_layer: int, dry_nan: bool, pos: int
) -> FloatArray:
    """
    Reads all values of one timestep.
    """
    with open(path, "rb") as f:
        f.seek(pos)
        a1d = np.empty(nlayer * ncells_per_layer, dtype=np.float64)
        for k in range(nlayer):
            f.seek(52, 1)  # skip kstp, kper, pertime
            a1d[k * ncells_per_layer : (k + 1) * ncells_per_layer] = np.fromfile(
                f, np.float64, ncells_per_layer
            )

    a2d = a1d.reshape((nlayer, ncells_per_layer))
    return _to_nan(a2d, dry_nan)


def open_hds(path: FilePath, d: Dict[str, Any], dry_nan: bool) -> xu.UgridDataArray:
    grid = d["grid"]
    nlayer, ncells_per_layer = d["nlayer"], d["ncells_per_layer"]
    filesize = os.path.getsize(path)
    ntime = filesize // (nlayer * (52 + (ncells_per_layer * 8)))
    times = read_times(path, ntime, nlayer, ncells_per_layer)
    coords = d["coords"]
    coords["time"] = times

    dask_list = []
    # loop over times and add delayed arrays
    for i in range(ntime):
        # TODO verify dimension order
        pos = i * (nlayer * (52 + ncells_per_layer * 8))
        a = dask.delayed(read_hds_timestep)(
            path, nlayer, ncells_per_layer, dry_nan, pos
        )
        x = dask.array.from_delayed(
            a, shape=(nlayer, ncells_per_layer), dtype=np.float64
        )
        dask_list.append(x)

    daskarr = dask.array.stack(dask_list, axis=0)
    da = xr.DataArray(
        daskarr, coords, ("time", "layer", grid.face_dimension), name=d["name"]
    )
    return xu.UgridDataArray(da, grid)


def open_imeth1_budgets(
    cbc_path: FilePath, grb_content: dict, header_list: List[cbc.Imeth1Header]
) -> xu.UgridDataArray:
    """
    Open the data for an imeth==1 budget section. Data is read lazily per
    timestep.

    Can be used for:

        * STO-SS
        * STO-SY
        * CSUB-CGELASTIC
        * CSUB-WATERCOMP

    Utilizes the shape information from the DIS GRB file to create a dense
    array; (lazily) allocates for the entire domain (all layers, faces)
    per timestep.

    Parameters
    ----------
    cbc_path: str, pathlib.Path
    grb_content: dict
    header_list: List[Imeth1Header]

    Returns
    -------
    xr.DataArray with dims ("time", "layer", face_dimension)
    """
    grid = grb_content["grid"]
    facedim = grid.face_dimension
    nlayer = grb_content["nlayer"]
    ncells_per_layer = grb_content["cells_per_layer"]
    coords = grb_content["coords"]
    budgets = cbc.open_imeth1_budgets(cbc_path, header_list)
    coords["time"] = budgets["time"]

    da = xr.DataArray(
        data=budgets.data.reshape((budgets["time"].size, nlayer, ncells_per_layer)),
        coords=coords,
        dims=("time", "layer", facedim),
        name=None,
    )
    return xu.UgridDataArray(da, grid)


def open_imeth6_budgets(
    cbc_path: FilePath, grb_content: dict, header_list: List[cbc.Imeth6Header]
) -> xu.UgridDataArray:
    """
    Open the data for an imeth==6 budget section.

    Uses the information of the DIS GRB file to create the properly sized dense
    xr.DataArrays (which store the entire domain). Doing so ignores the boundary
    condition internal index (id2) and any present auxiliary columns.

    Parameters
    ----------
    cbc_path: str, pathlib.Path
    grb_content: dict
    header_list: List[Imeth1Header]

    Returns
    -------
    xr.DataArray with dims ("time", "layer", "y", "x")
    """
    # Allocates dense arrays for the entire model domain
    dtype = np.dtype(
        [("id1", np.int32), ("id2", np.int32), ("budget", np.float64)]
        + [(name, np.float64) for name in header_list[0].auxtxt]
    )
    shape = (grb_content["nlayer"], grb_content["ncells_per_layer"])
    size = np.product(shape)
    dask_list = []
    time = np.empty(len(header_list), dtype=np.float64)
    for i, header in enumerate(header_list):
        time[i] = header.totim
        a = dask.delayed(cbc.read_imeth6_budgets)(
            cbc_path, header.nlist, dtype, header.pos, size, shape
        )
        x = dask.array.from_delayed(a, shape=shape, dtype=np.float64)
        dask_list.append(x)

    daskarr = dask.array.stack(dask_list, axis=0)
    coords = grb_content["coords"]
    coords["time"] = time
    name = header_list[0].text
    grid = grb_content["grid"]
    da = xr.DataArray(
        daskarr, coords, ("time", "layer", grid.face_dimension), name=name
    )
    return xu.UgridDataArray(da, grid)


@numba.njit
def disv_lower_index(
    ia: IntArray,
    ja: IntArray,
    ncells: int,
    nlayer: int,
    ncells_per_layer: int,
) -> IntArray:
    lower = np.full(ncells, -1, np.int64)
    for i in range(ncells):
        for nzi in range(ia[i], ia[i + 1]):
            nzi -= 1  # python is 0-based, modflow6 is 1-based
            j = ja[nzi] - 1  # python is 0-based, modflow6 is 1-based
            d = j - i
            if d < ncells_per_layer:  # upper, diagonal, horizontal
                continue
            elif d == ncells_per_layer:  # lower neighbor
                lower[i] = nzi
            else:  # skips one: must be pass through
                npassed = int(d / ncells_per_layer)
                for ipass in range(0, npassed):
                    lower[i + ipass * ncells_per_layer] = nzi

    return lower.reshape(nlayer, ncells_per_layer)


def expand_indptr(ia: IntArray):
    n = np.diff(ia)
    return np.repeat(np.arange(ia.size - 1), n)


def disv_horizontal_index(
    ia: IntArray,
    ja: IntArray,
    nlayer: int,
    ncells_per_layer: int,
    edge_face_connectivity: IntArray,
    fill_value: int,
    face_coordinates: FloatArray,
):
    # Allocate output array
    nedge = len(edge_face_connectivity)
    horizontal = np.full((nlayer, nedge), -1)

    # Grab the index values to the horizontal connections
    i = expand_indptr(ia)
    j = ja - 1
    d = j - i
    is_horizontal = (0 < d) & (d < ncells_per_layer)
    index = np.arange(j.size)[is_horizontal].reshape((nlayer, -1))

    # i -> j is pre-sorted (required by CSR structure); the edge_faces are repeated
    # per layer. Because i -> j is sorted in terms of face numbering, we need
    # only to figure out which order the edge_face_connectivity has.
    is_connection = edge_face_connectivity[:, 1] != fill_value
    edge_faces = edge_face_connectivity[is_connection]
    order = np.argsort(np.lexsort(edge_faces.T[::-1]))
    # Reshuffle for every layer
    index = index[:, order]

    # Now set the values in the output array
    horizontal[:, is_connection] = index

    # Compute unit components (x: u, y: v)
    edge_faces.sort(axis=1)
    u = np.full(nedge, np.nan)
    v = np.full(nedge, np.nan)
    xy = face_coordinates[edge_faces]
    dx = xy[:, 1, 0] - xy[:, 0, 0]
    dy = xy[:, 1, 1] - xy[:, 0, 1]
    t = np.sqrt(dx**2 + dy**2)
    u[is_connection] = dx / t
    v[is_connection] = dy / t
    return horizontal, u, v


def disv_to_horizontal_lower_indices(
    grb_content: dict,
) -> Tuple[xr.DataArray, xr.DataArray]:
    grid = grb_content["grid"]
    horizontal, u, v = disv_horizontal_index(
        ia=grb_content["ia"],
        ja=grb_content["ja"],
        nlayer=grb_content["nlayer"],
        ncells_per_layer=grb_content["ncells_per_layer"],
        edge_face_connectivity=grid.edge_face_connectivity,
        fill_value=grid.fill_value,
        face_coordinates=grid.face_coordinates,
    )
    lower = disv_lower_index(
        ia=grb_content["ia"],
        ja=grb_content["ja"],
        ncells=grb_content["ncells"],
        nlayer=grb_content["nlayer"],
        ncells_per_layer=grb_content["ncells_per_layer"],
    )

    # Compute unit_vector

    return (
        xr.DataArray(
            horizontal, grb_content["coords"], dims=["layer", grid.edge_dimension]
        ),
        xr.DataArray(u, dims=[grid.edge_dimension]),
        xr.DataArray(v, dims=[grid.edge_dimension]),
        xr.DataArray(lower, grb_content["coords"], dims=["layer", grid.face_dimension]),
    )


def disv_extract_lower_budget(
    budgets: xr.DataArray, index: xr.DataArray
) -> xr.DataArray:
    face_dimension = index.dims[-1]
    coords = dict(index.coords)
    coords["time"] = budgets["time"]
    # isel with a 3D array is extremely slow
    # this followed by the dask reshape is much faster for some reason.
    data = budgets.isel(linear_index=index.values.ravel()).data
    da = xr.DataArray(
        data=data.reshape((budgets["time"].size, *index.shape)),
        coords=coords,
        dims=("time", "layer", face_dimension),
        name="flow-ja-face",
    )
    return da.where(index >= 0, other=0.0)


def disv_extract_horizontal_budget(
    budgets: xr.DataArray, index: xr.DataArray
) -> xr.DataArray:
    """
    Grab horizontal flows from the flow-ja-face array.

    This could be done by a single .isel() indexing operation, but those
    are extremely slow in this case, which seems to be an xarray issue.

    Parameters
    ----------
    budgets: xr.DataArray of floats
        flow-ja-face array, dims ("time", "linear_index")
        The linear index enumerates cell-to-cell connections in this case, not
        the individual cells.
    index: xr.DataArray of ints
        index array with dims("layer", edge_dimension)

    Returns
    -------
    xr.DataArray of floats with dims ("time", "layer", edge_dimension)
    """
    edge_dimension = index.dims[-1]
    coords = dict(index.coords)
    coords["time"] = budgets["time"]
    # isel with a 3D array is extremely slow
    # this followed by the dask reshape is much faster for some reason.
    data = budgets.isel(linear_index=index.values.ravel()).data
    da = xr.DataArray(
        data=data.reshape((budgets["time"].size, *index.shape)),
        coords=coords,
        dims=("time", "layer", edge_dimension),
        name="flow-ja-face",
    )
    return da.where(index >= 0, other=0.0)


def disv_open_face_budgets(
    cbc_path: FilePath, grb_content: dict, header_list: List[cbc.Imeth1Header]
) -> Tuple[xu.UgridDataArray]:
    horizontal_index, u, v, lower_index = disv_to_horizontal_lower_indices(grb_content)
    budgets = cbc.open_imeth1_budgets(cbc_path, header_list)
    horizontal = disv_extract_horizontal_budget(budgets, horizontal_index)
    lower = disv_extract_horizontal_budget(budgets, lower_index)
    flow_x = -horizontal * u
    flow_y = -horizontal * v
    grid = grb_content["grid"]
    return (
        xu.UgridDataArray(horizontal, grid),
        xu.UgridDataArray(flow_x, grid),
        xu.UgridDataArray(flow_y, grid),
        xu.UgridDataArray(lower, grid),
    )


def open_cbc(
    cbc_path: FilePath, grb_content: Dict[str, Any], flowja: bool = False
) -> Dict[str, xu.UgridDataArray]:
    headers = cbc.read_cbc_headers(cbc_path)
    cbc_content = {}
    for key, header_list in headers.items():
        if key == "flow-ja-face":
            if flowja:
                flowja, ij = cbc.open_face_budgets_as_flowja(
                    cbc_path, header_list, grb_content
                )
                cbc_content["flow-ja-face"] = flowja
                cbc_content["connectivity"] = ij
            else:
                flow_xy, flow_x, flow_y, lower = disv_open_face_budgets(
                    cbc_path, grb_content, header_list
                )
                cbc_content["flow-horizontal-face"] = flow_xy
                cbc_content["flow-horizontal-face-x"] = flow_x
                cbc_content["flow-horizontal-face-y"] = flow_y
                cbc_content["flow-lower-face"] = lower
        elif isinstance(header_list[0], cbc.Imeth1Header):
            cbc_content[key] = open_imeth1_budgets(cbc_path, grb_content, header_list)
        elif isinstance(header_list[0], cbc.Imeth6Header):
            cbc_content[key] = open_imeth6_budgets(cbc_path, grb_content, header_list)

    return cbc_content


def grid_info(like: xu.UgridDataArray) -> Dict[str, Any]:
    grid = like.ugrid.grid
    facedim = grid.face_dimension
    return {
        "name": "head",
        "nlayer": like["layer"].size,
        "ncells_per_layer": like[facedim].size,
        "coords": {
            "layer": like["layer"],
            facedim: like[facedim],
        },
    }
