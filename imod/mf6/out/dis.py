import os
import struct
from typing import Any, BinaryIO, Dict, List, Tuple

import dask
import numba
import numpy as np
import xarray as xr

import imod

from . import cbc
from .common import FilePath, FloatArray, IntArray, _to_nan


# Binary Grid File / DIS Grids
# https://water.usgs.gov/water-resources/software/MODFLOW-6/mf6io_6.0.4.pdf#page=162
def read_grb(f: BinaryIO, ntxt: int, lentxt: int) -> Dict[str, Any]:
    # we don't need any information from the the text lines that follow,
    # they are definitions that aim to make the file more portable,
    # so let's skip straight to the binary data
    f.seek(ntxt * lentxt, 1)

    ncells = struct.unpack("i", f.read(4))[0]
    nlayer = struct.unpack("i", f.read(4))[0]
    nrow = struct.unpack("i", f.read(4))[0]
    ncol = struct.unpack("i", f.read(4))[0]
    nja = struct.unpack("i", f.read(4))[0]
    if ncells != (nlayer * nrow * ncol):
        raise ValueError(f"Invalid file {ncells} {nlayer} {nrow} {ncol}")
    xorigin = struct.unpack("d", f.read(8))[0]
    yorigin = struct.unpack("d", f.read(8))[0]
    f.seek(8, 1)  # skip angrot
    delr = np.fromfile(f, np.float64, ncol)
    delc = np.fromfile(f, np.float64, nrow)
    top_np = np.reshape(np.fromfile(f, np.float64, nrow * ncol), (nrow, ncol))
    bottom_np = np.reshape(np.fromfile(f, np.float64, ncells), (nlayer, nrow, ncol))
    ia = np.fromfile(f, np.int32, ncells + 1)
    ja = np.fromfile(f, np.int32, nja)
    idomain_np = np.reshape(np.fromfile(f, np.int32, ncells), (nlayer, nrow, ncol))
    icelltype_np = np.reshape(np.fromfile(f, np.int32, ncells), (nlayer, nrow, ncol))

    bounds = (xorigin, xorigin + delr.sum(), yorigin, yorigin + delc.sum())
    coords = imod.util._xycoords(bounds, (delr, -delc))
    top = xr.DataArray(top_np, coords, ("y", "x"), name="top")
    coords["layer"] = np.arange(1, nlayer + 1)
    dims = ("layer", "y", "x")
    bottom = xr.DataArray(bottom_np, coords, dims, name="bottom")
    idomain = xr.DataArray(idomain_np, coords, dims, name="idomain")
    icelltype = xr.DataArray(icelltype_np, coords, dims, name="icelltype")

    return {
        "distype": "dis",
        "top": top,
        "bottom": bottom,
        "coords": coords,
        "ncells": ncells,
        "nlayer": nlayer,
        "nrow": nrow,
        "ncol": ncol,
        "nja": nja,
        "ia": ia,
        "ja": ja,
        "idomain": idomain,
        "icelltype": icelltype,
    }


def read_times(
    path: FilePath, ntime: int, nlayer: int, nrow: int, ncol: int
) -> FloatArray:
    """
    Reads all total simulation times.
    """
    times = np.empty(ntime, dtype=np.float64)

    # Compute how much to skip to the next timestamp
    start_of_header = 16
    rest_of_header = 28
    data_single_layer = nrow * ncol * 8
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
    path: FilePath, nlayer: int, nrow: int, ncol: int, dry_nan: bool, pos: int
) -> FloatArray:
    """
    Reads all values of one timestep.
    """
    ncell_per_layer = nrow * ncol
    with open(path, "rb") as f:
        f.seek(pos)
        a1d = np.empty(nlayer * nrow * ncol, dtype=np.float64)
        for k in range(nlayer):
            f.seek(52, 1)  # skip kstp, kper, pertime
            a1d[k * ncell_per_layer : (k + 1) * ncell_per_layer] = np.fromfile(
                f, np.float64, nrow * ncol
            )

    a3d = a1d.reshape((nlayer, nrow, ncol))
    return _to_nan(a3d, dry_nan)


def open_hds(path: FilePath, d: Dict[str, Any], dry_nan: bool) -> xr.DataArray:
    nlayer, nrow, ncol = d["nlayer"], d["nrow"], d["ncol"]
    filesize = os.path.getsize(path)
    ntime = filesize // (nlayer * (52 + (nrow * ncol * 8)))
    times = read_times(path, ntime, nlayer, nrow, ncol)
    coords = d["coords"]
    coords["time"] = times

    dask_list = []
    # loop over times and add delayed arrays
    for i in range(ntime):
        # TODO verify dimension order
        pos = i * (nlayer * (52 + nrow * ncol * 8))
        a = dask.delayed(read_hds_timestep)(path, nlayer, nrow, ncol, dry_nan, pos)
        x = dask.array.from_delayed(a, shape=(nlayer, nrow, ncol), dtype=np.float64)
        dask_list.append(x)

    daskarr = dask.array.stack(dask_list, axis=0)
    return xr.DataArray(daskarr, coords, ("time", "layer", "y", "x"), name=d["name"])


def open_imeth1_budgets(
    cbc_path: FilePath, grb_content: dict, header_list: List[cbc.Imeth1Header]
) -> xr.DataArray:
    """
    Open the data for an imeth==1 budget section. Data is read lazily per
    timestep.

    Can be used for:

        * STO-SS
        * STO-SY
        * CSUB-CGELASTIC
        * CSUB-WATERCOMP

    Utilizes the shape information from the DIS GRB file to create a dense
    array; (lazily) allocates for the entire domain (all layers, rows, columns)
    per timestep.

    Parameters
    ----------
    cbc_path: str, pathlib.Path
    grb_content: dict
    header_list: List[Imeth1Header]

    Returns
    -------
    xr.DataArray with dims ("time", "layer", "y", "x")
    """
    nlayer = grb_content["nlayer"]
    nrow = grb_content["nrow"]
    ncol = grb_content["ncol"]
    budgets = cbc.open_imeth1_budgets(cbc_path, header_list)
    # Merge dictionaries
    coords = grb_content["coords"] | {"time": budgets["time"]}

    return xr.DataArray(
        data=budgets.data.reshape((budgets["time"].size, nlayer, nrow, ncol)),
        coords=coords,
        dims=("time", "layer", "y", "x"),
        name=budgets.name,
    )


def open_imeth6_budgets(
    cbc_path: FilePath,
    grb_content: dict,
    header_list: List[cbc.Imeth6Header],
    return_variable: str = "budget",
) -> xr.DataArray:
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
    shape = (grb_content["nlayer"], grb_content["nrow"], grb_content["ncol"])
    size = np.product(shape)
    dask_list = []
    time = np.empty(len(header_list), dtype=np.float64)
    for i, header in enumerate(header_list):
        time[i] = header.totim
        a = dask.delayed(cbc.read_imeth6_budgets_dense)(
            cbc_path, header.nlist, dtype, header.pos, size, shape, return_variable
        )
        x = dask.array.from_delayed(a, shape=shape, dtype=np.float64)
        dask_list.append(x)

    daskarr = dask.array.stack(dask_list, axis=0)
    coords = grb_content["coords"]
    coords["time"] = time
    name = header_list[0].text
    return xr.DataArray(daskarr, coords, ("time", "layer", "y", "x"), name=name)


@numba.njit
def dis_indices(
    ia: IntArray,
    ja: IntArray,
    ncells: int,
    nlayer: int,
    nrow: int,
    ncol: int,
) -> Tuple[IntArray, IntArray, IntArray]:
    """
    Infer type of connection via cell number comparison. Returns arrays that can
    be used for extracting right, front, and lower face flow from the
    flow-ja-face array.

    In a structured grid, using a linear index:
    * the right neighbor is +(1)
    * the front neighbor is +(number of cells in a column)
    * the lower neighbor is +(number of cells in a layer)
    * lower "pass-through" cells (idomain == -1) are multitude of (number of
      cells in a layer)

    Parameters
    ----------
    ia: Array of ints
        Row index of Compressed Sparse Row (CSR) connectivity matrix.
    ja: Array of ints
        Column index of CSR connectivity matrix. Every entry represents a
        cell-to-cell connection.
    ncells: int
    nlayer: int
    nrow: int
    ncol: int

    Returns
    -------
    right: 3D array of ints
    front: 3D array of ints
    lower: 3D array of ints
    """
    shape = (nlayer, nrow, ncol)
    ncells_per_layer = nrow * ncol
    right = np.full(ncells, -1, np.int64)
    front = np.full(ncells, -1, np.int64)
    lower = np.full(ncells, -1, np.int64)

    for i in range(ncells):
        for nzi in range(ia[i], ia[i + 1]):
            nzi -= 1  # python is 0-based, modflow6 is 1-based
            j = ja[nzi] - 1  # python is 0-based, modflow6 is 1-based
            d = j - i
            if d <= 0:  # left, back, upper
                continue
            elif d == 1:  # right neighbor
                right[i] = nzi
            elif d == ncol:  # front neighbor
                front[i] = nzi
            elif d == ncells_per_layer:  # lower neighbor
                lower[i] = nzi
            else:  # skips one: must be pass through
                npassed = int(d / ncells_per_layer)
                for ipass in range(0, npassed):
                    lower[i + ipass * ncells_per_layer] = nzi

    return right.reshape(shape), front.reshape(shape), lower.reshape(shape)


def dis_to_right_front_lower_indices(
    grb_content: dict,
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Infer the indices to extract right, front, and lower face flows from the
    flow-ja-face array.

    Parameters
    ----------
    grb_content: dict

    Returns
    -------
    right: xr.DataArray of ints with dims ("layer", "y", "x")
    front: xr.DataArray of ints with dims ("layer", "y", "x")
    lower: xr.DataArray of ints with dims ("layer", "y", "x")
    """
    right, front, lower = dis_indices(
        ia=grb_content["ia"],
        ja=grb_content["ja"],
        ncells=grb_content["ncells"],
        nlayer=grb_content["nlayer"],
        nrow=grb_content["nrow"],
        ncol=grb_content["ncol"],
    )
    return (
        xr.DataArray(right, grb_content["coords"], ("layer", "y", "x")),
        xr.DataArray(front, grb_content["coords"], ("layer", "y", "x")),
        xr.DataArray(lower, grb_content["coords"], ("layer", "y", "x")),
    )


def dis_extract_face_budgets(
    budgets: xr.DataArray, index: xr.DataArray
) -> xr.DataArray:
    """
    Grab right, front, or lower face flows from the flow-ja-face array.

    This could be done by a single .isel() indexing operation, but those
    are extremely slow in this case, which seems to be an xarray issue.

    Parameters
    ----------
    budgets: xr.DataArray of floats
        flow-ja-face array, dims ("time", "linear_index")
        The linear index enumerates cell-to-cell connections in this case, not
        the individual cells.
    index: xr.DataArray of ints
        right, front, or lower index array with dims("layer", "y", "x")

    Returns
    -------
    xr.DataArray of floats with dims ("time", "layer", "y", "x")
    """
    coords = dict(index.coords)
    coords["time"] = budgets["time"]
    # isel with a 3D array is extremely slow
    # this followed by the dask reshape is much faster for some reason.
    data = budgets.isel(linear_index=index.values.ravel()).data
    da = xr.DataArray(
        data=data.reshape((budgets["time"].size, *index.shape)),
        coords=coords,
        dims=("time", "layer", "y", "x"),
        name="flow-ja-face",
    )
    return da.where(index >= 0, other=0.0)


def dis_open_face_budgets(
    cbc_path: FilePath, grb_content: dict, header_list: List[cbc.Imeth1Header]
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Open the flow-ja-face, and extract right, front, and lower face flows.

    Parameters
    ----------
    cbc_path: str, pathlib.Path
    grb_content: dict
    header_list: List[Imeth1Header]

    Returns
    -------
    right: xr.DataArray of floats with dims ("time", "layer", "y", "x")
    front: xr.DataArray of floats with dims ("time", "layer", "y", "x")
    lower: xr.DataArray of floats with dims ("time", "layer", "y", "x")
    """
    right_index, front_index, lower_index = dis_to_right_front_lower_indices(
        grb_content
    )
    budgets = cbc.open_imeth1_budgets(cbc_path, header_list)
    right = dis_extract_face_budgets(budgets, right_index)
    front = dis_extract_face_budgets(budgets, front_index)
    lower = dis_extract_face_budgets(budgets, lower_index)
    return right, front, lower


# TODO: Currently assumes dis grb, can be checked & dispatched
def open_cbc(
    cbc_path: FilePath, grb_content: Dict[str, Any], flowja: bool = False
) -> Dict[str, xr.DataArray]:
    headers = cbc.read_cbc_headers(cbc_path)
    cbc_content = {}
    for key, header_list in headers.items():
        # TODO: validate homogeneity of header_list, ndat consistent, nlist consistent etc.
        if key == "flow-ja-face":
            if flowja:
                flowja, nm = cbc.open_face_budgets_as_flowja(
                    cbc_path, header_list, grb_content
                )
                cbc_content["flow-ja-face"] = flowja
                cbc_content["connectivity"] = nm
            else:
                right, front, lower = dis_open_face_budgets(
                    cbc_path, grb_content, header_list
                )
                cbc_content["flow-right-face"] = right
                cbc_content["flow-front-face"] = front
                cbc_content["flow-lower-face"] = lower
        else:
            if isinstance(header_list[0], cbc.Imeth1Header):
                cbc_content[key] = open_imeth1_budgets(
                    cbc_path, grb_content, header_list
                )
            elif isinstance(header_list[0], cbc.Imeth6Header):
                # for non cell flow budget terms, use auxiliary variables as return value
                if header_list[0].text.startswith("data-"):
                    for return_variable in header_list[0].auxtxt:
                        cbc_content[key + "-" + return_variable] = open_imeth6_budgets(
                            cbc_path, grb_content, header_list, return_variable
                        )
                else:
                    cbc_content[key] = open_imeth6_budgets(
                        cbc_path, grb_content, header_list
                    )

    return cbc_content


def grid_info(like: xr.DataArray) -> Dict[str, Any]:
    return {
        "nlayer": like["layer"].size,
        "nrow": like["y"].size,
        "ncol": like["x"].size,
        "coords": {
            "layer": like["layer"],
            "y": like["y"],
            "x": like["x"],
        },
    }
