import os
import struct
from typing import Any, BinaryIO, Dict, List, Tuple

import dask
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


def read_grb(f: BinaryIO, ntxt: int, lentxt: int) -> dict[str, Any]:
    # we don't need any information from the the text lines that follow,
    # they are definitions that aim to make the file more portable,
    # so let's skip straight to the binary data
    f.seek(ntxt * lentxt, 1)

    ncells = struct.unpack("i", f.read(4))[0]
    nlayer = struct.unpack("i", f.read(4))[0]
    ncell_per_layer = struct.unpack("i", f.read(4))[0]
    nvert = struct.unpack("i", f.read(4))[0]
    njavert = struct.unpack("i", f.read(4))[0]
    nja = struct.unpack("i", f.read(4))[0]
    if ncells != (nlayer * ncell_per_layer):
        raise ValueError(f"Invalid file {ncells} {nlayer} {ncell_per_layer}")
    _ = struct.unpack("d", f.read(8))[0]  # xorigin
    _ = struct.unpack("d", f.read(8))[0]  # yorigin
    f.seek(8, 1)  # skip angrot
    top_np = np.fromfile(f, np.float64, ncell_per_layer)
    bottom_np = np.reshape(
        np.fromfile(f, np.float64, ncells), (nlayer, ncell_per_layer)
    )
    vertices = np.reshape(np.fromfile(f, np.float64, nvert * 2), (nvert, 2))
    _ = np.fromfile(f, np.float64, ncell_per_layer)  # cellx
    _ = np.fromfile(f, np.float64, ncell_per_layer)  # celly
    # Python is 0-based; MODFLOW6 is Fortran 1-based
    iavert = np.fromfile(f, np.int32, ncell_per_layer + 1)
    javert = np.fromfile(f, np.int32, njavert)
    ia = np.fromfile(f, np.int32, ncells + 1)
    ja = np.fromfile(f, np.int32, nja)
    idomain_np = np.reshape(np.fromfile(f, np.int32, ncells), (nlayer, ncell_per_layer))
    icelltype_np = np.reshape(
        np.fromfile(f, np.int32, ncells), (nlayer, ncell_per_layer)
    )

    iavert, javert = _ugrid_iavert_javert(iavert, javert)
    face_nodes = scipy.sparse.csr_matrix((javert, javert, iavert))
    grid = xu.Ugrid2d(vertices[:, 0], vertices[:, 1], -1, face_nodes)

    top = xu.UgridDataArray(xr.DataArray(top_np, dims=["face"], name="top"), grid)
    coords = {"layer": np.arange(1, nlayer + 1)}
    dims = ("layer", "face")
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
        "ncell_per_layer": ncell_per_layer,
        "nja": nja,
        "ia": ia,
        "ja": ja,
        "idomain": xu.UgridDataArray(idomain, grid),
        "icelltype": xu.UgridDataArray(icelltype, grid),
    }


def read_times(
    path: FilePath, ntime: int, nlayer: int, ncell_per_layer: int
) -> FloatArray:
    """
    Reads all total simulation times.
    """
    times = np.empty(ntime, dtype=np.float64)

    # Compute how much to skip to the next timestamp
    start_of_header = 16
    rest_of_header = 28
    data_single_layer = ncell_per_layer * 8
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
    path: FilePath, nlayer: int, ncell_per_layer: int, dry_nan: bool, pos: int
) -> FloatArray:
    """
    Reads all values of one timestep.
    """
    with open(path, "rb") as f:
        f.seek(pos)
        a1d = np.empty(nlayer * ncell_per_layer, dtype=np.float64)
        for k in range(nlayer):
            f.seek(52, 1)  # skip kstp, kper, pertime
            a1d[k * ncell_per_layer : (k + 1) * ncell_per_layer] = np.fromfile(
                f, np.float64, ncell_per_layer
            )

    a2d = a1d.reshape((nlayer, ncell_per_layer))
    return _to_nan(a2d, dry_nan)


def open_hds(path: FilePath, d: dict[str, Any], dry_nan: bool) -> xu.UgridDataArray:
    nlayer, ncell_per_layer = d["nlayer"], d["nrow"], d["ncol"]
    filesize = os.path.getsize(path)
    ntime = filesize // (nlayer * (52 + (ncell_per_layer * 8)))
    times = read_times(path, ntime, nlayer, ncell_per_layer)
    d["coords"]["time"] = times

    dask_list = []
    # loop over times and add delayed arrays
    for i in range(ntime):
        # TODO verify dimension order
        pos = i * (nlayer * (52 + ncell_per_layer * 8))
        a = dask.delayed(read_hds_timestep)(path, nlayer, ncell_per_layer, dry_nan, pos)
        x = dask.array.from_delayed(
            a, shape=(nlayer, ncell_per_layer), dtype=np.float64
        )
        dask_list.append(x)

    daskarr = dask.array.stack(dask_list, axis=0)
    da = xr.DataArray(daskarr, d["coords"], ("time", "layer", "face"), name="head")
    return xu.UgriDataArray(da, d["grid"])


def open_imeth1_budgets(
    cbc_path: FilePath, grb_content: dict, header_list: List[cbc.Imeth1Header]
) -> xu.UgridDataArray:
    """
    Open the data for an imeth==1 budget section. Data is read lazily per
    timestep.

    Utilizes the shape information from the DIS GRB file to create a dense
    array; (lazily) allocates for the entire domain (all layers, edges)
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
    ncell_per_layer = grb_content["ncell_per_layer"]
    coords = grb_content["coords"]
    budgets = cbc.open_imeth1_budgets(cbc_path, header_list)
    coords["time"] = budgets["time"]

    da = xr.DataArray(
        data=budgets.data.reshape((budgets["time"].size, nlayer, ncell_per_layer)),
        coords=coords,
        dims=("time", "layer", "edge"),
        name="flow-ja-face",
    )
    return xu.UgridDataArray(da, grb_content["grid"])


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
    shape = (grb_content["nlayer"], grb_content["ncell_per_layer"])
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
    da = xr.DataArray(daskarr, coords, ("time", "layer", "face"), name=name)
    return xu.UgridDataArray(da, grb_content["grid"])


def open_cbc(
    cbc_path: FilePath, grb_content: Dict[str, Any]
) -> Dict[str, xu.UgridDataArray]:
    headers = cbc.read_cbc_headers(cbc_path)
    cbc_content = {}
    for key, header_list in headers.items():
        if isinstance(header_list[0], cbc.Imeth1Header):
            cbc_content[key] = open_imeth1_budgets(cbc_path, grb_content, header_list)
        elif isinstance(header_list[0], cbc.Imeth6Header):
            cbc_content[key] = open_imeth6_budgets(cbc_path, grb_content, header_list)

    return cbc_content
