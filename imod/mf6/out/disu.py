import os
import struct
from typing import Any, BinaryIO, Dict, List

import dask
import numpy as np
import xarray as xr
import xugrid as xu

from . import cbc
from .common import FilePath, _grb_text, _to_nan


def read_grb(f: BinaryIO, ntxt: int, lentxt: int) -> Dict[str, Any]:
    read_vertices = lentxt > 10

    if read_vertices:
        f.seek(10 * lentxt, 1)
        nvert = int(_grb_text(f).split()[-1])
        f.seek(3 * lentxt, 1)
        njavert = int(_grb_text(f).split()[-1])
    else:
        # we don't need any information from the the text lines that follow,
        # they are definitions that aim to make the file more portable,
        # so let's skip straight to the binary data
        f.seek(ntxt * lentxt, 1)

    ncells = struct.unpack("i", f.read(4))[0]
    nja = struct.unpack("i", f.read(4))[0]
    _ = struct.unpack("d", f.read(8))[0]  # xorigion
    _ = struct.unpack("d", f.read(8))[0]  # yorigin
    f.seek(8, 1)  # skip angrot
    top_np = np.fromfile(f, np.float64, ncells)
    bottom_np = np.fromfile(f, np.float64, ncells)
    ia = np.fromfile(f, np.int32, ncells + 1)
    ja = np.fromfile(f, np.int32, nja)
    idomain_np = np.fromfile(f, np.int32, ncells)
    icelltype_np = np.fromfile(f, np.int32, ncells)

    out = {
        "distype": "disu",
        "top": top_np,
        "bottom": bottom_np,
        "ncells": ncells,
        "nja": nja,
        "ia": ia,
        "ja": ja,
        "idomain": idomain_np,
        "icelltype": icelltype_np,
    }

    if read_vertices:
        out["vertices"] = np.reshape(np.fromfile(f, np.float64, nvert * 2), (nvert, 2))
        out["cellx"] = np.fromfile(f, np.float64, ncells)
        out["celly"] = np.fromfile(f, np.float64, ncells)
        out["iavert"] = np.fromfile(f, np.int32, ncells + 1)
        out["javert"] = np.fromfile(f, np.int32, njavert)

    return out


def read_times(path: FilePath, ntime: int, ncell: int):
    """
    Reads all total simulation times.
    """
    times = np.empty(ntime, dtype=np.float64)

    # Compute how much to skip to the next timestamp
    start_of_header = 16  # KSTP(4), KPER(4), PERTIM(8)
    rest_of_header = 28  # TEXT(16), NCOL(4), NROW(4), ILAY(4)
    data = ncell * 8
    nskip = rest_of_header + data + start_of_header
    with open(path, "rb") as f:
        f.seek(start_of_header)
        for i in range(ntime):
            times[i] = struct.unpack("d", f.read(8))[0]
            f.seek(nskip, 1)
    return times


def read_hds_timestep(path: FilePath, ncell: int, dry_nan: bool, pos: int):
    with open(path, "rb") as f:
        f.seek(pos + 52)
        a1d = np.fromfile(f, np.float64, ncell)
    return _to_nan(a1d, dry_nan)


def open_hds(path: FilePath, d: Dict[str, Any], dry_nan: bool) -> xr.DataArray:
    ncell = d["ncell"]
    filesize = os.path.getsize(path)
    ntime = filesize // (52 + ncell * 8)
    times = read_times(path, ntime, ncell)
    coords = d["coords"]
    coords["time"] = times

    dask_list = []
    for i in range(ntime):
        pos = i * (52 + ncell * 8)
        a = dask.delayed(read_hds_timestep)(path, ncell, dry_nan, pos)
        x = dask.array.from_delayed(a, shape=(ncell,), dtype=np.float64)
        dask_list.append(x)

    daskarr = dask.array.stack(dask_list, axis=0)
    return xr.DataArray(daskarr, coords, ("time", "node"), name=d["name"])


def open_imeth1_budgets(
    cbc_path: FilePath, grb_content: dict, header_list: List[cbc.Imeth1Header]
) -> xr.DataArray:
    raise NotImplementedError


def open_imeth6_budgets(
    cbc_path: FilePath, grb_content: dict, header_list: List[cbc.Imeth6Header]
) -> xr.DataArray:
    raise NotImplementedError


def open_cbc(
    cbc_path: FilePath, grb_content: Dict[str, Any]
) -> Dict[str, xu.UgridDataArray]:
    raise NotImplementedError
