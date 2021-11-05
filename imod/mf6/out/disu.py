import struct
from typing import Any, BinaryIO, Dict, List, NamedTuple, Union

import numpy as np
import xarray as xr
import xugrid as xu

from . import cbc
from .common import FilePath, FloatArray, IntArray, _grb_text


def read_grb(f: BinaryIO, ntxt: int, lentxt: int) -> dict[str, Any]:
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
    xorigin = struct.unpack("d", f.read(8))[0]
    yorigin = struct.unpack("d", f.read(8))[0]
    f.seek(8, 1)  # skip angrot
    top_np = np.fromfile(f, np.float64, ncells)
    bottom_np = np.fromfile(f, np.float64, ncells)
    ia = np.fromfile(f, np.int32, ncells + 1)
    ja = np.fromfile(f, np.int32, nja)
    idomain_np = np.fromfile(f, np.int32, ncells)
    icelltype_np = np.fromfile(f, np.int32, ncells)

    if read_vertices:
        vertices = np.reshape(np.fromfile(f, np.float64, nvert * 2), (nvert, 2))
        cellx = np.fromfile(f, np.float64, ncells)
        celly = np.fromfile(f, np.float64, ncells)
        iavert = np.fromfile(f, np.int32, ncells + 1)
        javert = np.fromfile(f, np.int32, njavert)

    return


def read_times(path: FilePath, ntime: int, ncell: int):
    raise NotImplementedError


def read_hds_timestep(
    path: FilePath, nlayer: int, ncell_per_layer: int, dry_nan: bool, pos: int
):
    raise NotImplementedError


def open_hds(path: FilePath, d: dict[str, Any], dry_nan: bool) -> xr.DataArray:
    raise NotImplementedError


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
