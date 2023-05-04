"""
Cell-by-cell flows
"""
import os
import struct
from collections import defaultdict
from typing import Any, BinaryIO, Dict, List, NamedTuple, Tuple, Union

import dask
import numpy as np
import xarray as xr

from .common import FilePath, FloatArray


class Imeth1Header(NamedTuple):
    kstp: int
    kper: int
    text: str
    ndim1: int
    ndim2: int
    ndim3: int
    imeth: int
    delt: float
    pertim: float
    totim: float
    pos: int


class Imeth6Header(NamedTuple):
    kstp: int
    kper: int
    text: str
    ndim1: int
    ndim2: int
    ndim3: int
    imeth: int
    delt: float
    pertim: float
    totim: float
    pos: int
    txt1id1: str
    txt2id1: str
    txt1id2: str
    txt2id2: str
    ndat: int
    auxtxt: List[str]
    nlist: int


def read_common_cbc_header(f: BinaryIO) -> Dict[str, Any]:
    """
    Read the common part (shared by imeth=1 and imeth6) of a CBC header section.
    """
    content = {}
    content["kstp"] = struct.unpack("i", f.read(4))[0]
    content["kper"] = struct.unpack("i", f.read(4))[0]
    content["text"] = f.read(16).decode("utf-8").strip().lower()
    content["ndim1"] = struct.unpack("i", f.read(4))[0]
    content["ndim2"] = struct.unpack("i", f.read(4))[0]
    content["ndim3"] = struct.unpack("i", f.read(4))[0]
    content["imeth"] = struct.unpack("i", f.read(4))[0]
    content["delt"] = struct.unpack("d", f.read(8))[0]
    content["pertim"] = struct.unpack("d", f.read(8))[0]
    content["totim"] = struct.unpack("d", f.read(8))[0]
    return content


def read_imeth6_header(f: BinaryIO) -> Dict[str, Any]:
    """
    Read the imeth=6 specific data of a CBC header section.
    """
    content = {}
    content["txt1id1"] = f.read(16).decode("utf-8").strip().lower()
    content["txt2id1"] = f.read(16).decode("utf-8").strip().lower()
    content["txt1id2"] = f.read(16).decode("utf-8").strip().lower()
    content["txt2id2"] = f.read(16).decode("utf-8").strip().lower()
    ndat = struct.unpack("i", f.read(4))[0]
    content["ndat"] = ndat
    content["auxtxt"] = [
        f.read(16).decode("utf-8").strip().lower() for _ in range(ndat - 1)
    ]
    content["nlist"] = struct.unpack("i", f.read(4))[0]
    return content


def read_cbc_headers(
    cbc_path: FilePath,
) -> Dict[str, List[Union[Imeth1Header, Imeth6Header]]]:
    """
    Read all the header data from a cell-by-cell (.cbc) budget file.

    All budget data for a MODFLOW6 model is stored in a single file. This
    function collects all header data, as well as the starting byte position of
    the actual budget data.

    This function groups the headers per TEXT record (e.g. "flow-ja-face",
    "drn", etc.). The headers are stored as a list of named tuples.
    flow-ja-face, storage-ss, and storage-sy are written using IMETH=1, all
    others with IMETH=6.

    Parameters
    ----------
    cbc_path: str, pathlib.Path
        Path to the budget file.

    Returns
    -------
    headers: Dict[List[UnionImeth1Header, Imeth6Header]]
        Dictionary containing a list of headers per TEXT record in the budget
        file.
    """
    headers = defaultdict(list)
    with open(cbc_path, "rb") as f:
        filesize = os.fstat(f.fileno()).st_size
        while f.tell() < filesize:
            header = read_common_cbc_header(f)
            if header["imeth"] == 1:
                datasize = (
                    # Multiply by -1 because ndim3 is stored as a negative for some reason.
                    # (ndim3 is the integer size of the third dimension)
                    header["ndim1"]
                    * header["ndim2"]
                    * header["ndim3"]
                    * -1
                ) * 8
                header["pos"] = f.tell()
                key = header["text"]
                headers[key].append(Imeth1Header(**header))
            elif header["imeth"] == 6:
                imeth6_header = read_imeth6_header(f)
                datasize = imeth6_header["nlist"] * (8 + imeth6_header["ndat"] * 8)
                header["pos"] = f.tell()
                key = imeth6_header["txt2id2"]
                headers[key].append(Imeth6Header(**header, **imeth6_header))
            else:
                raise ValueError(
                    f"Invalid imeth value in CBC file {cbc_path}. "
                    f"Should be 1 or 6, received: {header['imeth']}."
                )
            # Skip the data
            f.seek(datasize, 1)
    return headers


def read_imeth1_budgets(cbc_path: FilePath, count: int, pos: int) -> FloatArray:
    """
    Read the data for an imeth=1 budget section.

    Parameters
    ----------
    cbc_path: str, pathlib.Path
    count: int
        number of values to read
    pos:
        position in the file where the data for a timestep starts

    Returns
    -------
    1-D array of floats
    """
    with open(cbc_path, "rb") as f:
        f.seek(pos)
        timestep_budgets = np.fromfile(f, np.float64, count)
    return timestep_budgets


def open_imeth1_budgets(
    cbc_path: FilePath, header_list: List[Imeth1Header]
) -> xr.DataArray:
    """
    Open the data for an imeth==1 budget section. Data is read lazily per
    timestep. The cell data is not spatially labelled.

    Parameters
    ----------
    cbc_path: str, pathlib.Path
    header_list: List[Imeth1Header]

    Returns
    -------
    xr.DataArray with dims ("time", "linear_index")
    """
    # Gather times from the headers
    dask_list = []
    time = np.empty(len(header_list), dtype=np.float64)
    for i, header in enumerate(header_list):
        time[i] = header.totim
        count = header.ndim1 * header.ndim2 * header.ndim3 * -1
        a = dask.delayed(read_imeth1_budgets)(cbc_path, count, header.pos)
        x = dask.array.from_delayed(a, shape=(count,), dtype=np.float64)
        dask_list.append(x)

    return xr.DataArray(
        data=dask.array.stack(dask_list, axis=0),
        coords={"time": time},
        dims=("time", "linear_index"),
        name=header_list[0].text,
    )


def expand_indptr(ia) -> np.ndarray:
    n = np.diff(ia)
    return np.repeat(np.arange(ia.size - 1), n)


def open_face_budgets_as_flowja(
    cbc_path: FilePath, header_list: List[Imeth1Header], grb_content: Dict[str, Any]
) -> Tuple[xr.DataArray, xr.DataArray]:
    flowja = open_imeth1_budgets(cbc_path, header_list)
    flowja = flowja.rename({"linear_index": "connection"})
    n = expand_indptr(grb_content["ia"])
    m = grb_content["ja"] - 1
    nm = xr.DataArray(
        np.column_stack([n, m]),
        coords={"cell": ["n", "m"]},
        dims=["connection", "cell"],
    )
    return flowja, nm


def read_imeth6_budgets(
    cbc_path: FilePath, count: int, dtype: np.dtype, pos: int
) -> Any:
    """
    Read the data for an imeth==6 budget section for a single timestep.

    Returns a numpy structured array containing:
    * id1: the model cell number
    * id2: the boundary condition index
    * budget: the budget terms
    * and assorted auxiliary columns, if present

    Parameters
    ----------
    cbc_path: str, pathlib.Path
    count: int
        number of values to read
    dtype: numpy dtype
        Data type of the structured array. Contains at least "id1", "id2", and "budget".
        Optionally contains auxiliary columns.
    pos:
        position in the file where the data for a timestep starts

    Returns
    -------
    Numpy structured array of type dtype
    """
    with open(cbc_path, "rb") as f:
        f.seek(pos)
        table = np.fromfile(f, dtype, count)
    return table


def read_imeth6_budgets_dense(
    cbc_path: FilePath,
    count: int,
    dtype: np.dtype,
    pos: int,
    size: int,
    shape: tuple,
) -> FloatArray:
    """
    Read the data for an imeth==6 budget section.

    Utilizes the shape information from the DIS GRB file to create a dense numpy
    array. Always allocates for the entire domain (all layers, rows, columns).

    Parameters
    ----------
    cbc_path: str, pathlib.Path
    count: int
        number of values to read
    dtype: numpy dtype
        Data type of the structured array. Contains at least "id1", "id2", and "budget".
        Optionally contains auxiliary columns.
    pos: int
        position in the file where the data for a timestep starts
    size: int
        size of the entire model domain
    shape: tuple[int, int, int]
        Shape (nlayer, nrow, ncolumn) of entire model domain.

    Returns
    -------
    Three-dimensional array of floats
    """
    # Allocates a dense array for the entire domain
    out = np.zeros(size, dtype=np.float64)
    table = read_imeth6_budgets(cbc_path, count, dtype, pos)
    id1 = table["id1"] - 1  # Convert to 0 based index
    out[id1] = table["budget"]
    return out.reshape(shape)
