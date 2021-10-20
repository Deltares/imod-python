import os
import pathlib
import struct
from collections import defaultdict
from typing import Any, BinaryIO, Dict, List, NamedTuple, Tuple, Union

import dask
import numba
import numpy as np
import xarray as xr

import imod

# Type annotations
IntArray = np.ndarray
FloatArray = np.ndarray
FilePath = Union[str, pathlib.Path]


def _grb_text(f: BinaryIO, lentxt: int = 50):
    return f.read(lentxt).decode("utf-8").strip().lower()


# Binary Grid File / DIS Grids
# https://water.usgs.gov/water-resources/software/MODFLOW-6/mf6io_6.0.4.pdf#page=162
def open_disgrb(path):
    with open(path, "rb") as f:
        h1 = _grb_text(f)
        h2 = _grb_text(f)
        if h1 != "grid dis":
            raise ValueError(f'Expected "grid dis" file, got {h1}')
        if h2 != "version 1":
            raise ValueError(f"Only version 1 supported, got {h2}")

        ntxt = int(_grb_text(f).split()[1])
        lentxt = int(_grb_text(f).split()[1])

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
        delr = np.fromfile(f, np.float64, nrow)
        delc = np.fromfile(f, np.float64, ncol)
        # TODO verify dimension order
        top_np = np.reshape(np.fromfile(f, np.float64, nrow * ncol), (nrow, ncol))
        bottom_np = np.reshape(np.fromfile(f, np.float64, ncells), (nlayer, nrow, ncol))
        ia = np.fromfile(f, np.int32, ncells + 1)
        ja = np.fromfile(f, np.int32, nja)
        idomain_np = np.reshape(np.fromfile(f, np.int32, ncells), (nlayer, nrow, ncol))
        icelltype_np = np.reshape(
            np.fromfile(f, np.int32, ncells), (nlayer, nrow, ncol)
        )

    bounds = (xorigin, xorigin + delc.sum(), yorigin, yorigin + delr.sum())
    coords = imod.util._xycoords(bounds, (delc, -delr))
    top = xr.DataArray(top_np, coords, ("y", "x"), name="top")
    coords["layer"] = np.arange(1, nlayer + 1)
    dims = ("layer", "y", "x")
    bottom = xr.DataArray(bottom_np, coords, dims, name="bottom")
    idomain = xr.DataArray(idomain_np, coords, dims, name="idomain")
    icelltype = xr.DataArray(icelltype_np, coords, dims, name="icelltype")

    return {
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


def _to_nan(a, dry_nan):
    # TODO: this could really use a docstring?
    a[a == 1e30] = np.nan
    if dry_nan:
        a[a == -1e30] = np.nan
    return a


def _read_hds(path, nlayer, nrow, ncol, dry_nan, pos):
    """
    Reads all values of one timestep.
    """
    n_per_layer = nrow * ncol
    with open(path, "rb") as f:
        f.seek(pos)
        a1d = np.empty(nlayer * nrow * ncol, dtype=np.float64)
        for k in range(nlayer):
            f.seek(52, 1)  # skip kstp, kper, pertime
            a1d[k * n_per_layer : (k + 1) * n_per_layer] = np.fromfile(
                f, np.float64, nrow * ncol
            )

    a3d = a1d.reshape((nlayer, nrow, ncol))
    return _to_nan(a3d, dry_nan)


def _read_times(path, ntime, nlayer, nrow, ncol):
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


# Dependent Variable File / DIS Grids
# https://water.usgs.gov/water-resources/software/MODFLOW-6/mf6io_6.0.4.pdf#page=167
def open_hds(hds_path, grb_path, dry_nan=False):
    """
    Open head data
    """

    d = open_disgrb(grb_path)
    nlayer, nrow, ncol = d["nlayer"], d["nrow"], d["ncol"]
    filesize = os.path.getsize(hds_path)
    ntime = filesize // (nlayer * (52 + (nrow * ncol * 8)))
    times = _read_times(hds_path, ntime, nlayer, nrow, ncol)
    d["coords"]["time"] = times

    dask_list = []
    # loop over times and add delayed arrays
    for i in range(ntime):
        # TODO verify dimension order
        pos = i * (nlayer * (52 + nrow * ncol * 8))
        a = dask.delayed(_read_hds)(hds_path, nlayer, nrow, ncol, dry_nan, pos)
        x = dask.array.from_delayed(a, shape=(nlayer, nrow, ncol), dtype=np.float64)
        dask_list.append(x)

    daskarr = dask.array.stack(dask_list, axis=0)
    return xr.DataArray(daskarr, d["coords"], ("time", "layer", "y", "x"), name="head")


# -------------------
# Cell-by-cell flows
# -------------------
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


def _read_common_cbc_header(f: BinaryIO) -> Dict[str, Any]:
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


def _read_imeth6_header(f: BinaryIO) -> Dict[str, Any]:
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
            header = _read_common_cbc_header(f)
            if header["imeth"] == 1:
                datasize = header["ndim1"] * 8
                header["pos"] = f.tell()
                key = header["text"]
                headers[key].append(Imeth1Header(**header))
            elif header["imeth"] == 6:
                imeth6_header = _read_imeth6_header(f)
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


def _read_imeth1_budgets(cbc_path: FilePath, count: int, pos: int) -> FloatArray:
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


def _open_imeth1_budgets(
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
        a = dask.delayed(_read_imeth1_budgets)(cbc_path, header.ndim1, header.pos)
        x = dask.array.from_delayed(a, shape=(header.ndim1,), dtype=np.float64)
        dask_list.append(x)

    return xr.DataArray(
        data=dask.array.stack(dask_list, axis=0),
        coords={"time": time},
        dims=("time", "linear_index"),
        name=header_list[0].text,
    )


def _read_imeth6_budgets(
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


def _open_imeth6_budgets(
    cbc_path: FilePath, header_list: List[Imeth6Header]
) -> List[Any]:  # List[dask.delayed.Delayed]:
    """
    Open the data for an imeth==6 budget section. Data is read lazily per
    timestep.

    Does not convert the data to something that can be stored in a DataArray
    immediately. Rather returns a delayed numpy structured array.

    Parameters
    ----------
    cbc_path: str, pathlib.Path
    header_list: List[Imeth6Header]

    Returns
    -------
    List of dask Delayed structured arrays.
    """
    dtype = np.dtype(
        [("id1", np.int32), ("id2", np.int32), ("budget", np.float64)]
        + [(name, np.float64) for name in header_list[0].auxtxt]
    )

    dask_list = []
    for header in header_list:
        x = dask.delayed(_read_imeth6_budgets)(
            cbc_path, header.nlist, dtype, header.pos
        )
        dask_list.append(x)

    return dask_list


def _dis_read_imeth6_budgets(
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
    with open(cbc_path, "rb") as f:
        f.seek(pos)
        table = np.fromfile(f, dtype, count)
    out[table["id1"]] = table["budget"]
    return out.reshape(shape)


def _dis_open_imeth1_budgets(
    cbc_path: FilePath, grb_content: dict, header_list: List[Imeth1Header]
) -> xr.DataArray:
    """
    Open the data for an imeth==1 budget section. Data is read lazily per
    timestep.

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
    coords = grb_content["coords"]
    budgets = _open_imeth1_budgets(cbc_path, header_list)
    coords["time"] = budgets["time"]

    return xr.DataArray(
        data=budgets.data.reshape((budgets["time"].size, nlayer, nrow, ncol)),
        coords=coords,
        dims=("time", "layer", "y", "x"),
        name="flow-ja-face",
    )


def _dis_open_imeth6_budgets(
    cbc_path: FilePath, grb_content: dict, header_list: List[Imeth6Header]
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
        a = dask.delayed(_dis_read_imeth6_budgets)(
            cbc_path, header.nlist, dtype, header.pos, size, shape
        )
        x = dask.array.from_delayed(a, shape=shape, dtype=np.float64)
        dask_list.append(x)

    daskarr = dask.array.stack(dask_list, axis=0)
    coords = grb_content["coords"]
    coords["time"] = time
    name = header_list[0].text
    return xr.DataArray(daskarr, coords, ("time", "layer", "y", "x"), name=name)


@numba.njit
def _dis_indices(
    ia: IntArray,
    ja: IntArray,
    ncells: int,
    nlayer: int,
    nrow: int,
    ncol: int,
):
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


def _dis_to_right_front_lower_indices(
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
    right, front, lower = _dis_indices(
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


def _dis_extract_face_budgets(
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


def _dis_open_face_budgets(
    cbc_path: FilePath, grb_content: dict, header_list: List[Imeth1Header]
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
    right_index, front_index, lower_index = _dis_to_right_front_lower_indices(
        grb_content
    )
    budgets = _open_imeth1_budgets(cbc_path, header_list)
    right = _dis_extract_face_budgets(budgets, right_index)
    front = _dis_extract_face_budgets(budgets, front_index)
    lower = _dis_extract_face_budgets(budgets, lower_index)
    return right, front, lower


# TODO: Currently assumes dis grb, can be checked & dispatched
def open_cbc(cbc_path: FilePath, grb_path: FilePath) -> Dict[str, xr.DataArray]:
    """
    Open modflow6 cell-by-cell (.cbc) file.

    The data is lazily read per timestep and automatically converted into
    (dense) xr.DataArrays. The conversion is done via the information stored in
    the Binary Grid File (GRB).

    Currently only structured discretization (DIS) is supported. The flow-ja-face
    data is automatically converted into "right-face-flow", "front-face-flow" and
    "lower-face-flow".

    Parameters
    ----------
    cbc_path: str, pathlib.Path
        Path to the cell-by-cell flows file
    grb_path: str, pathlib.Path
        Path to the binary grid file

    Returns
    -------
    cbc_content: dict[str, xr.DataArray]
        DataArray contains float64 data of the budgets, with dimensions ("time",
        "layer", "y", "x").

    Examples
    --------

    Open a cbc file:

    >>> import imod
    >>> cbc_content = imod.mf6.open_cbc("budgets.cbc", "my-model.grb")

    Check the contents:

    >>> print(cbc_content.keys())

    Get the drainage budget, compute a time mean for the first layer:

    >>> drn_budget = cbc_content["drn]
    >>> mean = drn_budget.sel(layer=1).mean("time")

    """
    grb_content = imod.mf6.out.open_disgrb(grb_path)
    headers = read_cbc_headers(cbc_path)
    cbc_content = {}
    for key, header_list in headers.items():
        # TODO: validate homogeneity of header_list, ndat consistent, nlist consistent etc.
        if key == "flow-ja-face":
            right, front, lower = _dis_open_face_budgets(
                cbc_path, grb_content, header_list
            )
            cbc_content["right-face-flow"] = right
            cbc_content["front-face-flow"] = front
            cbc_content["lower-face-flow"] = lower
        else:
            if isinstance(header_list[0], Imeth1Header):
                cbc_content[key] = _dis_open_imeth1_budgets(
                    cbc_path, grb_content, header_list
                )
            elif isinstance(header_list[0], Imeth6Header):
                cbc_content[key] = _dis_open_imeth6_budgets(
                    cbc_path, grb_content, header_list
                )

    return cbc_content
