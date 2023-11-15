"""
Read MODFLOW6 output

The dis, disv, disu modules implement the following functions:

```python
Darray = Union[xr.DataArray, xu.UgridDataArray]

def read_grb(f: BinaryIO, ntxt: int, lentxt: int) -> Dict[str, Any]:
    return

def read_times(*args) -> FloatArray:
    return

def read_hds_timestep(*args) -> FloatArray:
    return

def open_hds(path: FilePath, d: Dict[str, Any], dry_nan: bool) -> Darray:
    return

def open_imeth1_budgets(
    cbc_path: FilePath, grb_content: dict, header_list: List["Imeth1Header"]
) -> Darray:
    return

def open_imeth6_budgets(
    cbc_path: FilePath, grb_content: dict, header_list: List["Imeth6Header"]
) -> Darray:
    return

def open_cbc(
    cbc_path: FilePath, grb_content: Dict[str, Any]
) -> Dict[str, Darray]:
    return
```

(These could be implemented via Reader classes, but why bother with mutable
state or a class with exclusively staticmethods?)
"""
from typing import Any, Callable, Dict, Union

import xarray as xr
import xugrid as xu

from . import dis, disu, disv
from .cbc import read_cbc_headers
from .common import FilePath, _grb_text

_READ_GRB = {
    "grid dis": dis.read_grb,
    "grid disv": disv.read_grb,
    "grid disu": disu.read_grb,
}

_OPEN_HDS = {
    "dis": dis.open_hds,
    "disv": disv.open_hds,
    "disu": disu.open_hds,
}

_OPEN_CBC = {
    "dis": dis.open_cbc,
    "disv": disv.open_cbc,
    "disu": disu.open_cbc,
}


def _get_function(d: Dict[str, Callable], key: str) -> Callable:
    try:
        func = d[key]
    except KeyError:
        valid_options = ", ".join(d.keys()).lower()
        raise ValueError(f"Expected one of {valid_options}, got: {key}")
    return func


def read_grb(path: FilePath) -> Dict[str, Any]:
    """
    Read the data in a MODFLOW6 binary grid (.grb) file.

    Parameters
    ----------
    path: Union[str, pathlib.Path]

    Returns
    -------
    grb_content: Dict[str, Any]
    """
    with open(path, "rb") as f:
        h1 = _grb_text(f)
        _read = _get_function(_READ_GRB, h1)
        h2 = _grb_text(f)
        if h2 != "version 1":
            raise ValueError(f"Only version 1 supported, got {h2}")
        ntxt = int(_grb_text(f).split()[1])
        lentxt = int(_grb_text(f).split()[1])
        d = _read(f, ntxt, lentxt)
    return d


def open_hds(
    hds_path: FilePath, grb_path: FilePath, dry_nan: bool = False
) -> Union[xr.DataArray, xu.UgridDataArray]:
    """
    Open modflow6 heads (.hds) file.

    The data is lazily read per timestep and automatically converted into
    (dense) xr.DataArrays or xu.UgridDataArrays, for DIS and DISV respectively.
    The conversion is done via the information stored in the Binary Grid file
    (GRB).


    Parameters
    ----------
    hds_path: Union[str, pathlib.Path]
    grb_path: Union[str, pathlib.Path]
    dry_nan: bool, default value: False.
        Whether to convert dry values to NaN.

    Returns
    -------
    head: Union[xr.DataArray, xu.UgridDataArray]
    """
    grb_content = read_grb(grb_path)
    grb_content["name"] = "head"
    distype = grb_content["distype"]
    _open = _get_function(_OPEN_HDS, distype)
    return _open(hds_path, grb_content, dry_nan)


def open_conc(
    ucn_path: FilePath, grb_path: FilePath, dry_nan: bool = False
) -> Union[xr.DataArray, xu.UgridDataArray]:
    """
    Open Modflow6 "Unformatted Concentration" (.ucn) file.

    The data is lazily read per timestep and automatically converted into
    (dense) xr.DataArrays or xu.UgridDataArrays, for DIS and DISV respectively.
    The conversion is done via the information stored in the Binary Grid file
    (GRB).

    Parameters
    ----------
    ucn_path: Union[str, pathlib.Path]
    grb_path: Union[str, pathlib.Path]
    dry_nan: bool, default value: False.
        Whether to convert dry values to NaN.

    Returns
    -------
    concentration: Union[xr.DataArray, xu.UgridDataArray]
    """
    grb_content = read_grb(grb_path)
    grb_content["name"] = "concentration"
    distype = grb_content["distype"]
    _open = _get_function(_OPEN_HDS, distype)
    return _open(ucn_path, grb_content, dry_nan)


def open_hds_like(
    path: FilePath,
    like: Union[xr.DataArray, xu.UgridDataArray],
    dry_nan: bool = False,
) -> Union[xr.DataArray, xu.UgridDataArray]:
    """
    Open modflow6 heads (.hds) file.

    The data is lazily read per timestep and automatically converted into
    DataArrays. Shape and coordinates are inferred from ``like``.

    Parameters
    ----------
    hds_path: Union[str, pathlib.Path]
    like: Union[xr.DataArray, xu.UgridDataArray]
    dry_nan: bool, default value: False.
        Whether to convert dry values to NaN.

    Returns
    -------
    head: Union[xr.DataArray, xu.UgridDataArray]
    """
    # TODO: check shape with hds metadata.
    if isinstance(like, xr.DataArray):
        d = dis.grid_info(like)
        return dis.open_hds(path, d, dry_nan)

    elif isinstance(like, xu.UgridDataArray):
        d = disv.grid_info(like)
        return disv.open_hds(path, d, dry_nan)

    else:
        raise TypeError(
            "like should be a DataArray or UgridDataArray, "
            f"received instead {type(like)}"
        )


def open_cbc(
    cbc_path: FilePath, grb_path: FilePath, flowja: bool = False
) -> Dict[str, Union[xr.DataArray, xu.UgridDataArray]]:
    """
    Open modflow6 cell-by-cell (.cbc) file.

    The data is lazily read per timestep and automatically converted into
    (dense) xr.DataArrays or xu.UgridDataArrays, for DIS and DISV respectively.
    The conversion is done via the information stored in the Binary Grid file
    (GRB).

    The ``flowja`` argument controls whether the flow-ja-face array (if present)
    is returned in grid form as "as is". By default ``flowja=False`` and the
    array is returned in "grid form", meaning:

        * DIS: in right, front, and lower face flow. All flows are placed in
          the cell.
        * DISV: in horizontal and lower face flow.the horizontal flows are
          placed on the edges and the lower face flow is placed on the faces.

    When ``flowja=True``, the flow-ja-face array is returned as it is found in
    the CBC file, with a flow for every cell to cell connection. Additionally,
    a ``connectivity`` DataArray is returned describing for every cell (n) its
    connected cells (m).

    Parameters
    ----------
    cbc_path: str, pathlib.Path
        Path to the cell-by-cell flows file
    grb_path: str, pathlib.Path
        Path to the binary grid file
    flowja: bool, default value: False
        Whether to return the flow-ja-face values "as is" (``True``) or in a
        grid form (``False``).

    Returns
    -------
    cbc_content: Dict[str, xr.DataArray]
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
    grb_content = read_grb(grb_path)
    distype = grb_content["distype"]
    _open = _get_function(_OPEN_CBC, distype)
    return _open(cbc_path, grb_content, flowja)
