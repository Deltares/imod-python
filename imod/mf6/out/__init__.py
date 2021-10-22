"""
Read MODFLOW6 output

The dis, disv, disu modules implement the following functions:

```python
Darray = Union[xr.DataArray, xu.UgridDataArray]

def read_grb(f: BinaryIO, ntxt: int, lentxt: int) -> dict[str, Any]:
    return

def read_times(*args) -> FloatArray:
    return

def read_hds_timestep(*args) -> FloatArray:
    return

def open_hds(path: FilePath, d: dict[str, Any], dry_nan: bool) -> Darray:
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
    "dis": dis.read_hds,
    "disv": disv.read_hds,
    "disu": disu.read_hds,
}

_OPEN_CBC = {
    "dis": dis.read_hds,
    "disv": disv.read_hds,
    "disu": disu.read_hds,
}


def _get_function(d: Dict[str, Callable], key: str) -> Callable:
    try:
        func = d[key]
    except KeyError:
        valid_options = ", ".join(d.keys()).lower()
        raise ValueError(f"Expected one of {valid_options}, got: {key}")
    return func


def read_grb(path: FilePath) -> dict[str, Any]:
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
    Open head data
    """
    grb_content = read_grb(grb_path)
    distype = grb_content["distype"]
    _open = _get_function(_OPEN_HDS, distype)
    return _open(hds_path, grb_content, dry_nan)


def open_cbc(
    cbc_path: FilePath, grb_path: FilePath
) -> Dict[str, Union[xr.DataArray, xu.UgridDataArray]]:
    """
    Open modflow6 cell-by-cell (.cbc) file.

    The data is lazily read per timestep and automatically converted into
    (dense) xr.DataArrays. The conversion is done via the information stored in
    the Binary Grid File (GRB).

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
    grb_content = read_grb(grb_path)
    distype = grb_content["distype"]
    _open = _get_function(_OPEN_CBC, distype)
    return _open(cbc_path, grb_content)
