from functools import wraps
from typing import Callable, ParamSpec, TypeVar

import pandas as pd

from imod.typing import DropVarsType, GridDataArray, Imod5DataDict, SelSettingsType
from imod.typing.grid import enforce_dim_order

_DROP_LAYER_KWARGS: SelSettingsType = {
    "layer": 0,
    "drop": True,
    "missing_dims": "ignore",
}

_DROP_VARS_KWARGS: DropVarsType = {
    "names": "layer",
    "errors": "ignore",
}

T = TypeVar("T")
P = ParamSpec("P")


def enforced_dim_order(func: Callable[P, T]) -> Callable[P, T]:
    """Decorator to enforce dimension order after function call"""

    @wraps(func)
    def decorator(*args: P.args, **kwargs: P.kwargs):
        x = func(*args, **kwargs)
        # Multiple grids returned
        if isinstance(x, tuple):
            return tuple(enforce_dim_order(i) for i in x)
        return enforce_dim_order(x)

    return decorator


def _drop_layer_dim_and_coord(
    da: GridDataArray,
) -> GridDataArray:
    """
    Drop the layer dimension and the layer coord if present from the given
    DataArray.
    """
    return da.isel(**_DROP_LAYER_KWARGS).compute().drop_vars(**_DROP_VARS_KWARGS)


def _drop_layer_if_dataarray(
    da: GridDataArray | pd.DataFrame,
) -> GridDataArray | pd.DataFrame:
    # There can be a dataframe in the cap data in case sprinkling is used with
    # IPF.
    if isinstance(da, pd.DataFrame):
        return da
    return _drop_layer_dim_and_coord(da)


def drop_layer_dim_cap_data(imod5_data: Imod5DataDict) -> Imod5DataDict:
    cap_data = imod5_data["cap"]
    return {"cap": {key: _drop_layer_if_dataarray(da) for key, da in cap_data.items()}}
