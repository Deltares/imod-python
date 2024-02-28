from typing import Any, Optional

import numpy as np
import xarray as xr



def where(condition, if_true, if_false, keep_nan: bool = True) -> xr.DataArray:
    """
    Wrapped version of xarray's ``.where``.

    This wrapped version does two differently:

    Firstly, it prioritizes the dimensions as: ``if_true > if_false > condition``.
    ``xarray.where(cond, a, b)`` will choose the dimension over ``a`` or ``b``.
    This may result in unwanted dimension orders such as ``("y", "x", "layer)``
    rather than ``("layer", "y', "x")``.

    Secondly, it preserves the NaN values of ``if_true`` by default.  If we
    wish to replace all values over 5 by 5, yet keep the NoData parts, this
    requires two operations with with xarray's ``where``.

    Parameters
    ----------
    condition: DataArray, Dataset
        Locations at which to preserve this object's values. dtype must be `bool`.
    if_true : scalar, DataArray or Dataset, optional
        Value to use for locations where ``cond`` is True.
    if_false : scalar, DataArray or Dataset, optional
        Value to use for locations where ``cond`` is False.
    keep_nan: bool, default: True
        Whether to keep the NaN values in place of ``if_true``.
    """
    xr_obj = (xr.DataArray, xr.Dataset)
    da_true = isinstance(if_true, xr_obj)
    da_false = isinstance(if_false, xr_obj)
    da_cond = isinstance(condition, xr_obj)

    # Give priority to where_true or where_false for broadcasting.
    if da_true:
        new = if_true.copy()
    elif da_false:
        new = xr.full_like(if_false, if_true)
    elif da_cond:
        new = xr.full_like(condition, if_true, dtype=type(if_true))
    else:
        raise ValueError(
            "at least one of {condition, if_true, if_false} should be a "
            "DataArray or Dataset"
        )

    new = new.where(condition, other=if_false)
    if keep_nan and da_true:
        new = new.where(if_true.notnull())

    return new


def replace(da: xr.DataArray, to_replace: Any, value: Any) -> xr.DataArray:
    """
    Replace values given in `to_replace` by `value`.

    Parameters
    ----------
    da: xr.DataArray
    to_replace: scalar or 1D array like
        Which values to replace. If to_replace and value are both array like,
        they must be the same length.
    value: scalar or 1D array like
        Value to replace any values matching `to_replace` with.

    Returns
    -------
    xr.DataArray
        DataArray after replacement.

    Examples
    --------

    Replace values of 1.0 by 10.0, and 2.0 by 20.0:

    >>> da = xr.DataArray([0.0, 1.0, 1.0, 2.0, 2.0])
    >>> replaced = imod.util.replace(da, to_replace=[1.0, 2.0], value=[10.0, 20.0])

    """
    from xarray.core.utils import is_scalar

    def _replace(
        a: np.ndarray, to_replace: np.ndarray, value: np.ndarray
    ) -> np.ndarray:
        flat = da.values.ravel()

        sorter = np.argsort(to_replace)
        insertion = np.searchsorted(to_replace, flat, sorter=sorter)
        indices = np.take(sorter, insertion, mode="clip")
        replaceable = to_replace[indices] == flat

        out = flat.copy()
        out[replaceable] = value[indices[replaceable]]
        return out.reshape(a.shape)

    if is_scalar(to_replace):
        if not is_scalar(value):
            raise TypeError("if to_replace is scalar, then value must be a scalar")
        if np.isnan(to_replace):
            return da.fillna(value)
        else:
            return da.where(da != to_replace, other=value)
    else:
        to_replace = np.asarray(to_replace)
        if to_replace.ndim != 1:
            raise ValueError("to_replace must be 1D or scalar")
        if is_scalar(value):
            value = np.full_like(to_replace, value)
        else:
            value = np.asarray(value)
            if to_replace.shape != value.shape:
                raise ValueError(
                    f"Replacement arrays must match in shape. "
                    f"Expecting {to_replace.shape} got {value.shape} "
                )

    _, counts = np.unique(to_replace, return_counts=True)
    if (counts > 1).any():
        raise ValueError("to_replace contains duplicates")

    isnan = np.isnan(to_replace)
    if isnan.any():
        i = np.nonzero(isnan)[0]
        da = da.fillna(value[i])

    return xr.apply_ufunc(
        _replace,
        da,
        kwargs={"to_replace": to_replace, "value": value},
        dask="parallelized",
        output_dtypes=[da.dtype],
    )


def values_within_range(
    da: xr.DataArray, min: Optional[float] = None, max: Optional[float] = None
) -> xr.DataArray | bool:
    """
    Find which values are within range.
    Function checks which values are unaffected by the clip method, to
    be able to deal with min and max values equal to None, which
    should be ignored.
    If both min and max are True, returns scalar True.

    Parameters
    ----------
    da: xr.DataArray
        DataArray to be checked
    min: float
        Minimum value, if None no minimum value is set
    max: float
        Maximum value, if None no maximum value is set

    Returns
    -------
    {bool, xr.DataArray}
        Boolean array with values which are within range as True.
    """
    if min is None and max is None:
        return True
    else:
        return da == da.clip(min=min, max=max)

