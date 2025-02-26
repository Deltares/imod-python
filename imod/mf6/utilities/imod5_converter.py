from typing import Optional, Union, cast

import numpy as np
import pandas as pd
import xarray as xr

from imod.common.utilities.regrid import RegridderWeightsCache, _regrid_package_data
from imod.mf6.package import Package
from imod.typing import GridDataDict, Imod5DataDict
from imod.typing.grid import full_like
from imod.util.dims import drop_layer_dim_cap_data
from imod.util.regrid_method_type import RegridMethodType


def convert_ibound_to_idomain(
    ibound: xr.DataArray, thickness: xr.DataArray
) -> xr.DataArray:
    # Convert IBOUND to IDOMAIN
    # -1 to 1, these will have to be filled with
    # CHD cells.
    idomain = np.abs(ibound)

    # Thickness <= 0 -> IDOMAIN = -1
    active_and_zero_thickness = (thickness <= 0) & (idomain > 0)
    # Don't make cells at top or bottom vpt, these should be inactive.
    # First, set all potential vpts to nan to be able to utilize ffill and bfill
    idomain_float = idomain.where(~active_and_zero_thickness)  # type: ignore[attr-defined]
    passthrough = (idomain_float.ffill("layer") > 0) & (
        idomain_float.bfill("layer") > 0
    )
    # Then fill nans where vertical passthrough with -1
    idomain_float = idomain_float.combine_first(
        full_like(idomain_float, -1.0, dtype=float).where(passthrough)
    )
    # Fill the remaining nans at tops and bottoms with 0
    return idomain_float.fillna(0).astype(int)


def convert_unit_rch_rate(rate: xr.DataArray) -> xr.DataArray:
    """Convert recharge from iMOD5's mm/d to m/d"""
    mm_to_m_conversion = 1e-3
    return rate * mm_to_m_conversion


def fill_missing_layers(
    source: xr.DataArray, full: xr.DataArray, fillvalue: Union[float | int]
) -> xr.DataArray:
    """
    This function takes a source grid in which the layer dimension is
    incomplete. It creates a result-grid which has the same layers as the "full"
    grid, which is assumed to have all layers. The result has the values in the
    source for the layers that are in the source. For the other layers, the
    fillvalue is assigned.
    """
    layer = full.coords["layer"]
    return source.reindex(layer=layer, fill_value=fillvalue)


def _well_from_imod5_cap_point_data(cap_data: GridDataDict) -> dict[str, np.ndarray]:
    raise NotImplementedError(
        "Assigning sprinkling wells with an IPF file is not supported, please specify them as IDF."
    )


def _well_from_imod5_cap_grid_data(cap_data: GridDataDict) -> dict[str, np.ndarray]:
    artificial_rch_type = cap_data["artificial_recharge"]
    layer = cap_data["artificial_recharge_layer"].astype(int)

    from_groundwater = (artificial_rch_type != 0).to_numpy()
    coords = artificial_rch_type.coords
    x_grid, y_grid = np.meshgrid(coords["x"].to_numpy(), coords["y"].to_numpy())

    data = {}
    data["layer"] = layer.data[from_groundwater]
    data["y"] = y_grid[from_groundwater]
    data["x"] = x_grid[from_groundwater]
    data["rate"] = np.zeros_like(data["x"])

    return data


def well_from_imod5_cap_data(imod5_data: Imod5DataDict) -> dict[str, np.ndarray]:
    """
    Abstraction data for sprinkling is defined in iMOD5 either with grids (IDF)
    or points (IPF) combined with a grid. Depending on the type, the function does
    different conversions

    - grids (IDF)
        The ``"artifical_recharge_layer"`` variable was defined as grid
        (IDF), this grid defines in which layer a groundwater abstraction
        well should be placed. The ``"artificial_recharge"`` grid contains
        types which point to the type of abstraction:
            * 0: no abstraction
            * 1: groundwater abstraction
            * 2: surfacewater abstraction
        The ``"artificial_recharge_capacity"`` grid/constant defines the
        capacity of each groundwater or surfacewater abstraction. This is an
        ``1:1`` mapping: Each grid cell maps to a separate well.

    - points with grid (IPF & IDF)
        The ``"artifical_recharge_layer"`` variable was defined as point
        data (IPF), this table contains wellids with an abstraction capacity
        and layer. The ``"artificial_recharge"`` grid contains a mapping of
        grid cells to wellids in the point data. The
        ``"artificial_recharge_capacity"`` is ignored as the abstraction
        capacity is already defined in the point data. This is an ``n:1``
        mapping: multiple grid cells can map to one well.
    """
    cap_data = cast(GridDataDict, drop_layer_dim_cap_data(imod5_data)["cap"])

    if isinstance(cap_data["artificial_recharge_layer"], pd.DataFrame):
        return _well_from_imod5_cap_point_data(cap_data)
    else:
        return _well_from_imod5_cap_grid_data(cap_data)


def regrid_imod5_pkg_data(
    cls: type[Package],
    imod5_pkg_data: GridDataDict,
    target_dis: Package,
    regridder_types: Optional[RegridMethodType],
    regrid_cache: RegridderWeightsCache,
) -> GridDataDict:
    """
    Regrid iMOD5 package data to target idomain. Optionally get regrid methods
    from class if not provided.
    """
    target_idomain = target_dis.dataset["idomain"]

    # set up regridder methods
    if regridder_types is None:
        regridder_types = cls.get_regrid_methods()
    # regrid the input data
    regridded_pkg_data = _regrid_package_data(
        imod5_pkg_data, target_idomain, regridder_types, regrid_cache, {}
    )
    return regridded_pkg_data
