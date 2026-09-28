from typing import Optional, Union, cast

import numpy as np
import pandas as pd
import xarray as xr

from imod.common.interfaces.imodel import IModel
from imod.common.interfaces.iregridpackage import IRegridPackage
from imod.common.utilities.dataclass_type import DataclassType
from imod.common.utilities.regrid import (
    _regrid_package_data,
    regrid_imod5_cap_and_bnd_data,
)
from imod.mf6.package import Package
from imod.mf6.regrid.regrid_schemes import ConstantHeadRegridMethod
from imod.mf6.utilities.mask import mask_topsystem
from imod.typing import GridDataArray, GridDataDict, Imod5DataDict
from imod.typing.grid import full_like
from imod.util.regrid import RegridderWeightsCache


def convert_ibound_to_idomain(
    ibound: xr.DataArray, thickness: xr.DataArray
) -> xr.DataArray:
    """
    Convert IBOUND array to IDOMAIN array. IBOUND -1 will be set to 1 in
    IDOMAIN. When the thickness is <= 0, IDOMAIN will be set to -1.

    Parameters
    ----------
    ibound : xr.DataArray
        The IBOUND array from iMOD5.
    thickness : xr.DataArray
        The thickness array of the model layers.

    Returns
    -------
    xr.DataArray
        The corresponding IDOMAIN array.
    """
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
    df_points = cap_data["artificial_recharge_layer"]
    data = {}
    # Order of columns is x, y, layer, the other columns are irrelevant here.
    data["x"] = df_points.iloc[:, 0].to_numpy().astype(float)
    data["y"] = df_points.iloc[:, 1].to_numpy().astype(float)
    data["layer"] = df_points.iloc[:, 2].to_numpy().astype(int)
    data["rate"] = np.zeros_like(data["x"], dtype=float)
    data["id"] = df_points.index.to_numpy()

    return data


def _well_from_imod5_cap_grid_data(cap_data: GridDataDict) -> dict[str, np.ndarray]:
    artificial_rch_type = cap_data["artificial_recharge"]
    layer = cap_data["artificial_recharge_layer"].astype(int)
    # Workaround to be able to later specify groundwater abstraction rates as
    # well as surface water abstraction rates in a single row, which is a
    # requirement of MetaSWAP. If artificial_rch_type == 1 (groundwater
    # abstraction), the surface water abstraction wells are set to 0.0 later.
    from_groundwater = (artificial_rch_type != 0).to_numpy()
    coords = artificial_rch_type.coords
    x_grid, y_grid = np.meshgrid(coords["x"].to_numpy(), coords["y"].to_numpy())

    data = {}
    data["layer"] = layer.data[from_groundwater]
    data["y"] = y_grid[from_groundwater]
    data["x"] = x_grid[from_groundwater]
    data["rate"] = np.zeros_like(data["x"])

    return data


def well_from_imod5_cap_data(
    imod5_data: Imod5DataDict,
    target_dis: Optional[IRegridPackage],
    regridder_types: DataclassType,
    regrid_cache: RegridderWeightsCache,
) -> dict[str, np.ndarray]:
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

    Parameters
    ----------
    imod5_data : Imod5DataDict
        The iMOD5 data containing the "cap" package with abstraction
        information.
    target_dis : Optional[IRegridPackage]
        The target discretization package for regridding the data. Required if
        the data is in grid format (IDF).
    regridder_types : DataclassType
        The regrid methods to use for regridding the data.
    regrid_cache : RegridderWeightsCache
        Cache for storing regridder weights to speed up repeated regridding
        operations.

    Returns
    -------
    dict[str, np.ndarray]
        A dictionary containing well information extracted from the iMOD5 cap
        data.
    """
    cap_data = imod5_data["cap"]
    has_ipf_well = isinstance(cap_data["artificial_recharge_layer"], pd.DataFrame)

    if has_ipf_well:
        return _well_from_imod5_cap_point_data(cap_data)
    else:
        if target_dis is None:
            raise ValueError(
                "target_dis must be provided when converting iMOD5 cap data "
                "from grids (IDF)"
            )
        cap_data_regridded = regrid_imod5_cap_and_bnd_data(
            imod5_data, target_dis, regridder_types, regrid_cache
        )["cap"]
        return _well_from_imod5_cap_grid_data(cap_data_regridded)


def regrid_imod5_pkg_data(
    pkg_type: Optional[type[Package]],
    imod5_pkg_data: GridDataDict,
    target_dis: Package,
    regridder_types: Optional[DataclassType],
    regrid_cache: RegridderWeightsCache,
) -> GridDataDict:
    """
    Regrid iMOD5 package data to target idomain. Optionally get regrid methods
    from class if not provided.

    Parameters
    ----------
    pkg_type:
        The type of the package being regridded. This is used to determine the
        appropriate regrid methods if regridder_types is not provided.
    imod5_pkg_data:
        The iMOD5 package data to be regridded.
    target_dis:
        The target discretization package containing the idomain to regrid to.
    regridder_types:
        Optional regrid methods to use for regridding. If not provided, they
        will be obtained from the pkg_type.
    regrid_cache:
        Cache for storing regridder weights to speed up repeated regridding
        operations.

    Returns
    -------
    GridDataDict
        The regridded iMOD5 package data.
    """
    if (pkg_type is None) and (regridder_types is None):
        raise ValueError(
            "Either pkg_type or regridder_types must be provided for regridding."
        )
    # set up regridder methods
    elif (pkg_type is not None) and (
        regridder_types is None
    ):  # check pkg_type not None for mypy
        regridder_types = pkg_type.get_regrid_methods()
    # For mypy to succeed
    regridder_types = cast(DataclassType, regridder_types)

    target_idomain = target_dis.dataset["idomain"]

    # regrid the input data
    regridded_pkg_data = _regrid_package_data(
        imod5_pkg_data, target_idomain, regridder_types, regrid_cache, {}
    )
    return regridded_pkg_data


def chd_cells_from_imod5_data(
    imod5_pkg_data: GridDataDict, target_idomain: GridDataArray
) -> GridDataDict:
    """
    Get CHD cells from iMOD5 package data based on IBOUND and target idomain.

    Parameters
    ----------
    imod5_pkg_data:
        The iMOD5 package data containing "head" and "ibound".
    target_idomain:
        The target idomain to filter active cells.

    Returns
    -------
    GridDataDict
        The filtered CHD cells with "head" values where IBOUND < 0 and target
        idomain > 0.
    """
    head = imod5_pkg_data["head"]
    ibound = imod5_pkg_data["ibound"]

    # select locations where ibound < 0
    head = head.where(ibound < 0)

    # select locations where idomain > 0
    head = head.where(target_idomain > 0)

    return {"head": head}


def mask_topsystem_packages_with_ibound(
    imod5_data: dict[str, dict[str, GridDataArray]],
    model: IModel,
    regridder_types: Optional[ConstantHeadRegridMethod],
    regrid_cache: RegridderWeightsCache,
    ignore_time_purge_empty: bool,
) -> None:
    """
    Mask all top system packages where IBOUND < 0. These locations are assigned
    a constant head.

    Parameters
    ----------
    imod5_data:
        The iMOD5 data containing the "bnd" package with "ibound".
    model:
        The target MODFLOW 6 model.
    regridder_types:
        Optional regrid methods to use for regridding. If not provided, default
        methods will be used.
    regrid_cache:
        Cache for storing regridder weights to speed up repeated regridding
        operations.
    ignore_time_purge_empty:
        Flag indicating whether to ignore time when purging empty cells.

    Returns
    -------
    None
        The function modifies the top system packages in the model in place.
    """

    if regridder_types is None:
        regridder_types = ConstantHeadRegridMethod()

    ibound = imod5_data["bnd"]["ibound"]
    regridded_ibound = regrid_imod5_pkg_data(
        pkg_type=None,
        imod5_pkg_data={"ibound": ibound},
        target_dis=model["dis"],
        regridder_types=regridder_types,
        regrid_cache=regrid_cache,
    )["ibound"]
    is_active = regridded_ibound >= 0

    mask_topsystem(model, is_active, ignore_time_purge_empty)
