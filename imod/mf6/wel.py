from __future__ import annotations

import abc
import itertools
import textwrap
from collections.abc import Iterable
from datetime import datetime
from typing import Any, Callable, Optional, Self, Sequence, Tuple, Union, cast

import cftime
import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
import xugrid as xu

import imod
from imod.common.interfaces.ipointdatapackage import IPointDataPackage
from imod.common.utilities.grid import broadcast_to_full_domain
from imod.common.utilities.layer import create_layered_top
from imod.common.utilities.schemata import validation_pkg_error_message
from imod.logging import init_log_decorator, logger
from imod.logging.logging_decorators import standard_log_decorator
from imod.logging.loglevel import LogLevel
from imod.mf6 import StructuredDiscretization, VerticesDiscretization
from imod.mf6.boundary_condition import (
    BoundaryCondition,
)
from imod.mf6.mf6_wel_adapter import Mf6Wel, concat_indices_to_cellid
from imod.mf6.package import Package
from imod.mf6.utilities.dataset import remove_inactive
from imod.mf6.utilities.imod5_converter import well_from_imod5_cap_data
from imod.mf6.validation_settings import ValidationSettings
from imod.mf6.write_context import WriteContext
from imod.prepare import assign_wells
from imod.prepare.cleanup import cleanup_wel, cleanup_wel_layered
from imod.schemata import (
    AllValueSchema,
    AnyNoDataSchema,
    DTypeSchema,
    EmptyIndexesSchema,
    ValidationError,
)
from imod.select.points import points_indices, points_values
from imod.typing import GridDataArray, Imod5DataDict, StressPeriodTimesType
from imod.typing.grid import is_spatial_grid, ones_like
from imod.util.expand_repetitions import average_timeseries, resample_timeseries
from imod.util.structured import values_within_range

ABSTRACT_METH_ERROR_MSG = "Method in abstract base class called"


def _assign_dims(arg: Any) -> Tuple | xr.DataArray:
    is_da = isinstance(arg, xr.DataArray)
    if is_da and "time" in arg.coords:
        if arg.ndim != 2:
            raise ValueError("time varying variable: must be 2d")
        if arg.dims[0] != "time":
            arg = arg.transpose()
        da = xr.DataArray(
            data=arg.values, coords={"time": arg["time"]}, dims=["time", "index"]
        )
        return da
    elif is_da:
        return "index", arg.values
    else:
        return "index", arg


def mask_2D(package: GridAgnosticWell, domain_2d: GridDataArray) -> GridAgnosticWell:
    point_active = points_values(domain_2d, x=package.x, y=package.y)

    is_inside_exterior = point_active > 0
    selection = package.dataset.loc[{"index": is_inside_exterior}]

    cls = type(package)
    return cls._from_dataset(selection)

def _move_item_to_index(
    lst: list[str], item: str, index: int
) -> list[str]:
    """Move item in list to specified index"""
    lst_copy = lst.copy()
    lst_copy.remove(item)
    lst_copy.insert(index, item)
    return lst_copy

def _df_groups_to_da_rates(
    unique_well_groups: Sequence[pd.api.typing.DataFrameGroupBy],
) -> xr.DataArray:
    # Convert dataframes all groups to DataArrays
    columns = list(unique_well_groups[0].columns)
    columns.remove("rate")
    # Enforce index to the front, to ensure gb_and_summed is correctly sorted by
    # index first, instead of y, x coords.
    columns = _move_item_to_index(columns, "index", 0)
    is_transient = "time" in columns
    # Move time to front if present
    if is_transient:
        columns =_move_item_to_index(columns, "time", 0)
        index_names = ["time", "index"]
    else:
        index_names = ["index"]

    gb_and_summed = pd.concat(unique_well_groups).groupby(columns).sum()
    # Unset multi-index, then set index to index_names
    df_temp = gb_and_summed.reset_index().set_index(index_names)
    da_rate = df_temp["rate"].to_xarray()
    # For safety: if index is still unordered, sort it.
    if not da_rate.indexes["index"].is_monotonic_increasing:
        da_rate = da_rate.sortby("index")
    return da_rate


def _prepare_well_rates_from_groups(
    pkg_data: dict,
    unique_well_groups: Sequence[pd.api.typing.DataFrameGroupBy],
    start_times: StressPeriodTimesType,
) -> xr.DataArray:
    """
    Prepare well rates from dataframe groups, grouped by unique well locations.
    Resample timeseries if ipf with associated text files.
    """
    has_associated = pkg_data["has_associated"]
    if has_associated:
        # Resample times per group
        unique_well_groups = [
            _process_timeseries(df_group, start_times)
            for df_group in unique_well_groups
        ]
    return _df_groups_to_da_rates(unique_well_groups)


def _process_timeseries(
    df_group: pd.api.typing.DataFrameGroupBy, start_times: StressPeriodTimesType
):
    if _is_steady_state(start_times):
        return average_timeseries(df_group)
    else:
        return resample_timeseries(df_group, start_times)


def _prepare_df_ipf_associated(
    pkg_data: dict, all_well_times: list[datetime]
) -> pd.DataFrame:
    """Prepare dataframe for an ipf with associated timeseries in a textfile."""
    # Validate if associated wells are assigned multiple layers, factors,
    # and additions.
    for entry in ["layer", "factor", "addition"]:
        uniques = set(pkg_data[entry])
        if len(uniques) > 1:
            raise ValueError(
                f"IPF with associated textfiles assigned multiple {entry}s: {uniques}"
            )
    # Validate if associated wells are defined only on first timestep or all
    # timesteps
    is_defined_all = len(set(all_well_times) - set(pkg_data["time"])) == 0
    is_defined_first = (len(pkg_data["time"]) == 1) & (
        pkg_data["time"][0] == all_well_times[0]
    )
    if not is_defined_all and not is_defined_first:
        raise ValueError(
            "IPF with associated textfiles assigned to wrong times. "
            "Should be assigned to all times or only first time. "
            f"PRJ times: {all_well_times}, package times: {pkg_data['time']}"
        )
    df = pkg_data["dataframe"][0]
    df["layer"] = pkg_data["layer"][0]
    return df


def _prepare_df_ipf_unassociated(
    pkg_data: dict, start_times: StressPeriodTimesType
) -> pd.DataFrame:
    """Prepare dataframe for an ipf with no associated timeseries."""
    is_steady_state = any(t is None for t in pkg_data["time"])
    if is_steady_state:
        index_dicts = [{"layer": lay} for lay in pkg_data["layer"]]
    else:
        index_dicts = [
            {"time": t, "layer": lay}
            for t, lay in zip(pkg_data["time"], pkg_data["layer"])
        ]
    # Concatenate dataframes, assign layer and times
    iter_dfs_dims = zip(pkg_data["dataframe"], index_dicts)
    df = pd.concat([df.assign(**index_dict) for df, index_dict in iter_dfs_dims])
    # Prepare multi-index dataframe to convert to a multi-dimensional DataArray
    # later.
    dimnames = list(index_dicts[0].keys())
    df_multi = df.set_index(dimnames + [df.index])
    df_multi.index = df_multi.index.set_names(dimnames + ["ipf_row"])
    # Temporarily convert to DataArray with 2 dimensions, as it allows for
    # multi-dimensional ffilling, instead pandas' ffilling the last value in a
    # column of the flattened table.
    ipf_row_index = pkg_data["dataframe"][0].index
    # Forward fill location columns, only reindex layer, filt_top and filt_bot
    # if present.
    cols_ffill_if_present = {"x", "y", "filt_top", "filt_bot"}
    cols_ffill = cols_ffill_if_present & set(df.columns)
    da_multi = df_multi.to_xarray()
    indexers = {"ipf_row": ipf_row_index}
    if not is_steady_state:
        if start_times == "steady-state":
            raise ValueError(
                "``start_times`` cannot be 'steady-state' for transient wells without associated timeseries."
            )
        indexers["time"] = start_times
    # Multi-dimensional reindex, forward fill well locations, fill well rates
    # with 0.0.
    df_ffilled = da_multi[cols_ffill].reindex(indexers, method="ffill").to_dataframe()
    df_fill_zero = da_multi["rate"].reindex(indexers, fill_value=0.0).to_dataframe()
    # Combine columns and reset dataframe back into a simple long table with
    # single index.
    df_out = pd.concat([df_ffilled, df_fill_zero], axis="columns")
    return df_out.reset_index().drop(columns="ipf_row")


def _unpack_package_data(
    pkg_data: dict, start_times: StressPeriodTimesType, all_well_times: list[datetime]
) -> pd.DataFrame:
    """Unpack package data to dataframe"""
    has_associated = pkg_data["has_associated"]
    if has_associated:
        return _prepare_df_ipf_associated(pkg_data, all_well_times)
    else:
        return _prepare_df_ipf_unassociated(pkg_data, start_times)


def get_all_imod5_prj_well_times(imod5_data: dict) -> list[datetime]:
    """Get all times a well data is defined on in a prj file"""
    wel_keys = [key for key in imod5_data.keys() if key.startswith("wel")]
    wel_times_per_pkg = [imod5_data[wel_key]["time"] for wel_key in wel_keys]
    # Flatten list
    wel_times_flat = itertools.chain.from_iterable(wel_times_per_pkg)
    # Get unique times by converting to set and sorting. ``sorted`` also
    # transforms set to a list again.
    return sorted(set(wel_times_flat))


def derive_cellid_from_points(
    dst_grid: GridDataArray,
    x: list,
    y: list,
    layer: list,
) -> GridDataArray:
    """
    Create DataArray with Modflow6 cell identifiers based on x, y coordinates
    in a dataframe. For structured grid this DataArray contains 3 columns:
    ``layer, row, column``. For unstructured grids, this contains 2 columns:
    ``layer, cell2d``.
    See also: https://water.usgs.gov/water-resources/software/MODFLOW-6/mf6io_6.4.0.pdf#page=35

    Note
    ----
    The "layer" coordinate should already be provided in the dataframe.
    To determine the layer coordinate based on screen depts, look at
    :func:`imod.prepare.wells.assign_wells`.

    Parameters
    ----------
    dst_grid: {xr.DataArray, xu.UgridDataArray}
        Destination grid to map the points to based on their x and y coordinates.
    x: {list, np.array}
        array-like with x-coordinates
    y: {list, np.array}
        array-like with y-coordinates
    layer: {list, np.array}
        array-like with layer-coordinates

    Returns
    -------
    cellid : xr.DataArray
        2D DataArray with a ``ncellid`` rows and 3 to 2 columns, depending
        on whether on a structured or unstructured grid."""

    # Find indices belonging to x, y coordinates
    indices_cell2d = points_indices(dst_grid, out_of_bounds="ignore", x=x, y=y)
    # Convert cell2d indices from 0-based to 1-based.
    indices_cell2d = {dim: index + 1 for dim, index in indices_cell2d.items()}
    # Prepare layer indices, for later concatenation

    if isinstance(dst_grid, xu.UgridDataArray):
        indices_layer = xr.DataArray(
            layer, coords=indices_cell2d["mesh2d_nFaces"].coords
        )
        face_dim = dst_grid.ugrid.grid.face_dimension
        indices_cell2d_dims = [face_dim]
        cell2d_coords = ["cell2d"]
    else:
        indices_layer = xr.DataArray(layer, coords=indices_cell2d["x"].coords)
        indices_cell2d_dims = ["y", "x"]
        cell2d_coords = ["row", "column"]

    # Prepare cellid array of the right shape.
    cellid_ls = [indices_layer] + [indices_cell2d[dim] for dim in indices_cell2d_dims]
    dim_cellid_coords = ["layer"] + cell2d_coords
    cellid = concat_indices_to_cellid(cellid_ls, dim_cellid_coords)
    # Assign extra coordinate names.
    xy_coords = {
        "x": ("ncellid", x),
        "y": ("ncellid", y),
    }
    cellid = cellid.assign_coords(coords=xy_coords)

    return cellid.astype(int)


def _is_steady_state(times: StressPeriodTimesType) -> bool:
    # Shortcut when not string, to avoid ambigious bitwise "and" operation when
    # its not.
    return isinstance(times, str) and times == "steady-state"


def _is_iterable_of_datetimes(times: StressPeriodTimesType) -> bool:
    return (
        isinstance(times, Iterable)
        and (len(times) > 0)
        and isinstance(times[0], (datetime, np.datetime64, pd.Timestamp))
    )


def _get_starttimes(
    times: StressPeriodTimesType,
) -> StressPeriodTimesType:
    if _is_steady_state(times):
        return times
    elif _is_iterable_of_datetimes(times):
        return cast(list[datetime], times[:-1])
    else:
        raise ValueError(
            "Only 'steady-state' or a list of datetimes are supported for ``times``."
        )


class GridAgnosticWell(BoundaryCondition, IPointDataPackage, abc.ABC):
    """
    Abstract base class for grid agnostic wells
    """

    _imod5_depth_colnames: list[str] = []
    _depth_colnames: list[tuple[str, type]] = []

    @property
    def x(self) -> npt.NDArray[np.float64]:
        return self.dataset["x"].values

    @property
    def y(self) -> npt.NDArray[np.float64]:
        return self.dataset["y"].values

    @classmethod
    def _is_grid_agnostic_package(cls) -> bool:
        """
        Returns True if this package does not depend on a grid, e.g. the
        :class:`imod.mf6.wel.Wel` package.
        """
        return True

    def _create_cellid(
        self, assigned_wells: pd.DataFrame, active: xr.DataArray
    ) -> GridDataArray:
        like = ones_like(active)

        # Groupby index and select first, to unset any duplicate records
        # introduced by the multi-indexed "time" dimension.
        unique_assigned_wells = assigned_wells.groupby("index").first()
        d_for_cellid = unique_assigned_wells[["x", "y", "layer"]].to_dict("list")

        return derive_cellid_from_points(like, **d_for_cellid)

    def _create_dataset_vars(
        self, assigned_wells: pd.DataFrame, cellid: xr.DataArray
    ) -> xr.Dataset:
        """
        Create dataset with all variables (rate, concentration), with a similar shape as the cellids.
        """
        data_vars = ["id", "rate"]
        if "concentration" in assigned_wells.columns:
            data_vars.append("concentration")

        ds_vars = assigned_wells[data_vars].to_xarray()
        # "rate" variable in conversion from multi-indexed DataFrame to xarray
        # DataArray results in duplicated values for "rate" along dimension
        # "species". Select first species to reduce this again.
        index_names = assigned_wells.index.names
        if "species" in index_names:
            ds_vars["rate"] = ds_vars["rate"].isel(species=0)

        # Carefully rename the dimension and set coordinates
        d_rename = {"index": "ncellid"}
        ds_vars = ds_vars.rename_dims(**d_rename).rename_vars(**d_rename)
        ds_vars = ds_vars.assign_coords(**{"ncellid": cellid.coords["ncellid"].values})

        return ds_vars

    def _render(self, directory, pkgname, globaltimes, binary):
        raise NotImplementedError(
            textwrap.dedent(
                f"""{self.__class__.__name__} is a grid-agnostic package and does not
            have a render method. To render the package, first convert to a
            Modflow6 package by calling pkg.to_mf6_pkg()"""
            )
        )

    def _write(
        self,
        pkgname: str,
        globaltimes: Union[list[np.datetime64], np.ndarray],
        write_context: WriteContext,
    ):
        raise NotImplementedError(
            "To write a wel package first convert it to a MF6 well using to_mf6_pkg."
        )

    def mask(self, domain: GridDataArray) -> GridAgnosticWell:
        """
        Mask wells based on two-dimensional domain. For three-dimensional
        masking: Wells falling in inactive cells are automatically removed in
        the call to write to Modflow 6 package. You can verify this by calling
        the ``to_mf6_pkg`` method.
        """

        # Drop layer coordinate if present, otherwise a layer coordinate is assigned
        # which causes conflicts downstream when assigning wells and deriving
        # cellids.
        domain_2d = domain.isel(layer=0, drop=True, missing_dims="ignore").drop_vars(
            "layer", errors="ignore"
        )
        return mask_2D(self, domain_2d)

    def to_mf6_pkg(
        self,
        idomain: GridDataArray,
        top: GridDataArray,
        bottom: GridDataArray,
        k: GridDataArray,
        validate: bool = False,
        strict_validation: bool = True,
    ) -> Mf6Wel:
        """
        Write package to Modflow 6 package.

        Based on the model grid and top and bottoms, cellids are determined.
        When well screens hit multiple layers, groundwater extractions are
        distributed based on layer transmissivities. Wells located in inactive
        cells are removed.

        Note
        ----
        The well distribution based on transmissivities assumes confined
        aquifers. If wells fall dry (and the rate distribution has to be
        recomputed at runtime), it is better to use the Multi-Aquifer Well
        package.

        Parameters
        ----------
        idomain: {xarry.DataArray, xugrid.UgridDataArray}
            Grid with active cells.
        top: {xarry.DataArray, xugrid.UgridDataArray}
            Grid with top of model layers.
        bottom: {xarry.DataArray, xugrid.UgridDataArray}
            Grid with bottom of model layers.
        k: {xarry.DataArray, xugrid.UgridDataArray}
            Grid with hydraulic conductivities.
        validate: bool, default True
            Run validation before converting
        strict_validation: bool, default True
            Set well validation strict:
            Throw error if well is removed entirely during its assignment to
            layers.

        Returns
        -------
        Mf6Wel
            Object with wells as list based input.
        """
        validation_context = ValidationSettings(
            validate=validate, strict_well_validation=strict_validation
        )
        return self._to_mf6_pkg(idomain, top, bottom, k, validation_context)

    def _to_mf6_pkg(
        self,
        idomain: GridDataArray,
        top: GridDataArray,
        bottom: GridDataArray,
        k: GridDataArray,
        validation_context: ValidationSettings,
    ) -> Mf6Wel:
        if validation_context.validate:
            errors = self._validate(self._write_schemata)
            if len(errors) > 0:
                message = validation_pkg_error_message(errors)
                raise ValidationError(message)

        wells_df = self._create_wells_df()
        nwells_df = len(wells_df["id"].unique())
        if nwells_df == 0:
            raise ValidationError(
                "No wells were assigned in package. None were present."
            )

        assigned_wells = self._assign_wells_to_layers(wells_df, idomain, top, bottom, k)
        filtered_assigned_well_ids = self._gather_filtered_well_ids(
            assigned_wells, wells_df
        )
        message_assign = self._to_mf6_package_information(
            filtered_assigned_well_ids, reason_text="permeability/thickness constraints"
        )
        error_on_well_removal = validation_context.strict_well_validation
        if error_on_well_removal and len(filtered_assigned_well_ids) > 0:
            logger.log(loglevel=LogLevel.ERROR, message=message_assign)
            raise ValidationError(message_assign)

        ds = xr.Dataset()
        ds["cellid"] = self._create_cellid(assigned_wells, idomain)

        ds_vars = self._create_dataset_vars(assigned_wells, ds["cellid"])
        ds = ds.assign(**ds_vars.data_vars)  # type: ignore[arg-type]

        ds = remove_inactive(ds, idomain)
        ds["save_flows"] = self["save_flows"].values[()]
        ds["print_flows"] = self["print_flows"].values[()]
        ds["print_input"] = self["print_input"].values[()]

        filtered_final_well_ids = self._gather_filtered_well_ids(ds, wells_df)
        if len(filtered_final_well_ids) > 0:
            reason_text = "inactive cells or permeability/thickness constraints"
            message_end = self._to_mf6_package_information(
                filtered_final_well_ids, reason_text=reason_text
            )
            logger.log(loglevel=LogLevel.WARNING, message=message_end)

        ds = ds.drop_vars("id")

        return Mf6Wel(**ds.data_vars)  # type: ignore[arg-type]

    def _gather_filtered_well_ids(
        self, well_data_filtered: pd.DataFrame | xr.Dataset, well_data: pd.DataFrame
    ) -> list[str]:
        # Work around performance issue with xarray isin for large datasets.
        if isinstance(well_data_filtered, xr.Dataset):
            filtered_ids = well_data_filtered["id"].to_dataframe()["id"]
        else:
            filtered_ids = well_data_filtered["id"]
        is_missing_id = ~well_data["id"].isin(filtered_ids.unique())
        return well_data["id"].loc[is_missing_id].unique()

    def _to_mf6_package_information(
        self, filtered_wells: list[str], reason_text: str
    ) -> str:
        message = textwrap.dedent(
            f"""Some wells were not placed in the MF6 well package. This can be
            due to {reason_text}.\n"""
        )
        if len(filtered_wells) < 10:
            message += "The filtered wells are: \n"
        else:
            message += " The first 10 unplaced wells are: \n"

        is_filtered = self.dataset["id"].compute().isin(filtered_wells)
        for i in range(min(10, len(filtered_wells))):
            ids = filtered_wells[i]
            x = self.dataset["x"].data[is_filtered][i]
            y = self.dataset["y"].data[is_filtered][i]
            message += f" id = {ids} x = {x}  y = {y} \n"
        return message

    def _create_wells_df(self) -> pd.DataFrame:
        raise NotImplementedError(ABSTRACT_METH_ERROR_MSG)

    def _assign_wells_to_layers(
        self,
        wells_df: pd.DataFrame,
        active: GridDataArray,
        top: GridDataArray,
        bottom: GridDataArray,
        k: GridDataArray,
    ) -> pd.DataFrame:
        raise NotImplementedError(ABSTRACT_METH_ERROR_MSG)

    @classmethod
    def _validate_imod5_depth_information(
        cls, key: str, pkg_data: dict, df: pd.DataFrame
    ) -> None:
        raise NotImplementedError(ABSTRACT_METH_ERROR_MSG)

    @classmethod
    def from_imod5_data(
        cls,
        key: str,
        imod5_data: dict[str, dict[str, GridDataArray]],
        times: StressPeriodTimesType,
        minimum_k: float = 0.0,
        minimum_thickness: float = 0.0,
    ) -> "GridAgnosticWell":
        """
        Convert wells to imod5 data, loaded with
        :func:`imod.formats.prj.open_projectfile_data`, to a Well object. As
        iMOD5 handles wells differently than iMOD Python normally does, some
        data transformations are made, which are outlined further.

        iMOD5 stores well information in IPF files and it supports two ways to
        specify injection/extraction rates:

            1. A timeseries of well rates, in an associated text file. We will
               call these "associated wells" further in this text.
            2. Constant rates in an IPF file, without an associated text file.
               We will call these "unassociated wells" further in this text.

        Depending on this, iMOD5 does different things, which we need to mimic
        in this method.

        *Associated wells*

        Wells with timeseries in an associated textfile are processed as
        follows.

        - Wells are validated if the following requirements are met:
            - Associated well entries in projectfile are defined on either all
              timestamps or just the first
            - Multiplication and addition factors need to remain constant through time
            - Same associated well cannot be assigned to multiple layers
        - The dataframe of the first projectfile timestamp is selected
        - Timeseries are processed based on the ``times`` argument of this method:
            - If ``times`` is a list of datetimes, rate timeseries are resampled
              with a time weighted mean to the simulation times. When simulation
              times fall outside well timeseries range, the last rate is forward
              filled.
            - If ``times = "steady-state"``, the simulation is assumed to be
              "steady-state" and an average rate is computed from the
              timeseries.
        - Projectfile timestamps are not used. Even if assigned to a
          "steady-state" timestamp, the resulting dataset still uses simulation
          times.

        *Unassociated wells*

        Wells without associated textfiles are processed as follows:

        - When a unassociated well disappears from the next time entry in the
          projectfile, the well is deactivated by setting its rate to 0.0. This
          is to prevent the well being activated again in case of any potential
          forward filling at a later stage by
          :meth:`imod.mf6.Modflow6Simulation.create_time_discretization`
        - Wells assigned to a "steady-state" entry in the projectfile will have
          no "time" dimension in the resulting dataset.
        - Times beyond the year 2261 are out of bounds for pandas. In associated
          timeseries these are ignored, instead the last stage is forward
          filled.

        .. note::
            In case you are wondering why is this so complicated? There are two
            main reasons:

            - iMOD5 is inconsistent in how it treats timeseries for grid data,
              compared to point data. Whereas grids are forward filled when
              there is no entry specified for a time entry, unassociated wells
              are deactivated. Associated wells, however, are forward filled.
            - Normally there are two levels in which times are defined: The
              simulation times, which are the requested times for the
              simulation, and projectfile times, on which data is defined. With
              associated ipfs, times are defined in three
              levels: There are simulation times (in iMOD5 in the ini
              file), there are projectfile times, and there are times
              defined in the associated textfiles on which data is defined.

        Parameters
        ----------

        key: str
            Name of the well system in the imod5 data
        imod5_data: dict
            iMOD5 data loaded from a projectfile with
            :func:`imod.formats.prj.open_projectfile_data`
        times: list[datetime] | Literal["steady-state"]
            Simulation times, a list of datetimes for transient simulations. Or
            the string ``"steady-state"`` for steady-state simulations.
        minimum_k: float, optional
            On creating point wells, no point wells will be placed in cells with
            a lower horizontal conductivity than this. Wells are placed when
            ``to_mf6_pkg`` is called.
        minimum_thickness: float, optional
            On creating point wells, no point wells will be placed in cells with
            a lower thickness than this. Wells are placed when ``to_mf6_pkg`` is
            called.
        """

        pkg_data = imod5_data[key]
        all_well_times = get_all_imod5_prj_well_times(imod5_data)

        start_times = _get_starttimes(times)  # Starts stress periods.
        df = _unpack_package_data(pkg_data, start_times, all_well_times)
        cls._validate_imod5_depth_information(key, pkg_data, df)

        # Groupby unique wells, to get dataframes per time.
        colnames_group = ["x", "y"] + cls._imod5_depth_colnames
        # Associated wells need additional grouping by id
        if pkg_data["has_associated"]:
            colnames_group.append("id")
        wel_index, well_groups_untagged = zip(*df.groupby(colnames_group))
        # Explictly sign an index to each group, so that the
        # DataArray of rates can be created with a unique index.
        unique_well_groups = [
            group.assign(index=i) for i, group in enumerate(well_groups_untagged)
        ]
        # Unpack wel indices by zipping
        varnames = [("x", float), ("y", float)] + cls._depth_colnames
        index_values = zip(*wel_index)
        cls_input: dict[str, Any] = {
            var: np.array(value, dtype=dtype)
            for (var, dtype), value in zip(varnames, index_values)
        }
        cls_input["rate"] = _prepare_well_rates_from_groups(
            pkg_data, unique_well_groups, start_times
        )
        cls_input["minimum_k"] = minimum_k
        cls_input["minimum_thickness"] = minimum_thickness

        return cls(**cls_input)

    def _cleanup(
        self,
        dis: StructuredDiscretization | VerticesDiscretization,
        cleanup_func: Callable,
        **cleanup_kwargs,
    ) -> None:
        # Work around mypy error, .data_vars cannot be used with xu.UgridDataset
        dict_to_broadcast: dict[str, GridDataArray] = dict(**dis.dataset)  # type: ignore
        # Top and bottom should be forced to grids with a x, y coordinates
        top, bottom = broadcast_to_full_domain(**dict_to_broadcast)
        # Collect point variable datanames
        point_varnames = list(self._write_schemata.keys())
        if "concentration" not in self.dataset.keys():
            point_varnames.remove("concentration")
        point_varnames.append("id")
        # Create dataset with purely point locations
        point_ds = self.dataset[point_varnames]
        # Take first item of irrelevant dimensions
        point_ds = point_ds.isel(time=0, species=0, drop=True, missing_dims="ignore")
        # Cleanup well dataframe
        wells = point_ds.to_dataframe()
        cleaned_wells = cleanup_func(wells, top.isel(layer=0), bottom, **cleanup_kwargs)
        # Select with ids in cleaned dataframe to drop points outside grid.
        well_ids = cleaned_wells.index
        dataset_cleaned = self.dataset.swap_dims({"index": "id"}).sel(id=well_ids)
        # Assign adjusted screen top and bottom
        if "screen_top" in cleaned_wells:
            dataset_cleaned["screen_top"] = cleaned_wells["screen_top"]
        if "screen_bottom" in cleaned_wells:
            dataset_cleaned["screen_bottom"] = cleaned_wells["screen_bottom"]
        # Ensure dtype of id is preserved
        id_type = self.dataset["id"].dtype
        dataset_cleaned = dataset_cleaned.swap_dims({"id": "index"}).reset_coords("id")
        dataset_cleaned["id"] = dataset_cleaned["id"].astype(id_type)
        # Override dataset
        self.dataset = dataset_cleaned


class Well(GridAgnosticWell):
    """
    Agnostic WEL package, which accepts x, y and a top and bottom of the well screens.

    This package can be written to any provided model grid.
    Any number of WEL Packages can be specified for a single groundwater flow model.
    https://water.usgs.gov/water-resources/software/MODFLOW-6/mf6io_6.0.4.pdf#page=63

    Parameters
    ----------

    y: list of floats or np.array of floats
        is the y location of the well.
    x: list of floats or np.array of floats
        is the x location of the well.
    screen_top: list of floats or np.array of floats
        is the top of the well screen.
    screen_bottom: list of floats or np.array of floats
        is the bottom of the well screen.
    rate: list of floats or xr.DataArray
        is the volumetric well rate. A positive value indicates well
        (injection) and a negative value indicates discharge (extraction) (q).
        If provided as DataArray, an ``"index"`` dimension is required and an
        optional ``"time"`` dimension and coordinate specify transient input.
        In the latter case, it is important that dimensions are in the order:
        ``("time", "index")``
    concentration: array of floats (xr.DataArray, optional)
        if this flow package is used in simulations also involving transport, then this array is used
        as the  concentration for inflow over this boundary.
    concentration_boundary_type: ({"AUX", "AUXMIXED"}, optional)
        if this flow package is used in simulations also involving transport, then this keyword specifies
        how outflow over this boundary is computed.
    id: list of Any, optional
        assign an identifier code to each well. if not provided, one will be generated
        Must be convertible to string, and unique entries.
    minimum_k: float, optional
        on allocating wells to the model, no filter segments will be placed in
        cells with a smaller horizontal conductivity than this. Defaults to 0.0.
    minimum_thickness: float, optional
        on allocating wells to the model, no filter segments will be placed in
        cells with a smaller thickness than this. Defaults to 0.0
    print_input: ({True, False}, optional)
        keyword to indicate that the list of well information will be written to
        the listing file immediately after it is read.
        Default is False.
    print_flows: ({True, False}, optional)
        Indicates that the list of well flow rates will be printed to the
        listing file for every stress period time step in which "BUDGET PRINT"
        is specified in Output Control. If there is no Output Control option
        and PRINT FLOWS is specified, then flow rates are printed for the last
        time step of each stress period.
        Default is False.
    save_flows: ({True, False}, optional)
        Indicates that well flow terms will be written to the file specified
        with "BUDGET FILEOUT" in Output Control.
        Default is False.
    observations: [Not yet supported.]
        Default is None.
    validate: {True, False}
        Flag to indicate whether the package should be validated upon
        initialization. This raises a ValidationError if package input is
        provided in the wrong manner. Defaults to True.
    repeat_stress: dict or xr.DataArray of datetimes, optional
        Used to repeat data for e.g. repeating stress periods such as
        seasonality without duplicating the values. If provided as dict, it
        should map new dates to old dates present in the dataset.
        ``{"2001-04-01": "2000-04-01", "2001-10-01": "2000-10-01"}`` if provided
        as DataArray, it should have dimensions ``("repeat", "repeat_items")``.
        The ``repeat_items`` dimension should have size 2: the first value is
        the "key", the second value is the "value". For the "key" datetime, the
        data of the "value" datetime will be used.

    Examples
    ---------

    >>> screen_top = [0.0, 0.0]
    >>> screen_bottom = [-2.0, -2.0]
    >>> y = [83.0, 77.0]
    >>> x = [81.0, 82.0]
    >>> rate = [1.0, 1.0]

    >>> imod.mf6.Well(x, y, screen_top, screen_bottom, rate)

    For a transient well:

    >>> weltimes = pd.date_range("2000-01-01", "2000-01-03")

    >>> rate_factor_time = xr.DataArray([0.5, 1.0], coords={"time": weltimes}, dims=("time",))
    >>> rate_transient = rate_factor_time * xr.DataArray(rate, dims=("index",))

    >>> imod.mf6.Well(x, y, screen_top, screen_bottom, rate_transient)
    """

    _pkg_id = "wel"

    _auxiliary_data = {"concentration": "species"}
    _init_schemata = {
        "screen_top": [DTypeSchema(np.floating)],
        "screen_bottom": [DTypeSchema(np.floating)],
        "y": [DTypeSchema(np.floating)],
        "x": [DTypeSchema(np.floating)],
        "rate": [DTypeSchema(np.floating)],
        "concentration": [DTypeSchema(np.floating)],
    }
    _write_schemata = {
        "screen_top": [AnyNoDataSchema(), EmptyIndexesSchema()],
        "screen_bottom": [
            AnyNoDataSchema(),
            EmptyIndexesSchema(),
            AllValueSchema("<=", "screen_top"),
        ],
        "y": [AnyNoDataSchema(), EmptyIndexesSchema()],
        "x": [AnyNoDataSchema(), EmptyIndexesSchema()],
        "rate": [AnyNoDataSchema(), EmptyIndexesSchema()],
        "concentration": [AnyNoDataSchema(), EmptyIndexesSchema()],
    }

    _imod5_depth_colnames: list[str] = ["filt_top", "filt_bot"]
    _depth_colnames: list[tuple[str, type]] = [
        ("screen_top", float),
        ("screen_bottom", float),
    ]

    @init_log_decorator()
    def __init__(
        self,
        x: np.ndarray | list[float],
        y: np.ndarray | list[float],
        screen_top: np.ndarray | list[float],
        screen_bottom: np.ndarray | list[float],
        rate: list[float] | xr.DataArray,
        concentration: Optional[list[float] | xr.DataArray] = None,
        concentration_boundary_type="aux",
        id: Optional[list[Any]] = None,
        minimum_k: float = 0.0,
        minimum_thickness: float = 0.0,
        print_input: bool = False,
        print_flows: bool = False,
        save_flows: bool = False,
        observations=None,
        validate: bool = True,
        repeat_stress: Optional[xr.DataArray] = None,
    ):
        if id is None:
            id = [str(i) for i in range(len(x))]
        else:
            set_id = set(id)
            if len(id) != len(set_id):
                raise ValueError("id's must be unique")
            id = [str(i) for i in id]
        dict_dataset = {
            "screen_top": _assign_dims(screen_top),
            "screen_bottom": _assign_dims(screen_bottom),
            "y": _assign_dims(y),
            "x": _assign_dims(x),
            "rate": _assign_dims(rate),
            "id": _assign_dims(id),
            "minimum_k": minimum_k,
            "minimum_thickness": minimum_thickness,
            "print_input": print_input,
            "print_flows": print_flows,
            "save_flows": save_flows,
            "observations": observations,
            "repeat_stress": repeat_stress,
            "concentration": concentration,
            "concentration_boundary_type": concentration_boundary_type,
        }
        super().__init__(dict_dataset)
        # Set index as coordinate
        index_coord = np.arange(self.dataset.sizes["index"])
        self.dataset = self.dataset.assign_coords(index=index_coord)
        self._validate_init_schemata(validate)

    def clip_box(
        self,
        time_min: Optional[cftime.datetime | np.datetime64 | str] = None,
        time_max: Optional[cftime.datetime | np.datetime64 | str] = None,
        layer_min: Optional[int] = None,
        layer_max: Optional[int] = None,
        x_min: Optional[float] = None,
        x_max: Optional[float] = None,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
        top: Optional[GridDataArray] = None,
        bottom: Optional[GridDataArray] = None,
    ) -> Self:
        """
        Clip a well package by a bounding box (time, layer, y, x).

        Parameters
        ----------
        time_min: optional, np.datetime64
            Start time to select. Data will be forward filled to this date. If
            time_min is before the start time of the dataset, data is
            backfilled.
        time_max: optional
            End time to select.
        layer_min: optional, int
            Ignored.
        layer_max: optional, int
            Ignored.
        x_min: optional, float
            Minimum x-coordinate to select.
        x_max: optional, float
            Maximum x-coordinate to select.
        y_min: optional, float
            Minimum y-coordinate to select.
        y_max: optional, float
            Maximum y-coordinate to select.
        top: optional, GridDataArray
            Grid of top used to clip well screen tops.
        bottom: optional, GridDataArray
            Grid of bottom used to clip well screen bottoms.

        Returns
        -------
        clipped : Well
            A new package that is clipped to the specified bounding box.

        Examples
        --------
        Slicing intervals may be half-bounded, by providing None:

        To select 500.0 <= x <= 1000.0:

        >>> pkg.clip_box(x_min=500.0, x_max=1000.0)

        To select x <= 1000.0:

        >>> pkg.clip_box(x_max=1000.0)``

        To select x >= 500.0:

        >>> pkg.clip_box(x_min=500.0)

        To select a time interval, you can use datetime64:

        >>> pkg.clip_box(time_min=np.datetime64("2020-01-01"), time_max=np.datetime64("2020-12-31"))

        To clip well screens:

        >>> pkg.clip_box(top=top_grid, bottom=bottom_grid)

        """
        if (layer_max or layer_min) and (top is None or bottom is None):
            raise ValueError(
                "When clipping by layer both the top and bottom should be defined"
            )

        if top is not None:
            # Bug in mypy when using unions in isInstance
            if not isinstance(top, GridDataArray) or "layer" not in top.coords:  # type: ignore
                top = create_layered_top(bottom, top)

        # The super method will select in the time dimension without issues.
        new = super().clip_box(time_min=time_min, time_max=time_max)

        ds = new.dataset

        z_max, in_bounds_z_max = self._find_well_value_at_layer(ds, top, layer_max)
        z_min, in_bounds_z_min = self._find_well_value_at_layer(ds, bottom, layer_min)

        # Prior to the actual clipping of z_max/z_min, replace the dataset in case a
        # spatial selection needs to be done when a spatial grid is present (top/bottom).
        ds = ds.loc[{"index": in_bounds_z_max & in_bounds_z_min}]

        if z_max is not None:
            ds["screen_top"] = ds["screen_top"].clip(None, z_max)
        if z_min is not None:
            ds["screen_bottom"] = ds["screen_bottom"].clip(z_min, None)

        # Initiate array of True with right shape to deal with case no spatial
        # selection needs to be done.
        in_bounds = np.full(ds.sizes["index"], True)
        # Select all variables along "index" dimension
        in_bounds &= values_within_range(ds["x"], x_min, x_max)
        in_bounds &= values_within_range(ds["y"], y_min, y_max)
        in_bounds &= values_within_range(ds["screen_top"], z_min, z_max)
        in_bounds &= values_within_range(ds["screen_bottom"], z_min, z_max)
        # remove wells where the screen bottom and top are the same
        in_bounds &= abs(ds["screen_bottom"] - ds["screen_top"]) > 1e-5
        # Replace dataset with reduced dataset based on booleans
        new.dataset = ds.loc[{"index": in_bounds}]

        return new

    @staticmethod
    def _find_well_value_at_layer(
        well_dataset: xr.Dataset, grid: GridDataArray, layer: Optional[int]
    ):
        value = None if layer is None else grid.isel(layer=layer)

        # if value is a grid select the values at the well locations and drop the dimensions
        if (value is not None) and is_spatial_grid(value):
            value = imod.select.points_values(
                value,
                x=well_dataset["x"].values,
                y=well_dataset["y"].values,
                out_of_bounds="ignore",
            )
            in_bounds = np.full(well_dataset.sizes["index"], False)
            in_bounds[value["index"]] = True
            value = value.drop_vars(lambda x: x.coords)
        else:
            in_bounds = np.full(well_dataset.sizes["index"], True)

        return value, in_bounds

    def _create_wells_df(self) -> pd.DataFrame:
        wells_df = self.dataset.to_dataframe()
        wells_df = wells_df.rename(
            columns={
                "screen_top": "top",
                "screen_bottom": "bottom",
            }
        )

        return wells_df

    @standard_log_decorator()
    def _validate(self, schemata: dict, **kwargs) -> dict[str, list[ValidationError]]:
        kwargs["screen_top"] = self.dataset["screen_top"]
        return Package._validate(self, schemata, **kwargs)

    def _assign_wells_to_layers(
        self,
        wells_df: pd.DataFrame,
        active: GridDataArray,
        top: GridDataArray,
        bottom: GridDataArray,
        k: GridDataArray,
    ) -> pd.DataFrame:
        # Ensure top, bottom & k
        # are broadcasted to 3d grid
        like = ones_like(active.compute())
        bottom = like * bottom.compute()
        top_2d = (like * top.compute()).sel(layer=1)
        top_3d = bottom.shift(layer=1).fillna(top_2d)
        k = like * k

        index_names = wells_df.index.names

        minimum_k = self.dataset["minimum_k"].item()
        minimum_thickness = self.dataset["minimum_thickness"].item()

        # Unset multi-index, because assign_wells cannot deal with
        # multi-indices which is returned by self.dataset.to_dataframe() in
        # case of a "time" and "species" coordinate.
        wells_df = wells_df.reset_index()

        assigned_wells = assign_wells(
            wells_df, top_3d, bottom, k, minimum_thickness, minimum_k, True
        )
        # Set multi-index again
        assigned_wells = assigned_wells.set_index(index_names).sort_index()

        return assigned_wells

    @classmethod
    def _validate_imod5_depth_information(
        cls, key: str, pkg_data: dict, df: pd.DataFrame
    ) -> None:
        if "layer" in pkg_data.keys() and (np.any(np.array(pkg_data["layer"]) != 0)):
            log_msg = textwrap.dedent(
                f"""
                In well {key} a layer was assigned, but this is not
                supported for imod.mf6.Well. Assignment will be done based on
                filter_top and filter_bottom, and the chosen layer
                ({pkg_data["layer"]}) will be ignored. To specify by layer, use
                imod.mf6.LayeredWell.
                """
            )
            logger.log(loglevel=LogLevel.WARNING, message=log_msg, additional_depth=2)

        if "filt_top" not in df.columns or "filt_bot" not in df.columns:
            log_msg = textwrap.dedent(
                f"""
                In well {key} the 'filt_top' and 'filt_bot' columns were
                not both found; this is not supported for import. To specify by
                layer, use imod.mf6.LayeredWell.
                """
            )
            logger.log(loglevel=LogLevel.ERROR, message=log_msg, additional_depth=2)
            raise ValueError(log_msg)

    @standard_log_decorator()
    def cleanup(self, dis: StructuredDiscretization | VerticesDiscretization):
        """
        Clean up package inplace. This method calls
        :func:`imod.prepare.cleanup_wel`, see documentation of that
        function for details on cleanup.

        dis: imod.mf6.StructuredDiscretization | imod.mf6.VerticesDiscretization
            Model discretization package.
        """
        minimum_thickness = float(self.dataset["minimum_thickness"])
        self._cleanup(dis, cleanup_wel, minimum_thickness=minimum_thickness)


class LayeredWell(GridAgnosticWell):
    """
    Agnostic WEL package, which accepts x, y and layers.

    This package can be written to any provided model grid, given that it has
    enough layers. Any number of WEL Packages can be specified for a single
    groundwater flow model.
    https://water.usgs.gov/water-resources/software/MODFLOW-6/mf6io_6.0.4.pdf#page=63

    Parameters
    ----------

    y: list of floats or np.array of floats
        is the y location of the well.
    x: list of floats or np.array of floats
        is the x location of the well.
    layer: list of ints or np.array of ints
        is the layer of the well.
    rate: list of floats or xr.DataArray
        is the volumetric well rate. A positive value indicates well
        (injection) and a negative value indicates discharge (extraction) (q).
        If provided as DataArray, an ``"index"`` dimension is required and an
        optional ``"time"`` dimension and coordinate specify transient input.
        In the latter case, it is important that dimensions are in the order:
        ``("time", "index")``
    concentration: array of floats (xr.DataArray, optional)
        if this flow package is used in simulations also involving transport, then this array is used
        as the  concentration for inflow over this boundary.
    concentration_boundary_type: ({"AUX", "AUXMIXED"}, optional)
        if this flow package is used in simulations also involving transport, then this keyword specifies
        how outflow over this boundary is computed.
    id: list of Any, optional
        assign an identifier code to each well. if not provided, one will be generated
        Must be convertible to string, and unique entries.
    minimum_k: float, optional
        on allocating wells to model cells, no wells will be placed in cells
        with a lower horizontal conductivity than this. Defaults to 0.0.
    minimum_thickness: float, optional
        on allocating wells to model cells, no wells will be placed in cells
        with a lower thickness than this. Defaults to 0.0.
    print_input: ({True, False}, optional)
        keyword to indicate that the list of well information will be written to
        the listing file immediately after it is read.
        Default is False.
    print_flows: ({True, False}, optional)
        Indicates that the list of well flow rates will be printed to the
        listing file for every stress period time step in which "BUDGET PRINT"
        is specified in Output Control. If there is no Output Control option
        and PRINT FLOWS is specified, then flow rates are printed for the last
        time step of each stress period.
        Default is False.
    save_flows: ({True, False}, optional)
        Indicates that well flow terms will be written to the file specified
        with "BUDGET FILEOUT" in Output Control.
        Default is False.
    observations: [Not yet supported.]
        Default is None.
    validate: {True, False}
        Flag to indicate whether the package should be validated upon
        initialization. This raises a ValidationError if package input is
        provided in the wrong manner. Defaults to True.
    repeat_stress: dict or xr.DataArray of datetimes, optional
        Used to repeat data for e.g. repeating stress periods such as
        seasonality without duplicating the values. If provided as dict, it
        should map new dates to old dates present in the dataset.
        ``{"2001-04-01": "2000-04-01", "2001-10-01": "2000-10-01"}`` if provided
        as DataArray, it should have dimensions ``("repeat", "repeat_items")``.
        The ``repeat_items`` dimension should have size 2: the first value is
        the "key", the second value is the "value". For the "key" datetime, the
        data of the "value" datetime will be used.

    Examples
    ---------

    >>> layer = [1, 2]
    >>> y = [83.0, 77.0]
    >>> x = [81.0, 82.0]
    >>> rate = [1.0, 1.0]

    >>> imod.mf6.LayeredWell(x, y, layer, rate)

    For a transient well:

    >>> weltimes = pd.date_range("2000-01-01", "2000-01-03")

    >>> rate_factor_time = xr.DataArray([0.5, 1.0], coords={"time": weltimes}, dims=("time",))
    >>> rate_transient = rate_factor_time * xr.DataArray(rate, dims=("index",))

    >>> imod.mf6.LayeredWell(x, y, layer, rate_transient)
    """

    _pkg_id = "wel"

    _auxiliary_data = {"concentration": "species"}
    _init_schemata = {
        "layer": [DTypeSchema(np.integer)],
        "y": [DTypeSchema(np.floating)],
        "x": [DTypeSchema(np.floating)],
        "rate": [DTypeSchema(np.floating)],
        "concentration": [DTypeSchema(np.floating)],
    }
    _write_schemata = {
        "layer": [AnyNoDataSchema(), EmptyIndexesSchema()],
        "y": [AnyNoDataSchema(), EmptyIndexesSchema()],
        "x": [AnyNoDataSchema(), EmptyIndexesSchema()],
        "rate": [AnyNoDataSchema(), EmptyIndexesSchema()],
        "concentration": [AnyNoDataSchema(), EmptyIndexesSchema()],
    }
    _imod5_depth_colnames: list[str] = ["layer"]
    _depth_colnames: list[tuple[str, type]] = [("layer", int)]

    @init_log_decorator()
    def __init__(
        self,
        x: np.ndarray | list[float],
        y: np.ndarray | list[float],
        layer: np.ndarray | list[int],
        rate: list[float] | xr.DataArray,
        concentration: Optional[list[float] | xr.DataArray] = None,
        concentration_boundary_type="aux",
        id: Optional[list[Any]] = None,
        minimum_k: float = 0.0,
        minimum_thickness: float = 0.0,
        print_input: bool = False,
        print_flows: bool = False,
        save_flows: bool = False,
        observations=None,
        validate: bool = True,
        repeat_stress: Optional[xr.DataArray] = None,
    ):
        if id is None:
            id = [str(i) for i in range(len(x))]
        else:
            set_id = set(id)
            if len(id) != len(set_id):
                raise ValueError("id's must be unique")
            id = [str(i) for i in id]
        dict_dataset = {
            "layer": _assign_dims(layer),
            "y": _assign_dims(y),
            "x": _assign_dims(x),
            "rate": _assign_dims(rate),
            "id": _assign_dims(id),
            "minimum_k": minimum_k,
            "minimum_thickness": minimum_thickness,
            "print_input": print_input,
            "print_flows": print_flows,
            "save_flows": save_flows,
            "observations": observations,
            "repeat_stress": repeat_stress,
            "concentration": concentration,
            "concentration_boundary_type": concentration_boundary_type,
        }
        super().__init__(dict_dataset)
        # Set index as coordinate
        index_coord = np.arange(self.dataset.sizes["index"])
        self.dataset = self.dataset.assign_coords(index=index_coord)
        self._validate_init_schemata(validate)

    def clip_box(
        self,
        time_min: Optional[cftime.datetime | np.datetime64 | str] = None,
        time_max: Optional[cftime.datetime | np.datetime64 | str] = None,
        layer_min: Optional[int] = None,
        layer_max: Optional[int] = None,
        x_min: Optional[float] = None,
        x_max: Optional[float] = None,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
        top: Optional[GridDataArray] = None,
        bottom: Optional[GridDataArray] = None,
    ) -> Self:
        """
        Clip a well package by a bounding box (time, layer, y, x).

        Parameters
        ----------
        time_min: optional, np.datetime64
            Start time to select. Data will be forward filled to this date. If
            time_min is before the start time of the dataset, data is
            backfilled.
        time_max: optional
            End time to select.
        layer_min: optional, int
            Minimum layer to select.
        layer_max: optional, int
            Maximum layer to select.
        x_min: optional, float
            Minimum x-coordinate to select.
        x_max: optional, float
            Maximum x-coordinate to select.
        y_min: optional, float
            Minimum y-coordinate to select.
        y_max: optional, float
            Maximum y-coordinate to select.
        top: optional, GridDataArray
            Ignored.
        bottom: optional, GridDataArray
            Ignored.

        Returns
        -------
        clipped : Well
            A new package that is clipped to the specified bounding box.

        Examples
        --------
        Slicing intervals may be half-bounded, by providing None:

        To select 500.0 <= x <= 1000.0:

        >>> pkg.clip_box(x_min=500.0, x_max=1000.0)

        To select x <= 1000.0:

        >>> pkg.clip_box(x_max=1000.0)``

        To select x >= 500.0:

        >>> pkg.clip_box(x_min=500.0)

        To select a time interval, you can use datetime64:

        >>> pkg.clip_box(time_min=np.datetime64("2020-01-01"), time_max=np.datetime64("2020-12-31"))

        """
        # The super method will select in the time dimension without issues.
        new = super().clip_box(time_min=time_min, time_max=time_max)

        ds = new.dataset

        # Initiate array of True with right shape to deal with case no spatial
        # selection needs to be done.
        in_bounds = np.full(ds.sizes["index"], True)
        # Select all variables along "index" dimension
        in_bounds &= values_within_range(ds["x"], x_min, x_max)
        in_bounds &= values_within_range(ds["y"], y_min, y_max)
        in_bounds &= values_within_range(ds["layer"], layer_min, layer_max)
        # Replace dataset with reduced dataset based on booleans
        new.dataset = ds.loc[{"index": in_bounds}]

        return new

    def _create_wells_df(self) -> pd.DataFrame:
        return self.dataset.to_dataframe()

    def _assign_wells_to_layers(
        self,
        wells_df: pd.DataFrame,
        active: GridDataArray,
        top: GridDataArray,
        bottom: GridDataArray,
        k: GridDataArray,
    ) -> pd.DataFrame:
        return wells_df

    @classmethod
    def _validate_imod5_depth_information(
        cls, key: str, pkg_data: dict, df: pd.DataFrame
    ) -> None:
        if np.any(np.array(pkg_data["layer"]) == 0):
            log_msg = textwrap.dedent(
                f"""
                Well {key} in projectfile is assigned to layer 0, but should be >
                0 for LayeredWell             
                """
            )
            logger.log(loglevel=LogLevel.ERROR, message=log_msg, additional_depth=2)
            raise ValueError(log_msg)

        if "layer" not in df.columns:
            log_msg = textwrap.dedent(
                f"""
                IPF file {key} has no layer assigned, but this is required
                for LayeredWell.
                """
            )
            logger.log(loglevel=LogLevel.ERROR, message=log_msg, additional_depth=2)
            raise ValueError(log_msg)

    @classmethod
    def from_imod5_cap_data(cls, imod5_data: Imod5DataDict):
        """
        Create LayeredWell from imod5_data in "cap" package. Abstraction data
        for sprinkling is defined in iMOD5 either with grids (IDF) or points
        (IPF) combined with a grid. Depending on the type, the function does
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
        imod5_data: dict[str, dict[str, GridDataArray]]
            dictionary containing the arrays mentioned in the project file as
            xarray datasets, under the key of the package type to which it
            belongs, as returned by
            :func:`imod.formats.prj.open_projectfile_data`.
        """
        data = well_from_imod5_cap_data(imod5_data)
        return cls(**data)  # type: ignore

    @standard_log_decorator()
    def cleanup(self, dis: StructuredDiscretization | VerticesDiscretization):
        """
        Clean up package inplace. This method calls
        :func:`imod.prepare.cleanup_wel_layered`, see documentation of that
        function for details on cleanup.

        dis: imod.mf6.StructuredDiscretization | imod.mf6.VerticesDiscretization
            Model discretization package.
        """
        self._cleanup(dis, cleanup_wel_layered)
