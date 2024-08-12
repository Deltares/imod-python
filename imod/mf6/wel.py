from __future__ import annotations

import abc
import itertools
import textwrap
import warnings
from datetime import datetime
from typing import Any, Optional, Tuple, Union

import cftime
import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
import xugrid as xu

import imod
from imod.logging import init_log_decorator, logger
from imod.logging.logging_decorators import standard_log_decorator
from imod.logging.loglevel import LogLevel
from imod.mf6.boundary_condition import (
    BoundaryCondition,
    DisStructuredBoundaryCondition,
    DisVerticesBoundaryCondition,
)
from imod.mf6.interfaces.ipointdatapackage import IPointDataPackage
from imod.mf6.mf6_wel_adapter import Mf6Wel
from imod.mf6.package import Package
from imod.mf6.utilities.dataset import remove_inactive
from imod.mf6.validation import validation_pkg_error_message
from imod.mf6.write_context import WriteContext
from imod.prepare import assign_wells
from imod.prepare.layer import create_layered_top
from imod.schemata import (
    AllValueSchema,
    AnyNoDataSchema,
    DTypeSchema,
    EmptyIndexesSchema,
    ValidationError,
)
from imod.select.points import points_indices, points_values
from imod.typing import GridDataArray
from imod.typing.grid import is_spatial_grid, ones_like
from imod.util.expand_repetitions import resample_timeseries
from imod.util.structured import values_within_range


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

    is_inside_exterior = point_active == 1
    selection = package.dataset.loc[{"index": is_inside_exterior}]

    cls = type(package)
    return cls._from_dataset(selection)


def _df_groups_to_da_rates(
    unique_well_groups: pd.api.typing.DataFrameGroupBy,
) -> xr.DataArray:
    # Convert dataframes all groups to DataArrays
    da_groups = [
        xr.DataArray(df_group["rate"], dims=("time"), coords={"time": df_group["time"]})
        for df_group in unique_well_groups
    ]
    # Assign index coordinates
    da_groups = [
        da_group.expand_dims(dim="index").assign_coords(index=[i])
        for i, da_group in enumerate(da_groups)
    ]
    # Concatenate datarrays along index dimension
    return xr.concat(da_groups, dim="index")


def _prepare_well_rates_from_groups(
    pkg_data: dict,
    unique_well_groups: pd.api.typing.DataFrameGroupBy,
    times: list[datetime],
) -> pd.DataFrame:
    """
    Prepare well rates from dataframe groups, grouped by unique well locations.
    Resample timeseries if ipf with associated text files.
    """
    has_associated = pkg_data["has_associated"]
    start_times = times[:-1]  # Starts stress periods.
    if has_associated:
        # Resample times per group
        unique_well_groups = [
            resample_timeseries(df_group, start_times)
            for df_group in unique_well_groups
        ]
    return _df_groups_to_da_rates(unique_well_groups)


def _prepare_df_ipf_associated(
    pkg_data: dict, start_times: list[datetime], all_well_times: list[datetime]
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
    pkg_data: dict, start_times: list[datetime]
) -> pd.DataFrame:
    """Prepare dataframe for an ipf with no associated timeseries."""
    # Concatenate dataframes, assign layer and times
    iter_dfs_dims = zip(pkg_data["dataframe"], pkg_data["time"], pkg_data["layer"])
    df = pd.concat([df.assign(time=t, layer=lay) for df, t, lay in iter_dfs_dims])
    # Prepare multi-index dataframe to convert to a multi-dimensional DataArray
    # later.
    df_multi = df.set_index(["time", df.index])
    df_multi.index = df_multi.index.set_names(["time", "ipf_row"])
    # Temporarily convert to DataArray with 2 dimensions, as it allows for
    # multi-dimensional ffilling, instead pandas' ffilling the last value in a
    # column of the flattened table.
    ipf_row_index = pkg_data["dataframe"][0].index
    cols_ffill = ["x", "y", "layer"]
    da_multi = df_multi.to_xarray()
    indexers = {"time": start_times, "ipf_row": ipf_row_index}
    # Multi-dimensional reindex, forward fill well locations, fill well rates
    # with 0.0.
    df_ffilled = da_multi[cols_ffill].reindex(indexers, method="ffill").to_dataframe()
    df_fill_zero = da_multi["rate"].reindex(indexers, fill_value=0.0).to_dataframe()
    # Combine columns and reset dataframe back into a simple long table with
    # single index.
    df_out = pd.concat([df_ffilled, df_fill_zero], axis="columns")
    return df_out.reset_index().drop(columns="ipf_row")


def _unpack_package_data(pkg_data: dict, times: list[datetime], all_well_times: list[datetime]) -> pd.DataFrame:
    """Unpack package data to dataframe"""
    start_times = times[:-1]  # Starts stress periods.
    has_associated = pkg_data["has_associated"]
    if has_associated:
        return _prepare_df_ipf_associated(pkg_data, start_times, all_well_times)
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
        

class GridAgnosticWell(BoundaryCondition, IPointDataPackage, abc.ABC):
    """
    Abstract base class for grid agnostic wells
    """

    _imod5_depth_colnames = []
    _depth_colnames = []

    @property
    def x(self) -> npt.NDArray[np.float64]:
        return self.dataset["x"].values

    @property
    def y(self) -> npt.NDArray[np.float64]:
        return self.dataset["y"].values

    @classmethod
    def is_grid_agnostic_package(cls) -> bool:
        return True

    def _create_cellid(
        self, wells_assigned: pd.DataFrame, active: xr.DataArray
    ) -> GridDataArray:
        like = ones_like(active)

        # Groupby index and select first, to unset any duplicate records
        # introduced by the multi-indexed "time" dimension.
        df_for_cellid = wells_assigned.groupby("index").first()
        d_for_cellid = df_for_cellid[["x", "y", "layer"]].to_dict("list")

        return self._derive_cellid_from_points(like, **d_for_cellid)

    def _create_dataset_vars(
        self, wells_assigned: pd.DataFrame, cellid: xr.DataArray
    ) -> xr.Dataset:
        """
        Create dataset with all variables (rate, concentration), with a similar shape as the cellids.
        """
        data_vars = ["id", "rate"]
        if "concentration" in wells_assigned.columns:
            data_vars.append("concentration")

        ds_vars = wells_assigned[data_vars].to_xarray()
        # "rate" variable in conversion from multi-indexed DataFrame to xarray
        # DataArray results in duplicated values for "rate" along dimension
        # "species". Select first species to reduce this again.
        index_names = wells_assigned.index.names
        if "species" in index_names:
            ds_vars["rate"] = ds_vars["rate"].isel(species=0)

        # Carefully rename the dimension and set coordinates
        d_rename = {"index": "ncellid"}
        ds_vars = ds_vars.rename_dims(**d_rename).rename_vars(**d_rename)
        ds_vars = ds_vars.assign_coords(**{"ncellid": cellid.coords["ncellid"].values})

        return ds_vars

    @staticmethod
    def _derive_cellid_from_points(
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
        cellid_ls = [indices_layer] + [
            indices_cell2d[dim] for dim in indices_cell2d_dims
        ]
        cellid = xr.concat(cellid_ls, dim="nmax_cellid")
        # Rename generic dimension name "index" to ncellid.
        cellid = cellid.rename(index="ncellid")
        # Put dimensions in right order after concatenation.
        cellid = cellid.transpose("ncellid", "nmax_cellid")
        # Assign extra coordinate names.
        coords = {
            "nmax_cellid": ["layer"] + cell2d_coords,
            "x": ("ncellid", x),
            "y": ("ncellid", y),
        }
        cellid = cellid.assign_coords(**coords)

        return cellid

    def render(self, directory, pkgname, globaltimes, binary):
        raise NotImplementedError(
            f"{self.__class__.__name__} is a grid-agnostic package and does not have a render method. To render the package, first convert to a Modflow6 package by calling pkg.to_mf6_pkg()"
        )

    def write(
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
        active: GridDataArray,
        top: GridDataArray,
        bottom: GridDataArray,
        k: GridDataArray,
        validate: bool = False,
        is_partitioned: bool = False,
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
        active: {xarry.DataArray, xugrid.UgridDataArray}
            Grid with active cells.
        top: {xarry.DataArray, xugrid.UgridDataArray}
            Grid with top of model layers.
        bottom: {xarry.DataArray, xugrid.UgridDataArray}
            Grid with bottom of model layers.
        k: {xarry.DataArray, xugrid.UgridDataArray}
            Grid with hydraulic conductivities.
        validate: bool
            Run validation before converting
        is_partitioned: bool
            Set to true if model has been partitioned

        Returns
        -------
        Mf6Wel
            Object with wells as list based input.
        """
        if validate:
            errors = self._validate(self._write_schemata)
            if len(errors) > 0:
                message = validation_pkg_error_message(errors)
                raise ValidationError(message)

        wells_df = self._create_wells_df()
        nwells_df = len(wells_df["id"].unique())
        wells_assigned = self._assign_wells_to_layers(wells_df, active, top, bottom, k)

        nwells_assigned = (
            0 if wells_assigned.empty else len(wells_assigned["id"].unique())
        )

        if nwells_df == 0:
            raise ValueError("No wells were assigned in package. None were present.")

        if not is_partitioned and nwells_df != nwells_assigned:
            raise ValueError(
                "One or more well(s) are completely invalid due to minimum conductivity and thickness constraints."
            )

        ds = xr.Dataset()
        ds["cellid"] = self._create_cellid(wells_assigned, active)

        ds_vars = self._create_dataset_vars(wells_assigned, ds["cellid"])
        ds = ds.assign(**ds_vars.data_vars)

        ds = remove_inactive(ds, active)
        ds["save_flows"] = self["save_flows"].values[()]
        ds["print_flows"] = self["print_flows"].values[()]
        ds["print_input"] = self["print_input"].values[()]

        filtered_wells = [
            id for id in wells_df["id"].unique() if id not in ds["id"].values
        ]
        if len(filtered_wells) > 0:
            message = self.to_mf6_package_information(filtered_wells)
            logger.log(loglevel=LogLevel.WARNING, message=message, additional_depth=2)

        ds = ds.drop_vars("id")

        return Mf6Wel(**ds.data_vars)

    def to_mf6_package_information(self, filtered_wells: pd.DataFrame) -> str:
        message = textwrap.dedent(
            """Some wells were not placed in the MF6 well package. This 
            can be due to inactive cells or permeability/thickness constraints.\n"""
        )
        if len(filtered_wells) < 10:
            message += "The filtered wells are: \n"
        else:
            message += " The first 10 unplaced wells are: \n"

        for i in range(min(10, len(filtered_wells))):
            ids = filtered_wells[i]
            x = self.dataset["x"][int(filtered_wells[i])].values[()]
            y = self.dataset["y"][int(filtered_wells[i])].values[()]
            message += f" id = {ids} x = {x}  y = {y} \n"
        return message

    def _create_wells_df(self) -> pd.DataFrame:
        raise NotImplementedError("Method in abstract base class called")

    def _assign_wells_to_layers(
        self,
        wells_df: pd.DataFrame,
        active: GridDataArray,
        top: GridDataArray,
        bottom: GridDataArray,
        k: GridDataArray,
    ) -> pd.DataFrame:
        raise NotImplementedError("Method in abstract base class called")

    @classmethod
    def _validate_imod5_depth_information(
        cls, key: str, pkg_data: dict, df: pd.DataFrame
    ) -> None:
        raise NotImplementedError("Method in abstract base class called")

    @classmethod
    def from_imod5_data(
        cls,
        key: str,
        imod5_data: dict[str, dict[str, GridDataArray]],
        times: list[datetime],
        minimum_k: float = 0.1,
        minimum_thickness: float = 1.0,
    ) -> "GridAgnosticWell":
        pkg_data = imod5_data[key]
        all_well_times = get_all_imod5_prj_well_times(imod5_data)

        df = _unpack_package_data(pkg_data, times, all_well_times)
        cls._validate_imod5_depth_information(key, pkg_data, df)

        # Groupby unique wells, to get dataframes per time.
        colnames_group = ["x", "y"] + cls._imod5_depth_colnames  # , "id"]
        wel_index, unique_well_groups = zip(*df.groupby(colnames_group))

        # Unpack wel indices by zipping
        varnames = [("x", float), ("y", float)] + cls._depth_colnames
        index_values = zip(*wel_index)
        cls_input = {
            var: np.array(value, dtype=dtype)
            for (var, dtype), value in zip(varnames, index_values)
        }
        cls_input["rate"] = _prepare_well_rates_from_groups(
            pkg_data, unique_well_groups, times
        )
        cls_input["minimum_k"] = minimum_k
        cls_input["minimum_thickness"] = minimum_thickness

        return cls(**cls_input)


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
        on creating point wells, no point wells will be placed in cells with a lower horizontal conductivity than this
    minimum_thickness: float, optional
        on creating point wells, no point wells will be placed in cells with a lower thickness than this
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
    repeat_stress: Optional[xr.DataArray] of datetimes
        Used to repeat data for e.g. repeating stress periods such as
        seasonality without duplicating the values. The DataArray should have
        dimensions ``("repeat", "repeat_items")``. The ``repeat_items``
        dimension should have size 2: the first value is the "key", the second
        value is the "value". For the "key" datetime, the data of the "value"
        datetime will be used. Can also be set with a dictionary using the
        ``set_repeat_stress`` method.

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
            AllValueSchema("<", "screen_top"),
        ],
        "y": [AnyNoDataSchema(), EmptyIndexesSchema()],
        "x": [AnyNoDataSchema(), EmptyIndexesSchema()],
        "rate": [AnyNoDataSchema(), EmptyIndexesSchema()],
        "concentration": [AnyNoDataSchema(), EmptyIndexesSchema()],
    }

    _imod5_depth_colnames = ["filt_top", "filt_bot"]
    _depth_colnames = [("screen_top", float), ("screen_bottom", float)]

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
        minimum_k: float = 0.1,
        minimum_thickness: float = 1.0,
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
        index_coord = np.arange(self.dataset.dims["index"])
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
    ) -> Package:
        """
        Clip a package by a bounding box (time, layer, y, x).

        The well package doesn't use the layer attribute to describe its depth and length.
        Instead, it uses the screen_top and screen_bottom parameters which corresponds with
        the z-coordinates of the top and bottom of the well. To go from a layer_min and
        layer_max to z-values used for clipping the well a top and bottom array have to be
        provided as well.

        Slicing intervals may be half-bounded, by providing None:

        * To select 500.0 <= x <= 1000.0:
          ``clip_box(x_min=500.0, x_max=1000.0)``.
        * To select x <= 1000.0: ``clip_box(x_min=None, x_max=1000.0)``
          or ``clip_box(x_max=1000.0)``.
        * To select x >= 500.0: ``clip_box(x_min = 500.0, x_max=None.0)``
          or ``clip_box(x_min=1000.0)``.

        Parameters
        ----------
        time_min: optional
        time_max: optional
        layer_min: optional, int
        layer_max: optional, int
        x_min: optional, float
        x_max: optional, float
        y_min: optional, float
        y_max: optional, float
        top: optional, GridDataArray
        bottom: optional, GridDataArray

        Returns
        -------
        sliced : Package
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

        z_max = self._find_well_value_at_layer(ds, top, layer_max)
        z_min = self._find_well_value_at_layer(ds, bottom, layer_min)

        if z_max is not None:
            ds["screen_top"] = ds["screen_top"].clip(None, z_max)
        if z_min is not None:
            ds["screen_bottom"] = ds["screen_bottom"].clip(z_min, None)

        # Initiate array of True with right shape to deal with case no spatial
        # selection needs to be done.
        in_bounds = np.full(ds.dims["index"], True)
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
            ).drop_vars(lambda x: x.coords)

        return value

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
        like = ones_like(active)
        bottom = like * bottom
        top_2d = (like * top).sel(layer=1)
        top_3d = bottom.shift(layer=1).fillna(top_2d)
        k = like * k

        index_names = wells_df.index.names

        minimum_k = self.dataset["minimum_k"].item()
        minimum_thickness = self.dataset["minimum_thickness"].item()

        # Unset multi-index, because assign_wells cannot deal with
        # multi-indices which is returned by self.dataset.to_dataframe() in
        # case of a "time" and "species" coordinate.
        wells_df = wells_df.reset_index()

        wells_assigned = assign_wells(
            wells_df, top_3d, bottom, k, minimum_thickness, minimum_k, True
        )
        # Set multi-index again
        wells_assigned = wells_assigned.set_index(index_names).sort_index()

        return wells_assigned

    @classmethod
    def _validate_imod5_depth_information(
        cls, key: str, pkg_data: dict, df: pd.DataFrame
    ) -> None:
        if "layer" in pkg_data.keys() and (np.any(pkg_data["layer"] != 0)):
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
        on creating point wells, no point wells will be placed in cells with a lower horizontal conductivity than this
    minimum_thickness: float, optional
        on creating point wells, no point wells will be placed in cells with a lower thickness than this
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
    repeat_stress: Optional[xr.DataArray] of datetimes
        Used to repeat data for e.g. repeating stress periods such as
        seasonality without duplicating the values. The DataArray should have
        dimensions ``("repeat", "repeat_items")``. The ``repeat_items``
        dimension should have size 2: the first value is the "key", the second
        value is the "value". For the "key" datetime, the data of the "value"
        datetime will be used. Can also be set with a dictionary using the
        ``set_repeat_stress`` method.

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
    _imod5_depth_colnames = ["layer"]
    _depth_colnames = [("layer", int)]

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
        minimum_k: float = 0.1,
        minimum_thickness: float = 1.0,
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
        index_coord = np.arange(self.dataset.dims["index"])
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
    ) -> Package:
        """
        Clip a package by a bounding box (time, layer, y, x).

        Slicing intervals may be half-bounded, by providing None:

        * To select 500.0 <= x <= 1000.0:
          ``clip_box(x_min=500.0, x_max=1000.0)``.
        * To select x <= 1000.0: ``clip_box(x_min=None, x_max=1000.0)``
          or ``clip_box(x_max=1000.0)``.
        * To select x >= 500.0: ``clip_box(x_min = 500.0, x_max=None.0)``
          or ``clip_box(x_min=1000.0)``.

        Parameters
        ----------
        time_min: optional
        time_max: optional
        layer_min: optional, int
        layer_max: optional, int
        x_min: optional, float
        x_max: optional, float
        y_min: optional, float
        y_max: optional, float
        top: optional, GridDataArray
        bottom: optional, GridDataArray

        Returns
        -------
        sliced : Package
        """
        # The super method will select in the time dimension without issues.
        new = super().clip_box(time_min=time_min, time_max=time_max)

        ds = new.dataset

        # Initiate array of True with right shape to deal with case no spatial
        # selection needs to be done.
        in_bounds = np.full(ds.dims["index"], True)
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
        if np.any(pkg_data["layer"] == 0):
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


class WellDisStructured(DisStructuredBoundaryCondition):
    """
    WEL package for structured discretization (DIS) models .
    Any number of WEL Packages can be specified for a single groundwater flow model.
    https://water.usgs.gov/water-resources/software/MODFLOW-6/mf6io_6.0.4.pdf#page=63

    .. warning::
        This class is deprecated and will be deleted in a future release.
        Consider changing your code to use the ``imod.mf6.Well`` package.

    Parameters
    ----------
    layer: list of int
        Model layer in which the well is located.
    row: list of int
        Row in which the well is located.
    column: list of int
        Column in which the well is located.
    rate: float or list of floats
        is the volumetric well rate. A positive value indicates well
        (injection) and a negative value indicates discharge (extraction) (q).
    concentration: array of floats (xr.DataArray, optional)
        if this flow package is used in simulations also involving transport, then this array is used
        as the  concentration for inflow over this boundary.
    concentration_boundary_type: ({"AUX", "AUXMIXED"}, optional)
        if this flow package is used in simulations also involving transport, then this keyword specifies
        how outflow over this boundary is computed.
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
    repeat_stress: Optional[xr.DataArray] of datetimes
        Used to repeat data for e.g. repeating stress periods such as
        seasonality without duplicating the values. The DataArray should have
        dimensions ``("repeat", "repeat_items")``. The ``repeat_items``
        dimension should have size 2: the first value is the "key", the second
        value is the "value". For the "key" datetime, the data of the "value"
        datetime will be used. Can also be set with a dictionary using the
        ``set_repeat_stress`` method.
    """

    _pkg_id = "wel"
    _period_data = ("layer", "row", "column", "rate")
    _keyword_map = {}
    _template = DisStructuredBoundaryCondition._initialize_template(_pkg_id)
    _auxiliary_data = {"concentration": "species"}

    _init_schemata = {
        "layer": [DTypeSchema(np.integer)],
        "row": [DTypeSchema(np.integer)],
        "column": [DTypeSchema(np.integer)],
        "rate": [DTypeSchema(np.floating)],
        "concentration": [DTypeSchema(np.floating)],
    }

    _write_schemata = {}

    @init_log_decorator()
    def __init__(
        self,
        layer,
        row,
        column,
        rate,
        concentration=None,
        concentration_boundary_type="aux",
        print_input=False,
        print_flows=False,
        save_flows=False,
        observations=None,
        validate: bool = True,
        repeat_stress=None,
    ):
        dict_dataset = {
            "layer": _assign_dims(layer),
            "row": _assign_dims(row),
            "column": _assign_dims(column),
            "rate": _assign_dims(rate),
            "print_input": print_input,
            "print_flows": print_flows,
            "save_flows": save_flows,
            "observations": observations,
            "repeat_stress": repeat_stress,
            "concentration": concentration,
            "concentration_boundary_type": concentration_boundary_type,
        }
        super().__init__(dict_dataset)
        self._validate_init_schemata(validate)

        warnings.warn(
            f"{self.__class__.__name__} is deprecated and will be removed in the v1.0 release."
            "Please adapt your code to use the imod.mf6.Well package",
            DeprecationWarning,
        )

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
    ) -> Package:
        """
        Clip a package by a bounding box (time, layer, y, x).

        Slicing intervals may be half-bounded, by providing None:

        * To select 500.0 <= x <= 1000.0:
          ``clip_box(x_min=500.0, x_max=1000.0)``.
        * To select x <= 1000.0: ``clip_box(x_min=None, x_max=1000.0)``
          or ``clip_box(x_max=1000.0)``.
        * To select x >= 500.0: ``clip_box(x_min = 500.0, x_max=None.0)``
          or ``clip_box(x_min=1000.0)``.

        Parameters
        ----------
        time_min: optional
        time_max: optional
        layer_min: optional, int
        layer_max: optional, int
        x_min: optional, float
        x_max: optional, float
        y_min: optional, float
        y_max: optional, float
        top: optional, GridDataArray
        bottom: optional, GridDataArray
        state_for_boundary: optional, GridDataArray

        Returns
        -------
        sliced : Package
        """
        # TODO: include x and y values.
        for arg in (
            layer_min,
            layer_max,
            x_min,
            x_max,
            y_min,
            y_max,
        ):
            if arg is not None:
                raise NotImplementedError("Can only clip_box in time for Well packages")

        # The super method will select in the time dimension without issues.
        new = super().clip_box(time_min=time_min, time_max=time_max)
        return new


class WellDisVertices(DisVerticesBoundaryCondition):
    """
    WEL package for discretization by vertices (DISV) models. Any number of WEL
    Packages can be specified for a single groundwater flow model.
    https://water.usgs.gov/water-resources/software/MODFLOW-6/mf6io_6.0.4.pdf#page=63

    .. warning::
        This class is deprecated and will be deleted in a future release.
        Consider changing your code to use the ``imod.mf6.Well`` package.

    Parameters
    ----------
    layer: list of int
        Modellayer in which the well is located.
    cell2d: list of int
        Cell in which the well is located.
    rate: float or list of floats
        is the volumetric well rate. A positive value indicates well (injection)
        and a negative value indicates discharge (extraction) (q).
    concentration: array of floats (xr.DataArray, optional)
        if this flow package is used in simulations also involving transport,
        then this array is used as the  concentration for inflow over this
        boundary.
    concentration_boundary_type: ({"AUX", "AUXMIXED"}, optional)
        if this flow package is used in simulations also involving transport,
        then this keyword specifies how outflow over this boundary is computed.
    print_input: ({True, False}, optional)
        keyword to indicate that the list of well information will be written to
        the listing file immediately after it is read. Default is False.
    print_flows: ({True, False}, optional)
        Indicates that the list of well flow rates will be printed to the
        listing file for every stress period time step in which "BUDGET PRINT"
        is specified in Output Control. If there is no Output Control option and
        PRINT FLOWS is specified, then flow rates are printed for the last time
        step of each stress period. Default is False.
    save_flows: ({True, False}, optional)
        Indicates that well flow terms will be written to the file specified
        with "BUDGET FILEOUT" in Output Control. Default is False.
    observations: [Not yet supported.]
        Default is None.
    validate: {True, False}
        Flag to indicate whether the package should be validated upon
        initialization. This raises a ValidationError if package input is
        provided in the wrong manner. Defaults to True.
    """

    _pkg_id = "wel"
    _period_data = ("layer", "cell2d", "rate")
    _keyword_map = {}
    _template = DisVerticesBoundaryCondition._initialize_template(_pkg_id)
    _auxiliary_data = {"concentration": "species"}

    _init_schemata = {
        "layer": [DTypeSchema(np.integer)],
        "cell2d": [DTypeSchema(np.integer)],
        "rate": [DTypeSchema(np.floating)],
        "concentration": [DTypeSchema(np.floating)],
    }

    _write_schemata = {}

    @init_log_decorator()
    def __init__(
        self,
        layer,
        cell2d,
        rate,
        concentration=None,
        concentration_boundary_type="aux",
        print_input=False,
        print_flows=False,
        save_flows=False,
        observations=None,
        validate: bool = True,
    ):
        dict_dataset = {
            "layer": _assign_dims(layer),
            "cell2d": _assign_dims(cell2d),
            "rate": _assign_dims(rate),
            "print_input": print_input,
            "print_flows": print_flows,
            "save_flows": save_flows,
            "observations": observations,
            "concentration": concentration,
            "concentration_boundary_type": concentration_boundary_type,
        }
        super().__init__(dict_dataset)
        self._validate_init_schemata(validate)

        warnings.warn(
            f"{self.__class__.__name__} is deprecated and will be removed in the v1.0 release."
            "Please adapt your code to use the imod.mf6.Well package",
            DeprecationWarning,
        )

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
    ) -> Package:
        """
        Clip a package by a bounding box (time, layer, y, x).

        Slicing intervals may be half-bounded, by providing None:

        * To select 500.0 <= x <= 1000.0:
          ``clip_box(x_min=500.0, x_max=1000.0)``.
        * To select x <= 1000.0: ``clip_box(x_min=None, x_max=1000.0)``
          or ``clip_box(x_max=1000.0)``.
        * To select x >= 500.0: ``clip_box(x_min = 500.0, x_max=None.0)``
          or ``clip_box(x_min=1000.0)``.

        Parameters
        ----------
        time_min: optional
        time_max: optional
        layer_min: optional, int
        layer_max: optional, int
        x_min: optional, float
        x_max: optional, float
        y_min: optional, float
        y_max: optional, float
        top: optional, GridDataArray
        bottom: optional, GridDataArray
        state_for_boundary: optional, GridDataArray

        Returns
        -------
        clipped: Package
        """
        # TODO: include x and y values.
        for arg in (
            layer_min,
            layer_max,
            x_min,
            x_max,
            y_min,
            y_max,
        ):
            if arg is not None:
                raise NotImplementedError("Can only clip_box in time for Well packages")

        # The super method will select in the time dimension without issues.
        new = super().clip_box(time_min=time_min, time_max=time_max)
        return new
