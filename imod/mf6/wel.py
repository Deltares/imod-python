from __future__ import annotations

import typing
import warnings
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
import xugrid as xu

from imod.mf6.auxiliary_variables import add_periodic_auxiliary_variable
from imod.mf6.boundary_condition import (
    BoundaryCondition,
    DisStructuredBoundaryCondition,
    DisVerticesBoundaryCondition,
)
from imod.mf6.interfaces.ipointdatapackage import IPointDataPackage
from imod.mf6.mf6_wel_adapter import Mf6Wel
from imod.mf6.package import Package
from imod.mf6.utilities.clip import clip_by_grid
from imod.mf6.utilities.dataset import remove_inactive
from imod.mf6.write_context import WriteContext
from imod.prepare import assign_wells
from imod.schemata import AllNoDataSchema, DTypeSchema
from imod.select.points import points_indices, points_values
from imod.typing import GridDataArray
from imod.typing.grid import ones_like
from imod.util import values_within_range


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


def mask_2D(package: Well, domain_2d: GridDataArray) -> Well:
    point_active = points_values(domain_2d, x=package.x, y=package.y)

    is_inside_exterior = point_active == 1
    selection = package.dataset.loc[{"index": is_inside_exterior}]

    cls = type(package)
    new = cls.__new__(cls)
    new.dataset = selection
    return new


class Well(BoundaryCondition, IPointDataPackage):
    """
    Agnostic WEL package, which accepts x, y and a top and bottom of the well screens.

    This package can be written to any provided model grid.
    Any number of WEL Packages can be specified for a single groundwater flow model.
    https://water.usgs.gov/water-resources/software/MODFLOW-6/mf6io_6.0.4.pdf#page=63

    Parameters
    ----------

    screen_top: float or list of floats
        is the top of the well screen.
    screen_bottom: float or list of floats
        is the bottom of the well screen.
    y: float or list of floats
        is the y location of the well.
    x: float or list of floats
        is the x location of the well.
    rate: float or list of floats
        is the volumetric well rate. A positive value indicates well
        (injection) and a negative value indicates discharge (extraction) (q).
    concentration: array of floats (xr.DataArray, optional)
        if this flow package is used in simulations also involving transport, then this array is used
        as the  concentration for inflow over this boundary.
    concentration_boundary_type: ({"AUX", "AUXMIXED"}, optional)
        if this flow package is used in simulations also involving transport, then this keyword specifies
        how outflow over this boundary is computed.
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
    """

    @property
    def x(self) -> npt.NDArray[float]:
        return self.dataset["x"].values

    @property
    def y(self) -> npt.NDArray[float]:
        return self.dataset["y"].values

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
        "y": [AllNoDataSchema()],
        "x": [AllNoDataSchema()],
    }

    _regrid_method = {}

    def __init__(
        self,
        x: List[float],
        y: List[float],
        screen_top: List[float],
        screen_bottom: List[float],
        rate: List[float],
        concentration: Optional[List[float] | xr.DataArray] = None,
        concentration_boundary_type="aux",
        id: Optional[List[int]] = None,
        minimum_k: float = 0.1,
        minimum_thickness: float = 1.0,
        print_input: bool = False,
        print_flows: bool = False,
        save_flows: bool = False,
        observations=None,
        validate: bool = True,
        repeat_stress: Optional[xr.DataArray] = None,
    ):
        super().__init__()
        self.dataset["screen_top"] = _assign_dims(screen_top)
        self.dataset["screen_bottom"] = _assign_dims(screen_bottom)
        self.dataset["y"] = _assign_dims(y)
        self.dataset["x"] = _assign_dims(x)
        self.dataset["rate"] = _assign_dims(rate)
        if id is None:
            id = np.arange(self.dataset["x"].size).astype(str)
        self.dataset["id"] = _assign_dims(id)
        self.dataset["minimum_k"] = minimum_k
        self.dataset["minimum_thickness"] = minimum_thickness

        self.dataset["print_input"] = print_input
        self.dataset["print_flows"] = print_flows
        self.dataset["save_flows"] = save_flows
        self.dataset["observations"] = observations
        self.dataset["repeat_stress"] = repeat_stress
        if concentration is not None:
            self.dataset["concentration"] = concentration
            self.dataset["concentration_boundary_type"] = concentration_boundary_type

        self._validate_init_schemata(validate)

    @classmethod
    def is_grid_agnostic_package(cls) -> bool:
        return True

    def clip_box(
        self,
        time_min=None,
        time_max=None,
        z_min=None,
        z_max=None,
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
    ) -> Well:
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
        z_min: optional, float
        z_max: optional, float
        x_min: optional, float
        x_max: optional, float
        y_min: optional, float
        y_max: optional, float

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
        in_bounds &= values_within_range(ds["screen_top"], None, z_max)
        in_bounds &= values_within_range(ds["screen_bottom"], z_min, None)
        # Replace dataset with reduced dataset based on booleans
        new.dataset = ds.loc[{"index": in_bounds}]

        return new

    def write(
        self,
        pkgname: str,
        globaltimes: npt.NDArray[np.datetime64],
        validate: bool,
        write_context: WriteContext,
        idomain: Union[xr.DataArray, xu.UgridDataArray],
        top: Union[xr.DataArray, xu.UgridDataArray],
        bottom: Union[xr.DataArray, xu.UgridDataArray],
        k: Union[xr.DataArray, xu.UgridDataArray],
    ) -> None:
        if validate:
            self._validate(self._write_schemata)
        mf6_package = self.to_mf6_pkg(
            idomain, top, bottom, k, write_context.is_partitioned
        )
        # TODO: make options like "save_flows" configurable. Issue github #623
        mf6_package.dataset["save_flows"] = True
        mf6_package.write(pkgname, globaltimes, write_context)

    def __create_wells_df(self) -> pd.DataFrame:
        wells_df = self.dataset.to_dataframe()
        wells_df = wells_df.rename(
            columns={
                "screen_top": "top",
                "screen_bottom": "bottom",
            }
        )

        return wells_df

    def __create_assigned_wells(
        self,
        wells_df: pd.DataFrame,
        active: GridDataArray,
        top: GridDataArray,
        bottom: GridDataArray,
        k: GridDataArray,
        minimum_k: float,
        minimum_thickness: float,
    ):
        # Ensure top, bottom & k
        # are broadcasted to 3d grid
        like = ones_like(active)
        bottom = like * bottom
        top_2d = (like * top).sel(layer=1)
        top_3d = bottom.shift(layer=1).fillna(top_2d)

        k = like * k

        index_names = wells_df.index.names

        # Unset multi-index, because assign_wells cannot deal with
        # multi-indices which is returned by self.dataset.to_dataframe() in
        # case of a "time" and "species" coordinate.
        wells_df = wells_df.reset_index()

        wells_assigned = assign_wells(
            wells_df, top_3d, bottom, k, minimum_thickness, minimum_k
        )
        # Set multi-index again
        wells_assigned = wells_assigned.set_index(index_names).sort_index()

        return wells_assigned

    def __create_dataset_vars(
        self, wells_assigned: pd.DataFrame, wells_df: pd.DataFrame, cellid: xr.DataArray
    ) -> xr.Dataset:
        """
        Create dataset with all variables (rate, concentration), with a similar shape as the cellids.
        """
        data_vars = ["rate"]
        if "concentration" in wells_assigned.columns:
            data_vars.append("concentration")

        ds_vars = wells_assigned[data_vars].to_xarray()
        # "rate" variable in conversion from multi-indexed DataFrame to xarray
        # DataArray results in duplicated values for "rate" along dimension
        # "species". Select first species to reduce this again.
        index_names = wells_df.index.names
        if "species" in index_names:
            ds_vars["rate"] = ds_vars["rate"].isel(species=0)

        # Carefully rename the dimension and set coordinates
        d_rename = {"index": "ncellid"}
        ds_vars = ds_vars.rename_dims(**d_rename).rename_vars(**d_rename)
        ds_vars = ds_vars.assign_coords(**{"ncellid": cellid.coords["ncellid"].values})

        return ds_vars

    def __create_cellid(self, wells_assigned: pd.DataFrame, active: xr.DataArray):
        like = ones_like(active)

        # Groupby index and select first, to unset any duplicate records
        # introduced by the multi-indexed "time" dimension.
        df_for_cellid = wells_assigned.groupby("index").first()
        d_for_cellid = df_for_cellid[["x", "y", "layer"]].to_dict("list")

        return self.__derive_cellid_from_points(like, **d_for_cellid)

    @staticmethod
    def __derive_cellid_from_points(
        dst_grid: GridDataArray,
        x: List,
        y: List,
        layer: List,
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
        indices_cell2d = dict((dim, index + 1) for dim, index in indices_cell2d.items())
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

    def to_mf6_pkg(
        self,
        active: Union[xr.DataArray, xu.UgridDataArray],
        top: Union[xr.DataArray, xu.UgridDataArray],
        bottom: Union[xr.DataArray, xu.UgridDataArray],
        k: Union[xr.DataArray, xu.UgridDataArray],
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
        Returns
        -------
        Mf6Wel
            Object with wells as list based input.
        """

        minimum_k = self.dataset["minimum_k"].item()
        minimum_thickness = self.dataset["minimum_thickness"].item()

        wells_df = self.__create_wells_df()
        wells_assigned = self.__create_assigned_wells(
            wells_df, active, top, bottom, k, minimum_k, minimum_thickness
        )

        nwells_df = len(wells_df["id"].unique())
        nwells_assigned = (
            0 if wells_assigned.empty else len(wells_assigned["id"].unique())
        )

        if nwells_df == 0:
            raise ValueError("No wells were assigned in package. None were present.")
        # @TODO: reinstate this check. issue github #621.
        if not is_partitioned and nwells_df != nwells_assigned:
            raise ValueError(
                "One or more well(s) are completely invalid due to minimum conductivity and thickness constraints."
            )

        ds = xr.Dataset()
        ds["cellid"] = self.__create_cellid(wells_assigned, active)

        ds_vars = self.__create_dataset_vars(wells_assigned, wells_df, ds["cellid"])
        ds = ds.assign(**dict(ds_vars.items()))

        ds = remove_inactive(ds, active)

        return Mf6Wel(**ds)

    def regrid_like(self, target_grid: GridDataArray, *_) -> Well:
        """
        The regrid_like method is irrelevant for this package as it is
        grid-agnostic, instead this method clips the package based on the grid
        exterior.
        """
        target_grid_2d = target_grid.isel(layer=0, drop=True, missing_dims="ignore")
        return clip_by_grid(self, target_grid_2d)

    def mask(self, domain: GridDataArray) -> Well:
        """
        Mask wells based on two-dimensional domain. For three-dimensional
        masking: Wells falling in inactive cells are automatically removed in
        the call to write to Modflow 6 package. You can verify this by calling
        the ``to_mf6_pkg`` method.
        """

        # Drop layer coordinate if present, otherwise a layer coordinate is assigned
        # which causes conflicts downstream when assigning wells and deriving
        # cellids.
        domain_2d = domain.isel(layer=0, drop=True, missing_dims="ignore").drop(
            "layer", errors="ignore"
        )
        return mask_2D(self, domain_2d)


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
        super().__init__()
        self.dataset["layer"] = _assign_dims(layer)
        self.dataset["row"] = _assign_dims(row)
        self.dataset["column"] = _assign_dims(column)
        self.dataset["rate"] = _assign_dims(rate)
        self.dataset["print_input"] = print_input
        self.dataset["print_flows"] = print_flows
        self.dataset["save_flows"] = save_flows
        self.dataset["observations"] = observations
        self.dataset["repeat_stress"] = repeat_stress

        if concentration is not None:
            self.dataset["concentration"] = concentration
            self.dataset["concentration_boundary_type"] = concentration_boundary_type
            add_periodic_auxiliary_variable(self)

        self._validate_init_schemata(validate)

        warnings.warn(
            f"{self.__class__.__name__} is deprecated and will be removed in the v1.0 release."
            "Please adapt your code to use the imod.mf6.Well package",
            DeprecationWarning,
        )

    def clip_box(
        self,
        time_min=None,
        time_max=None,
        layer_min=None,
        layer_max=None,
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
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
        x_min: optional, float
        y_max: optional, float
        y_max: optional, float

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
        super().__init__()
        self.dataset["layer"] = _assign_dims(layer)
        self.dataset["cell2d"] = _assign_dims(cell2d)
        self.dataset["rate"] = _assign_dims(rate)
        self.dataset["print_input"] = print_input
        self.dataset["print_flows"] = print_flows
        self.dataset["save_flows"] = save_flows
        self.dataset["observations"] = observations

        if concentration is not None:
            self.dataset["concentration"] = concentration
            self.dataset["concentration_boundary_type"] = concentration_boundary_type
            add_periodic_auxiliary_variable(self)

        self._validate_init_schemata(validate)

        warnings.warn(
            f"{self.__class__.__name__} is deprecated and will be removed in the v1.0 release."
            "Please adapt your code to use the imod.mf6.Well package",
            DeprecationWarning,
        )

    def clip_box(
        self,
        time_min=None,
        time_max=None,
        layer_min=None,
        layer_max=None,
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
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
        x_min: optional, float
        y_max: optional, float
        y_max: optional, float

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
