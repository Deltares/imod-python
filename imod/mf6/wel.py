import warnings
from typing import List, Union

import numpy as np
import xarray as xr
import xugrid as xu

from imod.mf6.pkgbase import (
    BoundaryCondition,
    DisStructuredBoundaryCondition,
    DisVerticesBoundaryCondition,
)
from imod.mf6.pkgbase_lowlevel import Mf6Bc, remove_inactive
from imod.prepare import assign_wells
from imod.schemata import DTypeSchema
from imod.select.points import points_indices


# FUTURE: There was an idea to autogenerate these object.
# This was relevant:
# https://github.com/Deltares/xugrid/blob/main/xugrid/core/wrap.py#L90
class Mf6Wel(Mf6Bc):
    _pkg_id = "wel"

    _period_data = ("cellid", "rate")
    _keyword_map = {}
    _template = Mf6Bc._initialize_template(_pkg_id)
    _auxiliary_data = {"concentration": "species"}

    _init_schemata = {
        "cellid": [DTypeSchema(np.integer)],
        "rate": [DTypeSchema(np.floating)],
        "concentration": [DTypeSchema(np.floating)],
    }
    _write_schemata = {}

    def __init__(
        self,
        cellid,
        rate,
        concentration=None,
        concentration_boundary_type="aux",
    ):
        super().__init__()
        self.dataset["cellid"] = cellid
        self.dataset["rate"] = rate

        if concentration is not None:
            self.dataset["concentration"] = concentration
            self.dataset["concentration_boundary_type"] = concentration_boundary_type
            self.add_periodic_auxiliary_variable()


class Well(BoundaryCondition):
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

    _auxiliary_data = {"concentration": "species"}
    _init_schemata = {
        "screen_top": [DTypeSchema(np.floating)],
        "screen_bottom": [DTypeSchema(np.floating)],
        "y": [DTypeSchema(np.floating)],
        "x": [DTypeSchema(np.floating)],
        "rate": [DTypeSchema(np.floating)],
        "concentration": [DTypeSchema(np.floating)],
    }
    _write_schemata = {}

    def __init__(
        self,
        screen_top,
        screen_bottom,
        y,
        x,
        rate,
        concentration=None,
        concentration_boundary_type="aux",
        id=None,
        print_input=False,
        print_flows=False,
        save_flows=False,
        observations=None,
        validate: bool = True,
        repeat_stress=None,
    ):
        super().__init__()
        self.dataset["screen_top"] = self.assign_dims(screen_top)
        self.dataset["screen_bottom"] = self.assign_dims(screen_bottom)
        self.dataset["y"] = self.assign_dims(y)
        self.dataset["x"] = self.assign_dims(x)
        self.dataset["rate"] = self.assign_dims(rate)
        if id is None:
            id = np.arange(self.dataset["x"].size).astype(str)
        self.dataset["id"] = self.assign_dims(id)

        self.dataset["print_input"] = print_input
        self.dataset["print_flows"] = print_flows
        self.dataset["save_flows"] = save_flows
        self.dataset["observations"] = observations
        self.dataset["repeat_stress"] = repeat_stress

        if concentration is not None:
            self.dataset["concentration"] = concentration
            self.dataset["concentration_boundary_type"] = concentration_boundary_type

        self._validate_init_schemata(validate)

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
    ) -> "Well":
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
        x_min: optional, float
        y_max: optional, float
        y_max: optional, float

        Returns
        -------
        sliced : Package
        """

        def is_within_range(da, min, max):
            """
            Find which elements are within range.
            Function checks which values are unaffected by the clip method, to
            be able to deal with min and max values equal to None, which
            should be ignored.
            """
            if min is None and max is None:
                return True
            else:
                return da == da.clip(min=min, max=max)

        # The super method will select in the time dimension without issues.
        new = super().clip_box(time_min=time_min, time_max=time_max)

        ds = new.dataset

        # Select all variables along "index" dimension
        in_bounds = is_within_range(ds["x"], x_min, x_max)
        in_bounds &= is_within_range(ds["y"], y_min, y_max)
        in_bounds &= is_within_range(ds["screen_top"], None, z_max)
        in_bounds &= is_within_range(ds["screen_bottom"], z_min, None)
        # Replace dataset with reduced dataset based on booleans
        new.dataset = ds.loc[{"index": in_bounds}]

        return new

    def to_mf6_pkg(self, active, top, bottom, k) -> Mf6Wel:
        """
        Write package to Modflow 6 package.

        Based on the model grid and top and bottoms, cellids are determined.
        When well screens hit multiple layers, groundwater extractions are
        distributed based on layer transmissivities.

        Parameters
        ----------

        """
        # Ensure top, bottom & k
        # are broadcasted to 3d grid
        like = xr.ones_like(active)
        top = like * top
        bottom = like * bottom
        k = like * k

        wells_df = self.dataset.to_dataframe()
        wells_df = wells_df.rename(
            columns={
                "screen_top": "top",
                "screen_bottom": "bottom",
            }
        )
        index_names = wells_df.index.names

        # Unset multi-index, because assign_wells cannot deal with
        # multi-indices which is returned by self.dataset.to_dataframe() in
        # case of a "time" and "species" coordinate.
        wells_df = wells_df.reset_index()
        wells_assigned = assign_wells(wells_df, top, bottom, k)
        # Set multi-index again
        wells_assigned = wells_assigned.set_index(index_names).sort_index()

        ds = xr.Dataset()
        # Groupby index and select first, to unset any duplicate records
        # introduced by the multi-indexed "time" dimension.
        df_for_cellid = wells_assigned.groupby("index").first()
        d_for_cellid = df_for_cellid[["x", "y", "layer"]].to_dict("list")
        ds["cellid"] = create_cellid(top, **d_for_cellid)

        data_vars = ["rate"]
        if "concentration" in wells_assigned.columns:
            data_vars.append("concentration")

        ds_vars = wells_assigned[data_vars].to_xarray()
        # "rate" variable in conversion from multi-indexed DataFrame to xarray
        # DataArray results in duplicated values for "rate" along dimension
        # "species". Select first species to reduce this again.
        if "species" in index_names:
            ds_vars["rate"] = ds_vars["rate"].isel(species=0)
        # Carefully rename the dimension and set coordinates before
        # assigning to dataset.
        d_rename = {"index": "ncellid"}
        ds_vars = ds_vars.rename_dims(**d_rename).rename_vars(**d_rename)
        ds_vars = ds_vars.assign_coords(**{"ncellid": ds.coords["ncellid"].values})
        ds = ds.assign(**dict(ds_vars.items()))
        # Remove wells defined in inactive cells
        # Cells outside grid have already been
        # removed in assign_wells.
        ds = remove_inactive(ds, active)

        return Mf6Wel(**ds)


class WellDisStructured(DisStructuredBoundaryCondition):
    """
    WEL package for structured discretization (DIS) models .
    Any number of WEL Packages can be specified for a single groundwater flow model.
    https://water.usgs.gov/water-resources/software/MODFLOW-6/mf6io_6.0.4.pdf#page=63

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
        self.dataset["layer"] = self.assign_dims(layer)
        self.dataset["row"] = self.assign_dims(row)
        self.dataset["column"] = self.assign_dims(column)
        self.dataset["rate"] = self.assign_dims(rate)
        self.dataset["print_input"] = print_input
        self.dataset["print_flows"] = print_flows
        self.dataset["save_flows"] = save_flows
        self.dataset["observations"] = observations
        self.dataset["repeat_stress"] = repeat_stress

        if concentration is not None:
            self.dataset["concentration"] = concentration
            self.dataset["concentration_boundary_type"] = concentration_boundary_type
            self.add_periodic_auxiliary_variable()

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
    ) -> "WellDisStructured":
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
        self.dataset["layer"] = self.assign_dims(layer)
        self.dataset["cell2d"] = self.assign_dims(cell2d)
        self.dataset["rate"] = self.assign_dims(rate)
        self.dataset["print_input"] = print_input
        self.dataset["print_flows"] = print_flows
        self.dataset["save_flows"] = save_flows
        self.dataset["observations"] = observations

        if concentration is not None:
            self.dataset["concentration"] = concentration
            self.dataset["concentration_boundary_type"] = concentration_boundary_type
            self.add_periodic_auxiliary_variable()

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
    ) -> "WellDisStructured":
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


def create_cellid(
    to_grid: Union[xr.DataArray, xu.UgridDataArray],
    x: List,
    y: List,
    layer: List,
) -> xr.DataArray:
    """

    Create DataArray with Modflow6 cell indices based on x, y coordinates
    in a dataframe. For structured grid this DataArray contains 3 columns:
    ``layer, row, column``. For unstructured grids, this contains 2 columns:
    ``layer, cell2d``.

    Note
    ----
    The "layer" coordinate should already be provided in the dataframe.
    To determine the layer coordinate based on screen depts, look at
    :func:`imod.prepare.wells.assign_wells`.

    Parameters
    ----------
    x: {list, np.array}
        array-like with x-coordinates
    y: {list, np.array}
        array-like with y-coordinates
    layer: {list, np.array}
        array-like with layer-coordinates
    to_grid: {xr.DataArray, xu.UgridDataArray}
        Grid to map the points to based on their x and y coordinates.

    Returns
    -------
    cellid : xr.DataArray
        2D DataArray with a ``ncellid`` rows and 3 to 2 columns, depending
        on whether on a structured or unstructured grid."""

    # Find indices belonging to x, y coordinates
    indices = points_indices(to_grid, out_of_bounds="ignore", x=x, y=y)
    # Prepare layer indices, for later concatenation
    indices_layer = xr.DataArray(layer, coords=indices["x"].coords)

    if isinstance(to_grid, xu.UgridDataArray):
        face_dim = to_grid.ugrid.grid.face_dimension
        indices_cell2d_dims = [face_dim]
        cell2d_coords = ["cell2d"]
    else:
        indices_cell2d_dims = ["y", "x"]
        cell2d_coords = ["row", "column"]
    # Convert cell2d indices from 0-based to 1-based.
    cellid_ls = [indices_layer] + [indices[dim] + 1 for dim in indices_cell2d_dims]
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
