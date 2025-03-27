from typing import Literal, Union

import numpy as np
import pandas as pd
import xarray as xr

from imod.logging import init_log_decorator
from imod.mf6.boundary_condition import BoundaryCondition
from imod.mf6.write_context import WriteContext
from imod.schemata import (
    AllValueSchema,
    DimsSchema,
    DTypeSchema,
)
from imod.select.points import points_indices
from imod.typing.grid import GridDataArray

STATIC_EDGE_SCHEMA = DimsSchema("{edge_dim}")
EDGE_SCHEMA = DimsSchema("{edge_dim}") | DimsSchema("time", "{edge_dim}")


def _detect_update(ds: xr.Dataset, edge_dim: str) -> np.ndarray:
    # Find at which times a variable changes.
    # Create suitable fill value so we don't convert arrays to a different dtype
    # None of the variables allow negative values, so -1 will result in a new value.
    fill_value = {var: np.astype(np.float64(-1), ds[var].dtype) for var in ds.data_vars}
    shifted = ds.shift({"time": 1}, fill_value=fill_value)
    mutation_ds = ds != shifted
    # Keep the values when no time dimension is present.
    for var in mutation_ds.data_vars:
        if "time" not in mutation_ds[var].dims:
            mutation_ds[var] = True
    melted = mutation_ds.to_dataframe().melt(id_vars=[edge_dim, "time"])
    rows_to_keep = melted["value"].to_numpy()
    return rows_to_keep


class StreamFlowRouting(BoundaryCondition):
    """
    Stream Flow Routing (SFR) package.

    Parameters
    ----------
    reach_length: array of floats (xr.DataArray)
        defines the reach length. Must be greater than zero.
    reach_width: array of floats (xr.DataArray)
        defines the reach width. Must be greater than zero.
    reach_gradient: array of floats (xr.DataArray)
        defines the stream gradient (slope) across the reach. Must be greater than zero.
    reach_top: array of floats (xr.DataArray)
        defines the bottom elevation of the reach.
    streambed_thickness: array of floats (xr.DataArray)
        defines the thickness of the reach streambed. Must be greater than zero if reach
        is connected to an underlying GWF cell.
    bedk: array of floats (xr.DataArray)
        defines the hydraulic conductivity of the reach streambed. Must be greater than zero
        if reach is connected to an underlying GWF cell.
    manning_n: array of floats (xr.DataArray)
        defines the Manning's roughness coefficient for the reach. Must be greater than zero.
    layer: array of integers (xr.DataArray), optional
        to which MODFLOW layer the reach is connected. A value of 0 can be specified for
        reaches that are not connected to the groundwater.
    upstream_fraction: array of floats (xr.DataArray), optional
        defines the fraction of upstream flow from each upstream reach that is applied as
        upstream inflow to the reach. Must be between 0 and 1, and sum of all fractions for
        reaches connected to same upstream reach must equal 1.
    status: array of strings (xr.DataArray), ({"active", "inactive", "simple"}), optional.
        defines the status of each reach. Can be "active", "inactive", or "simple".
        The SIMPLE status option simulates streamflow using a user-specified stage for
        a reach or a stage set to the top of the reach (depth = 0). In cases where the
        simulated leakage calculated using the specified stage exceeds the sum of
        inflows to the reach, the stage is set to the top of the reach and leakage is
        set equal to the sum of inflows. Upstream fractions should be updated if the
        status for one or more reaches is changed. For example, if one of two
        downstream connections for a reach is inactivated, the upstream fraction for
        the active and inactive downstream reach should be changed to 1.0 and 0.0,
        respectively, to ensure that the active reach receives all of the downstream
        outflow from the upstream reach. Default is ACTIVE for all reaches.
    inflow: array of floats (xr.DataArray, optional)
        defines the volumetric inflow rate for the streamflow routing reach.
        Default is zero for each reach.
    rainfall: array of floats (xr.DataArray, optional)
        defines the volumetric rate per unit area of water added by precipitation directly
        on the streamflow routing reach. Default is zero for each reach.
    evaporation: array of floats (xr.DataArray, optional)
        defines the volumetric rate per unit area of water subtracted by evaporation from
        the streamflow routing reach. Default is zero for each reach.
    runoff: array of floats (xr.DataArray, optional)
        defines the volumetric rate of diffuse overland runoff that enters the streamflow
        routing reach. Default is zero for each reach.
    stage: array of floats (xr.DataArray, optional)
        defines the stage for the reach. Only applied if reach uses simple routing option.
    diversion_source: array of integers (xr.DataArray, optional)
        defines the source reach of the diversion.
    diversion_target: array of integers (xr.DataArray, optional)
        defines the target reach of the diversion.
    diversion_priority: array of strings (xr.DataArray, optional)
        defines the type of diversion. One of "fraction", "excess", "threshold", "upto".
    diversion_flow: array of floats (xr.DataArray, optional)
        defines the diversion flow; the exact meaning depends on diversion_priority.

        * 'fraction': The amount of the diversion is computed as a fraction of
          the streamflow leaving the source reach. In this case, 0.0 <= DIVFLOW
          <= 1.0.

        * 'excess': A diversion is made only if the source reach flow exceeds the
           value of DIVFLOW. If this occurs, then the quantity of water
           diverted is the excess flow (reach flow − DIVFLOW) and the source
           reach flow is set equal to DIVFLOW. This represents a flood-control
           type of diversion, as described by Danskin and Hanson (2002).

        * 'threshold': If source reach flow is less than the specified
          diversion flow DIVFLOW, no water is diverted from reach IFNO. If flow in
          the source reach is greater than or equal to DIVFLOW, DIVFLOW is
          diverted and the source reach flow is set to the remainder (flow -
          DIVFLOW). This approach assumes that once flow in the stream is
          sufficiently low, diversions from the stream cease, and is the
          'priority' algorithm that originally was programmed into the STR1
          Package (Prudic, 1989).

        * 'upto': If source reach flow is greater than or equal to the
          specified diversion flow DIVFLOW, flow is reduced by DIVFLOW. If
          source reach flow is less than DIVFLOW, DIVFLOW is set to source
          reach flow and there will be no flow available for reaches connected
          to downstream end of the sourche reach.

    print_input: ({True, False}, optional)
        keyword to indicate that the list of SFR information will be written to the listing file
        immediately after it is read. Default is False.
    print_flows: ({True, False}, optional)
        keyword to indicate that the list of SFR flow rates will be printed to the listing file for
        every stress period time step. Default is False.
    save_flows: ({True, False}, optional)
        keyword to indicate that SFR flow terms will be written to the file specified with "BUDGET
        FILEOUT" in Output Control. Default is False.
    budget_fileout: ({"str"}, optional)
        path to output cbc-file for SFR budgets
    budgetcsv_fileout: ({"str"}, optional)
        path to output csv-file for summed budgets
    stage_fileout: ({"str"}, optional)
        path to output file for stream stages
    validate: {True, False}
        Flag to indicate whether the package should be validated upon
        initialization. This raises a ValidationError if package input is
        provided in the wrong manner. Defaults to True.
    """

    _init_schemata = {
        "reach_length": [
            DTypeSchema(np.floating),
            STATIC_EDGE_SCHEMA,
        ],
        "reach_width": [
            DTypeSchema(np.floating),
            STATIC_EDGE_SCHEMA,
        ],
        "reach_gradient": [
            DTypeSchema(np.floating),
            STATIC_EDGE_SCHEMA,
        ],
        "reach_top": [
            DTypeSchema(np.floating),
            STATIC_EDGE_SCHEMA,
        ],
        "streambed_thickness": [
            DTypeSchema(np.floating),
            STATIC_EDGE_SCHEMA,
        ],
        "bedk": [
            DTypeSchema(np.floating),
            EDGE_SCHEMA,
        ],
        "manning_n": [
            DTypeSchema(np.floating),
            EDGE_SCHEMA,
        ],
        "layer": [
            DTypeSchema(np.integer),
            EDGE_SCHEMA,
        ],
        "upstream_fraction": [
            DTypeSchema(np.floating),
            EDGE_SCHEMA | DimsSchema(),  # optional var
        ],
        "inflow": [
            DTypeSchema(np.floating),
            EDGE_SCHEMA | DimsSchema(),  # optional var
        ],
        "rainfall": [
            DTypeSchema(np.floating),
            EDGE_SCHEMA | DimsSchema(),  # optional var
        ],
        "evaporation": [
            DTypeSchema(np.floating),
            EDGE_SCHEMA | DimsSchema(),  # optional var
        ],
        "runoff": [
            DTypeSchema(np.floating),
            EDGE_SCHEMA | DimsSchema(),  # optional var
        ],
        "stage": [
            DTypeSchema(np.floating),
            EDGE_SCHEMA | DimsSchema(),  # optional var
        ],
        "diversion_source": [
            DTypeSchema(np.integer),
            DimsSchema("diversion") | DimsSchema(),
        ],  # optional var
        "diversion_target": [
            DTypeSchema(np.integer),
            DimsSchema("diversion") | DimsSchema(),
        ],  # optional var
        "diversion_priority": [
            DTypeSchema(str),
            DimsSchema("diversion") | DimsSchema(),
        ],  # optional var
        "diversion_flow": [
            DTypeSchema(np.floating),
            DimsSchema("time", "diversion") | DimsSchema("diversion") | DimsSchema(),
        ],  # optional var
        "print_flows": [DTypeSchema(np.bool_), DimsSchema()],
        "save_flows": [DTypeSchema(np.bool_), DimsSchema()],
    }

    _write_schemata = {
        "reach_length": [AllValueSchema(">", 0.0)],
        "reach_width": [AllValueSchema(">", 0.0)],
        "reach_gradient": [AllValueSchema(">", 0.0)],
        "streambed_thickness": [AllValueSchema(">", 0.0)],
        "bedk": [AllValueSchema(">", 0.0)],
        "manning_n": [AllValueSchema(">", 0.0)],
        "layer": [AllValueSchema(">=", 0.0)],
        "upstream_fraction": [
            AllValueSchema(">=", 0.0),
            AllValueSchema("<=", 1.0),
        ],
        "diversion_source": [AllValueSchema(">=", 0)],  # positive edge_index
        "diversion_target": [AllValueSchema(">=", 0)],  # positive edge_index
        "diversion_flow": [AllValueSchema(">=", 0.0)],
        # TODO: diversion_priority should be fraction, excess, threshold, upto
        # TODO: status should be active, inactive, simple
        "rainfall": [AllValueSchema(">=", 0.0)],
        "evaporation": [AllValueSchema(">=", 0.0)],
        "runoff": [AllValueSchema(">=", 0.0)],
    }

    _package_data = (
        "reach_length",
        "reach_width",
        "reach_gradient",
        "reach_top",
        "streambed_thickness",
        "bedk",
        "manning_n",
        "upstream_fraction",
    )

    _period_data = (
        "status",
        "bedk",
        "manning_n",
        "stage",
        "inflow",
        "rainfall",
        "evaporation",
        "runoff",
        "upstream_fraction",
    )

    _pkg_id = "sfr"

    _template = BoundaryCondition._initialize_template(_pkg_id)

    @init_log_decorator()
    def __init__(
        self,
        reach_length,
        reach_width,
        reach_gradient,
        reach_top,
        streambed_thickness,
        bedk,
        manning_n,
        layer=None,  # assign based on reach_top if None?
        upstream_fraction=None,  # assumes equal fraction when not defined
        status: Union[xr.DataArray, Literal["active", "inactive", "simple"]] = "active",
        inflow=None,
        rainfall=None,
        evaporation=None,
        runoff=None,
        stage=None,
        diversion_source=None,
        diversion_target=None,
        diversion_priority=None,
        diversion_flow=None,
        storage: bool = False,
        maximum_iterations: int = 100,
        maximum_depth_change: float = 1e-5,
        length_conversion: float = 1.0,  # assumes unit is meter!
        print_input: bool = False,
        print_flows: bool = False,
        save_flows: bool = False,
        budget_fileout=None,
        budgetcsv_fileout=None,
        stage_fileout=None,
        validate: bool = True,
    ):
        dict_dataset = {
            # Package data
            "reach_length": reach_length,
            "reach_width": reach_width,
            "reach_gradient": reach_gradient,
            "reach_top": reach_top,
            "streambed_thickness": streambed_thickness,
            "bedk": bedk,
            "manning_n": manning_n,
            "layer": layer,
            "upstream_fraction": upstream_fraction,
            # Stress period data
            "status": status,
            "inflow": inflow,
            "rainfall": rainfall,
            "evaporation": evaporation,
            "runoff": runoff,
            "stage": stage,
            # Diversions
            "diversion_source": diversion_source,
            "diversion_target": diversion_target,
            "diversion_priority": diversion_priority,
            "diversion_flow": diversion_flow,
            # Options
            "storage": storage,
            "maximum_iterations": maximum_iterations,
            "maximum_depth_change": maximum_depth_change,
            "length_conversion": length_conversion,
            "print_input": print_input,
            "print_flows": print_flows,
            "save_flows": save_flows,
            "budget_fileout": budget_fileout,
            "budgetcsv_fileout": budgetcsv_fileout,
            "stage_fileout": stage_fileout,
        }
        super().__init__(dict_dataset)
        self._validate_init_schemata(validate)

    # TODO: Borrowed and edited from lake package
    # Can probably be generalized further: DISV also uses something similar!
    def _write_table_section(
        self, f, dataframe: pd.DataFrame, title: str, index: bool = False
    ) -> None:
        """
        writes a dataframe to file. Used for the connection data and for the outlet data.
        """
        # TODO: might not need full precision on floating point numbers!
        # 4 digits is likely sufficient
        f.write(f"begin {title}\n")
        dataframe.to_csv(
            f,
            index=index,
            header=False,
            sep=" ",
            lineterminator="\n",
        )
        f.write(f"end {title}\n")
        return

    def _find_cellid(self, dst_grid: GridDataArray) -> pd.DataFrame:
        # Find the centroid of each reach
        network = self.dataset.ugrid.grid
        x, y = network.edge_coordinates.T
        # Find indices belonging to x, y coordinates
        indices_cell2d = points_indices(dst_grid, out_of_bounds="ignore", x=x, y=y)
        # Fill in values of 0 for all indices when unconnected.
        # Convert cell2d indices from 0-based to 1-based.
        layer = self.dataset["layer"].to_numpy()
        unconnected = layer == 0
        indices = np.stack(
            [layer]
            + [
                np.where(unconnected, 0, index + 1)
                for index in reversed(indices_cell2d.values())
            ],
            axis=1,
        )
        labels = ["layer"] + list(indices_cell2d.keys())
        return xr.DataArray(
            indices,
            dims=(network.edge_dimension, "dimension"),
            coords={"dimension": labels},
        )

    def _to_mf6_pkg(
        self,
        idomain,
        top,
        bottom,
        k,
        validate_context,
    ) -> "StreamFlowRouting":
        new = self.copy()
        new.dataset["cellid"] = self._find_cellid(idomain)
        return new

    def _derive_upstream_fraction(self, upstream_fraction_data) -> np.ndarray:
        network = self.dataset.ugrid.grid
        downstream_reaches = network.directed_edge_edge_connectivity
        upstream_reaches = downstream_reaches.transpose().tocsr()

        # Check topology: "X"-crossing (combination of confluence and bifurcation isn't allowed)
        n_upstream = upstream_reaches.getnnz(axis=1)
        n_downstream = downstream_reaches.getnnz(axis=1)
        complex_junction = (n_upstream > 1) & (n_downstream > 1)
        if complex_junction.any():
            n_complex = complex_junction.sum()
            # TODO: ideally we add a function to return these bad reaches?
            # Printing them here might give (too) large error messages.
            raise ValueError(
                "A reach should have either multiple upstream reaches (confluence)"
                "or multiple downstream reaches (furcation), but not both. "
                "For complex junctions, the upstream_fraction is ambiguous. "
                f"This SFR network contains {n_complex} junctions."
            )

        # TODO: efficiently check for None
        if upstream_fraction_data.isnull():
            # By default, assume equal split.
            n_downstream = downstream_reaches.getnnz(axis=1)
            fraction_downstream = 1.0 / n_downstream
            # Get upstream_fraction for each reach.
            # TODO: think this over once more
            fraction_upstream = np.ones(network.n_edge)
            fraction_upstream[downstream_reaches.indices] = fraction_downstream[
                upstream_reaches.indices
            ]
            return fraction_upstream
        else:
            # Validate that the provided data sum to one.
            # TODO: move into validation?
            # TODO: validate each time step if provided in transient form.
            if "time" in upstream_fraction_data.dims:
                upstream_fraction_data = upstream_fraction_data.isel(time=0)

            to_validate = downstream_reaches.copy()
            to_validate.data = upstream_fraction_data[downstream_reaches.indices]
            fraction_sum = to_validate.sum(axis=1)
            # TODO: ideally we add a function to return these bad reaches?
            not_one = ~np.isclose(fraction_sum, 1.0)
            if not_one.any():
                # TODO: ideally we add a function to return these bad reaches?
                # Printing them here might give (too) large error messages.
                raise ValueError(
                    "upstream_fraction does not sum to 1.0 for all reaches."
                )
            return upstream_fraction_data

    def _packagedata_dataframe(self) -> pd.DataFrame:
        dataset = self.dataset
        network = dataset.ugrid.grid
        cellid = dataset["cellid"].to_pandas()

        package_data_vars = list(self._package_data)
        # Derive upstream_fraction if not provided (simple networks)
        package_data_vars.remove("upstream_fraction")

        ds = self.dataset[package_data_vars]
        if "time" in ds.dims:
            ds = ds.isel(time=0)

        df = ds.to_dataframe()
        df["n_connection"] = network.edge_edge_connectivity.getnnz(axis=1)
        df["upstream_fraction"] = self._derive_upstream_fraction(
            dataset["upstream_fraction"]
        )
        # TODO: check for None efficiently
        if dataset["diversion_source"].isnull():
            df["number_of_diversions"] = 0
        else:
            df["number_of_diversions"] = np.bincount(
                dataset["diversion_source"], minlength=network.n_edge
            )
        dataframe = pd.concat((cellid, df), axis=1)
        # MODFLOW 6 is 1-based
        dataframe.index += 1
        return dataframe

    def _connectiondata_dataframe(self) -> pd.DataFrame:
        INT_MAX = np.iinfo(np.int64).max
        # MODFLOW 6 wants the upstream reaches numbered POSITIVE
        # and the downstream reaches numbered NEGATIVE
        network = self.dataset.ugrid.grid
        # Derive the connectivity for each reach (edge)
        downstream = network.directed_edge_edge_connectivity
        upstream = downstream.transpose().tocsr()
        sfr_upstream = network.format_connectivity_as_dense(upstream)
        sfr_downstream = network.format_connectivity_as_dense(downstream)
        sfr_connectivity = np.hstack((sfr_upstream + 1, -(sfr_downstream + 1)))
        sfr_fill = np.hstack((sfr_upstream == -1, sfr_downstream == -1))
        sfr_connectivity[sfr_fill] = INT_MAX
        # Use pandas Integer array for NA (masked) values.
        df = pd.DataFrame()
        for i, column in enumerate(sfr_connectivity.T):
            df[i] = pd.arrays.IntegerArray(
                # MODFLOW starts numbering at 1.
                values=column,
                mask=(column == INT_MAX),
            )
        # Increment the index by one, as MODFLOW numbers ifno from 1.
        df.index += 1
        return df

    def _crosssections_dataframe(self) -> Union[pd.DataFrame, None]:
        # TODO:
        return None

    def _diversions_dataframe(self) -> Union[pd.DataFrame, None]:
        diversion_source = self.dataset["diversion_source"]
        if diversion_source.isnull():
            return None
        diversion_df = self.dataset[
            ["diversion_source", "diversion_target", "diversion_priority"]
        ].to_dataframe()
        diversion_df["diversion_index"] = np.arange(len(diversion_source)) + 1
        # Make sure columns are returned in the right order
        return diversion_df[
            [
                "diversion_source",
                "diversion_index",
                "diversion_target",
                "diversion_priority",
            ]
        ]

    def _initialstages_dataframe(self) -> Union[pd.DataFrame, None]:
        stage = self.dataset["stage"]
        if stage.isnull():
            return None

        if "time" in stage.dims:
            initialstages = stage.isel(time=0)
        else:
            initialstages = stage
        df = initialstages.to_dataframe()
        df.index += 1
        return df

    def _perioddata_dataframe(self, globaltimes) -> Union[pd.DataFrame, None]:
        if "time" not in self.dataset.dims:
            return None
        period_ds = self.dataset[self._period_data]
        edge_dim = self.dataset.ugrid.grid.edge_dimension

        # These variables are both package data and period data. If they
        # aren't transient, the package data entry suffices, and they don't
        # need to be written in the period block.
        # Also drop optional transient entries.
        vars_to_drop = [
            var
            for var in ("bedk", "manning", "stage", "upstream_fraction")
            if "time" not in self.dataset[var].dims
        ] + [var for var in self._period_data if self.dataset[var] is None]
        period_ds = period_ds.drop_vars(vars_to_drop)
        if len(period_ds.data_vars) == 0:
            return None

        # Some variables can only be specified in the period block.
        # If all are constant over time, create a one-sized time dimension.
        if "time" not in period_ds:
            period_ds = period_ds.expand_dims("time")
            period_ds["time"] = [1]
        else:
            period_ds["time"] = np.searchsorted(period_ds["time"], globaltimes) + 1

        rows_to_keep = _detect_update(period_ds, edge_dim)
        period_df = period_ds.to_dataframe()
        updates = period_df.melt(id_vars=[edge_dim, "time"]).iloc[rows_to_keep]

        diversion_source = self.dataset["diversion_source"]
        if diversion_source is not None:
            # Diversion source contains the edge index
            # Detect changes of the flow
            diversion_ds = self.dataset[["diversion_flow"]]
            diversion_ds["time"] = (
                np.searchsorted(diversion_ds["time"], globaltimes) + 1
            )
            diversion_to_keep = _detect_update(diversion_ds, edge_dim)
            # Add the edge_index (IFNO), this'll end up as an index
            diversion_ds[edge_dim] = diversion_source
            diversion_ds["diversion_index"] = np.arange(len(diversion_source)) + 1
            diversion_df = diversion_ds.to_dataframe()
            # Concatenate the entries into a single column so it'll match the other period data.
            diversion_df["diversion_entry"] = (
                diversion_df["diversion_index"].astype(str)
                + " "
                + diversion_df["diversion_flow"].astype(str)
            )
            diversion_updates = (
                diversion_df[[edge_dim, "time", "diversion_entry"]]
                .melt(id_vars=[edge_dim, "time"])
                .iloc[diversion_to_keep]
            )
            updates = pd.concat((updates, diversion_updates))

        return updates

    def render(self, directory, pkgname, globaltimes, binary) -> str:
        d = {
            "storage": self.dataset["storage"].item(),
            "maximum_iterations": self.dataset["maximum_iterations"].item(),
            "maximum_depth_change": self.dataset["maximum_depth_change"].item(),
            "length_conversion": self.dataset["length_conversion"].item(),
            "time_conversion": 86_400.0,  # imod-python always assumes days!
            "print_input": self.dataset["print_input"].item(),
            "print_flows": self.dataset["print_flows"].item(),
            "save_flows": self.dataset["save_flows"].item(),
            # TODO:
            # "budget_fileout": self.dataset["budget_fileout"].item(),
            # "budgetcsv_fileout": self.dataset["budgetcsv_fileout"].item(),
            "stage_fileout": self.dataset["stage_fileout"].item(),
            "nreaches": self.dataset.ugrid.grid.n_edge,
        }
        return self._template.render(d)

    def write_blockfile(self, pkgname, globaltimes, write_context: WriteContext):
        dir_for_render = write_context.get_formatted_write_directory()
        content = self.render(
            dir_for_render, pkgname, globaltimes, write_context.use_binary
        )
        filename = write_context.write_directory / f"{pkgname}.{self._pkg_id}"

        packagedata_df = self._packagedata_dataframe()

        with open(filename, "w") as f:
            f.write(content)
            f.write("\n\n")

            connectiondata_df = self._connectiondata_dataframe()
            crosssection_df = self._crosssections_dataframe()
            packagedata_df = self._packagedata_dataframe()
            diversions_df = self._diversions_dataframe()
            initialstages_df = self._initialstages_dataframe()
            perioddata_df = self._perioddata_dataframe(globaltimes)

            self._write_table_section(
                f, packagedata_df, title="packagedata", index=True
            )
            if crosssection_df is not None:
                self._write_table_section(f, crosssection_df, title="crossections")
            self._write_table_section(
                f, connectiondata_df, title="connectiondata", index=True
            )
            if diversions_df is not None:
                self._write_table_section(f, diversions_df, title="diversions")
            if initialstages_df is not None:
                self._write_table_section(
                    f, initialstages_df, title="initialstages", index=True
                )
            if perioddata_df is not None:
                # Iterate over the periods
                for period_number, period_df in perioddata_df.groupby("time"):
                    self._write_table_section(
                        f, period_df, title=f"period {period_number}"
                    )

        return

    # Overloaded, neutralized methods
    # TODO: would be more convenient to move these into a single function to overload?

    def _package_data_to_sparse(self):
        return

    def fill_stress_perioddata(self):
        # this function is called from packagebase and should do nothing in this context
        return

    def _write_perioddata(self, directory, pkgname, binary):
        # this function is called from packagebase and should do nothing in this context
        return

    def is_splitting_supported(self) -> bool:
        return False

    def is_regridding_supported(self) -> bool:
        return False

    def is_clipping_supported(self) -> bool:
        return False
