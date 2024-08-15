import abc
import copy
import textwrap
import typing
from copy import deepcopy
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import cftime
import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
import xugrid as xu
from fastcore.dispatch import typedispatch

from imod.logging import init_log_decorator
from imod.mf6.boundary_condition import BoundaryCondition
from imod.mf6.interfaces.ilinedatapackage import ILineDataPackage
from imod.mf6.mf6_hfb_adapter import Mf6HorizontalFlowBarrier
from imod.mf6.package import Package
from imod.mf6.utilities.clip import clip_line_gdf_by_grid
from imod.mf6.utilities.grid import broadcast_to_full_domain
from imod.mf6.utilities.hfb import (
    _create_zlinestring_from_bound_df,
    _extract_hfb_bounds_from_zpolygons,
    _prepare_index_names,
)
from imod.schemata import EmptyIndexesSchema, MaxNUniqueValuesSchema
from imod.typing import GeoDataFrameType, GridDataArray, LineStringType
from imod.typing.grid import enforce_uda
from imod.util.imports import MissingOptionalModule

if TYPE_CHECKING:
    import geopandas as gpd
else:
    try:
        import geopandas as gpd
    except ImportError:
        gpd = MissingOptionalModule("geopandas")


if TYPE_CHECKING:
    import shapely
else:
    try:
        import shapely
    except ImportError:
        shapely = MissingOptionalModule("shapely")


@typedispatch
def _derive_connected_cell_ids(
    idomain: xr.DataArray, grid: xu.Ugrid2d, edge_index: np.ndarray
) -> xr.Dataset:
    """
    Derive the cell ids of the connected cells of an edge on a structured grid.

    Parameters
    ----------
    idomain: xr.DataArray
        The active domain
    grid :
        The unstructured grid of the domain
    edge_index :
        The indices of the edges from which the connected cell ids are computed

    Returns A dataset containing the cell_id1 and cell_id2 data variables. The  cell dimensions are stored in the
    cell_dims coordinates.
    -------

    """
    edge_faces = grid.edge_face_connectivity
    cell2d = edge_faces[edge_index]

    shape = (idomain["y"].size, idomain["x"].size)
    row_1, column_1 = np.unravel_index(cell2d[:, 0], shape)
    row_2, column_2 = np.unravel_index(cell2d[:, 1], shape)

    cell_ids = xr.Dataset()

    cell_ids["cell_id1"] = xr.DataArray(
        np.array([row_1 + 1, column_1 + 1]).T,
        coords={
            "edge_index": edge_index,
            "cell_dims1": ["row_1", "column_1"],
        },
    )

    cell_ids["cell_id2"] = xr.DataArray(
        np.array([row_2 + 1, column_2 + 1]).T,
        coords={
            "edge_index": edge_index,
            "cell_dims2": ["row_2", "column_2"],
        },
    )

    return cell_ids


@typedispatch  # type: ignore[no-redef]
def _derive_connected_cell_ids(
    _: xu.UgridDataArray, grid: xu.Ugrid2d, edge_index: np.ndarray
) -> xr.Dataset:
    """
    Derive the cell ids of the connected cells of an edge on an unstructured grid.

    Parameters
    ----------
    grid :
        The unstructured grid of the domain
    edge_index :
        The indices of the edges from which the connected cell ids are computed

    Returns A dataset containing the cell_id1 and cell_id2 data variables. The  cell dimensions are stored in the
    cell_dims coordinates.
    -------

    """
    edge_faces = grid.edge_face_connectivity
    cell2d = edge_faces[edge_index]

    cell2d_1 = cell2d[:, 0]
    cell2d_2 = cell2d[:, 1]

    cell_ids = xr.Dataset()

    cell_ids["cell_id1"] = xr.DataArray(
        np.array([cell2d_1 + 1]).T,
        coords={
            "edge_index": edge_index,
            "cell_dims1": ["cell2d_1"],
        },
    )

    cell_ids["cell_id2"] = xr.DataArray(
        np.array([cell2d_2 + 1]).T,
        coords={
            "edge_index": edge_index,
            "cell_dims2": ["cell2d_2"],
        },
    )

    return cell_ids


def to_connected_cells_dataset(
    idomain: GridDataArray,
    grid: xu.Ugrid2d,
    edge_index: np.ndarray,
    edge_values: dict,
) -> xr.Dataset:
    """
    Converts a cell edge grid with values defined on the edges to a dataset with the cell ids of the connected cells,
    the layer of the cells and the value of the edge. The connected cells are returned in cellid notation e.g.(row,
    column) for structured grid, (mesh2d_nFaces) for unstructured grids.

    Parameters
    ----------
    idomain: GridDataArray
        active domain
    grid: xu.Ugrid2d,
        unstructured grid containing the edge to cell array
    edge_index: np.ndarray
        indices of the edges for which the edge values will be converted to values in the connected cells
    edge_values: dict
        dictionary containing the value name and the edge values that are applied on the edges identified by the
        edge_index

    Returns
    ----------
        a dataset containing:
            - cell_id1
            - cell_id2
            - layer
            - value name

    """
    barrier_dataset = _derive_connected_cell_ids(idomain, grid, edge_index)

    for name, values in edge_values.items():
        barrier_dataset[name] = xr.DataArray(
            values,
            dims=["layer", "edge_index"],
            coords={
                "edge_index": edge_index,
                "layer": values.coords["layer"],
            },
        )

    barrier_dataset = barrier_dataset.stack(
        cell_id=("layer", "edge_index"), create_index=True
    )

    return barrier_dataset.dropna("cell_id")


def _make_linestring_from_polygon(
    dataframe: GeoDataFrameType,
) -> List[LineStringType]:
    """
    Make linestring from a polygon with one axis in the vertical dimension (z),
    and one axis in the horizontal plane (x & y dimension).
    """
    coordinates, index = shapely.get_coordinates(dataframe.geometry, return_index=True)
    df = pd.DataFrame(
        {"polygon_index": index, "x": coordinates[:, 0], "y": coordinates[:, 1]}
    )
    df = df.drop_duplicates().reset_index(drop=True)
    df = df.set_index("polygon_index")

    linestrings = [
        shapely.LineString(gb.values) for _, gb in df.groupby("polygon_index")
    ]

    return linestrings


def _select_dataframe_with_snapped_line_index(
    snapped_dataset: xr.Dataset, edge_index: np.ndarray, dataframe: GeoDataFrameType
):
    """
    Select dataframe rows with line indices of snapped edges. Usually, the
    broadcasting results in a larger dataframe where individual rows of input
    dataframe are repeated for multiple edges.
    """
    line_index = snapped_dataset["line_index"].values
    line_index = line_index[edge_index].astype(int)
    return dataframe.iloc[line_index]


def _extract_mean_hfb_bounds_from_dataframe(
    dataframe: GeoDataFrameType,
) -> Tuple[pd.Series, pd.Series]:
    """
    Extract hfb bounds from dataframe. Requires dataframe geometry to be of type
    shapely "Z Polygon".

    For the upper z bounds, function takes the average of the depth of the two
    upper nodes. The same holds for the lower z bounds, but then with the two
    lower nodes.

    As a visual representation, this happens for each z bound:

    .                 .
     \         >>>
      \   .    >>>     ---  .
       \ /     >>>         -
        .                 .

    """
    dataframe = _prepare_index_names(dataframe)

    if not dataframe.geometry.has_z.all():
        raise TypeError("GeoDataFrame geometry has no z, which is required.")

    lower, upper = _extract_hfb_bounds_from_zpolygons(dataframe)
    # Compute means inbetween nodes.
    index_names = lower.index.names
    lower_mean = lower.groupby(index_names)["z"].mean()
    upper_mean = upper.groupby(index_names)["z"].mean()

    # Assign to dataframe to map means to right index.
    df_to_broadcast = dataframe.copy()
    df_to_broadcast["lower"] = lower_mean
    df_to_broadcast["upper"] = upper_mean

    return df_to_broadcast["lower"], df_to_broadcast["upper"]


def _fraction_layer_overlap(
    snapped_dataset: xu.UgridDataset,
    edge_index: np.ndarray,
    dataframe: GeoDataFrameType,
    top: xu.UgridDataArray,
    bottom: xu.UgridDataArray,
) -> xr.DataArray:
    """
    Computes the fraction a barrier occupies inside a layer.
    """
    left, right = snapped_dataset.ugrid.grid.edge_face_connectivity[edge_index].T
    top_mean = _mean_left_and_right(top, left, right)
    bottom_mean = _mean_left_and_right(bottom, left, right)

    n_layer, n_edge = top_mean.shape
    layer_bounds = np.empty((n_edge, n_layer, 2), dtype=float)
    layer_bounds[..., 0] = typing.cast(np.ndarray, bottom_mean.values).T
    layer_bounds[..., 1] = typing.cast(np.ndarray, top_mean.values).T

    zmin, zmax = _extract_mean_hfb_bounds_from_dataframe(dataframe)
    hfb_bounds = np.empty((n_edge, n_layer, 2), dtype=float)
    hfb_bounds[..., 0] = zmin.values[:, np.newaxis]
    hfb_bounds[..., 1] = zmax.values[:, np.newaxis]

    overlap = _vectorized_overlap(hfb_bounds, layer_bounds)
    height = layer_bounds[..., 1] - layer_bounds[..., 0]
    # Avoid runtime warnings when diving by 0:
    height[height <= 0] = np.nan
    fraction = (overlap / height).T

    return xr.ones_like(top_mean) * fraction


def _mean_left_and_right(
    cell_values: xu.UgridDataArray, left: np.ndarray, right: np.ndarray
) -> xr.Dataset:
    """
    This method computes the mean value of cell pairs. The left array specifies the first cell, the right array
    the second cells. The mean is computed by (first_cell+second_cell/2.0)

    Parameters
    ----------
    cell_values: xu.UgridDataArray
        The array containing the data values of all the cells
    left :
        The array containing indices to the first cells
    right :
        The array containing indices to the second cells

    Returns
    -------
        The means of the cells

    """
    facedim = cell_values.ugrid.grid.face_dimension
    uda_left = cell_values.ugrid.obj.isel({facedim: left}).drop_vars(facedim)
    uda_right = cell_values.ugrid.obj.isel({facedim: right}).drop_vars(facedim)

    return xr.concat((uda_left, uda_right), dim="two").mean("two")


def _vectorized_overlap(bounds_a: np.ndarray, bounds_b: np.ndarray) -> np.ndarray:
    """
    Vectorized overlap computation. Returns the overlap of 2 vectors along the same axis.
    If there is no overlap zero will be returned.

            b1
            ▲
      a1    |
      ▲     |
      |     |
      |     ▼
      ▼     b0
      a0

    To compute the overlap of the 2 vectors the maximum of a0,b0, is subtracted from the minimum of a1,b1.

    Compare with:

    overlap = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    """
    return np.maximum(
        0.0,
        np.minimum(bounds_a[..., 1], bounds_b[..., 1])
        - np.maximum(bounds_a[..., 0], bounds_b[..., 0]),
    )


def _prepare_barrier_dataset_for_mf6_adapter(dataset: xr.Dataset) -> xr.Dataset:
    """
    Prepare barrier dataset for the initialization of Mf6HorizontalFlowBarrier.
    The dataset is expected to have "edge_index" and "layer" coordinates and a
    multi-index "cell_id" coordinate. The dataset contains as variables:
    "cell_id1", "cell_id2", and "hydraulic_characteristic".

    - Reset coords to get a coordless "cell_id" dimension instead of a multi-index coord
    - Assign "layer" as variable to dataset instead of as coord.
    """
    # Store layer to work around multiindex issue where dropping the edge_index
    # removes the layer as well.
    layer = dataset.coords["layer"].values
    # Drop leftover coordinate and reset cell_id.
    dataset = dataset.drop_vars("edge_index").reset_coords()
    # Attach layer again
    dataset["layer"] = ("cell_id", layer)
    return dataset


def _snap_to_grid_and_aggregate(
    barrier_dataframe: gpd.GeoDataFrame, grid2d: xu.Ugrid2d, vardict_agg: dict[str, str]
) -> tuple[xu.UgridDataset, npt.NDArray]:
    """
    Snap barrier dataframe to grid and aggregate multiple lines with a list of
    methods per variable.

    Parameters
    ----------
    barrier_dataframe: geopandas.GeoDataFrame
        GeoDataFrame with barriers, should have variable "line_index".
    grid2d: xugrid.Ugrid2d
        Grid to snap lines to
    vardict_agg: dict
        Mapping of variable name to aggregation method
    """
    snapping_df = xu.create_snap_to_grid_dataframe(
        barrier_dataframe, grid2d, max_snap_distance=0.5
    )
    # Map other variables to snapping_df with line indices
    line_index = snapping_df["line_index"]
    vars_to_snap = list(vardict_agg.keys())
    vars_to_snap.remove("line_index")  # snapping_df already has line_index
    for varname in vars_to_snap:
        snapping_df[varname] = barrier_dataframe[varname].iloc[line_index].to_numpy()
    # Aggregate to grid edges
    gb_edge = snapping_df.groupby("edge_index")
    # Initialize dataset and dataarray with the right shape and dims
    snapped_dataset = xu.UgridDataset(grids=[grid2d])
    new = xr.DataArray(np.full(grid2d.n_edge, np.nan), dims=[grid2d.edge_dimension])
    edge_index = np.array(list(gb_edge.indices.keys()))
    # Aggregate with different methods per variable
    for varname, method in vardict_agg.items():
        snapped_dataset[varname] = new.copy()
        snapped_dataset[varname].data[edge_index] = gb_edge[varname].aggregate(method)

    return snapped_dataset, edge_index


class BarrierType(Enum):
    HydraulicCharacteristic = 0
    Multiplier = 1
    Resistance = 2


class HorizontalFlowBarrierBase(BoundaryCondition, ILineDataPackage):
    _pkg_id = "hfb"

    _period_data = ()
    _init_schemata = {}
    _write_schemata = {"geometry": [EmptyIndexesSchema()]}

    def __init__(
        self,
        geometry: "gpd.GeoDataFrame",
        print_input: bool = False,
    ) -> None:
        dict_dataset = {"print_input": print_input}
        super().__init__(dict_dataset)
        self.line_data = geometry

    def _get_variable_names_for_gdf(self) -> list[str]:
        return [
            self._get_variable_name(),
            "geometry",
        ] + self._get_vertical_variables()

    @property
    def line_data(self) -> GeoDataFrameType:
        variables_for_gdf = self._get_variable_names_for_gdf()
        return gpd.GeoDataFrame(
            self.dataset[variables_for_gdf].to_dataframe(),
            geometry="geometry",
        )

    @line_data.setter
    def line_data(self, value: GeoDataFrameType) -> None:
        variables_for_gdf = self._get_variable_names_for_gdf()
        self.dataset = self.dataset.merge(
            value.to_xarray(), overwrite_vars=variables_for_gdf, join="right"
        )

    def render(self, directory, pkgname, globaltimes, binary):
        raise NotImplementedError(
            f"""{self.__class__.__name__} is a grid-agnostic package and does not have a render method. To render the
            package, first convert to a Modflow6 package by calling pkg.to_mf6_pkg()"""
        )

    def _netcdf_encoding(self):
        return {"geometry": {"dtype": "str"}}

    @classmethod
    def from_file(cls, path, **kwargs):
        instance = super().from_file(path, **kwargs)
        instance.dataset["geometry"] = shapely.wkt.loads(instance.dataset["geometry"])

        return instance

    def _compute_barrier_values(
        self, snapped_dataset, edge_index, idomain, top, bottom, k
    ):
        raise NotImplementedError()

    def _to_connected_cells_dataset(
        self,
        idomain: GridDataArray,
        top: GridDataArray,
        bottom: GridDataArray,
        k: GridDataArray,
    ) -> xr.Dataset:
        """
        Method does the following
        - forces input grids to unstructured
        - snaps lines to cell edges
        - remove edge values connected to cell edges
        - compute barrier values
        - remove edge values to inactive cells
        - finds connected cells in dataset

        Returns
        -------
        dataset with connected cells, containing:
            - cell_id1
            - cell_id2
            - layer
            - value name
        """
        top, bottom = broadcast_to_full_domain(idomain, top, bottom)
        k = idomain * k
        # Enforce unstructured
        unstructured_grid, top, bottom, k = (enforce_uda(grid) for grid in [idomain, top, bottom, k])
        snapped_dataset, edge_index = self._snap_to_grid(idomain)
        edge_index = self.__remove_invalid_edges(unstructured_grid, edge_index)

        barrier_values = self._compute_barrier_values(
            snapped_dataset, edge_index, idomain, top, bottom, k
        )
        barrier_values = self.__remove_edge_values_connected_to_inactive_cells(
            barrier_values, unstructured_grid, edge_index
        )

        if (barrier_values.size == 0) | np.isnan(barrier_values).all():
            raise ValueError(
                textwrap.dedent(
                    """
                    No barriers could be assigned to cell edges,
                    this is caused by either one of the following:
                    \t- Barriers fall completely outside the model grid
                    \t- Barriers were assigned to the edge of the model domain
                    \t- Barriers were assigned to edges of inactive cells
                    """
                )
            )

        return to_connected_cells_dataset(
            idomain,
            unstructured_grid.ugrid.grid,
            edge_index,
            {
                "hydraulic_characteristic": self.__to_hydraulic_characteristic(
                    barrier_values
                )
            },
        )

    def _to_mf6_pkg(self, barrier_dataset: xr.Dataset) -> Mf6HorizontalFlowBarrier:
        """
        Internal method, which does the following
        - final coordinate cleanup of barrier dataset
        - adds missing options to dataset

        Parameters
        ----------
        barrier_dataset: xr.Dataset
            Xarray dataset with dimensions "cell_dims1", "cell_dims2", "cell_id".
            Additional coordinates should be "layer" and "edge_index".

        Returns
        -------
        Mf6HorizontalFlowBarrier
        """
        barrier_dataset["print_input"] = self.dataset["print_input"]
        barrier_dataset = _prepare_barrier_dataset_for_mf6_adapter(barrier_dataset)
        return Mf6HorizontalFlowBarrier(**barrier_dataset.data_vars)

    def to_mf6_pkg(
        self,
        idomain: GridDataArray,
        top: GridDataArray,
        bottom: GridDataArray,
        k: GridDataArray,
    ) -> Mf6HorizontalFlowBarrier:
        """
        Write package to Modflow 6 package.

        Based on the model grid, top and bottoms, the layers in which the barrier belong are computed. If the
        barrier only partially occupies a layer an effective resistance or hydraulic conductivity for that layer is
        calculated. This calculation is skipped for the Multiplier type.

        Parameters
        ----------
        idomain: GridDataArray
             Grid with active cells.
        top: GridDataArray
            Grid with top of model layers.
        bottom: GridDataArray
            Grid with bottom of model layers.
        k: GridDataArray
            Grid with hydraulic conductivities.

        Returns
        -------
        Mf6HorizontalFlowBarrier
            Low level representation of the HFB package as MODFLOW 6 expects it.
        """
        barrier_dataset = self._to_connected_cells_dataset(idomain, top, bottom, k)
        return self._to_mf6_pkg(barrier_dataset)

    def is_empty(self) -> bool:
        if super().is_empty():
            return True

        linestrings = self.dataset["geometry"]
        only_empty_lines = all(ls.is_empty for ls in linestrings.values.ravel())
        return only_empty_lines

    def _resistance_layer(
        self,
        snapped_dataset: xu.UgridDataset,
        edge_index: np.ndarray,
        idomain: xu.UgridDataArray,
    ) -> xr.DataArray:
        """
        Returns layered xarray with barrier resistance distributed over layers
        """
        hfb_resistance = snapped_dataset[self._get_variable_name()].values[edge_index]
        hfb_layer = snapped_dataset["layer"].values[edge_index]
        nlay = idomain.layer.size
        model_layer = np.repeat(idomain.layer.values, hfb_resistance.size).reshape(
            (nlay, hfb_resistance.size)
        )
        data = np.where(model_layer == hfb_layer, hfb_resistance, np.nan)
        return xr.DataArray(
            data=data,
            coords={
                "layer": np.arange(nlay) + 1,
            },
            dims=["layer", "mesh2d_nFaces"],
        )

    def _resistance_layer_overlap(
        self,
        snapped_dataset: xu.UgridDataset,
        edge_index: np.ndarray,
        top: xu.UgridDataArray,
        bottom: xu.UgridDataArray,
        k: xu.UgridDataArray,
    ) -> xr.DataArray:
        """
        Computes the effective value of a barrier that partially overlaps a cell in the z direction.
        A barrier always lies on an edge in the xy-plane, however in doesn't have to fully extend in the z-direction to
        cover the entire layer. This method computes the effective resistance in that case.

                        Barrier
        ......................................  ▲     ▲
        .                @@@@                .  |     |
        .                @Rb@                .  | Lb  |
        .    Cell1       @@@@     Cell2      .  ▼     | H
        .                :  :                .        |
        .                :  :                .        |
        .................:..:.................        ▼
                k1                    k2

        The effective value of a partially emerged barrier in a layer is computed by:
        c_total = 1.0 / (fraction / Rb + (1.0 - fraction) / c_aquifer)
        c_aquifer = 1.0 / k_mean = 1.0 / ((k1 + k2) / 2.0)
        fraction = Lb / H

        """
        left, right = snapped_dataset.ugrid.grid.edge_face_connectivity[edge_index].T
        k_mean = _mean_left_and_right(k, left, right)

        resistance = self.__to_resistance(
            snapped_dataset[self._get_variable_name()]
        ).values[edge_index]

        dataframe = _select_dataframe_with_snapped_line_index(
            snapped_dataset, edge_index, self.line_data
        )
        fraction = _fraction_layer_overlap(
            snapped_dataset, edge_index, dataframe, top, bottom
        )

        c_aquifer = 1.0 / k_mean
        inverse_c = (fraction / resistance) + ((1.0 - fraction) / c_aquifer)
        c_total = 1.0 / inverse_c

        return self.__from_resistance(c_total)

    def __to_resistance(self, value: xu.UgridDataArray) -> xu.UgridDataArray:
        match self._get_barrier_type():
            case BarrierType.HydraulicCharacteristic:
                return 1.0 / value
            case BarrierType.Multiplier:
                return -1.0 / value
            case BarrierType.Resistance:
                return value

        raise ValueError(r"Unknown barrier type {barrier_type}")

    def __from_resistance(self, resistance: xr.DataArray) -> xr.DataArray:
        match self._get_barrier_type():
            case BarrierType.HydraulicCharacteristic:
                return 1.0 / resistance
            case BarrierType.Multiplier:
                return -1.0 / resistance
            case BarrierType.Resistance:
                return resistance

        raise ValueError(r"Unknown barrier type {barrier_type}")

    def __to_hydraulic_characteristic(self, value: xr.DataArray) -> xr.DataArray:
        match self._get_barrier_type():
            case BarrierType.HydraulicCharacteristic:
                return value
            case BarrierType.Multiplier:
                return -1.0 * value
            case BarrierType.Resistance:
                return 1.0 / value

        raise ValueError(r"Unknown barrier type {barrier_type}")

    @abc.abstractmethod
    def _get_barrier_type(self) -> BarrierType:
        raise NotImplementedError

    @abc.abstractmethod
    def _get_variable_name(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def _get_vertical_variables(self) -> list:
        raise NotImplementedError

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
    ) -> "HorizontalFlowBarrierBase":
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
        cls = type(self)
        new = cls._from_dataset(copy.deepcopy(self.dataset))
        new.line_data = self.line_data
        return new

    def mask(self, _) -> Package:
        """
        The mask method is irrelevant for this package as it is
        grid-agnostic, instead this method retuns a copy of itself.
        """
        return deepcopy(self)

    def _snap_to_grid(
        self, idomain: GridDataArray
    ) -> Tuple[xu.UgridDataset, np.ndarray]:
        variable_name = self._get_variable_name()
        has_layer = "layer" in self._get_vertical_variables()
        # Create geodataframe with barriers
        if has_layer:
            varnames_for_df = [variable_name, "geometry", "layer"]
        else:
            varnames_for_df = [variable_name, "geometry"]
        barrier_dataframe = gpd.GeoDataFrame(
            self.dataset[varnames_for_df].to_dataframe()
        )
        # Convert vertical polygon to linestring
        if not has_layer:
            lower, _ = _extract_hfb_bounds_from_zpolygons(barrier_dataframe)
            linestring = _create_zlinestring_from_bound_df(lower)
            barrier_dataframe["geometry"] = linestring["geometry"]
        # Clip barriers outside domain
        barrier_dataframe = clip_line_gdf_by_grid(
            barrier_dataframe, idomain.sel(layer=1)
        )
        # Prepare variable names and methods for aggregation
        vardict_agg = {"line_index": "first", variable_name: "sum"}
        if has_layer:
            vardict_agg["layer"] = "first"
        # Create grid from structured
        grid2d = enforce_uda(idomain.sel(layer=1)).grid

        return _snap_to_grid_and_aggregate(barrier_dataframe, grid2d, vardict_agg)

    @staticmethod
    def __remove_invalid_edges(
        unstructured_grid: xu.UgridDataArray, edge_index: np.ndarray
    ) -> np.ndarray:
        """
        Remove invalid edges indices. An edge is considered invalid when:
        - it lies on an exterior boundary (face_connectivity equals the grid fill value)
        - The corresponding connected cells are inactive
        """
        grid = unstructured_grid.ugrid.grid
        face_dimension = unstructured_grid.ugrid.grid.face_dimension
        face_connectivity = grid.edge_face_connectivity[edge_index]

        valid_edges = np.all(face_connectivity != grid.fill_value, axis=1)

        connected_cells = -np.ones((len(edge_index), 2))
        connected_cells[valid_edges, 0] = (
            unstructured_grid.sel(layer=1)
            .loc[{face_dimension: face_connectivity[valid_edges, 0]}]
            .values
        )
        connected_cells[valid_edges, 1] = (
            unstructured_grid.sel(layer=1)
            .loc[{face_dimension: face_connectivity[valid_edges, 1]}]
            .values
        )

        valid = (connected_cells > 0).all(axis=1)

        return edge_index[valid]

    @staticmethod
    def __remove_edge_values_connected_to_inactive_cells(
        values, unstructured_grid: xu.UgridDataArray, edge_index: np.ndarray
    ):
        face_dimension = unstructured_grid.ugrid.grid.face_dimension

        face_connectivity = unstructured_grid.ugrid.grid.edge_face_connectivity[
            edge_index
        ]
        connected_cells_left = unstructured_grid.loc[
            {face_dimension: face_connectivity[:, 0]}
        ]
        connected_cells_right = unstructured_grid.loc[
            {face_dimension: face_connectivity[:, 1]}
        ]

        return values.where(
            (connected_cells_left.drop(face_dimension) > 0)
            & (connected_cells_right.drop(face_dimension) > 0)
        )


class HorizontalFlowBarrierHydraulicCharacteristic(HorizontalFlowBarrierBase):
    """
    Horizontal Flow Barrier (HFB) package

    Input to the Horizontal Flow Barrier (HFB) Package is read from the file
    that has type "HFB6" in the Name File. Only one HFB Package can be
    specified for a GWF model.
    https://water.usgs.gov/water-resources/software/MODFLOW-6/mf6io_6.2.2.pdf

    Parameters
    ----------
    geometry: gpd.GeoDataFrame
        Dataframe that describes:
         - geometry: the geometries of the barriers,
         - hydraulic_characteristic: the hydraulic characteristic of the barriers
    print_input: bool

    Examples
    --------

    >>> x = [-10.0, 0.0, 10.0]
    >>> y = [10.0, 0.0, -10.0]
    >>> ztop = [10.0, 20.0, 15.0]
    >>> zbot = [-10.0, -20.0, 0.0]
    >>> polygons = linestring_to_trapezoid_zpolygons(x, y, ztop, zbot)
    >>> barrier_gdf = gpd.GeoDataFrame(
    >>>     geometry=polygons,
    >>>     data={
    >>>         "resistance": [1e-3, 1e-3],
    >>>     },
    >>> )
    >>> hfb = imod.mf6.HorizontalFlowBarrierHydraulicCharacteristic(barrier_gdf)

    """

    @init_log_decorator()
    def __init__(
        self,
        geometry: "gpd.GeoDataFrame",
        print_input=False,
    ):
        super().__init__(geometry, print_input)

    def _get_barrier_type(self):
        return BarrierType.HydraulicCharacteristic

    def _get_variable_name(self) -> str:
        return "hydraulic_characteristic"

    def _get_vertical_variables(self) -> list:
        return []

    def _compute_barrier_values(
        self, snapped_dataset, edge_index, idomain, top, bottom, k
    ):
        barrier_values = self._resistance_layer_overlap(
            snapped_dataset, edge_index, top, bottom, k
        )

        return barrier_values


class SingleLayerHorizontalFlowBarrierHydraulicCharacteristic(
    HorizontalFlowBarrierBase
):
    """
    Horizontal Flow Barrier (HFB) package

    Input to the Horizontal Flow Barrier (HFB) Package is read from the file
    that has type "HFB6" in the Name File. Only one HFB Package can be
    specified for a GWF model.
    https://water.usgs.gov/water-resources/software/MODFLOW-6/mf6io_6.2.2.pdf

    Parameters
    ----------
    geometry: gpd.GeoDataFrame
        Dataframe that describes:
         - geometry: the geometries of the barriers,
         - hydraulic_characteristic: the hydraulic characteristic of the
           barriers
         - layer: model layer for the barrier, only 1 single layer can be
           entered.
    print_input: bool

    Examples
    --------

    >>> barrier_x = [-1000.0, 0.0, 1000.0]
    >>> barrier_y = [500.0, 250.0, 500.0]
    >>> barrier_gdf = gpd.GeoDataFrame(
    >>>     geometry=[shapely.linestrings(barrier_x, barrier_y),],
    >>>     data={
    >>>         "hydraulic_characteristic": [1e-3,],
    >>>         "layer": [1,]
    >>>     },
    >>> )
    >>> hfb = imod.mf6.LayeredHorizontalFlowBarrierHydraulicCharacteristic(barrier_gdf)

    """

    _write_schemata = {
        "geometry": [EmptyIndexesSchema()],
        "layer": [MaxNUniqueValuesSchema(1)],
    }

    @init_log_decorator()
    def __init__(
        self,
        geometry: "gpd.GeoDataFrame",
        print_input=False,
    ):
        super().__init__(geometry, print_input)

    def _get_barrier_type(self):
        return BarrierType.HydraulicCharacteristic

    def _get_variable_name(self) -> str:
        return "hydraulic_characteristic"

    def _get_vertical_variables(self) -> list:
        return ["layer"]

    def _compute_barrier_values(
        self, snapped_dataset, edge_index, idomain, top, bottom, k
    ):
        barrier_values = self._resistance_layer(
            snapped_dataset,
            edge_index,
            idomain,
        )

        return barrier_values


class HorizontalFlowBarrierMultiplier(HorizontalFlowBarrierBase):
    """
    Horizontal Flow Barrier (HFB) package

    Input to the Horizontal Flow Barrier (HFB) Package is read from the file
    that has type "HFB6" in the Name File. Only one HFB Package can be
    specified for a GWF model.
    https://water.usgs.gov/water-resources/software/MODFLOW-6/mf6io_6.2.2.pdf

    If parts of the barrier overlap a layer the multiplier is applied to the entire layer.

    Parameters
    ----------
    geometry: gpd.GeoDataFrame
        Dataframe that describes:
         - geometry: the geometries of the barriers,
         - multiplier: the multiplier of the barriers
    print_input: bool

    Examples
    --------

    >>> x = [-10.0, 0.0, 10.0]
    >>> y = [10.0, 0.0, -10.0]
    >>> ztop = [10.0, 20.0, 15.0]
    >>> zbot = [-10.0, -20.0, 0.0]
    >>> polygons = linestring_to_trapezoid_zpolygons(x, y, ztop, zbot)
    >>> barrier_gdf = gpd.GeoDataFrame(
    >>>     geometry=polygons,
    >>>     data={
    >>>         "multiplier": [10.0, 10.0],
    >>>     },
    >>> )
    >>> hfb = imod.mf6.HorizontalFlowBarrierMultiplier(barrier_gdf)

    """

    @init_log_decorator()
    def __init__(
        self,
        geometry: "gpd.GeoDataFrame",
        print_input=False,
    ):
        super().__init__(geometry, print_input)

    def _get_barrier_type(self):
        return BarrierType.Multiplier

    def _get_variable_name(self) -> str:
        return "multiplier"

    def _get_vertical_variables(self) -> list:
        return []

    def _compute_barrier_values(
        self, snapped_dataset, edge_index, idomain, top, bottom, k
    ):
        dataframe = _select_dataframe_with_snapped_line_index(
            snapped_dataset, edge_index, self.line_data
        )
        fraction = _fraction_layer_overlap(
            snapped_dataset, edge_index, dataframe, top, bottom
        )

        barrier_values = (
            fraction.where(fraction)
            * snapped_dataset[self._get_variable_name()].values[edge_index]
        )

        return barrier_values


class SingleLayerHorizontalFlowBarrierMultiplier(HorizontalFlowBarrierBase):
    """
    Horizontal Flow Barrier (HFB) package

    Input to the Horizontal Flow Barrier (HFB) Package is read from the file
    that has type "HFB6" in the Name File. Only one HFB Package can be
    specified for a GWF model.
    https://water.usgs.gov/water-resources/software/MODFLOW-6/mf6io_6.2.2.pdf

    Parameters
    ----------
    geometry: gpd.GeoDataFrame
        Dataframe that describes:
         - geometry: the geometries of the barriers,
         - multiplier: the multiplier of the barriers
         - layer: model layer for the barrier, only 1 single layer can be
           entered.
    print_input: bool

    Examples
    --------

    >>> barrier_x = [-1000.0, 0.0, 1000.0]
    >>> barrier_y = [500.0, 250.0, 500.0]
    >>> barrier_gdf = gpd.GeoDataFrame(
    >>>     geometry=[shapely.linestrings(barrier_x, barrier_y),],
    >>>     data={
    >>>         "multiplier": [1.5,],
    >>>         "layer": [1,],
    >>>     },
    >>> )
    >>> hfb = imod.mf6.LayeredHorizontalFlowBarrierMultiplier(barrier_gdf)

    """

    _write_schemata = {
        "geometry": [EmptyIndexesSchema()],
        "layer": [MaxNUniqueValuesSchema(1)],
    }

    @init_log_decorator()
    def __init__(
        self,
        geometry: "gpd.GeoDataFrame",
        print_input=False,
    ):
        super().__init__(geometry, print_input)

    def _get_barrier_type(self):
        return BarrierType.Multiplier

    def _get_variable_name(self) -> str:
        return "multiplier"

    def _get_vertical_variables(self) -> list:
        return ["layer"]

    def _compute_barrier_values(
        self, snapped_dataset, edge_index, idomain, top, bottom, k
    ):
        barrier_values = self.__multiplier_layer(snapped_dataset, edge_index, idomain)

        return barrier_values

    def __multiplier_layer(
        self,
        snapped_dataset: xu.UgridDataset,
        edge_index: np.ndarray,
        idomain: xu.UgridDataArray,
    ) -> xr.DataArray:
        """
        Returns layered xarray with barrier multiplier distrubuted over layers
        """
        hfb_multiplier = snapped_dataset[self._get_variable_name()].values[edge_index]
        hfb_layer = snapped_dataset["layer"].values[edge_index]
        nlay = idomain.layer.size
        model_layer = np.repeat(idomain.layer.values, hfb_multiplier.shape[0]).reshape(
            (nlay, hfb_multiplier.shape[0])
        )
        data = np.where(model_layer == hfb_layer, hfb_multiplier, np.nan)
        return xr.DataArray(
            data=data,
            coords={
                "layer": np.arange(nlay) + 1,
            },
            dims=["layer", "mesh2d_nFaces"],
        )


class HorizontalFlowBarrierResistance(HorizontalFlowBarrierBase):
    """
    Horizontal Flow Barrier (HFB) package

    Input to the Horizontal Flow Barrier (HFB) Package is read from the file
    that has type "HFB6" in the Name File. Only one HFB Package can be
    specified for a GWF model.
    https://water.usgs.gov/water-resources/software/MODFLOW-6/mf6io_6.2.2.pdf

    Parameters
    ----------
    geometry: gpd.GeoDataFrame
        Dataframe that describes:
         - geometry: the geometries of the barriers,
         - resistance: the resistance of the barriers

    print_input: bool

    Examples
    --------

    >>> x = [-10.0, 0.0, 10.0]
    >>> y = [10.0, 0.0, -10.0]
    >>> ztop = [10.0, 20.0, 15.0]
    >>> zbot = [-10.0, -20.0, 0.0]
    >>> polygons = linestring_to_trapezoid_zpolygons(x, y, ztop, zbot)
    >>> barrier_gdf = gpd.GeoDataFrame(
    >>>     geometry=polygons,
    >>>     data={
    >>>         "resistance": [1e3, 1e3],
    >>>     },
    >>> )
    >>> hfb = imod.mf6.HorizontalFlowBarrierResistance(barrier_gdf)


    """

    @init_log_decorator()
    def __init__(
        self,
        geometry: "gpd.GeoDataFrame",
        print_input=False,
    ):
        super().__init__(geometry, print_input)

    def _get_barrier_type(self):
        return BarrierType.Resistance

    def _get_variable_name(self) -> str:
        return "resistance"

    def _get_vertical_variables(self) -> list:
        return []

    def _compute_barrier_values(
        self, snapped_dataset, edge_index, idomain, top, bottom, k
    ):
        barrier_values = self._resistance_layer_overlap(
            snapped_dataset, edge_index, top, bottom, k
        )

        return barrier_values


class SingleLayerHorizontalFlowBarrierResistance(HorizontalFlowBarrierBase):
    """
    Horizontal Flow Barrier (HFB) package

    Input to the Horizontal Flow Barrier (HFB) Package is read from the file
    that has type "HFB6" in the Name File. Only one HFB Package can be
    specified for a GWF model.
    https://water.usgs.gov/water-resources/software/MODFLOW-6/mf6io_6.2.2.pdf

    Parameters
    ----------
    geometry: gpd.GeoDataFrame
        Dataframe that describes:
         - geometry: the geometries of the barriers,
         - resistance: the resistance of the barriers
         - layer: model layer for the barrier, only 1 single layer can be
           entered.
    print_input: bool

    Examples
    --------

    >>> barrier_x = [-1000.0, 0.0, 1000.0]
    >>> barrier_y = [500.0, 250.0, 500.0]
    >>> barrier_gdf = gpd.GeoDataFrame(
    >>>     geometry=[shapely.linestrings(barrier_x, barrier_y),],
    >>>     data={
    >>>         "resistance": [1e3,],
    >>>         "layer": [2,],
    >>>     },
    >>> )
    >>> hfb = imod.mf6.LayeredHorizontalFlowBarrierResistance(barrier_gdf)


    """

    _write_schemata = {
        "geometry": [EmptyIndexesSchema()],
        "layer": [MaxNUniqueValuesSchema(1)],
    }

    @init_log_decorator()
    def __init__(
        self,
        geometry: "gpd.GeoDataFrame",
        print_input=False,
    ):
        super().__init__(geometry, print_input)

    def _get_barrier_type(self):
        return BarrierType.Resistance

    def _get_variable_name(self) -> str:
        return "resistance"

    def _get_vertical_variables(self) -> list:
        return ["layer"]

    def _compute_barrier_values(
        self, snapped_dataset, edge_index, idomain, top, bottom, k
    ):
        barrier_values = self._resistance_layer(
            snapped_dataset,
            edge_index,
            idomain,
        )
        return barrier_values

    @classmethod
    def from_imod5_dataset(
        cls, key: str, imod5_data: Dict[str, Dict[str, GridDataArray]]
    ):
        imod5_keys = list(imod5_data.keys())
        if key not in imod5_keys:
            raise ValueError("hfb key not present.")

        hfb_dict = imod5_data[key]
        if not list(hfb_dict.keys()) == ["geodataframe", "layer"]:
            raise ValueError("hfb is not a SingleLayerHorizontalFlowBarrierResistance")
        layer = hfb_dict["layer"]
        if layer == 0:
            raise ValueError(
                "assigning to layer 0 is not supported for "
                "SingleLayerHorizontalFlowBarrierResistance. "
                "Try HorizontalFlowBarrierResistance class."
            )
        geometry_layer = hfb_dict["geodataframe"]
        geometry_layer["layer"] = layer

        return cls(geometry_layer)
