import abc
import typing
from typing import Tuple

import geopandas as gpd
import numpy as np
import xarray as xr
import xugrid as xu

from imod.mf6.boundary_condition import BoundaryCondition
from imod.mf6.mf6_hfb_adapter import BarrierType, Mf6HorizontalFlowBarrier
from imod.typing.grid import GridDataArray, zeros_like


class HorizontalFlowBarrier(BoundaryCondition, abc.ABC):
    _pkg_id = "hfb"

    _period_data = ()
    _init_schemata = {}
    _write_schemata = {}

    _regrid_method = {}

    def __init__(
        self,
        geometry: gpd.GeoDataFrame,
        top: GridDataArray,
        bottom: GridDataArray,
        print_input: bool = False,
    ) -> None:
        super().__init__(locals())
        self.dataset["top"] = top
        self.dataset["bottom"] = bottom
        self.dataset["print_input"] = print_input

        self.dataset = self.dataset.merge(geometry.to_xarray())

    def to_mf6_pkg(
        self,
        active: GridDataArray,
        top: GridDataArray,
        bottom: GridDataArray,
        k: GridDataArray,
    ) -> Mf6HorizontalFlowBarrier:
        top, bottom, k = self.__broadcast_to_full_domain(active, top, bottom, k)
        top, bottom, k = (
            self.__to_unstructured(top, bottom, k)
            if isinstance(active, xr.DataArray)
            else [top, bottom, k]
        )
        snapped_dataset, edge_index = self.snap_to_grid(active)

        effective_value = self.effective_value(
            snapped_dataset, edge_index, top, bottom, k
        )

        barrier_values = (
            xr.ones_like(top["layer"]) * snapped_dataset[self._get_variable_name()]
        )
        barrier_values.values[:, edge_index] = effective_value

        return Mf6HorizontalFlowBarrier(
            self._get_barrier_type(),
            barrier_values,
            active,
            self.dataset["print_input"].values.item(),
        )

    def effective_value(
        self,
        snapped_dataset: xu.UgridDataset,
        edge_index: np.ndarray,
        top: xu.UgridDataArray,
        bottom: xu.UgridDataArray,
        k: xu.UgridDataArray,
    ) -> xr.DataArray:
        left, right = snapped_dataset.ugrid.grid.edge_face_connectivity[edge_index].T
        top_mean = HorizontalFlowBarrier.mean_left_and_right(top, left, right)
        bottom_mean = HorizontalFlowBarrier.mean_left_and_right(bottom, left, right)
        k_mean = HorizontalFlowBarrier.mean_left_and_right(k, left, right)

        n_layer, n_edge = top_mean.shape
        layer_bounds = np.empty((n_edge, n_layer, 2), dtype=float)
        layer_bounds[..., 0] = bottom_mean.values.T
        layer_bounds[..., 1] = top_mean.values.T

        hfb_bounds = np.empty((n_edge, n_layer, 2), dtype=float)
        hfb_bounds[..., 0] = (
            snapped_dataset["zbottom"].values[edge_index].reshape(n_edge, 1)
        )
        hfb_bounds[..., 1] = (
            snapped_dataset["ztop"].values[edge_index].reshape(n_edge, 1)
        )

        overlap = HorizontalFlowBarrier.vectorized_overlap(hfb_bounds, layer_bounds)
        height = layer_bounds[..., 1] - layer_bounds[..., 0]
        # Avoid runtime warnings when diving by 0:
        height[height <= 0] = np.nan
        fraction = (overlap / height).T

        resistance = self.to_resistance(
            snapped_dataset[self._get_variable_name()]
        ).values[edge_index]
        c_aquifer = 1.0 / k_mean
        inverse_c = (fraction / resistance) + ((1.0 - fraction) / c_aquifer)
        c_total = 1.0 / inverse_c

        return self.from_resistance(c_total)

    def to_resistance(self, value: xu.UgridDataArray) -> xu.UgridDataArray:
        match self._get_barrier_type():
            case BarrierType.HydraulicCharacteristic:
                return 1.0 / value
            case BarrierType.Multiplier:
                return -1.0 / value
            case BarrierType.Resistance:
                return value

        raise ValueError(r"Unknown barrier type {barrier_type}")

    def from_resistance(self, resistance: xr.DataArray) -> xr.DataArray:
        match self._get_barrier_type():
            case BarrierType.HydraulicCharacteristic:
                return 1.0 / resistance
            case BarrierType.Multiplier:
                return -1.0 / resistance
            case BarrierType.Resistance:
                return resistance

        raise ValueError(r"Unknown barrier type {barrier_type}")

    @staticmethod
    def mean_left_and_right(
        uda: xu.UgridDataArray, left: np.ndarray, right: np.ndarray
    ) -> xr.DataArray:
        facedim = uda.ugrid.grid.face_dimension
        uda_left = uda.ugrid.obj.isel({facedim: left}).drop_vars(facedim)
        uda_right = uda.ugrid.obj.isel({facedim: right}).drop_vars(facedim)
        return xr.concat((uda_left, uda_right), dim="two").mean("two")

    @staticmethod
    def vectorized_overlap(bounds_a: np.ndarray, bounds_b: np.ndarray):
        """
        Vectorized overlap computation.

        Compare with:

        overlap = max(0, min(a[1], b[1]) - max(a[0], b[0]))
        """
        return np.maximum(
            0.0,
            np.minimum(bounds_a[..., 1], bounds_b[..., 1])
            - np.maximum(bounds_a[..., 0], bounds_b[..., 0]),
        )

    @abc.abstractmethod
    def _get_barrier_type(self) -> BarrierType:
        raise NotImplementedError

    @abc.abstractmethod
    def _get_variable_name(self) -> str:
        raise NotImplementedError

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
        state_for_boundary=None,
    ) -> "HorizontalFlowBarrier":
        cls = type(self)
        new = cls.__new__(cls)
        new.dataset = self.dataset
        return new

    @staticmethod
    def __broadcast_to_full_domain(
        idomain: GridDataArray,
        top: GridDataArray,
        bottom: GridDataArray,
        k: GridDataArray,
    ) -> typing.Tuple[GridDataArray, GridDataArray, GridDataArray]:
        bottom = idomain * bottom
        top = (
            idomain * top
            if hasattr(top, "coords") and "layer" in top.coords
            else HorizontalFlowBarrier.__create_top(bottom, top)
        )
        k = idomain * k

        return top, bottom, k

    @staticmethod
    def __create_top(bottom: GridDataArray, top: GridDataArray) -> GridDataArray:
        new_top = zeros_like(bottom)
        new_top[0] = top
        new_top[1:] = bottom[0:-1].values

        return new_top

    @staticmethod
    def __to_unstructured(
        top: xr.DataArray, bottom: xr.DataArray, k: xr.DataArray
    ) -> Tuple[xu.UgridDataArray, xu.UgridDataArray, xu.UgridDataArray]:
        top = xu.UgridDataArray.from_structured(top)
        bottom = xu.UgridDataArray.from_structured(bottom)
        k = xu.UgridDataArray.from_structured(k)

        return top, bottom, k

    def snap_to_grid(
        self, idomain: GridDataArray
    ) -> Tuple[xu.UgridDataset, np.ndarray]:
        barrier_dataframe = self.dataset[
            [self._get_variable_name(), "geometry", "ztop", "zbottom"]
        ].to_dataframe()

        snapped_dataset, _ = xu.snap_to_grid(
            barrier_dataframe, grid=idomain, max_snap_distance=0.5
        )
        edge_index = np.argwhere(
            snapped_dataset[self._get_variable_name()].notnull().values
        ).ravel()

        return snapped_dataset, edge_index


class HorizontalFlowBarrierHydraulicCharacteristic(HorizontalFlowBarrier):
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
         - the geometries of the barriers,
         - the multiplier of the barriers
         - the top z-value of the barriers
         - the bottom z-value of the barriers
    top: GridDataArray
        DataArray that specifies the top z-value of the top layer
    bottom: GridDataArray
        DataArray that specifies the bottom z-value of all layers
    print_input: bool

    Examples
    --------

    >> barrier_x = [-1000.0, 0.0, 1000.0]
    >> barrier_y = [500.0, 250.0, 500.0]
    >> barrier_gdf = gpd.GeoDataFrame(
    >>     geometry=[shapely.linestrings(barrier_x, barrier_y),],
    >>     data={
    >>         "hydraulic_characteristic": [1e-3,],
    >>         "ztop": [10.0,],
    >>         "zbottom": [0.0,],
    >>     },
    >> )
    >>
    >> hfb = imod.mf6.HorizontalFlowBarrierResistance(barrier_gdf, top, bottom)

    """

    def __init__(
        self,
        geometry: gpd.GeoDataFrame,
        top: GridDataArray,
        bottom: GridDataArray,
        print_input=False,
    ):
        super().__init__(geometry, top, bottom, print_input)

    def _get_barrier_type(self):
        return BarrierType.HydraulicCharacteristic

    def _get_variable_name(self) -> str:
        return "hydraulic_characteristic"


class HorizontalFlowBarrierMultiplier(HorizontalFlowBarrier):
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
         - the geometries of the barriers,
         - the multiplier of the barriers
         - the top z-value of the barriers
         - the bottom z-value of the barriers
    top: GridDataArray
        DataArray that specifies the top z-value of the top layer
    bottom: GridDataArray
        DataArray that specifies the bottom z-value of all layers
    print_input: bool

    Examples
    --------

    >> barrier_x = [-1000.0, 0.0, 1000.0]
    >> barrier_y = [500.0, 250.0, 500.0]
    >> barrier_gdf = gpd.GeoDataFrame(
    >>     geometry=[shapely.linestrings(barrier_x, barrier_y),],
    >>     data={
    >>         "multiplier": [1.5,],
    >>         "ztop": [10.0,],
    >>         "zbottom": [0.0,],
    >>     },
    >> )
    >>
    >> hfb = imod.mf6.HorizontalFlowBarrierResistance(barrier_gdf, top, bottom)

    """

    def __init__(
        self,
        geometry: gpd.GeoDataFrame,
        top: GridDataArray,
        bottom: GridDataArray,
        print_input=False,
    ):
        super().__init__(geometry, top, bottom, print_input)

    def _get_barrier_type(self):
        return BarrierType.Multiplier

    def _get_variable_name(self) -> str:
        return "multiplier"


class HorizontalFlowBarrierResistance(HorizontalFlowBarrier):
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
         - the geometries of the barriers,
         - the resistance of the barriers
         - the top z-value of the barriers
         - the bottom z-value of the barriers
    top: GridDataArray
        DataArray that specifies the top z-value of the top layer
    bottom: GridDataArray
        DataArray that specifies the bottom z-value of all layers
    print_input: bool

    Examples
    --------

    >> barrier_x = [-1000.0, 0.0, 1000.0]
    >> barrier_y = [500.0, 250.0, 500.0]
    >> barrier_gdf = gpd.GeoDataFrame(
    >>     geometry=[shapely.linestrings(barrier_x, barrier_y),],
    >>     data={
    >>         "resistance": [1e3,],
    >>         "ztop": [10.0,],
    >>         "zbottom": [0.0,],
    >>     },
    >> )
    >>
    >> hfb = imod.mf6.HorizontalFlowBarrierResistance(barrier_gdf, top, bottom)


    """

    def __init__(
        self,
        geometry: gpd.GeoDataFrame,
        top: GridDataArray,
        bottom: GridDataArray,
        print_input=False,
    ):
        super().__init__(geometry, top, bottom, print_input)

    def _get_barrier_type(self):
        return BarrierType.Resistance

    def _get_variable_name(self) -> str:
        return "resistance"
