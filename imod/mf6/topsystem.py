import abc
from copy import deepcopy
from dataclasses import asdict
from typing import Optional, Self, cast

from imod.common.utilities.dataclass_type import DataclassType
from imod.mf6.aggregate.aggregate_schemes import EmptyAggregationMethod
from imod.mf6.boundary_condition import BoundaryCondition
from imod.mf6.dis import StructuredDiscretization
from imod.mf6.disv import VerticesDiscretization
from imod.mf6.npf import NodePropertyFlow
from imod.prepare.topsystem import (
    ALLOCATION_OPTION,
    DISTRIBUTING_OPTION,
    SimulationAllocationOptions,
    SimulationDistributingOptions,
)
from imod.typing import GridDataDict, GridDataset


def _handle_reallocate_arguments(
    pkg_id: str,
    has_conductance: bool,
    npf: Optional[NodePropertyFlow],
    allocation_option: Optional[ALLOCATION_OPTION],
    distributing_option: Optional[DISTRIBUTING_OPTION],
) -> tuple[ALLOCATION_OPTION, Optional[DISTRIBUTING_OPTION]]:
    if allocation_option is None:
        allocation_option = asdict(SimulationAllocationOptions())[pkg_id]
    elif allocation_option == ALLOCATION_OPTION.stage_to_riv_bot_drn_above:
        raise ValueError(
            f"Allocation option {allocation_option} is not supported for "
            "reallocation of boundary conditions."
        )
    if has_conductance and distributing_option is None:
        distributing_option = asdict(SimulationDistributingOptions())[pkg_id]
    if has_conductance and npf is None:
        raise ValueError(
            "NodePropertyFlow must be provided for packages with conductance variable."
        )
    return allocation_option, distributing_option


class TopSystemBoundaryCondition(BoundaryCondition, abc.ABC):
    """
    Base class to add some extra functionality for topsystem packages, such as
    RCH, DRN, RIV, and GHB.
    """

    _aggregate_method: DataclassType = EmptyAggregationMethod()

    def reallocate(
        self,
        dis: StructuredDiscretization | VerticesDiscretization,
        npf: Optional[NodePropertyFlow] = None,
        allocation_option: Optional[ALLOCATION_OPTION] = None,
        distributing_option: Optional[DISTRIBUTING_OPTION] = None,
    ) -> Self:
        """
        Reallocates topsystem data across layers and create new package with it.
        Aggregate data to planar data first, by taking either the mean for state
        variables (e.g. river stage), or the sum for fluxes and the
        conductance. Consequently allocate and distribute the planar data to the
        provided model layer schematization.

        Parameters
        ----------
        dis : StructuredDiscretization | VerticesDiscretization
            The discretization of the model to which the data should be
            reallocated.
        npf : NodePropertyFlow, optional
            The node property flow package of the model to which the conductance
            should be distributed (if applicable). Required for packages with a
            conductance variable.
        allocation_option : ALLOCATION_OPTION, optional
            The allocation option to use for the reallocation. If None, the
            default allocation option is taken from
            :class:`imod.prepare.SimulationAllocationOptions`.
        distributing_option : DISTRIBUTING_OPTION, optional
            The distributing option to use for the reallocation. Required for
            packages with a conductance variable. If None, the default is taken
            from :class:`imod.prepare.SimulationDistributingOptions`.

        Returns
        -------
        BoundaryCondition
            A new instance of the boundary condition class with the reallocated
            data. The original instance remains unchanged.
        """
        # Handle input arguments
        has_conductance = "conductance" in self.dataset.data_vars
        allocation_option, distributing_option = _handle_reallocate_arguments(
            self._pkg_id, has_conductance, npf, allocation_option, distributing_option
        )
        # Aggregate data to planar data first
        planar_data = self.aggregate_layers(self.dataset)
        # Then allocate and distribute the planar data to the model layers
        if has_conductance:
            npf = cast(NodePropertyFlow, npf)
            distributing_option = cast(DISTRIBUTING_OPTION, distributing_option)
            grid_dict = self._allocate_and_distribute_planar_data(
                planar_data, dis, npf, allocation_option, distributing_option
            )
        else:
            grid_dict = self._allocate_planar_data(planar_data, dis, allocation_option)
        # River package returns a tuple (second argument can also be Drainage
        # package)
        if isinstance(grid_dict, tuple):
            grid_dict, _ = grid_dict
        options = self._get_unfiltered_pkg_options({})
        data_dict = grid_dict | options
        return self.__class__(**data_dict)

    @classmethod
    def _allocate_and_distribute_planar_data(
        cls,
        planar_data: GridDataDict,
        dis: StructuredDiscretization | VerticesDiscretization,
        npf: NodePropertyFlow,
        allocation_option: ALLOCATION_OPTION,
        distributing_option: DISTRIBUTING_OPTION,
    ) -> tuple[GridDataDict, GridDataDict] | GridDataDict:
        raise NotImplementedError(
            "This method should be implemented in the specific boundary condition "
            "class that inherits from BoundaryCondition."
        )

    @classmethod
    def _allocate_planar_data(
        cls,
        planar_data: GridDataDict,
        dis: StructuredDiscretization | VerticesDiscretization,
        allocation_option: ALLOCATION_OPTION,
    ) -> tuple[GridDataDict, GridDataDict] | GridDataDict:
        raise NotImplementedError(
            "This method should be implemented in the specific boundary condition "
            "class that inherits from BoundaryCondition."
        )

    @classmethod
    def _get_aggregate_methods(cls) -> DataclassType:
        """
        Returns the aggregation methods used for aggregating data over layers
        into planar data.

        Returns
        -------
        DataclassType
            The aggregation methods used for the package.
        """
        return deepcopy(cls._aggregate_method)

    @classmethod
    def aggregate_layers(cls, dataset: GridDataset) -> GridDataDict:
        """
        Aggregate data over layers into planar dataset.

        Returns
        -------
        dict
            Dict of aggregated data arrays, where the keys are the variable
            names and the values are aggregated across the "layer" dimension.
        """
        aggr_methods = cls._get_aggregate_methods()
        if isinstance(aggr_methods, EmptyAggregationMethod):
            raise TypeError(
                f"Aggregation methods for {cls._pkg_id} package are not defined."
            )
        aggr_methods_dict = asdict(aggr_methods)
        planar_data = {
            key: dataset[key].reduce(func, dim="layer")
            for key, func in aggr_methods_dict.items()
            if key in dataset.data_vars
        }
        return planar_data
