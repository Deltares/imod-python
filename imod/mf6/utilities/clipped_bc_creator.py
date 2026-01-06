from typing import Optional, Tuple, TypeAlias

import xarray as xr

from imod.mf6 import ConstantConcentration, ConstantHead
from imod.select.grid import active_grid_boundary_xy
from imod.typing import GridDataArray
from imod.util.dims import enforced_dim_order

StateType: TypeAlias = ConstantHead | ConstantConcentration
StateClassType: TypeAlias = type[ConstantHead] | type[ConstantConcentration]


def _find_unassigned_grid_boundaries(
    active_grid_boundary: GridDataArray,
    boundary_conditions: list[StateType],
) -> GridDataArray:
    unassigned_grid_boundaries = active_grid_boundary.copy()
    for boundary_condition in boundary_conditions:
        # Fetch variable name from the first boundary condition, can be "head" or
        # "concentration".
        varname = boundary_condition._period_data[0]
        unassigned_grid_boundaries = (
            unassigned_grid_boundaries & boundary_condition[varname].isnull()
        )

    return unassigned_grid_boundaries


def _align_time_indexes_boundaries(
    state_for_clipped_boundary: GridDataArray,
    unassigned_grid_boundaries: GridDataArray,
) -> Optional[GridDataArray]:
    """
    Create an outer time index for aligning boundaries. Furthermore deal with
    cases where one or both boundaries don't have a time dimension. In a graphic
    way, we want this:

    State
       a-----b-----c
    Unassigned
    d-----e

    Needs to align to:
    d---a--e--b-----c
    """
    index_unassigned = unassigned_grid_boundaries.indexes
    index_state = state_for_clipped_boundary.indexes
    if "time" in index_unassigned and "time" in index_state:
        return index_unassigned["time"].join(index_state["time"], how="outer")
    elif "time" in index_unassigned:
        return index_unassigned["time"]
    elif "time" in index_state:
        return index_state["time"]
    else:
        return None


def _align_boundaries(
    state_for_clipped_boundary: GridDataArray,
    unassigned_grid_boundaries: Optional[GridDataArray],
) -> Tuple[GridDataArray, Optional[GridDataArray]]:
    """
    Customly align the state_for_clipped_boundary and unassigned grid boundaries.
    - "layer" dimension requires outer alignment
    - "time" requires reindexing with ffill
    - planar coordinates are expected to be aligned already
    """
    # Align dimensions
    if unassigned_grid_boundaries is not None:
        # Align along layer dimension with outer join, xarray API only supports
        # excluding dims, not specifying dims to align along, so we have to do
        # it this way.
        dims_to_exclude = set(state_for_clipped_boundary.dims) | set(
            unassigned_grid_boundaries.dims
        )
        dims_to_exclude.remove("layer")
        state_for_clipped_boundary, unassigned_grid_boundaries = xr.align(
            state_for_clipped_boundary,
            unassigned_grid_boundaries,
            join="outer",
            exclude=dims_to_exclude,
        )
        # Align along time dimension by finding the outer time indexes and then
        # reindexing with a ffill.
        outer_time_index = _align_time_indexes_boundaries(
            state_for_clipped_boundary, unassigned_grid_boundaries
        )
        if "time" in state_for_clipped_boundary.indexes:
            state_for_clipped_boundary = state_for_clipped_boundary.reindex(
                {"time": outer_time_index}, method="ffill"
            )
        if "time" in unassigned_grid_boundaries.indexes:
            unassigned_grid_boundaries = unassigned_grid_boundaries.reindex(
                {"time": outer_time_index}, method="ffill"
            )

    return state_for_clipped_boundary, unassigned_grid_boundaries


@enforced_dim_order
def _create_clipped_boundary_state(
    idomain: GridDataArray,
    state_for_clipped_boundary: GridDataArray,
    original_constant_head_boundaries: list[StateType],
):
    """Helper function to make sure dimension order is enforced"""
    active_grid_boundary = active_grid_boundary_xy(idomain > 0)
    unassigned_grid_boundaries = _find_unassigned_grid_boundaries(
        active_grid_boundary, original_constant_head_boundaries
    )

    state_for_clipped_boundary, unassigned_grid_boundaries = _align_boundaries(
        state_for_clipped_boundary, unassigned_grid_boundaries
    )

    return state_for_clipped_boundary.where(unassigned_grid_boundaries)


def create_clipped_boundary(
    idomain: GridDataArray,
    state_for_clipped_boundary: GridDataArray,
    original_constant_head_boundaries: list[StateType],
    pkg_type: StateClassType,
) -> StateType:
    """
    Create a ConstantHead/ConstantConcentration package on boundary cells that
    don't have any assigned to them. This is useful in combination with the
    clip_box method which can produce a domain with missing boundary conditions.

    Parameters
    ----------
    idomain:
        The clipped domain
    state_for_clipped_boundary :
        The values to be assigned to the created
        ConstantHead/ConstantConcentration package
    original_constant_head_boundaries :
        List of existing ConstantHead/ConstantConcentration boundaries

    Returns
    -------
        ConstantHead/ConstantConcentration package providing values for boundary
        cells that are not covered by other ConstantHead/ConstantConcentration
        packages

    """
    constant_state = _create_clipped_boundary_state(
        idomain,
        state_for_clipped_boundary,
        original_constant_head_boundaries,
    )

    return pkg_type(constant_state, print_input=True, print_flows=True, save_flows=True)
