from typing import Any, Optional, Tuple, TypeAlias, cast

import xarray as xr

from imod.common.interfaces.imodel import IModel
from imod.common.utilities.clip import clip_box_dataset
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


def _create_clipped_boundary_pkg(
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


def _create_boundary_condition_for_unassigned_boundary(
    model: IModel,
    state_for_boundary: Optional[GridDataArray],
    additional_boundaries: list[Optional[StateType]] = [None],
) -> Optional[StateType]:
    if state_for_boundary is None:
        return None

    pkg_type = cast(StateClassType, model._boundary_state_pkg_type)
    constant_state_packages = [
        pkg for _, pkg in model.items() if isinstance(pkg, pkg_type)
    ]

    filtered_boundaries: list[StateType] = [
        item for item in additional_boundaries or [] if item is not None
    ]

    constant_state_packages.extend(filtered_boundaries)

    return _create_clipped_boundary_pkg(
        model.domain, state_for_boundary, constant_state_packages, pkg_type
    )


def create_boundary_condition_clipped_boundary(
    original_model: IModel,
    clipped_model: IModel,
    state_for_boundary: Optional[GridDataArray],
    clip_box_args: tuple[Any, ...],
) -> Optional[StateType]:
    """
    Create a clipped boundary condition for a given state in the clipped model.
    The function takes the original model as a reference to determine where
    boundary conditions should NOT be placed, then applies this information to
    create the boundary condition in the clipped model.

    Parameters
    ----------
    original_model : IModel
        The original model containing the unassigned boundary condition.
    clipped_model : IModel
        The clipped model where the boundary condition will be applied.
    state_for_boundary : Optional[GridDataArray]
        The state array for the boundary condition.
    clip_box_args : tuple[Any, ...]
        Arguments defining the clipping box.

    Returns
    -------
    Optional[StateType]
        The clipped boundary condition package, or None if no boundary condition is created.
    """
    # Create temporary boundary condition for the original model boundary. This
    # is used later to see which boundaries can be ignored as they were already
    # present in the original model. We want to just end up with the boundary
    # created by the clip.
    unassigned_boundary_original_domain = (
        _create_boundary_condition_for_unassigned_boundary(
            original_model, state_for_boundary
        )
    )
    # Clip the unassigned boundary to the clipped model's domain, required to
    # avoid topological errors later.
    if unassigned_boundary_original_domain is not None:
        unassigned_boundary_clipped = unassigned_boundary_original_domain.clip_box(
            *clip_box_args
        )
    else:
        unassigned_boundary_clipped = None

    if state_for_boundary is not None:
        # Clip box as dataset, temporarily add variable name to convert to
        # dataset, then turn back into DataArray.
        state_cls = cast(StateClassType, original_model._boundary_state_pkg_type)
        varname = state_cls._period_data[0]
        state_for_boundary = state_for_boundary.to_dataset(name=varname)
        state_for_boundary_clipped = clip_box_dataset(
            state_for_boundary, *clip_box_args
        )[varname]
    else:
        state_for_boundary_clipped = None

    bc_constant_pkg = _create_boundary_condition_for_unassigned_boundary(
        clipped_model, state_for_boundary_clipped, [unassigned_boundary_clipped]
    )

    # Remove all indices before first timestep of state_for_clipped_boundary.
    # This to prevent empty dataarrays unnecessarily being made for these
    # indices, which can lead to them to be removed when purging empty packages
    # with ignore_time=True. Unfortunately, this is needs to be handled here and
    # not in _create_boundary_condition_for_unassigned_boundary, as otherwise
    # this function is called twice which could result in broadcasting errors in
    # the second call if the time domain of state_for_boundary and assigned
    # packages have no overlap.
    if (
        (state_for_boundary is not None)
        and (state_for_boundary.indexes.get("time") is not None)
        and (bc_constant_pkg is not None)
    ):
        start_time = state_for_boundary.indexes["time"][0]
        bc_constant_pkg.dataset = bc_constant_pkg.dataset.sel(
            time=slice(start_time, None)
        )

    return bc_constant_pkg
