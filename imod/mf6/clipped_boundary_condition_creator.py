import imod
from imod.mf6 import ConstantHead
from imod.select.grid import active_grid_boundary_xy
from imod.typing import GridDataArray


def create_clipped_boundary(
    idomain: GridDataArray,
    state_for_clipped_boundary: GridDataArray,
    original_constant_head_boundaries: list[ConstantHead],
) -> ConstantHead:
    """
    Create a ConstantHead package on boundary cells that don't have any assigned to them. This is useful in
    combination with the clip_box method which can produce a domain with missing boundary conditions.

    Parameters
    ----------
    idomain:
        The clipped domain
    state_for_clipped_boundary :
        The values to be assigned to the created ConstantHead package
    original_constant_head_boundaries :
        List of existing ConstantHead boundaries

    Returns
    -------
        ConstantHead package providing values for boundary cells that are not
        covered by other ConstantHead packages

    """
    active_grid_boundary = active_grid_boundary_xy(idomain > 0)
    unassigned_grid_boundaries = _find_unassigned_grid_boundaries(
        active_grid_boundary, original_constant_head_boundaries
    )

    constant_head = state_for_clipped_boundary.where(unassigned_grid_boundaries)

    return imod.mf6.ConstantHead(
        constant_head, print_input=True, print_flows=True, save_flows=True
    )


def _find_unassigned_grid_boundaries(
    active_grid_boundary: GridDataArray,
    boundary_conditions: list[ConstantHead],
) -> GridDataArray:
    unassigned_grid_boundaries = active_grid_boundary
    for boundary_condition in boundary_conditions:
        unassigned_grid_boundaries = (
            unassigned_grid_boundaries & boundary_condition["head"].isnull()
        )

    return unassigned_grid_boundaries
