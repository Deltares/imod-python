from typing import List

import numpy as np
from xugrid.core.wrap import UgridDataArray

import imod
from imod.mf6 import ConstantHead
from imod.select.grid import active_grid_boundary_xy


class ClippedBoundaryConditionCreator:
    def __init__(self, idomain: UgridDataArray) -> None:
        self.__idomain = idomain

    def create(
        self,
        state_for_boundary: UgridDataArray,
        boundary_conditions: List[ConstantHead],
    ) -> ConstantHead:
        self.__validate(boundary_conditions)

        active_grid_boundary = active_grid_boundary_xy(self.__idomain == 1)
        unassigned_grid_boundaries = self.__find_unassigned_grid_boundaries(
            active_grid_boundary, boundary_conditions
        )

        constant_head = state_for_boundary * unassigned_grid_boundaries.where(
            unassigned_grid_boundaries
        )

        return imod.mf6.ConstantHead(
            constant_head, print_input=True, print_flows=True, save_flows=True
        )

    @staticmethod
    def __validate(boundary_conditions: List[ConstantHead]) -> None:
        are_boundary_conditions_constant_head = all(
            isinstance(boundary_condition, imod.mf6.ConstantHead)
            for boundary_condition in boundary_conditions
        )

        if not are_boundary_conditions_constant_head:
            raise ValueError("Only ConstantHead boundary conditions are supported")

    @staticmethod
    def __find_unassigned_grid_boundaries(
        active_grid_boundary: UgridDataArray, boundary_conditions: List[ConstantHead]
    ) -> UgridDataArray:
        unassigned_grid_boundaries = active_grid_boundary
        for boundary_condition in boundary_conditions:
            unassigned_grid_boundaries = np.logical_and(
                unassigned_grid_boundaries, boundary_condition["head"].isnull()
            )

        return unassigned_grid_boundaries
