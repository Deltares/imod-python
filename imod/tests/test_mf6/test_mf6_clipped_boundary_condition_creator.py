from unittest.mock import MagicMock

import numpy as np
import pytest
import xugrid as xu
from xugrid.core.wrap import UgridDataArray

from imod.mf6 import ConstantConcentration, ConstantHead
from imod.mf6.clipped_boundary_condition_creator import ClippedBoundaryConditionCreator


def _get_boundaries(idomain):
    return xu.zeros_like(idomain.sel(layer=1), dtype=bool).ugrid.binary_dilation(
        border_value=True
    )


class TestClippedBoundaryConditionCreator:
    @pytest.mark.parametrize("number_of_clipped_cells", [15, 0, 36])
    def test_create(self, basic_unstructured_dis, number_of_clipped_cells):
        # Arrange.
        idomain, _, _ = basic_unstructured_dis

        head_value_original_domain = 1.0
        head_value_clipped_domain = 2.0

        original_boundary_location = _get_boundaries(idomain)
        original_boundary_location[
            original_boundary_location.size - number_of_clipped_cells :
        ] = False

        original_boundary_values = xu.full_like(
            idomain, head_value_original_domain, dtype=float
        ).where(original_boundary_location)

        original_boundary_constant_head_pkg = ConstantHead(
            original_boundary_values,
            print_input=True,
            print_flows=True,
            save_flows=True,
        )

        clipped_boundary_values = xu.full_like(
            idomain, head_value_clipped_domain, dtype=float
        )

        # Act.
        constant_head_pkg_clipped_domain = ClippedBoundaryConditionCreator.create(
            idomain, clipped_boundary_values, [original_boundary_constant_head_pkg]
        )

        # Assert.
        result_clipped_head = constant_head_pkg_clipped_domain["head"]

        for layer_index in range(1, 15):
            number_clipped_head_locations = np.count_nonzero(
                ~np.isnan(result_clipped_head.sel(layer=layer_index))
            )
            assert number_clipped_head_locations is number_of_clipped_cells
