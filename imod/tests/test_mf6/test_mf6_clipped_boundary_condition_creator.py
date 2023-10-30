import numpy as np
import pytest
import xarray as xr
import xugrid as xu

from imod.mf6 import ConstantHead
from imod.mf6.clipped_boundary_condition_creator import create_clipped_boundary
from imod.select.grid import grid_boundary_xy


def _remove_boundary_locations(boundary_locations, n_locations_to_remove):
    indexes = boundary_locations.values.nonzero()[0]
    boundary_locations[indexes[:n_locations_to_remove]] = False

    return boundary_locations


class TestClippedBoundaryConditionCreator:
    @pytest.mark.parametrize("n_clipped_cells", [15, 0, 36])
    def test_create_different_n_clipped_cells(self, circle_dis, n_clipped_cells):
        """
        This test validates that the number of assigned boundary cells in the ConstantHead package produced by the
        ClippedBoundaryConditionCreator class is as expected. The parameterized values are:
        - 15 : 15 boundary cells are available to the ClippedBoundaryConditionCreator
        - 0  : no boundary cells are available to the ClippedBoundaryConditionCreator
        - 36 : all boundary cells are available to the ClippedBoundaryConditionCreator
        """
        # Arrange.
        head_value_original_domain = 1.0
        head_value_clipped_domain = 2.0

        idomain, _, _ = circle_dis

        original_boundary_locations = grid_boundary_xy(idomain == 1)

        reduced_boundary_locations = _remove_boundary_locations(
            original_boundary_locations, n_clipped_cells
        )

        reduced_boundary_values = xu.full_like(
            idomain, head_value_original_domain, dtype=float
        ).where(reduced_boundary_locations)

        reduced_boundary_constant_head_pkg = ConstantHead(
            reduced_boundary_values,
            print_input=True,
            print_flows=True,
            save_flows=True,
        )

        clipped_boundary_values = xu.full_like(
            idomain, head_value_clipped_domain, dtype=float
        )

        # Act.
        constant_head_pkg_clipped_domain = create_clipped_boundary(
            idomain, clipped_boundary_values, [reduced_boundary_constant_head_pkg]
        )

        # Assert.
        result_clipped_head = constant_head_pkg_clipped_domain["head"]

        for layer_index in range(1, 15):
            number_clipped_head_locations = np.count_nonzero(
                ~np.isnan(result_clipped_head.sel(layer=layer_index))
            )
            assert number_clipped_head_locations is n_clipped_cells

    @pytest.mark.parametrize(
        "dis, grid_data_array", [("basic_unstructured_dis", xu), ("basic_dis", xr)]
    )
    def test_create_different_dis(self, dis, grid_data_array, request):
        # Arrange.
        n_clipped_cells = 9
        head_value_original_domain = 1.0
        head_value_clipped_domain = 2.0

        idomain, _, _ = request.getfixturevalue(dis)

        original_boundary_locations = grid_boundary_xy(idomain == 1)

        reduced_boundary_locations = _remove_boundary_locations(
            original_boundary_locations, n_clipped_cells
        )

        reduced_boundary_values = grid_data_array.full_like(
            idomain, head_value_original_domain, dtype=float
        ).where(reduced_boundary_locations)

        reduced_boundary_constant_head_pkg = ConstantHead(
            reduced_boundary_values,
            print_input=True,
            print_flows=True,
            save_flows=True,
        )

        clipped_boundary_values = grid_data_array.full_like(
            idomain, head_value_clipped_domain, dtype=float
        )

        # Act.
        constant_head_pkg_clipped_domain = create_clipped_boundary(
            idomain, clipped_boundary_values, [reduced_boundary_constant_head_pkg]
        )

        # Assert.
        result_clipped_head = constant_head_pkg_clipped_domain["head"]

        number_clipped_head_locations = np.count_nonzero(
            ~np.isnan(result_clipped_head.sel(layer=1))
        )
        assert number_clipped_head_locations is n_clipped_cells
