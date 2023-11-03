import numpy as np
import pytest
from pytest_cases import parametrize_with_cases

from imod.prepare import (
    get_lower_active_grid_cells,
    get_lower_active_layer_number,
    get_upper_active_grid_cells,
    get_upper_active_layer_number,
)
from imod.typing.grid import ones_like


@pytest.fixture(scope="function")
def active_structured(basic_dis):
    idomain, _, _ = basic_dis
    return idomain == 1


@pytest.fixture(scope="function")
def active_unstructured(basic_unstructured_dis):
    idomain, _, _ = basic_unstructured_dis
    return idomain == 1


@pytest.fixture(scope="function")
def layer(basic_dis):
    idomain, _, _ = basic_dis
    return idomain.coords["layer"]


class ActiveLayerCases:
    def case_all_active_structured(self, active_structured):
        upper_layer_number = ones_like(active_structured.sel(layer=1)).astype(int)
        lower_layer_number = upper_layer_number + 2
        return active_structured, upper_layer_number, lower_layer_number

    def case_all_active_unstructured(self, active_unstructured):
        upper_layer_number = ones_like(active_unstructured.sel(layer=1)).astype(int)
        lower_layer_number = upper_layer_number + 2
        return active_unstructured, upper_layer_number, lower_layer_number

    def case_structured(self, active_structured):
        active_structured[0, :, 1] = False
        active_structured[1, :, 2] = False
        active_structured[0, :, 3] = False
        active_structured[1, :, 3] = False
        active_structured[1, :, 4] = False
        active_structured[2, :, 4] = False
        active_structured[2, :, 5] = False
        upper_layer_number = ones_like(active_structured.sel(layer=1)).astype(int)
        lower_layer_number = upper_layer_number + 2
        upper_layer_number[:, 1] = 2
        upper_layer_number[:, 3] = 3
        lower_layer_number[:, 4] = 1
        lower_layer_number[:, 5] = 2
        return active_structured, upper_layer_number, lower_layer_number

    def case_unstructured(self, active_unstructured):
        active_unstructured[0, 1] = False
        active_unstructured[1, 2] = False
        active_unstructured[0, 3] = False
        active_unstructured[1, 3] = False
        active_unstructured[1, 4] = False
        active_unstructured[2, 4] = False
        active_unstructured[2, 5] = False
        upper_layer_number = ones_like(active_unstructured.sel(layer=1)).astype(int)
        lower_layer_number = upper_layer_number + 2
        upper_layer_number[1] = 2
        upper_layer_number[3] = 3
        lower_layer_number[4] = 1
        lower_layer_number[5] = 2
        return active_unstructured, upper_layer_number, lower_layer_number


@parametrize_with_cases("active_case", cases=ActiveLayerCases)
def test_get_lower_active_grid(active_case, layer):
    active, _, expected_lower_layer_nr = active_case
    is_lower_active = get_lower_active_grid_cells(active)

    expected_lower_active = layer == expected_lower_layer_nr

    np.testing.assert_array_equal(is_lower_active.values, expected_lower_active.values)


@parametrize_with_cases("active_case", cases=ActiveLayerCases)
def test_get_upper_active_grid(active_case, layer):
    active, expected_upper_layer_nr, _ = active_case
    is_upper_active = get_upper_active_grid_cells(active)

    expected_upper_active = layer == expected_upper_layer_nr

    np.testing.assert_array_equal(is_upper_active.values, expected_upper_active.values)


@parametrize_with_cases("active_case", cases=ActiveLayerCases)
def test_get_lower_layer_number(active_case):
    active, _, expected_lower_layer_nr = active_case
    is_lower_active = get_lower_active_layer_number(active)

    np.testing.assert_array_equal(
        is_lower_active.values, expected_lower_layer_nr.values
    )


@parametrize_with_cases("active_case", cases=ActiveLayerCases)
def test_get_upper_layer_number(active_case):
    active, expected_upper_layer_nr, _ = active_case
    is_upper_active = get_upper_active_layer_number(active)

    np.testing.assert_array_equal(
        is_upper_active.values, expected_upper_layer_nr.values
    )
