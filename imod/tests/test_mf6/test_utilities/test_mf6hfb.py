import geopandas as gpd
import numpy as np
import pytest
import shapely
import xarray as xr
from pytest_cases import parametrize_with_cases

from imod.mf6.hfb import (
    HorizontalFlowBarrierResistance,
    SingleLayerHorizontalFlowBarrierResistance,
)
from imod.mf6.utilities.mf6hfb import merge_hfb_packages
from imod.prepare.hfb import linestring_to_square_zpolygons


@pytest.mark.usefixtures("structured_flow_model")
@pytest.fixture(scope="function")
def modellayers_single_layer(structured_flow_model):
    model = structured_flow_model.clip_box(layer_max=1)
    dis = model["dis"]
    dis["bottom"] = dis["bottom"].isel(x=0, y=0, drop=True)
    npf = model["npf"]

    return {
        "idomain": dis["idomain"],
        "top": dis["top"],
        "bottom": dis["bottom"],
        "k": npf["k"],
    }


@pytest.mark.usefixtures("structured_flow_model")
@pytest.fixture(scope="function")
def modellayers(structured_flow_model):
    model = structured_flow_model
    dis = model["dis"]
    dis["bottom"] = dis["bottom"].isel(x=0, y=0, drop=True)
    npf = model["npf"]

    return {
        "idomain": dis["idomain"],
        "top": dis["top"],
        "bottom": dis["bottom"],
        "k": npf["k"],
    }


def make_layer_geometry(resistance, layer):
    barrier_y = [11.0, 5.0, -1.0]
    barrier_x = [5.0, 5.0, 5.0]

    geometry = gpd.GeoDataFrame(
        geometry=[shapely.linestrings(barrier_x, barrier_y)],
        data={
            "resistance": [resistance],
            "layer": [layer],
        },
    )
    return geometry


def make_layer_geometry__outside_domain(resistance, layer):
    barrier_y = [11.0, 5.0, -1.0]
    barrier_x = [-9990.0, -9990.0, -9990.0]

    geometry = gpd.GeoDataFrame(
        geometry=[shapely.linestrings(barrier_x, barrier_y)],
        data={
            "resistance": [resistance],
            "layer": [layer],
        },
    )
    return geometry


def make_depth_geometry(resistance, top, bot):
    barrier_y = [11.0, 5.0, -1.0]
    barrier_x = [5.0, 5.0, 5.0]
    barrier_ztop = [top, top]
    barrier_zbottom = [bot, bot]

    polygons = linestring_to_square_zpolygons(
        barrier_x, barrier_y, barrier_ztop, barrier_zbottom
    )

    geometry = gpd.GeoDataFrame(
        geometry=polygons,
        data={
            "resistance": [resistance, resistance],
        },
    )
    return geometry


def test_merge_three_hfbs__single_layer(modellayers_single_layer):
    """Merge three single layer hfbs, test for lenght"""
    # Arrange
    n_barriers = 3
    single_resistance = 400.0

    geometry = make_layer_geometry(single_resistance, 1)
    hfb_ls = [
        SingleLayerHorizontalFlowBarrierResistance(geometry) for _ in range(n_barriers)
    ]

    # Act
    mf6_hfb = merge_hfb_packages(hfb_ls, **modellayers_single_layer)

    # Assert
    assert mf6_hfb["cell_id"].shape == (6,)
    assert (mf6_hfb["layer"] == 1).all()
    expected_resistance = n_barriers * single_resistance
    assert (expected_resistance == 1 / mf6_hfb["hydraulic_characteristic"]).all()


def test_merge_three_hfbs__compare_single_hfb(modellayers_single_layer):
    """
    Merge three single layer hfbs, compare with one hfb with tripled
    resistance as created with a call to merge_hfb_packages.
    """
    # Arrange
    n_barriers = 3
    single_resistance = 400.0

    geometry = make_layer_geometry(single_resistance, 1)
    geometry_tripled = make_layer_geometry(n_barriers * single_resistance, 1)

    hfb_ls_triple = [
        SingleLayerHorizontalFlowBarrierResistance(geometry) for _ in range(n_barriers)
    ]
    hfb_ls_single = [SingleLayerHorizontalFlowBarrierResistance(geometry_tripled)]

    # Act
    mf6_hfb_three = merge_hfb_packages(hfb_ls_triple, **modellayers_single_layer)
    mf6_hfb_single = merge_hfb_packages(hfb_ls_single, **modellayers_single_layer)

    # Assert
    xr.testing.assert_equal(mf6_hfb_single.dataset, mf6_hfb_three.dataset)


def test_merge_three_hfbs__to_mf6_pkg_single_layer(modellayers_single_layer):
    """
    Merge three single layer hfbs, compare with one hfb with tripled
    resistance as with a call to to_mf6_pkg.
    """
    # Arrange
    n_barriers = 3
    single_resistance = 400.0

    geometry = make_layer_geometry(single_resistance, 1)
    geometry_tripled = make_layer_geometry(n_barriers * single_resistance, 1)

    hfb_ls_triple = [
        SingleLayerHorizontalFlowBarrierResistance(geometry) for _ in range(n_barriers)
    ]
    hfb_ls_single = [SingleLayerHorizontalFlowBarrierResistance(geometry_tripled)]

    # Act
    mf6_hfb_three = merge_hfb_packages(hfb_ls_triple, **modellayers_single_layer)
    mf6_hfb_single = hfb_ls_single[0].to_mf6_pkg(**modellayers_single_layer)

    # Assert
    xr.testing.assert_equal(mf6_hfb_single.dataset, mf6_hfb_three.dataset)


def test_merge_mixed_hfbs__single_layer(modellayers_single_layer):
    """Merge mix of layer hfb and depth hfb."""
    # Arrange
    n_barriers = 3
    single_resistance = 400.0

    top = float(modellayers_single_layer["top"].values)
    bot = modellayers_single_layer["bottom"].values[0]

    geometry = make_layer_geometry(single_resistance, 1)
    geometry_depth = make_depth_geometry(single_resistance, top, bot)

    hfb_ls_triple = [
        SingleLayerHorizontalFlowBarrierResistance(geometry),
        HorizontalFlowBarrierResistance(geometry_depth),
        HorizontalFlowBarrierResistance(geometry_depth),
    ]

    # Act
    mf6_hfb = merge_hfb_packages(hfb_ls_triple, **modellayers_single_layer)

    # Assert
    assert mf6_hfb["cell_id"].shape == (6,)
    assert (mf6_hfb["layer"] == 1).all()
    expected_resistance = n_barriers * single_resistance
    assert (expected_resistance == 1 / mf6_hfb["hydraulic_characteristic"]).all()


def test_merge_three_hfbs__multiple_single_layers(modellayers):
    """Merge three single layer hfbs at different layers"""
    # Arrange
    n_barriers = 3
    single_resistance = 400.0

    hfb_ls = [
        SingleLayerHorizontalFlowBarrierResistance(
            make_layer_geometry(single_resistance, i)
        )
        for i in range(1, n_barriers + 1)
    ]

    # Act
    mf6_hfb = merge_hfb_packages(hfb_ls, **modellayers)

    # Assert
    assert mf6_hfb["cell_id"].shape == (18,)
    assert np.all(np.unique(mf6_hfb["layer"]) == np.array([1, 2, 3]))
    expected_resistance = single_resistance
    assert (expected_resistance == 1 / mf6_hfb["hydraulic_characteristic"]).all()


def test_merge_mixed_hfbs__multiple_layer(modellayers):
    """
    Merge three single layer hfbs at different layers plus one hfb spread across
    the complete depth.
    """
    # Arrange
    n_barriers = 3
    single_resistance = 400.0

    hfb_ls = [
        SingleLayerHorizontalFlowBarrierResistance(
            make_layer_geometry(single_resistance, i)
        )
        for i in range(1, n_barriers + 1)
    ]
    hfb_ls.append(
        HorizontalFlowBarrierResistance(
            make_depth_geometry(single_resistance, 10.0, -3.0)
        )
    )

    # Act
    mf6_hfb = merge_hfb_packages(hfb_ls, **modellayers)

    # Assert
    assert mf6_hfb["cell_id"].shape == (18,)
    assert np.all(np.unique(mf6_hfb["layer"]) == np.array([1, 2, 3]))
    expected_resistance = 2 * single_resistance
    assert (expected_resistance == 1 / mf6_hfb["hydraulic_characteristic"]).all()


@pytest.mark.parametrize("strict_hfb_validation", [True, False])
@pytest.mark.parametrize("inactive_value", [0, -1])
def test_merge__middle_layer_inactive_domain(
    modellayers, strict_hfb_validation, inactive_value
):
    """
    Test where middle layer is deactivated, HFB assigned to that layer should be
    ignored.
    """
    # Arrange
    single_resistance = 400.0

    modellayers["idomain"].loc[2, :, :] = inactive_value

    hfb_ls = [
        SingleLayerHorizontalFlowBarrierResistance(
            make_layer_geometry(single_resistance, 1)
        ),
        SingleLayerHorizontalFlowBarrierResistance(
            make_layer_geometry(single_resistance, 2)
        ),
    ]

    # Act
    if strict_hfb_validation:
        pytest.xfail("Test expected to fail for hfb in inactive domain")

    mf6_hfb = merge_hfb_packages(
        hfb_ls, strict_hfb_validation=strict_hfb_validation, **modellayers
    )

    # Assert
    assert mf6_hfb.dataset.coords["cell_id"].shape == (6,)
    assert (mf6_hfb["layer"] == 1).all()
    expected_resistance = single_resistance
    assert (expected_resistance == 1 / mf6_hfb["hydraulic_characteristic"]).all()


class InactivityLabelCases:
    """
    Labels which will be set to inactive, and expected n_cellids
    """

    all = slice(None, None)

    def case_all(self):
        return (self.all, self.all, self.all), 0

    def case_half_left(self):
        return (self.all, self.all, slice(None, 5.0)), 0

    def case_half_right(self):
        return (self.all, self.all, slice(5.0, None)), 0

    def case_strip_left(self):
        return (self.all, self.all, slice(None, 1.0)), 6

    def case_strip_right(self):
        return (self.all, self.all, slice(9.0, None)), 6


@parametrize_with_cases("inactive_labels, n_cellids", cases=InactivityLabelCases)
@pytest.mark.parametrize("strict_hfb_validation", [True, False])
@pytest.mark.parametrize("inactive_value", [0, -1])
def test_merge__single_layer_inactive_domain(
    modellayers_single_layer,
    strict_hfb_validation,
    inactive_value,
    inactive_labels,
    n_cellids,
):
    """
    Test with single inactive layer, HFB assigned to that layer should be
    ignored.
    """
    # Arrange
    single_resistance = 400.0

    modellayers_single_layer["idomain"].loc[inactive_labels] = inactive_value

    hfb_ls = [
        SingleLayerHorizontalFlowBarrierResistance(
            make_layer_geometry(single_resistance, 1)
        ),
        SingleLayerHorizontalFlowBarrierResistance(
            make_layer_geometry(single_resistance, 1)
        ),
    ]

    # Act
    if strict_hfb_validation and n_cellids == 0:
        pytest.xfail("Strict hfb validation and no cellids could be allocated")

    mf6_hfb = merge_hfb_packages(
        hfb_ls, strict_hfb_validation=strict_hfb_validation, **modellayers_single_layer
    )

    # Assert
    assert mf6_hfb.dataset.coords["cell_id"].shape == (n_cellids,)
