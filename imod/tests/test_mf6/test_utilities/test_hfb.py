import geopandas as gpd
import numpy as np
import pytest
import shapely
import xarray as xr

from imod.mf6.hfb import (
    HorizontalFlowBarrierResistance,
    SingleLayerHorizontalFlowBarrierResistance,
)
from imod.mf6.utilities.hfb import merge_hfb_packages


@pytest.mark.usefixtures("structured_flow_model")
@pytest.fixture(scope="function")
def modellayers_single_layer(structured_flow_model):
    # TODO: Refactor to just use structured grid.
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
    # TODO: Refactor to just use structured grid.
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


def make_depth_geometry(resistance, top, bot):
    barrier_y = [11.0, 5.0, -1.0]
    barrier_x = [5.0, 5.0, 5.0]

    geometry = gpd.GeoDataFrame(
        geometry=[shapely.linestrings(barrier_x, barrier_y)],
        data={
            "resistance": [resistance],
            "ztop": [top],
            "zbottom": [bot],
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

    top = modellayers_single_layer["top"].values
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
