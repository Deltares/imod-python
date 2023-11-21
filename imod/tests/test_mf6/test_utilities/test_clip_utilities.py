from unittest.mock import MagicMock

import geopandas as gpd
import numpy as np
import pytest
import shapely
import xarray as xr
import xugrid as xu
from shapely.testing import assert_geometries_equal

import imod
from imod.mf6 import HorizontalFlowBarrierResistance
from imod.mf6.interfaces.ipackagebase import IPackageBase
from imod.mf6.utilities.clip import clip_by_grid
from imod.mf6.utilities.grid import broadcast_to_full_domain


@pytest.fixture(scope="function")
def horizontal_flow_barrier():
    ztop = -5.0
    zbottom = -135.0

    barrier_y = [70.0, 40.0, 0.0]
    barrier_x = [120.0, 40.0, 0.0]

    geometry = gpd.GeoDataFrame(
        geometry=[shapely.linestrings(barrier_x, barrier_y)],
        data={
            "resistance": [1e3],
            "ztop": [ztop],
            "zbottom": [zbottom],
        },
    )
    return HorizontalFlowBarrierResistance(geometry)


def test_clip_by_grid_convex_grid(basic_dis):
    # Arrange
    x_min = 35.0
    y_min = 55.0

    idomain, top, bottom = basic_dis
    top, bottom = broadcast_to_full_domain(idomain, top, bottom)
    pkg = imod.mf6.StructuredDiscretization(top.sel(layer=1), bottom, idomain)

    active = idomain.sel(layer=1, drop=True)
    active = active.where((active.x > x_min) & (active.y > y_min), -1)

    # Act
    clipped_pkg = clip_by_grid(pkg, active)

    # Assert
    assert pkg.dataset.data_vars.keys() == clipped_pkg.dataset.data_vars.keys()
    assert clipped_pkg.dataset.x.min() > x_min
    assert clipped_pkg.dataset.y.min() > y_min

    expected_idomain_shape = active.where(active > 0, 0, drop=True).shape
    assert clipped_pkg.dataset["idomain"].sel(layer=1).shape == expected_idomain_shape


def test_clip_by_grid_concave_grid(basic_dis):
    # Arrange
    x_start_cut = 35.0
    y_start_cut = 55.0

    idomain, top, bottom = basic_dis
    top, bottom = broadcast_to_full_domain(idomain, top, bottom)
    pkg = imod.mf6.StructuredDiscretization(top.sel(layer=1), bottom, idomain)

    active = idomain.sel(layer=1, drop=True)
    active = active.where((active.x > x_start_cut) & (active.y > y_start_cut), -1)
    active = active * -1
    active = active.where(active > 0, 0)

    # Act
    clipped_pkg = clip_by_grid(pkg, active)

    # Assert
    assert pkg.dataset.data_vars.keys() == clipped_pkg.dataset.data_vars.keys()

    expected_idomain = active.where(active > 0, 0, drop=True)
    expected_idomain_shape = expected_idomain.shape
    assert clipped_pkg.dataset["idomain"].sel(layer=1).shape == expected_idomain_shape
    assert (
        clipped_pkg.dataset["idomain"].sel(layer=1, drop=True) == expected_idomain
    ).all()


def test_clip_by_grid_unstructured_grid(basic_unstructured_dis):
    # Arrange
    idomain, top, bottom = basic_unstructured_dis
    top, bottom = broadcast_to_full_domain(idomain, top, bottom)
    pkg = imod.mf6.VerticesDiscretization(top.sel(layer=1), bottom, idomain)

    active = idomain.sel(layer=1, drop=True)
    active = active.where(active.grid.face_x > 0, -1)

    # Act
    clipped_pkg = clip_by_grid(pkg, active)

    # Assert
    assert pkg.dataset.data_vars.keys() == clipped_pkg.dataset.data_vars.keys()

    clipped_active_cells = clipped_pkg.dataset["idomain"].sel(layer=1).count()
    expected_active_cells = active.where(active > 0).count()

    assert clipped_active_cells == expected_active_cells


def test_clip_by_grid_wrong_grid_type():
    # Arrange
    pkg = MagicMock(spec_set=IPackageBase)
    active = "wrong type"

    # Act/Assert
    with pytest.raises(TypeError):
        _ = clip_by_grid(pkg, active)


def test_clip_by_grid_with_line_data_package__structured(
    basic_dis, horizontal_flow_barrier
):
    # Arrange
    idomain, _, _ = basic_dis
    active = idomain.sel(layer=1, drop=True)

    # Act
    hfb_clipped = clip_by_grid(horizontal_flow_barrier, active)

    # Assert
    with pytest.raises(AssertionError):
        assert_geometries_equal(
            hfb_clipped["geometry"].item(), horizontal_flow_barrier["geometry"].item()
        )

    x, y = hfb_clipped["geometry"].item().xy
    np.testing.assert_allclose(x, np.array([90.0, 40.0, 0.0]))
    np.testing.assert_allclose(y, np.array([58.75, 40.0, 0.0]))


def test_clip_by_grid_with_line_data_package__unstructured(
    basic_dis, horizontal_flow_barrier
):
    # Arrange
    idomain, _, _ = basic_dis
    active = idomain.sel(layer=1, drop=True)
    active_uda = xu.UgridDataArray.from_structured(active)

    # Act
    hfb_clipped = clip_by_grid(horizontal_flow_barrier, active_uda)

    # Assert
    with pytest.raises(AssertionError):
        assert_geometries_equal(
            hfb_clipped["geometry"].item(), horizontal_flow_barrier["geometry"].item()
        )

    x, y = hfb_clipped["geometry"].item().xy
    np.testing.assert_allclose(x, np.array([90.0, 40.0, 0.0]))
    np.testing.assert_allclose(y, np.array([58.75, 40.0, 0.0]))


def test_clip_by_grid__structured_grid_full(
    basic_dis, well_high_lvl_test_data_stationary
):
    """All wells are included within the structured grid bounds"""
    # Arrange
    idomain, _, _ = basic_dis
    wel = imod.mf6.Well(*well_high_lvl_test_data_stationary, print_flows=True)

    # Act
    wel_clipped = clip_by_grid(wel, idomain)

    # Assert
    assert isinstance(wel_clipped, imod.mf6.Well)
    assert wel_clipped.dataset["rate"].shape == wel.dataset["rate"].shape
    # Test if options are copied
    assert wel_clipped.dataset["print_flows"] == wel.dataset["print_flows"]


def test_clip_by_grid__structured_grid_clipped(
    basic_dis, well_high_lvl_test_data_stationary
):
    """Half of the wells are included within the structured grid bounds"""
    # Arrange
    idomain, _, _ = basic_dis
    wel = imod.mf6.Well(*well_high_lvl_test_data_stationary, print_flows=True)
    # Clip grid so that xmax is set to 70.0 instead of 90.0
    idomain_selected = idomain.where(idomain.x < 70.0, -1)

    # Act
    wel_clipped = clip_by_grid(wel, idomain_selected)

    # Assert
    assert isinstance(wel_clipped, imod.mf6.Well)
    assert wel_clipped.dataset["rate"].shape == (4,)
    # Test if options are copied
    assert wel_clipped.dataset["print_flows"] == wel.dataset["print_flows"]


def test_clip_by_grid__unstructured_grid_full(
    basic_dis, well_high_lvl_test_data_stationary
):
    """All the wells are included within the unstructured grid bounds"""
    # Arrange
    idomain, _, _ = basic_dis
    wel = imod.mf6.Well(*well_high_lvl_test_data_stationary, print_flows=True)
    idomain_ugrid = xu.UgridDataArray.from_structured(idomain)

    # Act
    wel_clipped = clip_by_grid(wel, idomain_ugrid)

    # Assert
    assert isinstance(wel_clipped, imod.mf6.Well)
    assert wel_clipped.dataset["rate"].shape == wel.dataset["rate"].shape
    # Test if options are copied
    assert wel_clipped.dataset["print_flows"] == wel.dataset["print_flows"]


def test_clip_by_grid__unstructured_grid_clipped(
    basic_dis, well_high_lvl_test_data_stationary
):
    """Half of the wells are included within the unstructured grid bounds"""
    # Arrange
    idomain, _, _ = basic_dis
    wel = imod.mf6.Well(*well_high_lvl_test_data_stationary, print_flows=True)
    # Clip grid so that xmax is set to 70.0 instead of 90.0
    idomain_selected = idomain.sel(x=slice(None, 70.0))
    idomain_ugrid = xu.UgridDataArray.from_structured(idomain_selected)

    # Act
    wel_clipped = clip_by_grid(wel, idomain_ugrid)

    # Assert
    assert isinstance(wel_clipped, imod.mf6.Well)
    assert wel_clipped.dataset["rate"].shape == (4,)
    # Test if options are copied
    assert wel_clipped.dataset["print_flows"] == wel.dataset["print_flows"]


def test_clip_by_grid_contains_non_grid_data_variables(basic_dis):
    # Arrange
    x_min = 35.0
    y_min = 55.0

    idomain, _, _ = basic_dis
    k = xr.full_like(idomain, 1.0, dtype=float)

    pkg = imod.mf6.NodePropertyFlow(
        k=k,
        icelltype=0,
        variable_vertical_conductance=True,
        dewatered=True,
        perched=True,
        save_flows=True,
    )

    active = idomain.sel(layer=1, drop=True)
    active = active.where((active.x > x_min) & (active.y > y_min), -1)

    # Act
    clipped_pkg = clip_by_grid(pkg, active)

    # Assert
    assert pkg.dataset["icelltype"] == clipped_pkg.dataset["icelltype"]
    assert (
        pkg.dataset["variable_vertical_conductance"]
        == clipped_pkg.dataset["variable_vertical_conductance"]
    )
    assert pkg.dataset["dewatered"] == clipped_pkg.dataset["dewatered"]
    assert pkg.dataset["perched"] == clipped_pkg.dataset["perched"]
    assert pkg.dataset["save_flows"] == clipped_pkg.dataset["save_flows"]
