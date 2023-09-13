from unittest.mock import MagicMock

import pytest
import xugrid as xu

import imod
from imod.mf6 import HorizontalFlowBarrierResistance
import geopandas as gpd
from imod.mf6.interfaces.ilinedatapackage import ILineDataPackage
from imod.mf6.interfaces.ipackagebase import IPackageBase
from imod.mf6.utilities.clip_utilities import clip_by_grid
from imod.mf6.utilities.grid_utilities import broadcast_to_full_domain


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

    expected_idomain_shape = active.where(active > 0, -1, drop=True).shape
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

    # Act
    clipped_pkg = clip_by_grid(pkg, active)

    # Assert
    assert pkg.dataset.data_vars.keys() == clipped_pkg.dataset.data_vars.keys()

    expected_idomain = active.where(active > 0, -1, drop=True)
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


def test_clip_by_grid_with_line_data_package(basic_dis):
    # Arrange
    idomain, top, bottom = basic_dis
    pkg = HorizontalFlowBarrierResistance(gpd.GeoDataFrame())

    active = idomain.sel(layer=1, drop=True)

    # Act/Assert
    with pytest.raises(NotImplementedError):
        _ = clip_by_grid(pkg, active)


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
    idomain_selected = idomain.sel(x=slice(None, 70.0))

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
