# tests/test_prepare/test_topsystem_layer_preservation.py

import numpy as np
import pytest
import xarray as xr

from imod.prepare import LayerRegridder
from imod.prepare.topsystem import (
    ALLOCATION_OPTION,
    allocate_drn_cells,
    allocate_rch_cells,
    allocate_riv_cells,
)
from imod.typing import GridDataArray


def make_model_grid(n_layers, nrow=10, ncol=10, dx=100.0):
    x = np.arange(ncol) * dx
    y = np.arange(nrow) * -dx
    layer = np.arange(1, n_layers + 1)

    top = xr.DataArray(
        np.stack([np.full((nrow, ncol), -float(k)) for k in range(n_layers)]),
        {"layer": layer, "y": y, "x": x},
        ("layer", "y", "x"),
    )
    bottom = top - 1.0
    active = xr.full_like(top, True, dtype=bool)
    return active, top, bottom


def make_sparse_riv(nrow=10, ncol=10, dx=100.0, active_layer=1):
    """Planar river stage/bottom_elevation - genuinely intersects only
    ~1 layer of a deep model."""
    x = np.arange(ncol) * dx
    y = np.arange(nrow) * -dx
    stage = xr.DataArray(np.full((nrow, ncol), -0.2), {"y": y, "x": x}, ("y", "x"))
    bottom_elevation = xr.DataArray(
        np.full((nrow, ncol), -0.8), {"y": y, "x": x}, ("y", "x")
    )
    return stage, bottom_elevation


@pytest.fixture(params=[2, 10, 30])
def n_layers(request):
    return request.param


def n_nonempty_layers(
    da: GridDataArray, spatial_dims: tuple[str, ...] = ("y", "x")
) -> int:
    """
    Number of layers that contain at least one meaningfully "present" value.

    Handles the fact that boolean arrays get upcast to float by xarray's
    .where() (NaN has no bool representation) - after such a coercion,
    False becomes 0.0, which must still be treated as "no data", not as
    a valid float value.
    """
    reduce_dims = [d for d in da.dims if d != "layer"]

    if da.dtype == bool:
        has_data_per_layer = da.any(dim=reduce_dims)
    else:
        # Treat both NaN and 0.0 as "no data" - this covers arrays that
        # started boolean and were upcast to float by .where()/masking.
        is_present = (~da.isnull()) & (da != 0)
        has_data_per_layer = is_present.any(dim=reduce_dims)

    return int(has_data_per_layer.sum())


def reindex_to_full_layers(
    da: xr.DataArray, full_layer: xr.DataArray, dtype
) -> xr.DataArray:
    """
    Re-expand a layer-trimmed result back onto the full model layer
    coordinate, so it can be compared against expectations written for
    the untrimmed (drop_empty_layers=False) behaviour.
    """
    fill_value = False if dtype is bool else np.nan
    return da.reindex(layer=full_layer, fill_value=fill_value)


def take_nth_layer_column(grid, n):
    if "time" in grid.dims:
        grid = grid.isel(time=-1)
    return grid.values[:, n, n]


@pytest.fixture
def basic_riv_inputs():
    nlayer, nrow, ncol = 4, 3, 3
    layer = np.array([1, 2, 3, 4])
    y = np.arange(nrow) * -10.0
    x = np.arange(ncol) * 10.0

    top = xr.DataArray(
        np.stack([np.full((nrow, ncol), -float(k)) for k in range(nlayer)]),
        {"layer": layer, "y": y, "x": x},
        ("layer", "y", "x"),
    )
    bottom = top - 1.0
    active = xr.full_like(top, True, dtype=bool)

    # Stage/bottom_elevation only intersect layer 1: planar, no layer dim.
    stage = xr.DataArray(np.full((nrow, ncol), -0.2), {"y": y, "x": x}, ("y", "x"))
    bottom_elevation = xr.DataArray(
        np.full((nrow, ncol), -0.8), {"y": y, "x": x}, ("y", "x")
    )
    return active, top, bottom, stage, bottom_elevation, layer


def test_allocate_riv_cells_drop_empty_layers_matches_full(basic_riv_inputs):
    active, top, bottom, stage, bottom_elevation, layer = basic_riv_inputs

    full, _ = allocate_riv_cells(
        ALLOCATION_OPTION.stage_to_riv_bot,
        active,
        top,
        bottom,
        stage,
        bottom_elevation,
        drop_empty_layers=False,
    )
    trimmed, _ = allocate_riv_cells(
        ALLOCATION_OPTION.stage_to_riv_bot,
        active,
        top,
        bottom,
        stage,
        bottom_elevation,
        drop_empty_layers=True,
    )

    # Trimmed result should have fewer (or equal) layers than the full one.
    assert trimmed.sizes["layer"] <= full.sizes["layer"]

    # Once re-expanded, trimmed result must be identical to the full one.
    re_expanded = reindex_to_full_layers(trimmed, full["layer"], dtype=bool)
    xr.testing.assert_equal(re_expanded, full)


def test_allocate_drn_cells_drop_empty_layers_matches_full(basic_riv_inputs):
    active, top, bottom, _, elevation, layer = basic_riv_inputs

    full = allocate_drn_cells(
        ALLOCATION_OPTION.at_elevation,
        active,
        top,
        bottom,
        elevation,
        drop_empty_layers=False,
    )
    trimmed = allocate_drn_cells(
        ALLOCATION_OPTION.at_elevation,
        active,
        top,
        bottom,
        elevation,
        drop_empty_layers=True,
    )

    assert trimmed.sizes["layer"] <= full.sizes["layer"]
    re_expanded = reindex_to_full_layers(trimmed, full["layer"], dtype=bool)
    xr.testing.assert_equal(re_expanded, full)


def test_allocate_rch_cells_drop_empty_layers_matches_full(basic_riv_inputs):
    active, _, _, _, _, layer = basic_riv_inputs
    nrow, ncol = active.sizes["y"], active.sizes["x"]
    rate = xr.DataArray(
        np.full((nrow, ncol), 0.001), {"y": active.y, "x": active.x}, ("y", "x")
    )
    active2d_active = (
        active  # active already has layer dim, at_first_active uses it directly
    )

    full = allocate_rch_cells(
        ALLOCATION_OPTION.at_first_active,
        active2d_active,
        rate,
        drop_empty_layers=False,
    )
    trimmed = allocate_rch_cells(
        ALLOCATION_OPTION.at_first_active, active2d_active, rate, drop_empty_layers=True
    )

    assert trimmed.sizes["layer"] <= full.sizes["layer"]
    re_expanded = reindex_to_full_layers(trimmed, full["layer"], dtype=bool)
    xr.testing.assert_equal(re_expanded, full)


class TestAllocationLayerCount:
    def test_allocate_riv_cells_does_not_grow_beyond_real_extent(self, n_layers):
        """
        The allocated result may keep the full model `layer` coordinate
        (that part is unavoidable, see investigation doc), but the number
        of layers that actually contain True values should not depend on
        total model layer count - it should stay pinned to how many
        layers the stage/bottom_elevation genuinely intersect (here: 1).
        """
        active, top, bottom = make_model_grid(n_layers)
        stage, bottom_elevation = make_sparse_riv()

        riv_cells, _ = allocate_riv_cells(
            ALLOCATION_OPTION.stage_to_riv_bot,
            active,
            top,
            bottom,
            stage,
            bottom_elevation,
        )

        assert n_nonempty_layers(riv_cells) == 1, (
            "Number of allocated layers should not scale with total model "
            f"layers (got layers with True values for n_layers={n_layers})"
        )


class TestRegridLayerCount:
    def test_regrid_preserves_sparse_layer_count(self, n_layers):
        """
        A source array pre-trimmed to 2 real layers should not become
        denser after regridding onto a destination grid with n_layers
        model layers - only 2 destination layers should end up non-nan.
        """
        real_layers = 2
        _, src_top, src_bot = make_model_grid(n_layers)
        _, dst_top, dst_bot = make_model_grid(n_layers)  # same discretization here

        # Sparse source: only first `real_layers` layers have data, rest all-nan.
        source = xr.full_like(src_top, np.nan)
        source.values[:real_layers] = 1.0

        regridder = LayerRegridder(method="mean")
        result = regridder.regrid(source, src_top, src_bot, dst_top, dst_bot)

        assert n_nonempty_layers(result) == real_layers, (
            "Regridding introduced extra non-nan layers beyond the "
            f"source's real extent (n_layers={n_layers})"
        )

    def test_regrid_sparse_input_matches_dense_input_result(self, n_layers):
        """
        Regression guard: regridding a package pre-trimmed to its real
        layers should give the same numerical result as regridding the
        same package padded out to the full model layer range with nan.
        This is the property that allows allocation to safely trim layers
        before regridding without changing behaviour.
        """
        real_layers = 2
        _, src_top, src_bot = make_model_grid(n_layers)
        _, dst_top, dst_bot = make_model_grid(n_layers)

        dense_source = xr.full_like(src_top, np.nan)
        dense_source.values[:real_layers] = 1.0

        sparse_source = dense_source.isel(layer=slice(0, real_layers))
        sparse_top = src_top.isel(layer=slice(0, real_layers))
        sparse_bot = src_bot.isel(layer=slice(0, real_layers))

        regridder = LayerRegridder(method="mean")
        dense_result = regridder.regrid(
            dense_source, src_top, src_bot, dst_top, dst_bot
        )
        sparse_result = regridder.regrid(
            sparse_source, sparse_top, sparse_bot, dst_top, dst_bot
        )

        xr.testing.assert_allclose(dense_result, sparse_result)


class TestClipLayerCount:
    def test_clip_by_grid_preserves_sparse_layers(self, n_layers):
        """
        Clipping a package to a smaller planar extent should not
        reintroduce layers that had no data before clipping.
        """
        active, top, bottom = make_model_grid(n_layers)
        stage, bottom_elevation = make_sparse_riv()

        riv_cells, _ = allocate_riv_cells(
            ALLOCATION_OPTION.stage_to_riv_bot,
            active,
            top,
            bottom,
            stage,
            bottom_elevation,
        )

        # Clip to a smaller planar window.
        x_slice = slice(0, 500.0)
        y_slice = slice(0.0, -500.0)
        clipped = riv_cells.sel(x=x_slice, y=y_slice)

        assert n_nonempty_layers(clipped) <= n_nonempty_layers(riv_cells), (
            "Clipping should never increase the number of non-empty layers"
        )


class TestMaskLayerCount:
    def test_mask_does_not_densify_layers(self, n_layers):
        """
        Masking with idomain (full n_layers) should not turn a
        sparse-layer package dense via alignment/broadcasting.
        """
        active, top, bottom = make_model_grid(n_layers)
        stage, bottom_elevation = make_sparse_riv()

        riv_cells, _ = allocate_riv_cells(
            ALLOCATION_OPTION.stage_to_riv_bot,
            active,
            top,
            bottom,
            stage,
            bottom_elevation,
        )

        idomain = active.astype(int)  # full n_layers, all active
        masked = riv_cells.where(idomain > 0)

        assert n_nonempty_layers(masked) == n_nonempty_layers(riv_cells), (
            "Masking against a full-layer idomain changed the number of "
            "non-empty layers - likely due to alignment/broadcasting"
        )


class TestSplitLayerCount:
    def test_split_preserves_sparse_layers_per_partition(self, n_layers):
        """
        Partitioning by a planar label array should not force a
        sparse-layer package to become dense in any partition.
        """
        active, top, bottom = make_model_grid(n_layers)
        stage, bottom_elevation = make_sparse_riv()

        riv_cells, _ = allocate_riv_cells(
            ALLOCATION_OPTION.stage_to_riv_bot,
            active,
            top,
            bottom,
            stage,
            bottom_elevation,
        )

        # Simple 2-partition planar label: left half / right half.
        label = xr.zeros_like(riv_cells.isel(layer=0, drop=True), dtype=int)
        ncol = label.sizes["x"]
        label[:, ncol // 2 :] = 1

        for part in [0, 1]:
            part_mask = label == part
            partitioned = riv_cells.where(part_mask)
            assert n_nonempty_layers(partitioned) <= n_nonempty_layers(riv_cells), (
                f"Partition {part} has more non-empty layers than the "
                "original unpartitioned array"
            )
