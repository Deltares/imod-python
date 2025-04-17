.. currentmodule:: imod.prepare

Prepare model input
-------------------

.. autosummary::
    :toctree: generated/prepare

    Regridder
    LayerRegridder
    Voxelizer

    fill
    laplace_interpolate

    polygonize

    reproject

    rasterize
    gdal_rasterize
    celltable
    rasterize_celltable

    zonal_aggregate_polygons
    zonal_aggregate_raster

    linestring_to_square_zpolygons
    linestring_to_trapezoid_zpolygons

    assign_wells

    get_lower_active_grid_cells
    get_lower_active_layer_number
    get_upper_active_grid_cells
    get_upper_active_layer_number
    create_layered_top

    ALLOCATION_OPTION
    DISTRIBUTING_OPTION
    SimulationAllocationOptions
    SimulationDistributingOptions
    allocate_drn_cells
    allocate_ghb_cells
    allocate_rch_cells
    allocate_riv_cells
    c_leakage
    c_radial
    distribute_drn_conductance
    distribute_ghb_conductance
    distribute_riv_conductance
    split_conductance_with_infiltration_factor

    cleanup_drn
    cleanup_ghb
    cleanup_riv
    cleanup_wel
    cleanup_wel_layered
