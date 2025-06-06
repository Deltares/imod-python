"""
Prepare model input.

A various selection of functions to mangle your data from one form into another,
such that it will fit into your model. This includes
:func:`imod.prepare.reproject` for reprojecting grids, and
:func:`imod.prepare.rasterize` to create grids from vector files.

Naturally you are free to write your own functions or one of the many existing
ones from other packages. As long as you keep your data in the expected
``xarray.DataArray`` and ``pandas.DataFrame`` formats, this will work. In some
cases, such as :class:`imod.prepare.Regridder`, these methods are optimized for
speed by making use of the Numba compiler, to be able to regrid large datasets.
"""

from imod.common.utilities.layer import create_layered_top
from imod.prepare import hfb, spatial, subsoil, surface_water
from imod.prepare.cleanup import (
    cleanup_drn,
    cleanup_ghb,
    cleanup_riv,
    cleanup_wel,
    cleanup_wel_layered,
)
from imod.prepare.hfb import (
    linestring_to_square_zpolygons,
    linestring_to_trapezoid_zpolygons,
)
from imod.prepare.layer import (
    get_lower_active_grid_cells,
    get_lower_active_layer_number,
    get_upper_active_grid_cells,
    get_upper_active_layer_number,
)
from imod.prepare.layerregrid import LayerRegridder
from imod.prepare.partition import create_partition_labels
from imod.prepare.regrid import Regridder
from imod.prepare.reproject import reproject
from imod.prepare.spatial import (
    celltable,
    fill,
    gdal_rasterize,
    laplace_interpolate,
    polygonize,
    rasterize,
    rasterize_celltable,
    zonal_aggregate_polygons,
    zonal_aggregate_raster,
)
from imod.prepare.topsystem import (
    ALLOCATION_OPTION,
    DISTRIBUTING_OPTION,
    SimulationAllocationOptions,
    SimulationDistributingOptions,
    allocate_drn_cells,
    allocate_ghb_cells,
    allocate_rch_cells,
    allocate_riv_cells,
    c_leakage,
    c_radial,
    distribute_drn_conductance,
    distribute_ghb_conductance,
    distribute_riv_conductance,
    split_conductance_with_infiltration_factor,
)
from imod.prepare.voxelize import Voxelizer
from imod.prepare.wells import assign_wells
