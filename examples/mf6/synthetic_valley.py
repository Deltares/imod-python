"""
Synthetic valley
================

This example rebuilds the "Synthetic Valley" problem (also known as the
McDonald Valley problem) with imod-python. The problem was introduced in the
FloPy paper by `Hughes and others (2023)
<https://doi.org/10.1111/gwat.13327>`_ and is used in the `MODFLOW 6 example
suite <https://github.com/MODFLOW-ORG/modflow6-examples>`_ as
``ex-gwt-synthetic-valley``, and in the USGS MODFLOW and FloPy workshops. It is
a useful test case because it combines a lot of features that practitioners use
every day.

The model represents an alluvial valley, surrounded by low permeability
bedrock, with:

* a **Voronoi grid** (``DISV``) with local refinement around a river, a
  lake and three pumping wells;
* **six layers**: two unconfined aquifer layers, a confining unit that only
  exists in the north, and three layers of a lower aquifer;
* topography and hydraulic conductivity from **raster files**;
* **recharge** and **evapotranspiration**;
* three **pumping wells** in the lower aquifer;
* a **lake** (``LAK``) with rainfall and evaporation;
* **drains** below the lake area;
* a **river** that crosses the southern half of the valley.

This example is written for people that know MODFLOW 6 and want to see how
the same model looks in imod-python. For every step, we mention the MODFLOW 6
package that is being created.

.. note::

    This is not an exact copy of the original model, which also contains a
    solute transport model. The differences are:

    * We only build the **flow model**.
    * imod-python currently has no Streamflow Routing (``SFR``) package. The
      river is represented by the River (``RIV``) package instead, with a
      prescribed stage. There is no routing of stream flow.
    * imod-python does not support the linear scaling of drain conductance
      with depth (``AUXDEPTHNAME`` in the ``DRN`` package). Our drains use a
      constant conductance.
    * The grid is generated with `pandamesh
      <https://deltares.github.io/pandamesh/>`_ instead of Triangle through
      FloPy, so the cells are not exactly the same as in the original model.

    The point of this example is to show the imod-python workflow, not to
    reproduce the numbers of the original publication. As a check, we ran the
    same grid and input with FloPy, including an ``SFR`` package and scaled
    drains. The heads differ by 0.07 m on average, and by at most 0.23 m, for
    heads that range from -2 to 5 m (the lowest heads are located in the cells
    with the wells).
"""

# sphinx_gallery_thumbnail_number = 3

# %%
# Imports
# -------
#
# imod-python builds on `xarray <https://docs.xarray.dev>`_ for gridded data
# and on `xugrid <https://deltares.github.io/xugrid/>`_ for unstructured grids.
# `pandamesh <https://deltares.github.io/pandamesh/>`_ is used to generate the
# mesh, and `pooch <https://www.fatiando.org/pooch/>`_ to download the input
# rasters.

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandamesh as pm
import pandas as pd
import pooch
import shapely
import xarray as xr
import xugrid as xu
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

import imod

# %%
# Problem definition
# ------------------
#
# The original model uses a mix of feet and meters. We work in meters and days
# throughout. The geometry of the model features is defined in feet in the
# original example, so we convert it first.

FT = 0.3048  # meter per foot

nlay = 6
rainfall = 0.0025  # m/d
evaporation = 0.0019  # m/d, maximum rate
extinction_depth = 1.0  # m
initial_head = 11.0  # m
simulation_length = 10957.5  # d (30 years)

lake_xy = (
    np.array(
        [
            (1500, 18500),
            (3500, 18500),
            (3500, 15500),
            (4000, 15500),
            (4000, 14500),
            (4500, 14500),
            (4500, 12000),
            (2500, 12000),
            (2500, 12500),
            (2000, 12500),
            (2000, 14000),
            (1500, 14000),
            (1500, 15000),
            (1000, 15000),
            (1000, 18000),
            (1500, 18000),
        ]
    )
    * FT
)
river_xy = np.array([(4250, 8750), (4250, 0)]) * FT  # flows north to south
river_box_xy = np.array([(3500, 0), (3500, 9500), (5000, 9500), (5000, 0)]) * FT
well_xy = np.array([(7250, 17250), (7750, 2750), (2750, 3750)]) * FT
well_names = ["P3", "P1", "P2"]
well_rates = [-1900.0, -7600.0, -7600.0]  # m3/d

domain = shapely.box(0.0, 0.0, 12500 * FT, 20000 * FT)
lake = shapely.Polygon(lake_xy)
river = shapely.LineString(river_xy)
river_box = shapely.Polygon(river_box_xy)

# %%
# Generating the Voronoi grid
# ---------------------------
#
# The original example uses a Voronoi grid, refined around the river, the lake
# and the wells. This is a perfect fit for the MODFLOW 6 ``DISV`` package.
#
# Here, we generate the mesh with pandamesh. It takes a GeoDataFrame with
# polygons and a ``cellsize`` column. The cell size is the edge length of the
# triangles, so we ask for fine triangles in the refinement zones, and for
# coarser triangles in the rest of the domain. Points are forced into the
# triangulation: we use them to place a vertex every 25 m along the river, and
# at the location of each well.

well_zones = [shapely.Point(xy).buffer(100.0) for xy in well_xy]
background = domain.difference(shapely.union_all([lake, river_box, *well_zones]))

n_river = int(np.ceil(river.length / 25.0))
river_points = [
    river.interpolate(distance)
    for distance in np.linspace(0.0, river.length, n_river + 1)[:-1]
]
well_points = [shapely.Point(xy) for xy in well_xy]

zones = gpd.GeoDataFrame(
    {"cellsize": [150.0, 45.0, 40.0] + [35.0] * len(well_zones)},
    geometry=[background, lake, river_box, *well_zones],
)
point_geometry = [*river_points, *well_points]
points = gpd.GeoDataFrame(
    {"cellsize": np.full(len(point_geometry), np.nan)}, geometry=point_geometry
)
mesh_input = pd.concat([zones, points], ignore_index=True)

triangles = pm.TriangleMesher(mesh_input).generate_ugrid()

# %%
# The Voronoi diagram of the triangulation is the dual of the triangulation:
# every triangle vertex becomes the center of a Voronoi cell. This is a proper
# "CVFD" grid for MODFLOW 6: the line between two cell centers is perpendicular
# to the shared cell face.
#
# Triangulations with very regular spacing contain groups of four or more
# points that lie on a common circle. This results in Voronoi vertices that
# coincide up to rounding errors, and thereby in cell faces with an (almost)
# zero length. These are unwanted, since MODFLOW 6 input files are written with
# finite precision. We merge vertices closer than a millimeter together.


def merge_close_nodes(grid, tolerance=1.0e-3):
    xy = np.column_stack([grid.node_x, grid.node_y])
    pairs = cKDTree(xy).query_pairs(tolerance, output_type="ndarray")
    n = len(xy)
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    _, first = np.unique(labels, return_index=True)
    new_xy = xy[first]

    faces = np.full(grid.face_node_connectivity.shape, grid.fill_value)
    for i, row in enumerate(grid.face_node_connectivity):
        nodes = labels[row[row != grid.fill_value]]
        keep = nodes != np.roll(nodes, 1)  # drop consecutive repeated nodes
        faces[i, : keep.sum()] = nodes[keep]
    return xu.Ugrid2d(new_xy[:, 0], new_xy[:, 1], grid.fill_value, faces)


grid = merge_close_nodes(triangles.tesselate_circumcenter_voronoi())
print(f"The grid has {grid.n_face} cells per layer.")

# %%
fig, axes = plt.subplots(1, 3, figsize=(12, 7), gridspec_kw={"width_ratios": [1, 1, 1]})
xmin, ymin, xmax, ymax = grid.bounds
views = [
    ("Entire grid", (xmin, xmax, ymin, ymax)),
    ("River and wells", (900, 1700, 0, 700)),
    ("Lake", (200, 1500, 3600, 5700)),
]
for ax, (title, (x0, x1, y0, y1)) in zip(axes, views):
    grid.plot(ax=ax, color="grey", linewidth=0.3)
    ax.plot(*lake.exterior.xy, color="tab:blue")
    ax.plot(*river.xy, color="tab:cyan", linewidth=2)
    ax.scatter(*well_xy.T, color="tab:red", zorder=3)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect(1.0)
    ax.set_title(title)
fig.tight_layout()

# %%
# Loading the input rasters
# -------------------------
#
# The land surface elevation, the bottom of the aquifer, the hydraulic
# conductivity of the aquifer and the confining unit, and the location of the
# lake are provided as GeoTIFFs in the MODFLOW 6 examples repository. These are
# 40 by 25 rasters with a cell size of 500 feet (152.4 m).

base_url = (
    "https://github.com/MODFLOW-ORG/modflow6-examples/raw/master/data/"
    "ex-gwt-synthetic-valley"
)
raster_files = {
    "top": ("top_SI.tif", "md5:781155bdcc2b9914e1cad6b10de0e9c7"),
    "bottom": ("bottom_SI.tif", "md5:00b4a39fbf5180e65c0367cdb6f15c93"),
    "k_aquifer": ("k_aq_SI.tif", "md5:d233e5c393ab6c029c63860d73818856"),
    "k_clay": ("k_clay_SI.tif", "md5:a08999c37f42b35884468e4ef896d5f9"),
    "lake": ("lake_location_SI.tif", "md5:38600d6f0eef7c033ede278252dc6343"),
}
rasters = {}
for name, (fname, known_hash) in raster_files.items():
    path = pooch.retrieve(
        url=f"{base_url}/{fname}",
        fname=fname,
        known_hash=known_hash,
        path=pooch.os_cache("imod-python"),
    )
    rasters[name] = imod.rasterio.open(path)

rasters["top"]

# %%
# Regridding to the unstructured grid
# ------------------------------------
#
# The rasters need to be converted to data on the Voronoi grid. xugrid provides
# regridders for this purpose.
#
# * Elevations are interpolated linearly (barycentric interpolation on the
#   triangulated raster cell centers). Cells at the edge of the domain are
#   outside the raster cell centers and are filled by nearest neighbor.
# * Hydraulic conductivity and lake location are categorical-like data: we
#   take the value of the raster cell that contains the cell center.


def drop_raster_coords(da):
    return da.drop_vars(["dx", "dy"], errors="ignore")


def regrid_linear(da):
    linear = xu.BarycentricInterpolator(da, grid).regrid(da)
    nearest = xu.CentroidLocatorRegridder(da, grid).regrid(da)
    return drop_raster_coords(linear.fillna(nearest))


def regrid_nearest(da):
    return drop_raster_coords(xu.CentroidLocatorRegridder(da, grid).regrid(da))


top = regrid_linear(rasters["top"])
bottom_aquifer = regrid_linear(rasters["bottom"])
k_aquifer = regrid_nearest(rasters["k_aquifer"])
k_clay = regrid_nearest(rasters["k_clay"])
is_lake = regrid_nearest(rasters["lake"]) == 1.0

# %%
# The confining unit (layer 3) is present where the clay conductivity raster
# has a value lower than 60 m/d. Elsewhere the raster holds a placeholder value.
# In those locations, layer 3 is made inactive.

has_clay = k_clay < 60.0

fig, axes = plt.subplots(2, 2, figsize=(10, 12))
for ax, (da, title, kwargs) in zip(
    axes.ravel(),
    [
        (top, "Land surface (m)", {"cmap": "terrain"}),
        (bottom_aquifer, "Bottom of aquifer (m)", {"cmap": "viridis"}),
        (k_aquifer, "Hydraulic conductivity of aquifer (m/d)", {"cmap": "plasma"}),
        (has_clay.astype(float), "Confining unit present", {"cmap": "Greys"}),
    ],
):
    da.ugrid.plot(ax=ax, **kwargs)
    ax.set_title(title)
    ax.set_aspect(1.0)
fig.tight_layout()

# %%
# Discretization - DISV
# ---------------------
#
# Now we have all the data to set up the model. We start by creating an
# :class:`imod.mf6.GroundwaterFlowModel`. Since the water table is located in
# the model layers, the model needs the Newton-Raphson formulation, just like
# in the original model.
#
# The MODFLOW 6 ``DISV`` package is created with
# :class:`imod.mf6.VerticesDiscretization`. imod-python requires:
#
# * ``top``: the top of the model, a 2D array;
# * ``bottom``: the bottom of every layer, a 3D array;
# * ``idomain``: a 3D array that is 1 for active cells, and -1 for cells that
#   should be skipped vertically, but connect the cells above and below.
#
# Note that you do not provide vertices or ``cell2d`` information: this is
# derived from the grid of the data. The layer information is carried by the
# ``layer`` coordinate.

layer = np.arange(1, nlay + 1)

# The bottom elevations are, in feet: -5 (layer 1), -50 (layer 2), -51
# (layer 3, the confining unit, which is a thin layer), and -100 (layer 4).
# Layers 5 and 6 divide the remaining thickness down to the bottom of the
# aquifer in two equal parts. Where there is no confining unit, layer 3 has
# a bottom at -50 feet, and is inactive.
bottom_layers = [
    xu.full_like(top, -5.0 * FT),
    xu.full_like(top, -50.0 * FT),
    xu.full_like(top, -51.0 * FT).where(has_clay, -50.0 * FT),
    xu.full_like(top, -100.0 * FT),
    bottom_aquifer + 0.5 * (-100.0 * FT - bottom_aquifer),
    bottom_aquifer,
]
bottom = xu.concat(bottom_layers, dim="layer").assign_coords(layer=layer)

ones = xu.full_like(top, 1, dtype=int)
idomain = xu.concat(
    [ones, ones, ones.where(has_clay, -1), ones, ones, ones], dim="layer"
).assign_coords(layer=layer)

gwf_model = imod.mf6.GroundwaterFlowModel(newton=True, under_relaxation=True)
gwf_model["disv"] = imod.mf6.VerticesDiscretization(
    top=top, bottom=bottom, idomain=idomain
)

# %%
# Node property flow - NPF
# ------------------------
#
# The hydraulic conductivity is the same in all layers, except in the confining
# unit. Vertical anisotropy is 4 in the aquifer layers. The first layer is
# convertible (``icelltype=1``), the other layers are confined. The XT3D option
# is used since a Voronoi grid is not perfectly orthogonal everywhere.
#
# Many array inputs of MODFLOW 6 can be given per layer. In imod-python, you
# can do so by providing a DataArray with a layer coordinate. xarray broadcasts
# it where necessary. Cells that are inactive (``idomain`` of -1 or 0) should be
# ``NaN`` in the data.

per_layer = lambda values: xr.DataArray(  # noqa: E731
    values, coords={"layer": layer}, dims=("layer",)
)
is_aquifer_layer = per_layer([True, True, False, True, True, True])

dims = ("layer", grid.face_dimension)
k_horizontal = k_aquifer.where(is_aquifer_layer, k_clay).where(idomain > 0)
k_vertical = (0.25 * k_aquifer).where(is_aquifer_layer, k_clay).where(idomain > 0)
k_horizontal = k_horizontal.transpose(*dims)
k_vertical = k_vertical.transpose(*dims)

gwf_model["npf"] = imod.mf6.NodePropertyFlow(
    icelltype=per_layer([1, 0, 0, 0, 0, 0]),
    k=k_horizontal,
    k33=k_vertical,
    xt3d_option=True,
    save_flows=True,
    save_specific_discharge=True,
    save_saturation=True,
)

# %%
# Initial conditions and storage - IC and STO
# -------------------------------------------
#
# The model is a steady-state model. In MODFLOW 6, a model is steady-state when
# no storage package is present, or when the storage package specifies a
# steady-state stress period. imod-python always requires a storage package; to
# get a steady-state simulation set ``transient=False``. The storage
# parameters are therefore never used.

gwf_model["ic"] = imod.mf6.InitialConditions(start=initial_head)
gwf_model["sto"] = imod.mf6.SpecificStorage(
    specific_storage=1.0e-5,
    specific_yield=0.15,
    transient=False,
    convertible=0,
)

# %%
# Recharge and evapotranspiration - RCH and EVT
# ----------------------------------------------
#
# These are the ``RCH`` and ``EVT`` packages of MODFLOW 6. imod-python takes
# a 2D array with a single layer: the recharge and the evapotranspiration are
# applied to the first layer. For ``EVT``, a flux is moved to an underlying cell
# if the specified cell is inactive, unless ``fixed_cell=True``.
#
# The :class:`imod.mf6.Evapotranspiration` package expects the maximum rate, the
# ET surface and the extinction depth. The simple linear ET function of the
# original model is obtained by not providing the optional ``proportion_rate``
# and ``proportion_depth``.

top_layer = top.assign_coords(layer=1)

gwf_model["rch"] = imod.mf6.Recharge(
    rate=xu.full_like(top_layer, rainfall),
    save_flows=True,
)
gwf_model["evt"] = imod.mf6.Evapotranspiration(
    surface=top_layer,
    rate=xu.full_like(top_layer, evaporation),
    depth=xu.full_like(top_layer, extinction_depth),
    save_flows=True,
)

# %%
# Wells - WEL
# -----------
#
# The three wells extract from the lower aquifer (layer 6). In imod-python wells
# are specified by an x and y coordinate together with a screen top and bottom
# in elevation. This is converted to the cell and layer in which the well is
# located when writing the model, and the rate is distributed over layers
# according to the transmissivity.
#
# Here, the screens span the bottom five meters of the model.

well_cell = grid.locate_points(well_xy)
well_bottom = bottom_aquifer.values[well_cell]

gwf_model["wel"] = imod.mf6.Well(
    x=well_xy[:, 0],
    y=well_xy[:, 1],
    screen_top=well_bottom + 5.0,
    screen_bottom=well_bottom,
    rate=well_rates,
    id=well_names,
    save_flows=True,
)

# %%
# Drains - DRN
# ------------
#
# The original model has drains in the lake area, with a conductance that is
# derived from a vertical hydraulic conductivity of 0.03048 m/d and a bed
# thickness of 0.3048 m. imod-python expects a drain *conductance* per cell, so
# we multiply by the area of the cell. Note that xugrid computes the area of the
# cells for us.

drain_k = 0.03048  # m/d
drain_thickness = 0.3048  # m

cell_area = xu.UgridDataArray(
    xr.DataArray(grid.area, dims=(grid.face_dimension,)),
    grid,
)
drain_conductance = (drain_k * cell_area / drain_thickness).where(is_lake)
drain_elevation = top.where(is_lake)

gwf_model["drn"] = imod.mf6.Drainage(
    elevation=drain_elevation.expand_dims(layer=[1]),
    conductance=drain_conductance.expand_dims(layer=[1]),
    save_flows=True,
)

# %%
# River - RIV
# -----------
#
# In the original model, the river is a streamflow routing network of reaches,
# each connected to a cell of the grid. Since SFR is not available in
# imod-python, we use the River package instead. A River boundary requires a
# stage, a conductance, and the bottom elevation of the river bed.
#
# We can reuse most of the SFR input for this:
#
# * the reaches are the grid cells that are crossed by the river line; we
#   obtain the length of the river in each cell by intersecting the line with the
#   grid;
# * the conductance of the river bed is the same in all reaches per unit length:
#   it is 1.8 million cubic feet per day over the entire length of the river;
# * the bed top elevation drops 0.0002 m/m going downstream, starting from -0.3
#   feet at the upstream end;
# * the bed is 0.3048 m thick, and 3.048 m wide.
#
# The stage is the only thing that SFR would solve from the Manning equation. For
# the River package we have to provide it. A fixed depth is a poor
# approximation, because the river gains water along its path so its depth
# increases. We make the depth grow linearly from 0.02 m at the upstream end
# to 0.38 m at the downstream end. These values are chosen based on the stage
# that the original SFR model computes.

_, river_faces, river_segments = grid.intersect_edges(river_xy[np.newaxis])
segment_length = np.linalg.norm(river_segments[:, 1] - river_segments[:, 0], axis=1)
distance = np.cumsum(segment_length) - 0.5 * segment_length  # to segment midpoint
total_length = segment_length.sum()

bed_thickness = 0.3048
bed_top = -0.3 * FT - 0.0002 * distance
depth = np.interp(distance, [0.0, total_length], [0.02, 0.38])
total_conductance = 1_800_000.0 * FT**3  # m3/d
conductance = total_conductance * segment_length / total_length


def river_array(values):
    """Place values for the river cells in a 2D (single layer) grid array."""
    data = np.full(grid.n_face, np.nan)
    data[river_faces] = values
    da = xu.UgridDataArray(xr.DataArray(data, dims=(grid.face_dimension,)), grid)
    return da.expand_dims(layer=[1])


gwf_model["riv"] = imod.mf6.River(
    stage=river_array(bed_top + depth),
    conductance=river_array(conductance),
    bottom_elevation=river_array(bed_top - bed_thickness),
    save_flows=True,
)

# %%
# Lake - LAK
# ----------
#
# The lake sits on top of the model. It is connected vertically to the cells of
# the first layer that are located below the lake. We use the
# ``LakeData`` class to describe one lake. All the grid-based arguments hold
# data in the connected cells and ``NaN`` elsewhere. This is the same as for the
# other grid-based packages. The values that describe the lake bed are the bed
# leakance (1/d), and "connection" geometry. The latter is only required for
# horizontal connections of lakes, and is ignored by MODFLOW 6 for vertical
# connections.
#
# Time series (here: rainfall and evaporation) are supplied as arrays with a
# ``time`` dimension.

VERTICAL = 1  # connection type of the lake with the cells below


def lake_array(value):
    return xu.full_like(top, value).where(is_lake).expand_dims(layer=[1])


start_time = np.datetime64("2000-01-01")


def timeseries(value):
    return xr.DataArray([value], coords={"time": [start_time]}, dims=("time",))


lake_data = imod.mf6.lak.LakeData(
    starting_stage=11.3 * FT,
    boundname="Blue_Lake",
    connection_type=lake_array(VERTICAL),
    bed_leak=lake_array(0.0013),
    top_elevation=lake_array(0.0),
    bot_elevation=lake_array(0.0),
    connection_length=lake_array(1.0),
    connection_width=lake_array(1.0),
    rainfall=timeseries(rainfall),
    evaporation=timeseries(evaporation),
)
gwf_model["lak"] = imod.mf6.Lake.from_lakes_and_outlets(
    [lake_data],
    print_flows=True,
    save_flows=True,
)

# %%
# Output control - OC
# -------------------

gwf_model["oc"] = imod.mf6.OutputControl(save_head="all", save_budget="all")

# %%
# Putting it together: the simulation
# -----------------------------------
#
# The simulation holds the model, the solver (``IMS``) and the time discretization
# (``TDIS``). We follow the "simple" complexity settings of the original model
# for the solver. Since a Newton model produces an asymmetric matrix, we have to
# use BiCGSTAB.
#
# The time discretization is created from a single stress period. For the
# steady-state model, the length of this period has no effect on the results.

simulation = imod.mf6.Modflow6Simulation("synthetic_valley")
simulation["gwf"] = gwf_model
simulation["solver"] = imod.mf6.Solution(
    modelnames=["gwf"],
    print_option="summary",
    outer_dvclose=1.0e-3,
    outer_maximum=25,
    inner_maximum=50,
    inner_dvclose=1.0e-3,
    inner_rclose=0.1,
    linear_acceleration="bicgstab",
    preconditioner_levels=7,
    preconditioner_drop_tolerance=1.0e-3,
    relaxation_factor=0.0,
)
simulation["time_discretization"] = imod.mf6.TimeDiscretization(
    timestep_duration=xr.DataArray(
        [simulation_length], coords={"time": [start_time]}, dims=("time",)
    )
)

# %%
# Write and run the model
# -----------------------
#
# Writing validates the input, and writes the MODFLOW 6 input files. Note that
# you can also inspect the generated files in the model directory.
# Running requires the MODFLOW 6 executable to be on your PATH.

modeldir = imod.util.temporary_directory()
simulation.write(modeldir, binary=True)
simulation.run()

# %%
# Results
# -------
#
# The heads are returned as a xugrid ``UgridDataArray`` with a ``layer``
# dimension. We plot the first layer (the water table aquifer) and the lowest
# layer, in which the wells are located.

head = simulation.open_head().isel(time=0)

fig, axes = plt.subplots(1, 2, figsize=(11, 8))
for ax, lay in zip(axes, [1, 6]):
    head.sel(layer=lay).ugrid.plot(
        ax=ax, cmap="viridis", vmin=0.0, vmax=5.0, cbar_kwargs={"label": "head (m)"}
    )
    ax.plot(*lake.exterior.xy, color="white", linewidth=1)
    ax.plot(*river.xy, color="tab:cyan", linewidth=2)
    ax.scatter(*well_xy.T, color="tab:red", zorder=3)
    ax.set_title(f"Head in layer {lay}")
    ax.set_aspect(1.0)
fig.tight_layout()

# %%
# Wells pump from the deepest layer, and the effect of the extraction is seen in
# the head in this layer. A cross-section through the valley from west to east
# shows how the heads in the aquifers relate to the land surface. The section is
# located at a distance of 4400 m from the southern boundary, so it crosses the
# lake and the confining unit.

y_section = 4400.0
start, end = (xmin, y_section), (xmax, y_section)
surface_section = top.ugrid.intersect_line(start, end)
distance_section = surface_section["mesh2d_s"]

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(distance_section, surface_section, color="black", label="land surface")
for lay in [1, 3, 6]:
    section = head.sel(layer=lay).ugrid.intersect_line(start, end)
    ax.plot(section["mesh2d_s"], section, label=f"head layer {lay}")
ax.set_xlabel("x (m)")
ax.set_ylabel("elevation (m)")
ax.legend()
ax.set_title(f"West-east cross-section at y = {y_section:.0f} m")

# %%
# Water balance
# -------------
#
# Because we asked for budget output for every package (``save_flows=True``), we
# can sum these up and compare the water coming in and out of the model. Positive
# values are flows into the model; negative values are flows out of the model.
# For a steady-state model, they should cancel each other.

budget = simulation.open_flow_budget()
packages = ["rch", "evt", "wel", "drn", "riv", "lak"]
flows = {name: float(budget[f"{name}_{name}"].isel(time=0).sum()) for name in packages}
water_balance = pd.DataFrame(
    {"flow (m3/d)": flows}, index=pd.Index(packages, name="package")
)
water_balance.loc["total"] = water_balance["flow (m3/d)"].sum()
water_balance.round(1)

# %%
# The sum of all flows (the last row) is very small compared to the individual
# terms, so the model is well balanced. The recharge is balanced by the wells,
# the drains, the river, and by evapotranspiration. The river is a net drain of
# the groundwater system. In the original model, this water would be routed
# downstream by the SFR package. In this version, it leaves the model at the
# River boundary. The lake slightly supplies the aquifer.

fig, ax = plt.subplots(figsize=(7, 4))
water_balance["flow (m3/d)"].iloc[:-1].plot.barh(ax=ax, color="tab:blue")
ax.axvline(0.0, color="black", linewidth=0.5)
ax.set_xlabel("flow into the model (m3/d)")
fig.tight_layout()

# %%
# From here
# ---------
#
# The original example continues with a groundwater transport model, coupled with the
# flow model by the ``FMI`` and ``SSM`` packages. imod-python has classes for
# transport models too: see :class:`imod.mf6.GroundwaterTransportModel`, and the
# ``transport_2d`` example for the workflow.
#
# Also have a look at the ``lake`` example for a lake with outlets, and at the
# ``hondsrug`` example for a larger regional model that is built from data of
# the Dutch subsurface.
