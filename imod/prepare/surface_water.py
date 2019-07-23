import numpy as np
import shapely.geometry as sg
import xarray as xr

# since geopandas is a big dependency that is sometimes hard to install
# and not always required, we made this an optional dependency
try:
    import geopandas as gpd
except ImportError:
    pass

import imod


def raster_to_features(raster):
    """
    Parameters
    ----------
    raster : xarray.DataArray
        containing coordinates x and y, uniformly spaced.
    """
    # generate shapes of cells to use for intersection
    dx, xmin, xmax, dy, ymin, ymax = imod.util.spatial_reference(raster)
    a_dx = abs(dx)
    a_dy = abs(dy)
    # "start" corners
    xs = np.arange(xmin, xmax, a_dx)
    ys = np.arange(ymin, ymax, a_dy)
    yy_s, xx_s = np.meshgrid(ys, xs)
    # "end" corners
    xe = np.arange(xmin + dx, xmax + dy, a_dx)
    ye = np.arange(ymin + dy, ymax + dy, a_dy)
    yy_e, xx_e = np.meshgrid(ye, xe)

    A = xx_s.flatten()
    B = xx_e.flatten()
    C = yy_s.flatten()
    D = yy_e.flatten()
    squares = [
        sg.Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
        for x1, x2, y1, y2 in zip(A, B, C, D)
    ]
    features = gpd.GeoDataFrame()
    features.geometry = squares
    return features


def coordinate_index(x, xmin, xmax, dx):
    """
    Takes care of reversed coordinates, typically y with a negative value for dy.
    """
    if dx < 0:
        xi = ((xmax - x) / abs(dx)).astype(int)
    else:
        xi = ((x - xmin) / abs(dx)).astype(int)
    return xi


def rivers(rivers_lines, width_column, depth_column, dem):
    # TODO check for model tops and bots
    # TODO replace ``dem`` by ``like`` or something more general
    buffered = []
    for _, row in rivers_lines.iterrows():
        width = row[width_column]
        row.geometry = row.geometry.buffer(width / 2.0)
        buffered.append(row)
    rivers_polygons = gpd.GeoDataFrame(buffered)

    # intersection
    gridshape = raster_to_features(dem)
    # TODO: probably replace by writing to shapefile
    # then call org2ogr intersect instead
    # since this requires geopandas-cython to perform acceptably.
    river_cells = gpd.overlay(rivers_polygons, gridshape, how="intersection")

    centroids = gpd.GeoDataFrame()
    centroids.geometry = river_cells.centroid
    centroids["x"] = centroids.geometry.x
    centroids["y"] = centroids.geometry.y
    centroids["area"] = river_cells.area
    centroids["depth"] = river_cells[depth_column]
    # calculate indices in grid out
    dx, xmin, xmax, dy, ymin, ymax = imod.util.spatial_reference(dem)
    centroids["xi"] = coordinate_index(centroids["x"].values, xmin, xmax, dx)
    centroids["yi"] = coordinate_index(centroids["y"].values, ymin, ymax, dy)

    # fill in outgoing grids
    nrow, ncol = dem.y.size, dem.x.size
    area = np.full((nrow, ncol), 0.0)

    # ensure it's within raster area
    centroids = centroids[(centroids["yi"] >= 0) & (centroids["yi"] < nrow)]
    centroids = centroids[(centroids["xi"] >= 0) & (centroids["xi"] < ncol)]

    # for area weighted depth
    depth_x_area = np.full((nrow, ncol), 0.0)
    for i, j, a, d in zip(
        centroids["yi"], centroids["xi"], centroids["area"], centroids["depth"]
    ):
        area[i, j] += a
        depth_x_area[i, j] += a * d
    depth = depth_x_area / area

    river_resistance = 100.0  # TODO
    conductance = xr.full_like(dem, area / river_resistance)
    is_river = conductance > 0.0
    depth_da = xr.full_like(dem, depth)
    stage = dem - 0.15 * depth_da  # TODO
    bottom = dem - depth_da
    infiltration_factor = xr.full_like(dem, 1.0)

    conductance = conductance.where(is_river)
    stage = stage.where(is_river)
    bottom = bottom.where(is_river)
    infiltration_factor = infiltration_factor.where(is_river)

    return conductance, stage, bottom, infiltration_factor
