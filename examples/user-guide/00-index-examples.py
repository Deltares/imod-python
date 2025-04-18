# %%
# Let's first create some raster data to work with. 
import xugrid as xu
import xarray as xr
import numpy as np

elevation_uda = xu.data.elevation_nl()

part = elevation_uda.ugrid.sel(
    x=slice(100_000.0, 200_000.0),
    y=slice(400_000.0, 500_000.0)
)

# %%
# Make structured grid
import imod
resolution = 500.0
xmin, ymin, xmax, ymax = part.ugrid.grid.bounds
structured_grid = imod.util.empty_2d(resolution, xmin, xmax, resolution, ymin, ymax)
# Interpolate
regridder = xu.BarycentricInterpolator(part, structured_grid)
interpolated_elevation = regridder.regrid(part)
# Plot
interpolated_elevation.plot.imshow()

# %%
# Save to GeoTIFF
import imod
import rioxarray

# Create a temporary directory to store the data
tmp_dir = imod.util.path.temporary_directory()
tmp_dir.mkdir(parents=True, exist_ok=True)
interpolated_elevation.rio.to_raster(tmp_dir / "elevation.tif")

# %%
# Seamlessly integrate your GIS rasters or meshes with MODFLOW6, by using `xarray`_
# and `xugrid`_ arrays, for structured and unstructured grids respectively, to
# create grid-based model packages. 

import imod
# Open Geotiff with elevation data as xarray DataArray
elevation = imod.rasterio.open(tmp_dir / "elevation.tif")
elevation.load()
# Create idomain grid
layer_template = xr.DataArray([1, 1, 1], dims=('layer',), coords={'layer': [1, 2, 3]})
idomain = layer_template * xr.ones_like(elevation).astype(int)
# Compute bottom elevations of model layers
layer_thickness = xr.DataArray([10.0, 20.0, 10.0], dims=('layer',), coords={'layer': [1, 2, 3]})
bottom = elevation - layer_thickness.cumsum(dim='layer')
# Create MODFLOW 6 DIS package
dis_pkg = imod.mf6.StructuredDiscretization(
    idomain=idomain, top=elevation, bottom=bottom.transpose("layer", "y", "x")
)

# %%
# Assign wells based on x, y coordinates and filter screen depths, instead of
# layer, row and column:
import pandas as pd

screen_top = [0.0, 0.0]
screen_bottom = [-4.0, -10.0]
x = [150_200.0, 160_800.0]
y = [450_300.0, 460_200.0]
# Create transient well package, rates every 2 days
weltimes = pd.date_range("2000-01-01", "2000-01-03", freq="2D")
rate = xr.DataArray([[0.5, 1.0], [2.5, 3.0]], coords={"time": weltimes}, dims=("time","index"))
wel_pkg = imod.mf6.Well(x=x, y=y, screen_top=screen_top, screen_bottom=screen_bottom, rate=rate)
# iMOD Python will take care of the rest and assign the wells to the correct model
# layers upon writing the model. It will furthermore distribute well rates based
# on transmissivities. To verify how wells will be assigned to model layers before
# writing the entire simulation, you can use the following command:
wel_mf6_pkg = wel_pkg.to_mf6_pkg(idomain, elevation, bottom, k=1.0)
# Wells have been distributed across two model layers based on screen depths.
print(wel_mf6_pkg["cellid"])
# Well rates have been distributed based on screen overlap
print(wel_mf6_pkg["rate"])

# %%
# MODFLOW6 requires that all stress periods are defined in the time discretization
# package. However, usually boundary conditions are defined at insconsistent
# times. iMOD Python can help you to create a time discretization package that is
# consistent, based on all the unique times assigned to the boundary conditions.

simulation = imod.mf6.Modflow6Simulation("example")
simulation["gwf"] = imod.mf6.GroundwaterFlowModel()
simulation["gwf"]["dis"] = dis_pkg
simulation["gwf"]["wel"] = wel_pkg
simulation.create_time_discretization(additional_times=["2000-01-07"])
# Note that timesteps in well package are also inserted in the time
# discretization
print(simulation["time_discretization"].dataset)

# %%
# Regrid MODFLOW6 models to different grids, even from structured to unstructured
# grids. iMOD Python takes care of properly scaling the input parameters. You can
# also configure scaling methods yourself for each input parameter, for example
# when you want to upscale drainage elevations with the minimum instead of the
# average.

new_unstructured_grid = part

sim_regridded = simulation.regrid_like("regridded", new_unstructured_grid)
# Notice that discretization has converted to VerticesDiscretization (DISV)
print(sim_regridded["gwf"]["dis"])

# %%
# Clip models to a bounding box.

sim_clipped = simulation.clip_box(x_min=125_000, x_max=175_000, y_min=425_000, y_max=475_000)

# %%
