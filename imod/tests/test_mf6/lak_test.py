
import numpy as np
import xarray as xr
from imod.mf6.pkgbase import AdvancedBoundaryCondition

from imod.mf6.lake_package.lak import Lake


shape = nlay, nrow, ncol = 3, 9, 9

dx = 10.0
dy = -10.0
xmin = 0.0
xmax = dx * ncol
ymin = 0.0
ymax = abs(dy) * nrow
dims = ("layer", "y", "x")

layer = np.arange(1, nlay + 1)
y = np.arange(ymax, ymin, dy) + 0.5 * dy
x = np.arange(xmin, xmax, dx) + 0.5 * dx
coords = {"layer": layer, "y": y, "x": x}

idomain = xr.DataArray(np.ones(shape, dtype=np.int32), coords=coords, dims=dims)
lake = xr.full_like(idomain, dtype=np.float32, fill_value=0.)

#create 3 lakes

lakenames = ["Ijsselmeer","Vinkeveense plas", "Reeuwijkse plas"]
lake_starting_stages = [1.,2.,3.]
lake_bed_elevations = [2,3,4]
nlake =3
dimensions = ["lake_name"]
coordinates = {"lake_name": lakenames}

lake_array_layout = xr.DataArray(np.ones(nlake, dtype=np.float32), coords=coordinates, dims=dimensions)
bed_elevations = xr.full_like(lake_array_layout, fill_value=0, dtype=np.float32)
bed_elevations.data = lake_bed_elevations

boundnames = xr.full_like(lake_array_layout, fill_value="", dtype=np.str0)
boundnames.data = lakenames

starting_stages = xr.full_like(lake_array_layout, fill_value=0, dtype=np.float32)
starting_stages.data = lake_starting_stages

lake_numbers =  xr.full_like(lake_array_layout, fill_value=0, dtype=np.int32)
lake_numbers.data = np.arange(0, nlake)

# create 6 connections
'''
connection_nr  lake_nr cell_id connection_type bed_leak bottom_elevation top_elevation connection_width connection_length
1               1      3,4,1     vertical        0.2        -1              0             0.1             0.2
2               1      4,4,1     vertical        0.3        -2              0.1           0.2             0.3
3               1      3,5,1     vertical        0.4        -3              -0.1          0.3             0.4
4               2      17,4,1    horizontal     None        -4              0.2           0.5             0.6
5               2      18,4,1    horizontal     None        -5              -0.2          0.6             0.7
6               3      23,25,1   embeddedv      None        -6              0             0.7             0.8
'''

nconnect = 6
dimensions = ["connection_nr"]
coordinates = {"connection_nr": np.arange(0,nconnect)}
connection_array_layout = xr.DataArray(np.ones(nconnect, dtype=np.float32), coords=coordinates, dims=dimensions)
connection_lake_number = xr.full_like(connection_array_layout, fill_value=0, dtype=np.int32)
connection_lake_number.data = [1,1,1,2,2,3]
connection_type  = xr.full_like(connection_array_layout, fill_value=0, dtype=np.str0)
connection_type.data = ["vertical", "vertical", "vertical", "horizontal", "horizontal", "embeddedv"]
connection_bed_leak  = xr.full_like(connection_array_layout, fill_value=0, dtype=np.float32)
connection_bed_leak.data  = [0.2, 0.3, 0.4, -1,-1,-1]

connection_cell_id_index  = xr.full_like(connection_array_layout, fill_value=0, dtype=np.int32)
connection_cell_id_index.data  = [3,4,3,17,18, 23]
connection_cell_id_layer  = xr.full_like(connection_array_layout, fill_value=0, dtype=np.int32)
connection_cell_id_layer.data  = [1,1,2,1,1,1]

connection_bottom_elevation  = xr.full_like(connection_array_layout, fill_value=0, dtype=np.float32)
connection_bottom_elevation.data = [-1,-2,-3,-4,-5,-6]
connection_top_elevation  = xr.full_like(connection_array_layout, fill_value=0, dtype=np.float32)
connection_top_elevation.data = [0,0.1,-0.1,0.2,-0.2,0]
connection_width  = xr.full_like(connection_array_layout, fill_value=0, dtype=np.float32)
connection_width.data = [-1,-2,-3,-4,-5,-6]
connection_length  = xr.full_like(connection_array_layout, fill_value=0, dtype=np.float32)
connection_length.data = [-1,-2,-3,-4,-5,-6]


lake = Lake(lake_numbers, starting_stages, bed_elevations,boundnames, connection_lake_number,connection_cell_id_index,None, connection_cell_id_layer,connection_type,  connection_bed_leak,
connection_bottom_elevation, connection_top_elevation, connection_width, connection_length)
lake.render(None,None, None, False)