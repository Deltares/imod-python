"""
Regridding
==========

Most MF6 packages have spatial data arrays as input. These arrays 
are discrete: they are defined over the simulation grid and contain
values associated to each cell.

Regridding these package means: create a new package with
spatial data arrays defined over a different grid. Computing what
the values in these new arrays should be, is done by xugrid.

xugrid will compute a regridded array based on
-the original array 
-the original discretization (this is described in the coordinates of the original arry)
-the new discretization
-a regridding method

More information on the available regridding methods can be found in the xugrid documentation

The regridding method that should be used depends on the property being regridded. For example
a thermodynamically intensive property (whose value would not increase on a coarser discretization)
such as temperature or density can be regridded by an averaging approach (for upscaling) or sampling (for downscaling).
Extensive properties ( whose values would increase on choosing a coarser discretization) include mass and pore volume:
a bigger gridblock will contain more pore volume if the porosity is the same, and the regridding method should be chosen to reflect that.
Finally regridding methods for some physical parameters such as conductivity 


Available methods include

"""
