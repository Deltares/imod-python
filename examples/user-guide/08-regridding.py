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
a thermodynamically intensive property (whose value do not depend intrinsically on the grid block size)
such as temperature or density can be regridded by an averaging approach (for upscaling) or sampling (for downscaling).
Extensive properties ( whose values do depend on the grid block size) include mass and pore volume, and the regridding
method should be chosen to reflect that.
Finally regridding methods for conductivity-like properties follow the rules for parallel or serial resistors.

Note that the different regridding methods may have a different output domain when regridding, in the sense that



Obtaining the final (i)domain
=============================
In many real-world models, some cells will be inactive or marked as "vertical passthrough" (VPT) in the idomain
array of the simulation. Some packages require that all cells that are inactictive or VPT in idomain
are excluded from the package as well. An example is the npf package: cells that are inactive or VPT, should
not have conductivity data in the npf package. 
Therefore at the end of the regridding process, a final step consists in enforcing consistency between those
idomain and all the packages. This is a 2-step process:
1) for cells that do not have inputs in crucial packages like npf or storage, idomain will be set to inactive.
2) for cells that are marked as inactive or VPT in idomain, all package inputs will be removed from all the packages
 

Regridding using default methods
================================

The regrid_like function is available on packages, models and simulations.
When the default methods are acceptable, regridding the whole simulation is the most convenient 
from a user-perspective. 
When non-default methods are used for one or more packages,  these should be regridded separately.
In that case, the most convenient approach is likely:
-pop the packages that should use non-default methods from the source simulation 
(except if it are packages that are required for the simulation to be regridded)
-regrid the source simulation: this takes care of all the packages that should use default methods. 
-regrid the package(s) where you want to use non-standard rergridding methods indivudually
starting from the packages in the source simulation
-insert the custom-regridded packages to the regridded simulation

Regridding using non-default methods
====================================


Regridding boundary conditions
==============================
Special care must be taken when regridding boundary conditions, and it is recommended that users 
verify the balance output of a regridded simulation and compare it to the original model.
If the regridded simulation is a good representation of the original simulation, the mass contributions
on the balance by the different boundary conditions should be comparable in both simulations.
To achieve this, it may be nevessary to tweak the input or the regridding methods.
An example of this is upscaling recharge (so the target grid has coarser cells than the source grid). 
Its default method is averaging, with the following rules:
- if a cell in the source grid is inactive in the source recharge package (meaning no recharge), it will not count when averaging.
So if a target cell has partial overlap with one source recharge cell, and the rest of the target cell has no overlap with any
source active recharge cell, it will get the recharge of the one cell it has overlap with. But since the target cell is larger, 
this effectively means the regridded recharge will be more in the regridded simulation than it was in the source simulation
- but we do the same regridding this time assigning a zero recharge to cells without recharge then the averaging will
take the zero-recharge cells into account and the regridded recharge will be the same as the source recharge.



If we have a recharge package containing only those cells
that receive recharge and we regrid it with default methods, then cells in the target grid will get a
different amount of recharge

"""
