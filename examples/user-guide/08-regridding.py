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
https://deltares.github.io/xugrid/user_guide.html

The regridding method that should be used depends on the property being
regridded. For example a thermodynamically intensive property (whose value do
not depend intrinsically on the grid block size) such as temperature or density
can be regridded by an averaging approach (for upscaling) or sampling (for
downscaling). Extensive properties (whose values do depend on the grid block
size) include (water) mass and pore volume of a gridblock, and the regridding method should be chosen
to reflect that. Finally regridding methods for conductivity-like properties
follow the rules for parallel or serial resistors- at least when the tensor rotation angles
are constant or comparable in the involved gridblocks.

Note that the different regridding methods may have a different output domain
when regridding: if the original array has no-data values in some cells, then
the output array may have no-data values as well, and where these end up depends
on the chosen regridding method. Also note that regridding is only possible in
the xy-plane, and not across the layer dimension. The output array will have the
same number of layers as the input array. 



Obtaining the final (i)domain
============================= 

In many real-world models, some cells will be inactive or marked as "vertical
passthrough" (VPT) in the idomain array of the simulation. Some packages require
that all cells that are inactictive or VPT in idomain are excluded from the
package as well. An example is the npf package: cells that are inactive or VPT
in idomain, should not have conductivity data in the npf package. Therefore at
the end of the regridding process, a final step consists in enforcing
consistency between those of idomain and all the packages. This is a 2-step
process:

1) for cells that do not have inputs in crucial packages like npf or storage,
   idomain will be set to inactive.
2) for cells that are marked as inactive or VPT in idomain, all package inputs
   will be removed from all the packages
 
This synchronization step between idomain and the packages is automated, and it
is carried out when calling regrid_like on the simulation object or the model
object. There are 2 caveats:
1) the synchronization between idomain and the package domains is done on the
   model-level. If we have a simulation containing both a flow model and a
   transport model then idomain for flow is determined independent from that for
   transport. These models may therefore end up using different domains (this
   may lead to undesired results, so a manual synchronization may be necessary
   between the flow regridding and the transport regridding) This manual
   synchronization can be done using the "mask_all_packages" function- this
   function removes input from all packages that are marked as inactive or VPT
   in the idomain passed to this method.
2) The idomain/packages synchronization step is carried out when regridding a
   model, and when regridding a model it will use default methods for all the
   packages. So if you have regridded some packages yourself with non-default
   methods, then these are not taken into account during this synchonization
   step. 

Regridding using default methods
================================

The regrid_like function is available on packages, models and simulations.
When the default methods are acceptable, regridding the whole simulation is the most convenient 
from a user-perspective. 

Regridding using non-default methods
====================================
When non-default methods are used for one or more packages,  these should be regridded separately.
In that case, the most convenient approach is likely:
-pop the packages that should use non-default methods from the source simulation 
(except if it are packages that are required for the simulation to be regridded)
-regrid the source simulation: this takes care of all the packages that should use default methods. 
-regrid the package(s) where you want to use non-standard rergridding methods indivudually
starting from the packages in the source simulation
-insert the custom-regridded packages to the regridded simulation

in code, consider an example where we want to regrid the leakage package using non default methods:
then we would do the following:

# remove the leakage package from the model, and obtain it as a variable
original_leakage_package = original_simulation["GWF_1"].pop("leakage") 

# regrid the simulation(without leakage)
regridded_simulation = original_simulation.regrid_like("regridded!", new_grid)

# regrid the leakage package
regridder_types = {"k": (RegridderType.CENTROIDLOCATOR, None)}
old_grid = original_leakage_package ["head"]  # just take any array of the package to use as the old grid
regrid_context = RegridderWeightsCache(original_leakage_package["k"], new_grid)
regridded_leakage = original_leakage_package.regrid_like(new_grid)
    new_grid,
    regrid_context=regrid_context,
    regridder_types=regridder_types,
)

# add the leakage package to the regridded simulation
regridded_simulation["GWF_1]["leak"] = regridded_leakage



Regridding boundary conditions
============================== 
Special care must be taken when regridding boundary conditions, and it is
recommended that users verify the balance output of a regridded simulation and
compare it to the original model. If the regridded simulation is a good
representation of the original simulation, the mass contributions on the balance
by the different boundary conditions should be comparable in both simulations.
To achieve this, it may be nevessary to tweak the input or the regridding
methods. An example of this is upscaling recharge (so the target grid has
coarser cells than the source grid). Its default method is averaging, with the
following rules:
- if a cell in the source grid is inactive in the source recharge package
  (meaning no recharge), it will not count when averaging. So if a target cell
  has partial overlap with one source recharge cell, and the rest of the target
  cell has no overlap with any source active recharge cell, it will get the
  recharge of the one cell it has overlap with. But since the target cell is
  larger, this effectively means the regridded recharge will be more in the
  regridded simulation than it was in the source simulation
- but we do the same regridding this time assigning a zero recharge to cells
  without recharge then the averaging will take the zero-recharge cells into
  account and the regridded recharge will be the same as the source recharge.

We created this code snippet to print all default methods
"""

from imod.tests.fixtures.package_instance_creation import ALL_PACKAGE_INSTANCES
from pandas import DataFrame

regrid_method_setup = {'package name': [], 'array name ': [] , "method name": [], "function name": []}
regrid_method_table = DataFrame(regrid_method_setup)

counter = 0
for pkg in ALL_PACKAGE_INSTANCES:
   if hasattr(pkg, "_regrid_method"):
      package_name = type(pkg).__name__
      regrid_methods = pkg._regrid_method
      for array_name in regrid_methods.keys():
         method_name = regrid_methods[array_name][0].name
         function_name = ""
         if len( regrid_methods[array_name]) > 0:
            function_name = regrid_methods[array_name][1]
         regrid_method_table.loc[counter] = (package_name,array_name,method_name, function_name)
         counter = counter + 1 
print(regrid_method_table.to_string())

