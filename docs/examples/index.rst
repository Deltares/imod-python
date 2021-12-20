Examples
========

The imod package can be used in many ways. By and large, imod **enables**
functionality rather than implementing it. Geopandas, pandas, and xarray
provide most of the functionality; imod provides the link to the MODFLOW
specific file formats.

Sometimes, commonly used functionality (in groundwater modeling) is not
available in existing packages, or is insufficiently fast or convenient. For
these situations, imod provides a number of additional functions. These
functions revolve around the same principal data structures (geopandas, pandas,
xarray) and rarely introduce new data structures.

The examples below follow this rationale: a great deal of demonstrated
funtionality is e.g. "vanilla" xarray functionaly, but used for the goal of
convenient groundwater modeling.

.. toctree::
   :titlesonly:
   :hidden:
   
   mf6/index.rst
   imodflow/index.rst
   imod-wq/index.rst
