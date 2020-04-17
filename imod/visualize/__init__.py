"""
Module to provide functions to create customized plots.

These functions build on existing libraries, and merely serve as a shorthand for
plots that may be useful to evaluate groundwater models. All
``xarray.DataArray`` and ``pandas.DataFrame`` objects have ``.plot()`` or
``.imshow()`` methods to plot them directly.
"""

from imod.visualize.cross_sections import cross_section
from imod.visualize.pyvista import (GridAnimation3D, StaticGridAnimation3D,
                                    grid_3d, line_3d)
from imod.visualize.spatial import imshow_topview, plot_map, read_imod_legend
from imod.visualize.waterbalance import waterbalance_barchart
