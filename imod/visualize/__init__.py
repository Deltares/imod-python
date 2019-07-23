"""
Module to provide functions to create customized plots.

These functions build on existing libraries, and merely serve as a shorthand for
plots that may be useful to evaluate groundwater models. All
``xarray.DataArray`` and ``pandas.DataFrame`` objects have ``.plot()`` or
``.imshow()`` methods to plot them directly.
"""

from imod.visualize.spatial import plot_map, read_imod_legend
