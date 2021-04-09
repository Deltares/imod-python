import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from mpl_toolkits.axes_grid1 import make_axes_locatable

import imod
from imod.visualize import common


def _meshcoords(da, continuous=True):
    """
    Generate coordinates for pcolormesh, or fill_between

    Parameters
    ----------
    da : xr.DataArray
        The array to plot
    continuous : bool, optional
        Whether the layers are connected, such that the bottom of layer N is
        the top of layer N + 1.

    Returns
    -------
    X : np.array
        x coordinates of mesh
    Y : np.array
        y coordinates of mesh
    C : np.array
        values of the mesh
    """
    dims = tuple(da.dims)
    if not dims[0] == "layer":
        # Switch 'm around
        dims = (dims[1], dims[0])

    # Ensure dimensions are in the right order
    da = da.transpose(*dims, transpose_coords=True)
    data = da.values
    xcoord = da.dims[1]
    dx, xmin, xmax = imod.util.coord_reference(da[xcoord])
    if isinstance(dx, (int, float)):
        if dx < 0.0:
            dx = abs(dx)
            data = data[:, ::-1]
        X = np.arange(xmin + dx, xmax, dx)
    else:  # assuming dx is an array, non-equidistant dx
        if (dx < 0.0).all():
            dx = abs(dx)
            data = data[:, ::-1]
        elif (dx > 0.0).all():
            pass
        else:
            raise ValueError(f"{xcoord} is not monotonic")
        X = (xmin + dx[0]) + dx[1:].cumsum()

    # if dimensions of top and bottom are 1D (voxels), promote to 2D
    if len(da["top"].dims) == 1 and len(da["bottom"].dims) == 1:
        top = np.repeat(np.atleast_2d(da.top), len(da[xcoord]), axis=0).T
        da = da.assign_coords(top=(("layer", xcoord), top))
        bot = np.repeat(np.atleast_2d(da.bottom), len(da[xcoord]), axis=0).T
        da = da.assign_coords(bottom=(("layer", xcoord), bot))

    if continuous:
        Y = np.vstack([da["top"].isel(layer=0).values, da["bottom"].values])
    else:
        nrow, ncol = da["top"].shape
        Y = np.empty((nrow * 2, ncol))
        Y[0::2] = da["top"].values
        Y[1::2] = da["bottom"].values

    Y = np.repeat(Y, 2, 1)
    nodata = np.isnan(Y)
    Y[nodata] = 0.0

    X = np.hstack([xmin, np.repeat(X, 2), xmax])
    X = np.full_like(Y, X)

    if continuous:
        nrow, ncol = Y.shape
        C = np.full((nrow - 1, ncol - 1), np.nan)
        C[:, 0::2] = data
    else:
        _, ncol = Y.shape
        C = np.full((nrow, ncol - 1), np.nan)
        C[:, 0::2] = data

    return X, Y, C, nodata


def _plot_aquitards(aquitards, ax, kwargs_aquitards):
    """
    Overlay aquitards on ax

    Parameters
    ----------
    aquitards : xr.DataArray
        DataArray containing location of aquitards
        NaN's and zeros are treated as locations without aquitard
    ax : matplotlib.Axes object
    kwargs_aquitards : dict
        keyword arguments for ax.fill_between()
    """
    if kwargs_aquitards is None:
        kwargs_aquitards = {"alpha": 0.5, "facecolor": "grey"}
    X_aq, Y_aq, C_aq, _ = _meshcoords(aquitards, continuous=False)
    C_aq.astype(np.float)
    for j, i in enumerate(range(0, X_aq.shape[0] - 1, 2)):
        Y_i = Y_aq[i : i + 2]
        C_i = C_aq[j]
        C_i[C_i == 0.0] = np.nan
        nodata = np.repeat(np.isnan(C_i[0::2]), 2)
        Y_i[:, nodata] = np.nan
        ax.fill_between(X_aq[0], Y_i[0], Y_i[1], **kwargs_aquitards)


def cross_section(
    da,
    colors,
    levels,
    layers=False,
    aquitards=None,
    kwargs_pcolormesh={},
    kwargs_colorbar={},
    kwargs_aquitards=None,
    return_cmap_norm=False,
    fig=None,
    ax=None,
):
    """
    Wraps matplotlib.pcolormesh to draw cross-sections, drawing cell boundaries
    accurately. Aquitards can be plotted on top of the cross-section, by providing
    a DataArray with the aquitard location for `aquitards`.

    Parameters
    ----------
    da : xr.DataArray
        Two dimensional DataArray containing data of the cross section. One
        dimension must be "layer", and the second dimension will be used as the
        x-axis for the cross-section.

        Coordinates "top" and "bottom" must be present, and must have at least the
        "layer" dimension (voxels) or both the "layer" and x-coordinate dimension.

        *Use imod.select.cross_section_line() or cross_section_linestring() to obtain
        the required DataArray.*
    colors : list of str, or list of RGB tuples
        Matplotlib acceptable list of colors. Length N.
        Accepts both tuples of (R, G, B) and hexidecimal (e.g. "#7ec0ee").

        Looking for good colormaps? Try: http://colorbrewer2.org/
        Choose a colormap, and use the HEX JS array.
    levels : listlike of floats or integers
        Boundaries between the legend colors/classes. Length: N - 1.
    layers : boolean, optional
        Whether to draw lines separating the layers.
    aquitards : xr.DataArray, optional
        Datarray containing data on location of aquitard layers.
    kwargs_pcolormesh : dict
        Other optional keyword arguments for matplotlib.pcolormesh.
    kwargs_colorbar : dict
        Optional keyword argument ``whiten_triangles`` whitens respective colorbar triangle if
        data is not larger/smaller than legend_levels-range. Defaults to True.
        Other arguments are forwarded to fig.colorbar()
    kwargs_aquitards: dict
        These arguments are forwarded to matplotlib.fill_between to draw the
        aquitards.
    return_cmap_norm : boolean, optional
        Return the cmap and norm of the plot, default False
    fig : matplotlib Figure instance, optional
        Figure to write plot to. If not supplied, a Figure instance is created
    ax : matplotlib Axes instance, optional
        Axes to write plot to. If not supplied, an Axes instance is created

    Returns
    -------
    fig : matplotlib.figure
    ax : matplotlig.ax
    if return_cmap_norm == True:
    cmap : matplotlib.colors.ListedColormap
    norm : matplotlib.colors.BoundaryNorm

    Examples
    --------

    Basic cross section:

    >>> imod.visualize.cross_section(da, colors, levels)

    Aquitards can be styled in multiple ways. For a transparent grey overlay
    (the default):

    >>> kwargs_aquitards = {"alpha": 0.5, "facecolor": "grey"}
    >>> imod.visualize.cross_section(da, colors, levels, aquitards=aquitards, kwargs_aquitards)

    For a hatched overlay:

    >>> kwargs_aquitards = {"hatch": "/", "edgecolor": "k"}
    >>> imod.visualize.cross_section(da, colors, levels, aquitards=aquitards, kwargs_aquitards)
    """
    da = da.copy(deep=False)
    if aquitards is not None:
        aquitards = aquitards.copy(deep=False)

    if len(da.dims) != 2:
        raise ValueError("DataArray must be 2D")
    if "layer" not in da.dims:
        raise ValueError('DataArray must contain dimension "layer"')
    if "top" not in da.coords:
        raise ValueError('DataArray must contain coordinate "top"')
    if "bottom" not in da.coords:
        raise ValueError('DataArray must contain coordinate "bottom"')
    if len(da["top"].dims) > 2:
        raise ValueError('"top" coordinate be 1D or 2D')
    if len(da["bottom"].dims) > 2:
        raise ValueError('"bottom" coordinate be 1D or 2D')

    # Read legend settings
    cmap, norm = common._cmapnorm_from_colorslevels(colors, levels)

    # cbar kwargs
    settings_cbar = {"ticks": levels, "extend": "both"}

    # Find a unit in the raster to use in the colorbar label
    try:
        settings_cbar["label"] = da.attrs["units"]
    except (KeyError, AttributeError):
        try:
            settings_cbar["label"] = da.attrs["unit"]
        except (KeyError, AttributeError):
            pass

    whiten_triangles = True
    if kwargs_colorbar is not None:
        whiten_triangles = kwargs_colorbar.pop("whiten_triangles", True)
        settings_cbar.update(kwargs_colorbar)

    # pcmesh kwargs
    settings_pcmesh = {"cmap": cmap, "norm": norm}
    if kwargs_pcolormesh is not None:
        settings_pcmesh.update(kwargs_pcolormesh)

    if fig is None or ax is None:
        fig, ax = plt.subplots()

    # Plot raster
    X, Y, C, nodata = _meshcoords(da, continuous=True)
    ax1 = ax.pcolormesh(X, Y, C, **settings_pcmesh)
    # Plot aquitards if applicable
    if aquitards is not None:
        _plot_aquitards(aquitards, ax, kwargs_aquitards)

    if layers:
        Y[nodata] = np.nan
        for y in Y:
            ax.step(x=X[0], y=y)

    # Make triangles white if data is not larger/smaller than legend_levels-range
    if whiten_triangles:
        if float(da.max().compute()) < levels[-1]:
            ax1.cmap.set_over("#FFFFFF")
        if float(da.min().compute()) > levels[0]:
            ax1.cmap.set_under("#FFFFFF")

    # Add colorbar
    divider = make_axes_locatable(ax)
    cbar_ax = divider.append_axes("right", size="5%", pad="5%")
    fig.colorbar(ax1, cax=cbar_ax, **settings_cbar)

    if not return_cmap_norm:
        return fig, ax
    else:
        return fig, ax, cmap, norm


def streamfunction(da, ax, n_streamlines=10, kwargs_contour={}):
    """
    Wraps matplotlib.contour to draw stream lines. Function can be used to draw
    stream lines on top of a cross-section.

    Parameters
    ----------
    da : xr.DataArray
        Two dimensional DataArray containing data of the cross section. One
        dimension must be "layer", and the second dimension will be used as the
        x-axis for the cross-section.

        Coordinates "top" and "bottom" must be present, and must have at least the
        "layer" dimension (voxels) or both the "layer" and x-coordinate dimension.

        *Use imod.evaluate.streamfunction_line() or streamfunction_linestring() to obtain
        the required DataArray.*
    ax : matplotlib Axes instance
        Axes to write plot to.
    n_streamlines : int or array_like
        Determines the number and positions of the contour lines / regions.

        If an int n, use n data intervals; i.e. draw n+1 contour lines. The level heights are automatically chosen.

        If array-like, draw contour lines at the specified levels. The values must be in increasing order.
    kwargs_contour : dict
        Other optional keyword arguments for matplotlib.contour.

    Returns
    -------
    matplotlib.contour.QuadContourSet
        The drawn contour lines.
    """

    da = da.copy(deep=False)
    if len(da.dims) != 2:
        raise ValueError("DataArray must be 2D")
    if "layer" not in da.dims:
        raise ValueError('DataArray must contain dimension "layer"')
    if "top" not in da.coords:
        raise ValueError('DataArray must contain coordinate "top"')
    if "bottom" not in da.coords:
        raise ValueError('DataArray must contain coordinate "bottom"')
    if len(da["top"].dims) > 2:
        raise ValueError('"top" coordinate be 1D or 2D')
    if len(da["bottom"].dims) > 2:
        raise ValueError('"bottom" coordinate be 1D or 2D')

    # add a row of zeros at the bottom, with thickness zero
    # moved here i/o imod.evaluate to not bother user with
    # additional layer hampering coord assignment
    zeros = xr.zeros_like(da.isel(layer=-1))
    zeros.coords["layer"] += 1
    zeros.coords["top"] = zeros.coords["bottom"]  # layer of zero thickness
    zeros.coords["bottom"] = zeros.coords["bottom"] - 0.1
    da = xr.concat((da, zeros), dim="layer")

    # _meshcoords returns mesh edges, but useful here to allow 1D/2D
    # dimensions in da
    # go back to cell centers in 2D
    X = np.vstack(da.shape[0] * [da.s])
    Y = 0.5 * da.top + 0.5 * da.bottom
    if Y.ndim == 1:
        # promote to 2D
        Y2 = xr.concat(da.shape[1] * [Y], dim=da.s)

    settings_contour = {
        "colors": "w",
        "linestyles": "solid",
        "linewidths": 0.8,
        "zorder": 10,
    }
    settings_contour.update(kwargs_contour)

    CS = ax.contour(X, Y, da.values, n_streamlines, **settings_contour)
    return CS


def quiver(u, v, ax, kwargs_quiver={}):
    """
    Wraps matplotlib.quiver to draw quiver plots. Function can be used to draw
    flow quivers on top of a cross-section.

    Parameters
    ----------
    u : xr.DataArray
        Two dimensional DataArray containing u component of quivers. One
        dimension must be "layer", and the second dimension will be used as the
        x-axis for the cross-section.

        Coordinates "top" and "bottom" must be present, and must have at least the
        "layer" dimension (voxels) or both the "layer" and x-coordinate dimension.

        *Use imod.evaluate.quiver_line() or quiver_linestring() to obtain
        the required DataArray.*
    v : xr.DataArray
        Two dimensional DataArray containing v component of quivers. One
        dimension must be "layer", and the second dimension will be used as the
        x-axis for the cross-section.

        Coordinates "top" and "bottom" must be present, and must have at least the
        "layer" dimension (voxels) or both the "layer" and x-coordinate dimension.

        *Use imod.evaluate.quiver_line() or quiver_linestring() to obtain
        the required DataArray.*
    ax : matplotlib Axes instance
        Axes to write plot to.
    kwargs_quiver : dict
        Other optional keyword arguments for matplotlib.quiver.

    Returns
    -------
    matplotlib.quiver.Quiver
        The drawn quivers.

    Examples
    --------

    First: apply evaluate.quiver_line to get the u and v components of the quivers
    from a three dimensional flow field. Assign top and bottom coordinates if these are
    not already present in the flow field data arrays.

    >>> u, v = imod.evaluate.quiver_line(right, front, lower, start, end)
    >>> u.assign_coords(top=top, bottom=bottom)
    >>> v.assign_coords(top=top, bottom=bottom)

    The quivers can then be plotted over a cross section created by imod.visualize.cross_section():

    >>> imod.visualize.quiver(u, v, ax)

    Quivers can easily overwhelm your plot, so it is a good idea to 'thin out' some of the quivers:

    >>> # Only plot quivers at every 5th cell and every 3rd layer
    >>> thin = {"s": slice(0, None, 5), "layer": slice(0, None, 3)}
    >>> imod.visualize.quiver(u.isel(**thin), v.isel(**thin), ax)
    """

    for da, name in zip([u, v], ["u", "v"]):
        if len(da.dims) != 2:
            raise ValueError(f"DataArray {name} must be 2D")
        if "layer" not in da.dims:
            raise ValueError(f'DataArray {name} must contain dimension "layer"')
        if "top" not in da.coords:
            raise ValueError(f'DataArray {name} must contain coordinate "top"')
        if "bottom" not in da.coords:
            raise ValueError(f'DataArray {name} must contain coordinate "bottom"')
        if len(da["top"].dims) > 2:
            raise ValueError(f'"top" coordinate in dataarray {name} must be 1D or 2D')
        if len(da["bottom"].dims) > 2:
            raise ValueError(
                f'"bottom" coordinate in dataarray {name} must be 1D or 2D'
            )

    # do not draw quivers for cells that are only thinly sliced (ds > 25% of cellsize)
    dsmin = 0.25 * u.dx
    u = u.where(u.ds > dsmin)
    v = v.where(v.ds > dsmin)

    # _meshcoords returns mesh edges, but useful here to allow 1D/2D
    # dimensions in da
    # go back to cell centers in 2D
    X, _ = np.meshgrid(u.s.values, u.layer.values)
    Y = 0.5 * u.top + 0.5 * u.bottom
    if Y.ndim == 1:
        # promote to 2D
        Y = xr.concat(u.shape[1] * [Y], dim=u.s)

    settings_quiver = {
        "color": "w",
        "scale": None,  # autoscaling
        "linestyle": "-",
        "zorder": 10,
    }
    settings_quiver.update(kwargs_quiver)

    Q = ax.quiver(X, Y, u.values, v.values, **settings_quiver)
    return Q
