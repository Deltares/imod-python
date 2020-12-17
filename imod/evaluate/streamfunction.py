import numpy as np
import xarray as xr

import imod


def _streamfunction(sfrf, sfff, dx, dy):
    # Scale total flux to length of cell traversed
    # --> no, this gives really akward patterns
    scale_factor = (
        1.0  # (sfrf.dx ** 2 + sfff.dy ** 2) ** 0.5 / (dx ** 2 + dy ** 2) ** 0.5
    )
    sfrfi = sfrf * -scale_factor  # FRF is negative eastward, so flip
    sfffi = sfff * scale_factor  # FFF is positive northward

    # calculate projection of flux onto cross section line
    # Fi_proj = (Fi dotproduct Si) / |Si|
    # keep sign
    Fi_dot_si = sfrfi * sfrf.dx + sfffi * sfff.dy
    Fi_proj = Fi_dot_si / Fi_dot_si.ds

    # calculate stream function by cumsumming from bottom upwards
    # method taken from https://olsthoorn.readthedocs.io/en/latest/07_stream_lines.html
    layindex = Fi_proj.layer.values
    Fi_proj = (
        Fi_proj.reindex(layer=layindex[::-1])
        .cumsum(dim="layer")
        .reindex(layer=layindex)
    )
    return Fi_proj


def streamfunction_line(frf, fff, start, end):
    """Obtain the streamfunction for a line cross section through
    a three-dimensional flow field. The streamfunction is obtained
    by first projecting the horizontal flow components onto the provided
    cross-section. The streamfunction can be contoured to visualize stream lines.
    Stream lines are an efficient way to visualize groundwater flow.

    Note, however, that the streamfunction is only defined in 2D, non-diverging,
    steady-state flow without sources and sinks. These assumption are violated even
    in a 2D model, but even more so in the approach followed here. Flow perpendicular
    to the cross-section will not be visualized. It is up to the user to choose
    cross-sections as perpendicular to the main flow direction as possible.

    The 2D streamfunction and stream line visualization is based on work of Em. Prof. Olsthoorn.

    Parameters
    ----------
    frf: `xarray.DataArray`
        Three- (or higher) dimensional dataarray of flow component along the rows (FLOW RIGHT FACE).
    fff: `xarray.DataArray`
        Three- (or higher) dimensional dataarray of flow component along the columns (FLOW FRONT FACE).
    start: (2, ) array_like
        A latitude-longitude pair designating the start point of the cross
        section.
    end: (2, ) array_like
        A latitude-longitude pair designating the end point of the cross
        section.

    Returns
    -------
    `xarray.DataArray`
        The streamfunction projected on the cross-section between start and end coordinate,
        with new dimension "s" along the cross-section. The cellsizes along "s" are given in
        the "ds" coordinate.
    """
    # interpolate frf and fff to cell center
    frf = 0.5 * frf + 0.5 * frf.shift({"x": -1})
    fff = 0.5 * fff + 0.5 * fff.shift({"y": -1})
    # get cross section of frf and fff
    sfrf = imod.select.cross_section_line(frf, start, end)
    sfff = imod.select.cross_section_line(fff, start, end)
    return _streamfunction(sfrf, sfff, frf.dx, fff.dy)


def streamfunction_linestring(frf, fff, linestring):
    """Obtain the streamfunction for a linestring cross section through
    a three-dimensional flow field. The streamfunction is obtained
    by first projecting the horizontal flow components onto the provided
    cross-section. The streamfunction can be contoured to visualize stream lines.
    Stream lines are an efficient way to visualize groundwater flow.

    Note, however, that the streamfunction is only defined in 2D, non-diverging,
    steady-state flow without sources and sinks. These assumption are violated even
    in a 2D model, but even more so in the approach followed here. Flow perpendicular
    to the cross-section will not be visualized. It is up to the user to choose
    cross-sections as perpendicular to the main flow direction as possible.

    The 2D streamfunction and stream line visualization is based on work of Em. Prof. Olsthoorn.

    Parameters
    ----------
    frf: `xarray.DataArray`
        Three- (or higher) dimensional dataarray of flow component along the rows (FLOW RIGHT FACE).
    fff: `xarray.DataArray`
        Three- (or higher) dimensional dataarray of flow component along the columns (FLOW FRONT FACE).

    linestring : shapely.geometry.LineString
        Shapely geometry designating the linestring along which to sample the
        cross section.

    Returns
    -------
    `xarray.DataArray`
        The streamfunction projected on the cross-section defined by provided linestring,
        with new dimension "s" along the cross-section. The cellsizes along "s" are given in
        the "ds" coordinate.
    """

    # interpolate frf and fff to cell center
    frf = 0.5 * frf + 0.5 * frf.shift({"x": -1})
    fff = 0.5 * fff + 0.5 * fff.shift({"y": -1})
    # get cross section of frf and fff
    sfrf = imod.select.cross_section_linestring(frf, linestring)
    sfff = imod.select.cross_section_linestring(fff, linestring)
    return _streamfunction(sfrf, sfff, frf.dx, fff.dy)


def _quiver(sfrf, sfff, sflf):  # , dx, dy, dz):
    # Scale total flux to length of cell traversed
    # --> no, this gives really akward patterns -> also for quivers? here it might seem like a good idea
    scale_factor = (
        1.0  # (sfrf.dx ** 2 + sfff.dy ** 2) ** 0.5 / (dx ** 2 + dy ** 2) ** 0.5
    )
    sfrfi = sfrf * -scale_factor  # FRF is negative eastward, so flip
    sfffi = sfff * scale_factor  # FFF is positive northward
    sflfi = sflf * scale_factor  # FLF is positive upward

    # calculate projection of flux onto cross section plane
    # cross section plane is given by? u: <dx, dy, 0>, v: <0, 0, 1>
    # Fi_proj_u = (Fi dotproduct Si) / |Si| * (Si / |Si|)
    # unitvector not necessary: return u and v as scalars
    # keep sign
    Fi_dot_u = sfrfi * sfrf.dx + sfffi * sfff.dy  # + sflfi * 0.
    Fi_proj_u = Fi_dot_u / Fi_dot_u.ds
    Fi_proj_v = sflfi  # no projection necessary for v

    return Fi_proj_u, Fi_proj_v


def quiver_line(frf, fff, flf, start, end):
    """Obtain the u and v components for quiver plots for a line cross section
    through a three-dimensional flux field. The u and v components are obtained
    by first projecting the threedimensional flux components onto the provided
    cross-section.

    Parameters
    ----------
    frf: `xarray.DataArray`
        Three- (or higher) dimensional dataarray of flow component along the rows (FLOW RIGHT FACE).
    fff: `xarray.DataArray`
        Three- (or higher) dimensional dataarray of flow component along the columns (FLOW FRONT FACE).
    flf: `xarray.DataArray`
        Three- (or higher) dimensional dataarray of flow component along the layers (FLOW LOWER FACE).
    start: (2, ) array_like
        A latitude-longitude pair designating the start point of the cross
        section.
    end: (2, ) array_like
        A latitude-longitude pair designating the end point of the cross
        section.

    Returns
    -------
    u: `xarray.DataArray`
        The u component (x-component) of the flow projection on the cross-section between start and end coordinate,
        with new dimension "s" along the cross-section. The cellsizes along "s" are given in
        the "ds" coordinate.
    v: `xarray.DataArray`
        The v component (y-component) of the flow projection on the cross-section between start and end coordinate,
        with new dimension "s" along the cross-section. The cellsizes along "s" are given in
        the "ds" coordinate.

    Notes
    -----
    Use imod.evaluate.flow_velocity() first to obtain groundwater velocities
    as input for this function. Velocity in x direction is, however, inverted and must
    be re-inverted before using as input here.
    """
    # interpolate frf and fff to cell center
    frf = 0.5 * frf + 0.5 * frf.shift({"x": -1})
    fff = 0.5 * fff + 0.5 * fff.shift({"y": -1})
    # get cross section of frf, fff and flf
    sfrf = imod.select.cross_section_line(frf, start, end)
    sfff = imod.select.cross_section_line(fff, start, end)
    sflf = imod.select.cross_section_line(flf, start, end)
    return _quiver(sfrf, sfff, sflf)


def quiver_linestring(frf, fff, flf, linestring):
    """Obtain the u and v components for quiver plots for a linestring cross section
    through a three-dimensional flow field. The u and v components are obtained
    by first projecting the threedimensional flow components onto the provided
    cross-section.

    Parameters
    ----------
    frf: `xarray.DataArray`
        Three- (or higher) dimensional dataarray of flow component along the rows (FLOW RIGHT FACE).
    fff: `xarray.DataArray`
        Three- (or higher) dimensional dataarray of flow component along the columns (FLOW FRONT FACE).
    flf: `xarray.DataArray`
        Three- (or higher) dimensional dataarray of flow component along the layers (FLOW LOWER FACE).
    linestring : shapely.geometry.LineString
        Shapely geometry designating the linestring along which to sample the
        cross section.

    Returns
    -------
    u: `xarray.DataArray`
        The u component (x-component) of the flow projection on the cross-section between start and end coordinate,
        with new dimension "s" along the cross-section. The cellsizes along "s" are given in
        the "ds" coordinate.
    v: `xarray.DataArray`
        The v component (y-component) of the flow projection on the cross-section between start and end coordinate,
        with new dimension "s" along the cross-section. The cellsizes along "s" are given in
        the "ds" coordinate.

    Notes
    -----
    Use imod.evaluate.flow_velocity() first to obtain groundwater velocities
    as input for this function. Velocity in x direction is, however, inverted and must
    be re-inverted before using as input here.
    """
    # interpolate frf and fff to cell center
    frf = 0.5 * frf + 0.5 * frf.shift({"x": -1})
    fff = 0.5 * fff + 0.5 * fff.shift({"y": -1})
    # get cross section of frf, fff and flf
    sfrf = imod.select.cross_section_linestring(frf, linestring)
    sfff = imod.select.cross_section_linestring(fff, linestring)
    sflf = imod.select.cross_section_linestring(flf, linestring)
    return _quiver(sfrf, sfff, sflf)
